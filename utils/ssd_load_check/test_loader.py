"""Correctness and bandwidth of the connector's SSD layerwise loader.

    python3 test_loader.py check <c_ops .so> [gpu] [data dir]
    python3 test_loader.py check-evict <c_ops .so> [gpu] [data dir]
    python3 test_loader.py bench <c_ops .so> [gpus, e.g. 6,7] [data root]
    python3 test_loader.py bench-wb <c_ops .so> [gpus] [data root]
    python3 test_loader.py bench-dram <c_ops .so> [gpus]

bench-wb: bench with every SSD value also copied to a write-back buffer (the
SSD-hit write-back to DRAM). Its load time against bench's is what the copies
cost the load; wb_tail is how long after the last layer they finish.

bench-dram: the same, every value in cudaHostRegister'ed memory (a Mooncake
DRAM replica), loaded by the DRAM LayerwiseLoader with all layers enqueued at
once: the H2D time with no compute to pace it.

check: values written as Mooncake bucket records ([key bytes][value], records
in random order, so values sit at unaligned offsets; the last one ends at end
of file), a mix of SSD values and cudaHostRegister'ed host values, loaded with
start_layers_ssd + step_layer_ssd on a separate compute stream, compared byte
for byte with the old path (copy_ into a GPU buffer + single_layer_kv_transfer).
The same loader then serves the next case, a missing file and a truncated one.
Cases marked wb also pass write-back buffers for their SSD values and check
them byte for byte, and that a failed load still releases wait_writeback.

check-evict: exclusive unblocking. The paged KV holds old KV, some of the
loaded blocks are lent: start_evict copies their old KV to replicas
(cudaHostRegister'ed) while the load overwrites them, each layer's scatter
waiting for that layer's copy. Checks the replicas and the paged KV byte for
byte (DRAM and SSD loaders), then times the DRAM load with 0 to all of its
blocks lent.

bench: every value on SSD, files fsync'ed so the O_DIRECT reads go to disk. One
process per GPU (one per TP rank); all of them write their files first, then
load at the same moment behind a barrier, as the two ranks of a prefill do.
Prints each rank's load time (start -> last layer on the GPU) and the combined
SSD bandwidth. The loader has 64 I/O threads and 256 MiB of slots (gpu_connectors.py).
"""
import importlib.util
import multiprocessing as mp
import os
import random
import statistics
import sys
import threading
import time

import numpy as np
import torch

L, NH, HS, BS, CH = 48, 2, 128, 512, 512
TOK = 2 * NH * HS * 2  # bytes per token per layer, bf16
dt = torch.bfloat16
BENCH_TOKENS = (4096, 8192, 16384, 32768, 65536)
REPS = 3
ops = CONN = COMPUTE = FMT = cudart = None
DATA = ""


def setup(so_path, gpu, data):
    global ops, CONN, COMPUTE, FMT, cudart, DATA
    spec = importlib.util.spec_from_file_location("lmcache.c_ops", so_path)
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    sys.modules["lmcache.c_ops"] = ops
    from lmcache.v1.gpu_connector import gpu_connectors as gc

    gc.lmc_ops = ops
    torch.cuda.set_device(gpu)
    cudart = torch.cuda.cudart()
    DATA = data
    FMT = ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
    COMPUTE = torch.cuda.Stream()
    CONN = gc.VLLMPagedMemLayerwiseGPUConnector.__new__(gc.VLLMPagedMemLayerwiseGPUConnector)
    CONN.hidden_dim_size, CONN.num_layers, CONN.dtype = NH * HS, L, dt
    CONN.device, CONN.element_size, CONN.use_mla = torch.device("cuda"), 2, False
    CONN.load_stream, CONN.gpu_kv_format = torch.cuda.Stream(), FMT
    CONN.initialize_kvcaches_ptr = lambda **kw: None
    CONN.use_gpu, CONN.gpu_buffer_allocator = False, None  # format set above
    CONN._ssd, CONN._ssd_lock = None, threading.Lock()
    CONN.evict_stream, CONN._evictor = torch.cuda.Stream(), None


def write_buckets(values, keys_per_file, tag):
    """values: {(l, c): bytes}. Returns {(l, c): (path, value offset, size)}."""
    os.makedirs(DATA, exist_ok=True)
    items = list(values.items())
    random.shuffle(items)
    locs = {}
    for f in range(0, len(items), keys_per_file):
        path = os.path.join(DATA, f"{tag}_{f // keys_per_file}.bucket")
        off = 0
        with open(path, "wb") as fh:
            for (l, c), data in items[f : f + keys_per_file]:
                key = os.urandom(random.randint(60, 200))
                fh.write(key)
                fh.write(data)
                locs[(l, c)] = (path, off + len(key), len(data))
                off += len(key) + len(data)
            fh.flush()
            os.fsync(fh.fileno())
    return locs


class Case:
    """One request's data, written and registered, ready to load."""

    def __init__(self, n_tokens, skip, ssd_frac, tag, check=True, break_with=None,
                 writeback=False, late=False):
        self.late = late
        self.keep = []
        nb = (n_tokens + BS - 1) // BS * 2 + 4
        self.shape = (2, nb, BS, NH, HS)
        self.kv = [torch.zeros(self.shape, dtype=dt, device="cuda") for _ in range(L)]
        self.slots = torch.randperm(nb * BS, device="cuda")[:n_tokens].to(torch.int64)
        first = skip // CH
        self.start, self.end = first * CH, n_tokens
        self.sizes = [min(CH, n_tokens - s) for s in range(self.start, self.end, CH)]
        nc = len(self.sizes)
        rng = np.random.default_rng(random.getrandbits(32))
        shared = None if check else rng.integers(0, 256, CH * TOK, dtype=np.uint8)
        # The stored value may be longer than what is loaded (a full chunk).
        self.data = {(l, c): rng.integers(0, 256, CH * TOK, dtype=np.uint8) if check else shared
                     for l in range(L) for c in range(nc)}
        on_ssd = {k for k in self.data if random.random() < ssd_frac}
        locs = write_buckets({k: self.data[k].tobytes() for k in on_ssd}, L, tag)
        self.paths = sorted({p for p, _, _ in locs.values()})
        pidx = {p: i for i, p in enumerate(self.paths)}
        self.fidx = torch.full((L, nc), -1, dtype=torch.int32)
        self.offs = torch.zeros((L, nc), dtype=torch.int64)
        for (l, c), (p, o, _) in locs.items():
            self.fidx[l, c], self.offs[l, c] = pidx[p], o
        for k in self.data.keys() - on_ssd:
            a = np.empty(CH * TOK, dtype=np.uint8)
            assert int(cudart.cudaHostRegister(a.ctypes.data, a.nbytes, 0)) == 0
            a[:] = self.data[k]
            self.keep.append(a)
            self.offs[k] = a.ctypes.data
        self.ssd_bytes = sum(self.sizes[c] * TOK for (_, c) in on_ssd)
        self.wb, self.wb_bufs, self.wb_ok, self.wb_tail_ms = None, {}, None, 0.0
        if writeback:
            self.wb = torch.zeros((L, nc), dtype=torch.int64)
            for l, c in on_ssd:
                buf = np.zeros(self.sizes[c] * TOK, dtype=np.uint8)
                self.wb_bufs[(l, c)] = buf
                self.wb[l, c] = buf.ctypes.data
        if break_with == "missing" and self.paths:
            os.unlink(self.paths[0])
        if break_with == "truncated" and self.paths:
            os.truncate(self.paths[-1], 4096)

    def load(self):
        """Returns load ms, or the loader's error message."""
        CONN.kvcaches = self.kv
        self.first_ms = 0.0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen = None
        try:
            with torch.cuda.stream(COMPUTE):
                CONN.start_layers_ssd(self.start, self.end, self.sizes, self.paths,
                                      self.fidx, self.offs, False,
                                      writeback=None if self.late else self.wb,
                                      slot_mapping=self.slots)
                gen = CONN.ssd_generation()
                if self.late and self.wb is not None:
                    # Some reads are already under way: only the rest are armed.
                    self.armed = CONN.set_ssd_writeback(gen, self.wb)
                for l in range(L):
                    CONN.step_layer_ssd(l, True)
                    if l == 0:
                        # Layer 0 handed to the GPU: how long the first
                        # attention layer of a forward would wait.
                        self.first_ms = (time.perf_counter() - t0) * 1e3
            t_last = time.perf_counter()
            if self.wb is not None:
                self.wb_ok = CONN.wait_ssd_writeback(gen)
                self.wb_tail_ms = (time.perf_counter() - t_last) * 1e3
            torch.cuda.synchronize()
        except RuntimeError as e:
            if self.wb is not None and gen is not None:
                self.wb_ok = CONN.wait_ssd_writeback(gen)
            torch.cuda.synchronize()
            return str(e).splitlines()[0]
        return (time.perf_counter() - t0) * 1e3

    def load_dram(self):
        """Load every (host) value with the DRAM LayerwiseLoader, all layers
        enqueued at once so the copy runs at its own pace. Returns load ms."""
        CONN.kvcaches = self.kv
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(COMPUTE):
            CONN.start_layers(self.start, self.end, self.sizes, self.offs, L, False,
                              slot_mapping=self.slots)
            for l in range(L):
                CONN.step_layer(l, True)
                if l == 0:
                    self.first_ms = (time.perf_counter() - t0) * 1e3
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3

    def matches(self):
        buf = torch.empty(self.end - self.start, 2, NH * HS, dtype=dt, device="cuda")
        flat = buf.view(-1).view(torch.uint8)
        ok = True
        for l in range(L):
            ref = torch.zeros(self.shape, dtype=dt, device="cuda")
            o = 0
            for c, t in enumerate(self.sizes):
                flat[o : o + t * TOK].copy_(torch.from_numpy(self.data[(l, c)][: t * TOK]))
                o += t * TOK
            ops.single_layer_kv_transfer(buf, ref, self.slots[self.start : self.end],
                                         ops.TransferDirection.H2D, FMT, token_major=True)
            ok &= torch.equal(ref.view(torch.int16), self.kv[l].view(torch.int16))
        if self.wb is not None:
            check = self.wb_bufs.items()
            if self.late:
                check = [(k, b) for k, b in check if self.armed[k[0], k[1]]]
            ok &= bool(self.wb_ok) and all(
                np.array_equal(buf, self.data[k][: buf.size]) for k, buf in check)
        return ok

    def close(self):
        for a in self.keep:
            cudart.cudaHostUnregister(a.ctypes.data)
        for p in self.paths:
            if os.path.exists(p):
                os.unlink(p)


def check():
    fails = 0
    for n, skip, frac, wb in ((512, 0, 1.0, False), (4096, 0, 1.0, False),
                              (4096 + 200, 0, 0.5, False), (8192, 1024, 0.7, False),
                              (16384, 0, 0.05, False), (2048, 0, 0.0, False),
                              (4096, 0, 1.0, True), (4096 + 200, 0, 0.5, True),
                              (8192, 1024, 0.7, True), (16384, 0, 0.05, True)):
        case = Case(n, skip, frac, f"chk{n}_{skip}_{frac}_{wb}", writeback=wb)
        res = case.load()
        ok = not isinstance(res, str) and case.matches()
        case.close()
        print(f"check tokens={n:6d} skip={skip:5d} ssd_frac={frac:.2f} wb={wb} ok={ok} {res}")
        fails += not ok
    for n in (16384, 32768):
        case = Case(n, 0, 1.0, f"late{n}", writeback=True, late=True)
        res = case.load()
        ok = not isinstance(res, str) and case.matches()
        armed = int(case.armed.sum())
        case.close()
        print(f"check late writeback tokens={n:6d} armed {armed}/{len(case.wb_bufs)} "
              f"ok={ok and armed > 0} {res}")
        fails += not (ok and armed > 0)
    for how in ("missing", "truncated"):
        for wb in (False, True):
            case = Case(4096, 0, 1.0, f"bad_{how}_{wb}", break_with=how, writeback=wb)
            res = case.load()
            case.close()
            # A failed load never reports its write-back done (and must not hang;
            # a missing file already fails in start, before any copy).
            ok = isinstance(res, str) and case.wb_ok is not True
            print(f"check {how} wb={wb} ok={ok}: {res}")
            fails += not ok
    case = Case(4096, 0, 1.0, "after_errors")
    res = case.load()
    ok = not isinstance(res, str) and case.matches()
    case.close()
    print(f"check after errors ok={ok}")
    print("FAILS", fails + (not ok))


class EvictCase(Case):
    """A load into whole blocks holding old KV, `n_lent` of them lent: their
    old KV must reach its replicas (cudaHostRegister'ed, as a Mooncake
    segment) before the load overwrites them."""

    def __init__(self, n_tokens, ssd_frac, n_lent, tag, check=True):
        super().__init__(n_tokens, 0, ssd_frac, tag, check=check)
        nb = self.shape[1]
        blocks = torch.randperm(nb)[: (n_tokens + BS - 1) // BS]
        self.slots = (blocks.unsqueeze(1) * BS + torch.arange(BS)).flatten()[:n_tokens]
        self.slots = self.slots.to(torch.int64).cuda()
        for t in self.kv:
            t.copy_(torch.randint(-2**15, 2**15, t.shape, dtype=torch.int16,
                                  device="cuda").view(dt))
        self.kv0 = [t.clone() for t in self.kv]
        lent = blocks[torch.randperm(len(blocks))[:n_lent]]
        self.lent_slots = (lent.unsqueeze(1) * BS + torch.arange(BS)).flatten()
        self.reps = {}
        self.ptrs = torch.zeros((L, n_lent), dtype=torch.int64)
        for l in range(L):
            for c in range(n_lent):
                a = np.zeros(CH * TOK, dtype=np.uint8)
                assert int(cudart.cudaHostRegister(a.ctypes.data, a.nbytes, 0)) == 0
                self.keep.append(a)
                self.reps[(l, c)] = a
                self.ptrs[l, c] = a.ctypes.data

    def run(self, evict=True):
        """Load (DRAM loader if no value is on SSD), lent blocks evicted first
        when `evict`. Returns load ms."""
        CONN.kvcaches = self.kv
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(COMPUTE):
            events, done = (CONN.start_evict(self.lent_slots, self.ptrs) if evict
                            else (None, None))
            if self.paths:
                CONN.start_layers_ssd(self.start, self.end, self.sizes, self.paths,
                                      self.fidx, self.offs, False, evict_events=events,
                                      slot_mapping=self.slots)
                step = CONN.step_layer_ssd
            else:
                CONN.start_layers(self.start, self.end, self.sizes, self.offs, L, False,
                                  evict_events=events, slot_mapping=self.slots)
                step = CONN.step_layer
            for l in range(L):
                step(l, True)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        if done is not None:
            done.synchronize()
        return ms

    def matches_evict(self):
        buf = torch.empty(self.end - self.start, 2, NH * HS, dtype=dt, device="cuda")
        flat = buf.view(-1).view(torch.uint8)
        old = torch.empty(len(self.lent_slots), 2, NH * HS, dtype=dt, device="cuda")
        n_lent = self.ptrs.shape[1]
        ok_kv = ok_rep = True
        for l in range(L):
            ref = self.kv0[l].clone()
            o = 0
            for c, t in enumerate(self.sizes):
                flat[o : o + t * TOK].copy_(torch.from_numpy(self.data[(l, c)][: t * TOK]))
                o += t * TOK
            ops.single_layer_kv_transfer(buf, ref, self.slots[self.start : self.end],
                                         ops.TransferDirection.H2D, FMT, token_major=True)
            ok_kv &= torch.equal(ref.view(torch.int16), self.kv[l].view(torch.int16))
            ops.single_layer_kv_transfer(old, self.kv0[l], self.lent_slots.cuda(),
                                         ops.TransferDirection.D2H, FMT, token_major=True)
            want = old.view(torch.uint8).view(n_lent, -1).cpu().numpy()
            for c in range(n_lent):
                ok_rep &= np.array_equal(self.reps[(l, c)], want[c])
        return ok_kv, ok_rep


def check_evict():
    """Lent blocks: replicas hold their old KV, the paged KV the new load."""
    fails = 0
    for n, frac, lent in ((4096, 0.0, 3), (16384, 0.0, 16), (16384 + 200, 0.0, 1),
                          (4096, 1.0, 3), (16384, 0.5, 16), (8192, 0.3, 8)):
        case = EvictCase(n, frac, lent, f"ev{n}_{frac}_{lent}")
        ms = case.run()
        ok_kv, ok_rep = case.matches_evict()
        case.close()
        print(f"check-evict tokens={n:6d} ssd_frac={frac:.1f} lent={lent:3d} "
              f"kv_ok={ok_kv} replicas_ok={ok_rep} load={ms:.1f}ms")
        fails += not (ok_kv and ok_rep)
    print("FAILS", fails)
    # What evicting costs the load (DRAM loader, all values host-resident):
    # the D2H runs alongside the H2D, the scatter of a layer waits for it.
    for n in (8192, 32768):
        for lent in (0, n // BS // 4, n // BS // 2, n // BS):
            ts = []
            for rep in range(5):
                case = EvictCase(n, 0.0, max(lent, 1), f"evb{n}_{lent}_{rep}", check=False)
                ts.append(case.run(evict=lent > 0))
                case.close()
            print(f"evict-cost tokens={n:6d} lent={lent:3d} blocks "
                  f"load median={statistics.median(ts[1:]):6.2f}ms")


def bench_worker(so_path, gpu, data, barrier, out, tier):
    setup(so_path, gpu, data)
    frac = 0.0 if tier == "dram" else 1.0
    wb = tier == "ssd-wb"
    warm = Case(512, 0, frac, "warmup", writeback=wb)  # threads, pinned arena
    warm.load() if frac else warm.load_dram()
    warm.close()
    for n in BENCH_TOKENS:
        for rep in range(REPS):
            case = Case(n, 0, frac, f"bench{n}_{rep}", check=False, writeback=wb)
            nbytes = sum(case.sizes) * TOK * L
            barrier.wait()
            ms = case.load() if frac else case.load_dram()
            barrier.wait()
            out.put((gpu, n, rep, ms, nbytes, case.first_ms, case.wb_tail_ms))
            case.close()


def bench(so_path, gpus, root, tier="ssd"):
    ctx = mp.get_context("spawn")
    barrier, out = ctx.Barrier(len(gpus)), ctx.Queue()
    procs = [ctx.Process(target=bench_worker,
                         args=(so_path, g, os.path.join(root, f"gpu{g}"), barrier, out,
                               tier))
             for g in gpus]
    for p in procs:
        p.start()
    res = [out.get() for _ in range(len(gpus) * len(BENCH_TOKENS) * REPS)]
    for p in procs:
        p.join()
    print(f"bench tier={tier} gpus={gpus} (median of {REPS})")
    for n in BENCH_TOKENS:
        per_rep = []
        for rep in range(REPS):
            rows = [r for r in res if r[1] == n and r[2] == rep]
            per_rep.append((max(r[3] for r in rows), sum(r[4] for r in rows),
                            max(r[5] for r in rows), max(r[6] for r in rows)))
        ms, nbytes, _, _ = sorted(per_rep)[REPS // 2]
        first = statistics.median(r[2] for r in per_rep)
        tail = statistics.median(r[3] for r in per_rep)
        gib = nbytes / 2**30
        print(f"bench tokens={n:6d} ssd={gib:6.2f}GiB load={ms:8.1f}ms first_layer={first:6.2f}ms "
              f"bw={gib / ms * 1e3:6.2f}GiB/s c={ms / n * 1e3:6.2f}ms/Ktok"
              + (f" wb_tail={tail:5.2f}ms" if tier == "ssd-wb" else ""))


if __name__ == "__main__":
    mode, so = sys.argv[1], sys.argv[2]
    if mode in ("check", "check-evict"):
        gpu = int(sys.argv[3]) if len(sys.argv) > 3 else 7
        setup(so, gpu, sys.argv[4] if len(sys.argv) > 4 else f"/root/ssd_load_check_data/gpu{gpu}")
        check() if mode == "check" else check_evict()
    else:
        gpus = [int(g) for g in (sys.argv[3] if len(sys.argv) > 3 else "6,7").split(",")]
        bench(so, gpus, sys.argv[4] if len(sys.argv) > 4 else "/root/ssd_load_check_data",
              {"bench-dram": "dram", "bench-wb": "ssd-wb"}.get(mode, "ssd"))
