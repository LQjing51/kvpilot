"""Per-GPU host-to-device bandwidth when several GPUs read host DRAM at once.

One process per GPU, like one prefill TP rank: an anonymous host buffer touched
under numactl --membind and then cudaHostRegister'ed (as Mooncake does with its
segment), copied with lmcache c_ops.batched_memcpy_h2d (cudaMemcpyBatchAsync,
the call LayerwiseLoader issues). All processes of a case start at the same
instant and copy for --seconds; each reports its own rate.

    python3 h2d_bw.py 0,1@local 0-7@local 0,1@remote
    python3 h2d_bw.py --patterns layer --seconds 10 0-7@any

A case is GPUs @ placement of each GPU's host buffer: local = the GPU's NUMA
node, remote = the other node, any = no numactl (first touch wherever the
process runs, as in the service). Patterns: bulk = 256 MiB contiguous copies;
layer = one cudaMemcpyBatchAsync of 64 scattered 512 KiB values (one layer of
one 512-token chunk on one TP rank of Qwen3-30B-A3B; 64 chunks = a 32K prefix)
into a 32 MiB staging buffer, two buffers alternating.
"""
import argparse
import collections
import ctypes
import json
import mmap
import os
import random
import subprocess
import sys
import tempfile
import time

GIB = 1 << 30
MIB = 1 << 20
VALUE = 512 * 1024
VALUES_PER_LAYER = 64
BULK = 256 * MIB


def gpu_nodes():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        text=True)
    nodes = {}
    for line in out.strip().splitlines():
        idx, bus = (s.strip() for s in line.split(","))
        with open(f"/sys/bus/pci/devices/{bus[-12:].lower()}/numa_node") as f:
            nodes[int(idx)] = int(f.read())
    return nodes


def parse_gpus(spec):
    gpus = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        gpus += range(int(lo), int(hi or lo) + 1)
    return gpus


def node_pages(ptr):
    """Pages per NUMA node of the mapping holding ptr."""
    best, fields = -1, []
    with open("/proc/self/numa_maps") as f:
        for line in f:
            parts = line.split()
            start = int(parts[0], 16)
            if best < start <= ptr:
                best, fields = start, parts[1:]
    return {k: int(v) for k, v in
            (p.split("=", 1) for p in fields if p[:1] == "N" and "=" in p)}


def worker(a):
    import torch
    import lmcache.c_ops as ops

    torch.cuda.set_device(0)
    size = a.host_gib * GIB
    buf = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    ptr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    host = torch.frombuffer(buf, dtype=torch.uint8)
    host.fill_(1)
    err = torch.cuda.cudart().cudaHostRegister(ptr, size, 0)
    if int(err) != 0:
        raise SystemExit(f"cudaHostRegister failed: {err}")
    pages = node_pages(ptr)

    rng = random.Random(a.gpu)
    if a.pattern == "bulk":
        dsts = [torch.empty(BULK, dtype=torch.uint8, device="cuda")]
        ops_list = [([ptr + o], [BULK]) for o in range(0, size - BULK + 1, BULK)]
    else:
        dsts = [torch.empty(VALUE * VALUES_PER_LAYER, dtype=torch.uint8, device="cuda")
                for _ in range(2)]
        nvals = size // VALUE
        ops_list = [([ptr + VALUE * rng.randrange(nvals) for _ in range(VALUES_PER_LAYER)],
                     [VALUE] * VALUES_PER_LAYER) for _ in range(256)]
    per_op = sum(ops_list[0][1])

    stream = torch.cuda.Stream()
    pending = collections.deque()
    n = 0
    with torch.cuda.stream(stream):
        for i in range(16):
            ops.batched_memcpy_h2d(*ops_list[i % len(ops_list)], dsts[i % len(dsts)])
        stream.synchronize()
        open(os.path.join(a.sync_dir, f"ready{a.gpu}"), "w").close()
        go = os.path.join(a.sync_dir, "go")
        while not os.path.exists(go):
            time.sleep(0.002)
        with open(go) as f:
            start_at = float(f.read())
        while time.time() < start_at:
            pass
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        t0 = time.time()
        ev0.record(stream)
        while time.time() - t0 < a.seconds:
            for _ in range(4):
                ops.batched_memcpy_h2d(*ops_list[n % len(ops_list)], dsts[n % len(dsts)])
                n += 1
            ev = torch.cuda.Event()
            ev.record(stream)
            pending.append(ev)
            if len(pending) > 2:
                pending.popleft().synchronize()
        ev1.record(stream)
        stream.synchronize()
        t1 = time.time()
    torch.cuda.cudart().cudaHostUnregister(ptr)
    nbytes = n * per_op
    print(json.dumps({
        "gpu": a.gpu, "pattern": a.pattern, "bytes": nbytes,
        "wall_s": t1 - t0, "gpu_s": ev0.elapsed_time(ev1) / 1e3,
        "gbps": nbytes / (t1 - t0) / 1e9, "t0": t0, "t1": t1, "pages": pages,
    }), flush=True)


def run_case(gpus, mem, pattern, a, nodes):
    tmp = tempfile.mkdtemp(prefix="h2d_bw_")
    procs = []
    for g in gpus:
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--gpu", str(g),
               "--pattern", pattern, "--seconds", str(a.seconds),
               "--host-gib", str(a.host_gib), "--sync-dir", tmp]
        if mem != "any":
            node = nodes[g] if mem == "local" else 1 - nodes[g]
            cmd = ["numactl", f"--cpunodebind={nodes[g]}", f"--membind={node}"] + cmd
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True))
    while sum(os.path.exists(os.path.join(tmp, f"ready{g}")) for g in gpus) < len(gpus):
        dead = [p for p in procs if p.poll() is not None]
        if dead:
            for p in procs:
                p.kill()
            raise SystemExit(f"worker exited early ({gpus}@{mem} {pattern})")
        time.sleep(0.05)
    with open(os.path.join(tmp, "go.tmp"), "w") as f:
        f.write(repr(time.time() + 0.5))
    os.replace(os.path.join(tmp, "go.tmp"), os.path.join(tmp, "go"))
    res = []
    for p in procs:
        out = p.communicate()[0]
        res.append(json.loads([l for l in out.splitlines() if l.startswith("{")][-1]))
    overlap = min(r["t1"] for r in res) - max(r["t0"] for r in res)
    total = sum(r["gbps"] for r in res)
    per = " ".join(f"{r['gpu']}:{r['gbps']:.1f}" for r in res)
    nodes_used = sorted({k for r in res for k, v in r["pages"].items() if v > 1000})
    print(f"{','.join(map(str, gpus)):>16}@{mem:<6} {pattern:<5} total {total:6.1f} GB/s  "
          f"per-GPU min {min(r['gbps'] for r in res):5.1f} max {max(r['gbps'] for r in res):5.1f}  "
          f"[{per}]  overlap {overlap:.2f}/{a.seconds}s mem {'+'.join(nodes_used)}", flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps({"gpus": gpus, "mem": mem, "pattern": pattern,
                                "seconds": a.seconds, "results": res}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", help="GPUS@local|remote|any, e.g. 0-7@local")
    ap.add_argument("--patterns", default="bulk,layer")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--host-gib", type=int, default=4)
    ap.add_argument("--out", help="append raw per-worker results (JSONL)")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--pattern")
    ap.add_argument("--sync-dir")
    a = ap.parse_args()
    if a.worker:
        worker(a)
        return
    nodes = gpu_nodes()
    for case in a.cases:
        spec, _, mem = case.partition("@")
        for pattern in a.patterns.split(","):
            run_case(parse_gpus(spec), mem or "local", pattern, a, nodes)


if __name__ == "__main__":
    main()
