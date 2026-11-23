"""HBM <-> DRAM transfer symmetry: does loading (H2D) cost what writing back
(D2H) does?

    python3 bench.py                # GPU 0 alone
    python3 bench.py --gpus 0,1     # two GPUs at once, as the two ranks of a TP-2 prefill

copy: pinned host <-> device copies on the copy engine, 4 KiB .. 512 MiB,
  median over repetitions; each direction alone, then both at once.
kv: chunks in Qwen3-30B-A3B's TP-2 layout (512 tokens x 2 KV heads x 128 dims,
  bf16: 512 KiB per chunk per layer), all 48 layers. write = gather (paged KV
  -> GPU staging) + D2H, as the exclusive writer does; load = H2D + scatter
  (GPU staging -> paged KV), the same kernel in the other direction.
  write->DRAM d2h is the store path now: per layer gather, then one batched D2H
  (lmc_ops.batched_memcpy_d2h) straight into each value's destination, as into
  Mooncake replicas. write->DRAM memmove is the path it replaced: gather + D2H
  into one of two pinned buffers, then a one-thread memmove of each value
  while the next layer's D2H runs; memmove is that host copy alone.
"""
import argparse
import multiprocessing as mp
import statistics
import time

import torch

KIB, MIB = 1 << 10, 1 << 20
SIZES = [4 * KIB, 64 * KIB, 512 * KIB, 4 * MIB, 24 * MIB, 96 * MIB, 512 * MIB]
LAYERS, BS, NH, HS = 48, 512, 2, 128


def med_ms(fn, reps):
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        out.append(e0.elapsed_time(e1))
    return statistics.median(out)


def fmt_size(s):
    return f"{s // MIB} MiB" if s >= MIB else f"{s // KIB} KiB"


def copy_bench(dev, rows):
    for s in SIZES:
        host = torch.empty(s, dtype=torch.uint8, pin_memory=True)
        gpu = torch.empty(s, dtype=torch.uint8, device=dev)
        reps = 200 if s <= MIB else 30 if s <= 96 * MIB else 10
        for _ in range(3):
            gpu.copy_(host, non_blocking=True)
            host.copy_(gpu, non_blocking=True)
        h2d = med_ms(lambda: gpu.copy_(host, non_blocking=True), reps)
        d2h = med_ms(lambda: host.copy_(gpu, non_blocking=True), reps)
        rows.append(("copy", fmt_size(s), h2d * 1e3, s / h2d / 1e6, d2h * 1e3, s / d2h / 1e6))
    # Both directions at once, on two streams.
    s = 512 * MIB
    hin, hout = (torch.empty(s, dtype=torch.uint8, pin_memory=True) for _ in range(2))
    gin, gout = (torch.empty(s, dtype=torch.uint8, device=dev) for _ in range(2))
    s1, s2 = torch.cuda.Stream(dev), torch.cuda.Stream(dev)
    walls = []
    for _ in range(10):
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.cuda.stream(s1):
            gin.copy_(hin, non_blocking=True)
        with torch.cuda.stream(s2):
            hout.copy_(gout, non_blocking=True)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t)
    w = statistics.median(walls) * 1e3
    rows.append(("copy both", "512 MiB each", w * 1e3, s / w / 1e6, w * 1e3, s / w / 1e6))


def kv_bench(dev, rows):
    from lmcache.utils import EngineType
    import lmcache.c_ops as lmc_ops
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemLayerwiseGPUConnector
    from lmcache.v1.gpu_connector.utils import normalize_kv_and_discover_format

    nb = 72
    kv = [torch.randn(2, nb, BS, NH, HS, dtype=torch.bfloat16, device=dev) for _ in range(LAYERS)]
    gc = VLLMPagedMemLayerwiseGPUConnector(
        hidden_dim_size=NH * HS, num_layers=LAYERS, use_gpu=True, chunk_size=BS,
        dtype=torch.bfloat16, device=dev,
    )
    fmt, kvs = normalize_kv_and_discover_format(
        kv, EngineType.VLLM, layout_hints=getattr(gc, "layout_hints", None))
    for k in (1, 8, 64):
        n = k * BS
        slots = (torch.arange(k).unsqueeze(1) * BS + torch.arange(BS)).flatten().to(dev)
        stage = torch.empty(gc.get_shape(n), dtype=torch.bfloat16, device=dev)
        host = torch.empty(gc.get_shape(n * LAYERS), dtype=torch.bfloat16, pin_memory=True)
        size = n * gc.token_bytes() * LAYERS

        def gather():
            for l in range(LAYERS):
                lmc_ops.single_layer_kv_transfer(
                    stage, kvs[l], slots, lmc_ops.TransferDirection.D2H, fmt, True)

        def scatter():
            for l in range(LAYERS):
                lmc_ops.single_layer_kv_transfer(
                    stage, kvs[l], slots, lmc_ops.TransferDirection.H2D, fmt, True)

        def d2h():
            for l in range(LAYERS):
                host[l * n:(l + 1) * n].copy_(stage, non_blocking=True)

        def h2d():
            for l in range(LAYERS):
                stage.copy_(host[l * n:(l + 1) * n], non_blocking=True)

        def write():
            for l in range(LAYERS):
                lmc_ops.single_layer_kv_transfer(
                    stage, kvs[l], slots, lmc_ops.TransferDirection.D2H, fmt, True)
                host[l * n:(l + 1) * n].copy_(stage, non_blocking=True)

        def load():
            for l in range(LAYERS):
                stage.copy_(host[l * n:(l + 1) * n], non_blocking=True)
                lmc_ops.single_layer_kv_transfer(
                    stage, kvs[l], slots, lmc_ops.TransferDirection.H2D, fmt, True)

        for f in (write, load):
            f()
        t = {name: med_ms(f, 10) for name, f in
             (("write", write), ("load", load), ("gather", gather), ("scatter", scatter),
              ("d2h", d2h), ("h2d", h2d))}
        label = f"{k} chunk{'s' if k > 1 else ''} x {LAYERS} layers"
        rows.append(("kv load/write", label, t["load"] * 1e3, size / t["load"] / 1e6,
                     t["write"] * 1e3, size / t["write"] / 1e6))
        rows.append(("  scatter/gather", label, t["scatter"] * 1e3, size / t["scatter"] / 1e6,
                     t["gather"] * 1e3, size / t["gather"] / 1e6))
        rows.append(("  H2D/D2H only", label, t["h2d"] * 1e3, size / t["h2d"] / 1e6,
                     t["d2h"] * 1e3, size / t["d2h"] / 1e6))

        nbytes = BS * gc.token_bytes()
        dst = torch.empty(gc.get_shape(n * LAYERS), dtype=torch.bfloat16, pin_memory=True)
        hb = [torch.empty(gc.get_shape(n), dtype=torch.bfloat16, pin_memory=True) for _ in range(2)]
        gb = [stage, torch.empty_like(stage)]
        evs = [torch.cuda.Event(), torch.cuda.Event()]
        stream = torch.cuda.Stream(dev)

        def flush(l, b):
            evs[b].synchronize()
            src, base = hb[b].data_ptr(), dst.data_ptr() + l * n * gc.token_bytes()
            lmc_ops.batched_memcpy([src + c * nbytes for c in range(k)],
                                   [base + c * nbytes for c in range(k)], [nbytes] * k)

        def write_dram():
            pending = None
            for l in range(LAYERS):
                b = l % 2
                with torch.cuda.stream(stream):
                    lmc_ops.single_layer_kv_transfer(
                        gb[b], kvs[l], slots, lmc_ops.TransferDirection.D2H, fmt, True)
                    hb[b].copy_(gb[b], non_blocking=True)
                    evs[b].record(stream)
                if pending is not None:
                    flush(*pending)
                pending = (l, b)
            flush(*pending)

        def memmove():
            for l in range(LAYERS):
                src, base = host.data_ptr() + l * n * gc.token_bytes(), dst.data_ptr() + l * n * gc.token_bytes()
                lmc_ops.batched_memcpy([src + c * nbytes for c in range(k)],
                                       [base + c * nbytes for c in range(k)], [nbytes] * k)

        def wall_ms(f, reps=5):
            out = []
            for _ in range(reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                f()
                torch.cuda.synchronize()
                out.append((time.perf_counter() - t0) * 1e3)
            return statistics.median(out)

        def write_d2h():
            with torch.cuda.stream(stream):
                for l in range(LAYERS):
                    lmc_ops.single_layer_kv_transfer(
                        stage, kvs[l], slots, lmc_ops.TransferDirection.D2H, fmt, True)
                    base = dst.data_ptr() + l * n * gc.token_bytes()
                    lmc_ops.batched_memcpy_d2h(
                        stage, [base + c * nbytes for c in range(k)], [nbytes] * k)

        write_dram()
        write_d2h()
        wd, mm, wb = wall_ms(write_dram), wall_ms(memmove), wall_ms(write_d2h)
        rows.append(("  write->DRAM d2h", label, float("nan"), float("nan"), wb * 1e3, size / wb / 1e6))
        rows.append(("  write->DRAM memmove", label, float("nan"), float("nan"), wd * 1e3, size / wd / 1e6))
        rows.append(("  memmove", label, float("nan"), float("nan"), mm * 1e3, size / mm / 1e6))


def run(gpu, barrier, results):
    torch.cuda.set_device(gpu)
    dev = torch.device(f"cuda:{gpu}")
    rows = []
    barrier.wait()
    copy_bench(dev, rows)
    barrier.wait()
    kv_bench(dev, rows)
    results[gpu] = rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0")
    a = ap.parse_args()
    gpus = [int(g) for g in a.gpus.split(",")]
    ctx = mp.get_context("spawn")
    barrier, results = ctx.Barrier(len(gpus)), ctx.Manager().dict()
    procs = [ctx.Process(target=run, args=(g, barrier, results)) for g in gpus]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    for g in gpus:
        print(f"== GPU {g} ({len(gpus)} GPU(s) running at once)")
        print(f"{'test':20s} {'size':22s} {'H2D/load us':>12s} {'GB/s':>7s} {'D2H/write us':>13s} {'GB/s':>7s}")
        for test, size, a_us, a_bw, b_us, b_bw in results[g]:
            print(f"{test:20s} {size:22s} {a_us:12.1f} {a_bw:7.1f} {b_us:13.1f} {b_bw:7.1f}")


if __name__ == "__main__":
    main()
