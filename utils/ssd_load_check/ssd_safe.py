"""How many requests of an agent trace are ssd-safe, from the measured SSD load.

    python3 ssd_safe.py [prefill-0.log of the ramp run] [per_request.jsonl ...]

ssd-safe, same criterion as dram-safe in paper-figure/toolagent-dram-safe-per-
request: with the whole reusable prefix loaded from SSD, the prefill takes < 5%
longer than with it in HBM, i.e. (e2e_ssd - compute) / compute < 5%.

Model of e2e_ssd, read off the 45-group end-to-end grid (1P2D, prefill
cudagraph [64..1024], SsdLayerwiseLoader with the read ramp):
  load-bound     e2e = E(p)          the SSD reads plus a tail (last layers'
                                     compute, request set-up); fit e0 + e1 * p
                                     on the reps whose load span > compute
  compute-bound  e2e = C + delta(p)  the forward waits for layer 0 once:
                                     delta(p) = FL(p) + r, FL(p) = first-layer
                                     ready time from test_loader.py bench, r
                                     fit on 4K/1024, the one cleanly
                                     compute-bound cell
  e2e_ssd = max(C + delta(p), E(p))
Compute C of a trace request:
  suffix <= 1024  cudagraph: a * s * (p + s/2) + b * s_pad + d, fit on the
                  grid's HBM e2e; s_pad = capture size the forward is padded to
  suffix > 1024   eager under this config: the request's measured e2e_hbm from
                  the per-request experiment (an older H20; eager e2e on the two
                  machines agrees within 3% at suffix 1024)
"""
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "logs/ssd-load-check-ramp-logs/prefill-0.log"
TRACES = [Path(p) for p in sys.argv[2:]] or [
    ROOT / "paper-figure/toolagent-dram-safe-per-request/per_request.jsonl",
    ROOT / "paper-figure/toolagent-turn5plus-dram-safe-per-request/per_request.jsonl",
]
CAPTURE = [64, 128, 256, 512, 1024]
SAFE = 0.05
# test_loader.py bench (2 ranks, 64 threads, ramp), median first-layer ready
# time over three runs, 2026-11-02; one-off outliers (16K 30.3, 64K 33.6) left out.
FIRST_LAYER_MS = {4096: 3.60, 8192: 4.54, 16384: 7.40, 32768: 12.63, 65536: 18.55}


def lstsq(xs, ys):
    """Least squares for y = X @ w, X rows given; returns w (tiny normal-eq solve)."""
    n = len(xs[0])
    ata = [[sum(x[i] * x[j] for x in xs) for j in range(n)] for i in range(n)]
    aty = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(n)]
    for i in range(n):  # Gauss-Jordan
        piv = ata[i][i]
        ata[i] = [v / piv for v in ata[i]]
        aty[i] /= piv
        for k in range(n):
            if k != i:
                f = ata[k][i]
                ata[k] = [a - f * b for a, b in zip(ata[k], ata[i])]
                aty[k] -= f * aty[i]
    return aty


def grid():
    """(p, s, rep) -> dict(C, ssd, load) from the ramp run."""
    rows = [json.loads(x) for x in (HERE / "trace.jsonl").read_text().splitlines()]
    e2e, load = {}, defaultdict(list)
    for line in LOG.read_text(errors="replace").splitlines():
        if m := re.search(r"KVPilot prefill req=(\d{8})-\S+ .*?e2e=([\d.]+)ms", line):
            e2e[int(m[1])] = float(m[2])
        elif m := re.search(r"KVPilot load req=(\d{8})-\S+ .*?load=([\d.]+)ms ssd=(\d+)", line):
            load[int(m[1])].append(float(m[2]))
    by = defaultdict(dict)
    for i, r in enumerate(rows):
        by[(r["prefix"], r["suffix"], r["rep"])][r["tier"]] = i
    return {k: dict(C=e2e[t[1]], ssd=e2e[t[4]], load=max(load[t[4]])) for k, t in by.items()}


def fit(g):
    a, b, d = lstsq([[s * (p + s / 2) / 1e6, s / 1e3, 1.0] for p, s, _ in g],
                    [v["C"] for v in g.values()])
    lb = [(p, v["ssd"]) for (p, s, _), v in g.items() if v["load"] > v["C"] + 5]
    e1, e0 = lstsq([[p / 1e3, 1.0] for p, _ in lb], [y for _, y in lb])
    pts = sorted(FIRST_LAYER_MS.items())
    f1, f0 = lstsq([[p / 1e3, 1.0] for p, _ in pts], [y for _, y in pts])
    fl = lambda p: f0 + f1 * p / 1e3  # noqa: E731
    r = st.median(v["ssd"] - v["C"] - fl(p) for (p, s, _), v in g.items()
                  if (p, s) == (4096, 1024))
    return dict(a=a, b=b, d=d, e0=e0, e1=e1, f0=f0, f1=f1, r=r)


def delta_fn(k, scale_delta=1.0, shift_delta=0.0):
    return lambda p: scale_delta * (k["f0"] + k["f1"] * p / 1e3 + k["r"]) + shift_delta


def e_fn(k, scale_e=1.0):
    return lambda p: scale_e * (k["e0"] + k["e1"] * p / 1e3)


def model(k, scale_delta=1.0, scale_e=1.0, shift_delta=0.0):
    delta, e = delta_fn(k, scale_delta, shift_delta), e_fn(k, scale_e)
    return lambda p, c: max(c + delta(p), e(p))


def compute(k, p, s, e2e_hbm, scale_eager=1.0):
    if s <= 1024:
        s_pad = next(x for x in CAPTURE if x >= s)
        return k["a"] * s * (p + s / 2) / 1e6 + k["b"] * s_pad / 1e3 + k["d"]
    return scale_eager * e2e_hbm


def validate(g, k):
    pred = model(k)
    print("\n== validation on the grid (45 reps; C = measured HBM e2e of the same rep)")
    print(f"{'prefix':>6} {'suffix':>6} {'meas dly':>9} {'pred dly':>9} {'meas %':>7} {'pred %':>7} safe(m/p)")
    errs, agree = [], 0
    cells = defaultdict(list)
    for (p, s, rep), v in sorted(g.items()):
        m_dly, p_dly = v["ssd"] - v["C"], pred(p, v["C"]) - v["C"]
        errs.append(p_dly - m_dly)
        agree += (m_dly / v["C"] < SAFE) == (p_dly / v["C"] < SAFE)
        cells[(p, s)].append((m_dly, p_dly, v["C"]))
    for (p, s), xs in sorted(cells.items()):
        m = st.median(x[0] for x in xs)
        q = st.median(x[1] for x in xs)
        c = st.median(x[2] for x in xs)
        print(f"{p:6d} {s:6d} {m:9.1f} {q:9.1f} {100 * m / c:6.1f}% {100 * q / c:6.1f}%"
              f"  {'Y' if m / c < SAFE else 'n'}/{'Y' if q / c < SAFE else 'n'}")
    ae = sorted(abs(x) for x in errs)
    print(f"delay error: median |err| {st.median(ae):.1f} ms, p90 {ae[int(0.9 * len(ae))]:.1f} ms, "
          f"max {ae[-1]:.1f} ms; ssd-safe verdict agrees on {agree}/{len(errs)} reps")
    cerr = [abs(compute(k, p, s, 0) / v["C"] - 1) for (p, s, _), v in g.items()]
    print(f"compute fit (cudagraph): median rel err {100 * st.median(cerr):.1f}%, max {100 * max(cerr):.1f}%")


def bins(v, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return i
    return len(edges) - 2


def label(edges, i):
    k = lambda x: f"{x // 1024}K" if x >= 1024 else str(x)  # noqa: E731
    lo, hi = edges[i], edges[i + 1]
    return f">{k(lo - 1)}" if hi >= 10**9 else f"{k(lo)}-{k(hi - 1)}"


def classify(k, rows, **scales):
    pred = model(k, scales.get("scale_delta", 1.0), scales.get("scale_e", 1.0),
                 scales.get("shift_delta", 0.0))
    out = []
    for r in rows:
        c = compute(k, r["prefix"], r["suffix"], r["e2e_hbm"], scales.get("scale_eager", 1.0))
        out.append((pred(r["prefix"], c) - c) / c < SAFE)
    return out


def reasons(k, rows):
    """Per request: 'safe', 'load' (E(p) > 1.05 C: the load itself is longer)
    or 'wait' (the load hides, but the layer-0 wait alone is >= 5% of C)."""
    delta, e = delta_fn(k), e_fn(k)
    out = []
    for r in rows:
        p = r["prefix"]
        c = compute(k, p, r["suffix"], r["e2e_hbm"])
        if e(p) > (1 + SAFE) * c:
            out.append("load")
        elif delta(p) >= SAFE * c:
            out.append("wait")
        else:
            out.append("safe")
    return out


def report(k, path):
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    safe = classify(k, rows)
    n = len(rows)
    print(f"\n== {path.parent.name}: {sum(safe)} / {n} requests ssd-safe ({100 * sum(safe) / n:.1f}%); "
          f"dram-safe measured on the old path: {100 * sum(r['dram_safe'] for r in rows) / n:.1f}%")
    for name, key, edges in (
        ("suffix", "suffix", [1, 65, 257, 1025, 2049, 4097, 16385, 10**9]),
        ("prefix", "prefix", [512, 4096, 8192, 16384, 32768, 65536, 10**9]),
    ):
        cnt, ok = defaultdict(int), defaultdict(int)
        for r, sf in zip(rows, safe):
            i = bins(r[key], edges)
            cnt[i] += 1
            ok[i] += sf
        print(f"  by {name}: " + "  ".join(
            f"{label(edges, i)}: {ok[i]}/{cnt[i]} ({100 * ok[i] / cnt[i]:.0f}%)" for i in sorted(cnt)))
    why = reasons(k, rows)
    assert [w == "safe" for w in why] == safe
    print(f"  not safe because the load is longer than compute: {why.count('load')}; "
          f"load hidden but the layer-0 wait >= 5% of compute: {why.count('wait')}")
    d4 = delta_fn(k)(4096)
    print(f"  sensitivity (ssd-safe %; delta(4K) = {d4:.1f} ms in the fit):")
    for desc, sc in (
        ("delta(4K) = 3.3 (fastest rep)", dict(shift_delta=3.3 - d4)),
        ("delta(4K) = 8.1 (slowest rep)", dict(shift_delta=8.1 - d4)),
        ("delta x0.5", dict(scale_delta=0.5)), ("delta x2", dict(scale_delta=2.0)),
        ("E x0.9", dict(scale_e=0.9)), ("E x1.1", dict(scale_e=1.1)),
        ("eager compute x0.95", dict(scale_eager=0.95)),
        ("eager compute x1.05", dict(scale_eager=1.05)),
    ):
        s = classify(k, rows, **sc)
        print(f"    {desc:30s} {sum(s):6d}  {100 * sum(s) / n:5.1f}%")
    return sum(safe), n


def main():
    g = grid()
    k = fit(g)
    print("fit: compute(cudagraph) = %.2f*s*(p+s/2)/1e6 + %.2f*s_pad/1e3 + %.1f ms" % (k["a"], k["b"], k["d"]))
    print("     load-bound E(p) = %.1f + %.2f * p/1K ms   (load span + tail)" % (k["e0"], k["e1"]))
    print("     delta(p) = first layer %.2f + %.3f * p/1K  + relaunch %.1f ms" % (k["f0"], k["f1"], k["r"]))
    validate(g, k)
    for path in TRACES:
        report(k, path)


main()
