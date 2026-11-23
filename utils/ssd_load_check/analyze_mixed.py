"""gpu_stall vs e2e delay under mixed DRAM + SSD hits (make_trace_mixed.py).

    python3 analyze_mixed.py <log dir> <trace_mixed.jsonl>

delay = e2e of a loading test - e2e of the [1] test of the same (prefix,
suffix, rep); compared with that test's gpu_stall request by request. A test
counts only if it loaded the whole prefix and each TP rank read exactly the
planned share from SSD (ssd_values = SSD chunks x layers); others are dropped
and counted.
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BLOCK, LAYERS = 512, 48
PREFILL = re.compile(r"KVPilot prefill req=(\d{8})-\S+ .*?hit=(\d+) dram=(\d+) "
                     r"e2e=([\d.]+)ms.*? gpu_stall=([\d.]+)ms")
RETRIEVE = re.compile(r"KVPilot prelookup retrieve req=(\d{8})-\S+ chunks=(\d+) ssd_values=(\d+)")

log, trace = Path(sys.argv[1]), Path(sys.argv[2])
log = log / "prefill-0.log" if log.is_dir() else log
pf, ret = {}, defaultdict(list)
for line in log.read_text(errors="replace").splitlines():
    if m := PREFILL.search(line):
        pf[int(m[1])] = dict(hit=int(m[2]), dram=int(m[3]), e2e=float(m[4]), gs=float(m[5]))
    elif m := RETRIEVE.search(line):
        ret[int(m[1])].append((int(m[2]), int(m[3])))

rows = [json.loads(x) for x in trace.read_text().splitlines()]
groups = defaultdict(dict)
for i, r in enumerate(rows):
    if r["tier"]:
        groups[(r["prefix"], r["suffix"], r["rep"])][r["ssd_frac"]] = i

med = statistics.median
q = lambda v, x: sorted(v)[min(len(v) - 1, int(x * (len(v) - 1) + 0.5))]
cells, err, bad = defaultdict(lambda: defaultdict(list)), defaultdict(list), defaultdict(int)
for (p, s, _), g in sorted(groups.items()):
    base = pf.get(g.get(None))
    if base is None or base["hit"] < p or base["dram"]:
        bad["hbm baseline"] += 1
        continue
    for f, i in g.items():
        if f is None:
            continue
        x, want = pf.get(i), int(p * f) // BLOCK * LAYERS
        if x is None or x["dram"] < p or not ret[i] or any(v != want for _, v in ret[i]):
            bad[f"ssd_frac {f}"] += 1
            continue
        d = x["e2e"] - base["e2e"]
        cells[(p, s, f)]["delay"].append(d)
        cells[(p, s, f)]["gs"].append(x["gs"])
        err[(f, "short" if s <= 1024 else "long")].append(x["gs"] - d)

fracs = sorted({f for _, _, f in cells})
print("delay / gpu_stall (ms, median of reps) by SSD share of the prefix")
print(f"{'prefix':>7} {'suffix':>6} | " + " | ".join(f"{f:>4} dly    gs" for f in fracs))
for p, s in sorted({(p, s) for p, s, _ in cells}):
    vals = []
    for f in fracs:
        c = cells.get((p, s, f))
        vals.append(f"{med(c['delay']):8.1f} {med(c['gs']):5.1f}" if c else f"{'-':>8} {'-':>5}")
    print(f"{p:7d} {s:6d} | " + " | ".join(vals))
print("\ngpu_stall - delay per request (ms); short = suffix <= 1024 (cudagraph)")
for (f, band), v in sorted(err.items()):
    a = [abs(x) for x in v]
    print(f"  ssd_frac {f:<4} {band:5s} median {med(v):+6.1f}  |err| p50 {q(a, .5):5.1f} "
          f"p90 {q(a, .9):5.1f} max {max(a):5.1f}  n={len(v)}")
print("dropped:", dict(bad) or "none")
