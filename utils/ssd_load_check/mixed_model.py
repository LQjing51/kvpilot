"""Does a split hit (HBM head, then DRAM, then SSD) load and prefill as the
blockwise time model says?

    python3 mixed_model.py <log dir> <trace.jsonl> [coeffs.json]

Traces: make_trace_mixed.py (DRAM + SSD, no HBM head) or make_trace_hbmmix.py
(HBM head, then DRAM, then SSD). Each group (prefix p, suffix s, rep) has a
[1] test (compute only) and tests with the prefix split (H, D, X) tokens
across HBM / DRAM / SSD. A test counts only if it hit exactly H in HBM, loaded
D + X, and each TP rank read exactly X / 512 x 48 values from SSD.

1. Load time (`KVPilot load ... load=`, the slower TP rank, median of reps).
   Pure DRAM and pure SSD loads (H = 0) are fitted linear in loaded tokens
   per suffix; a split load is predicted as
     max:      max(dram(D + X), ssd(X))       every value is copied, SSD part read
     additive: dram(D) + ssd(X) - dram(0)
2. Exposed time, e2e - e2e of the group's [1] test, against the blockwise
   model's (vllm/v1/core/kvpilot_time_model, default coefficients Figure3
   fit_times2_swe.json): TimeModel.e2e(p, s, D, X) - compute(p, s).
3. The safe decision blockwise makes, exposed / compute < thresh, from the
   model and from the measurement.
"""
import importlib.util
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# The module file alone: importing the vllm package needs torch.
_spec = importlib.util.spec_from_file_location(
    "kvpilot_time_model", ROOT / "vllm-dev/vllm/v1/core/kvpilot_time_model.py")
_tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tm)
TimeModel = _tm.TimeModel

BLOCK, LAYERS = 512, 48
PREFILL = re.compile(r"KVPilot prefill req=(\d{8})-\S+ .*?hit=(\d+) dram=(\d+) "
                     r"e2e=([\d.]+)ms")
RETRIEVE = re.compile(r"KVPilot prelookup retrieve req=(\d{8})-\S+ chunks=(\d+) ssd_values=(\d+)")
LOAD = re.compile(r"KVPilot load req=(\d{8})-\S+ tokens=(\d+) .*?load=([\d.]+)ms ssd=(\d+)")

log, trace = Path(sys.argv[1]), Path(sys.argv[2])
coeffs = sys.argv[3] if len(sys.argv) > 3 else str(
    ROOT / "paper-figure/Figure3-request-safe/fit_times2_swe.json")
log = log / "prefill-0.log" if log.is_dir() else log
pf, ret, load = {}, defaultdict(list), defaultdict(list)
for line in log.read_text(errors="replace").splitlines():
    if m := PREFILL.search(line):
        pf[int(m[1])] = dict(hit=int(m[2]), dram=int(m[3]), e2e=float(m[4]))
    elif m := RETRIEVE.search(line):
        ret[int(m[1])].append(int(m[3]))
    elif m := LOAD.search(line):
        load[int(m[1])].append(float(m[3]))


def split_of(r):
    if "split" in r:
        return tuple(r["split"])
    p, f = r["prefix"], r["ssd_frac"]
    if f is None:
        return (p, 0, 0)
    x = int(p * f) // BLOCK * BLOCK
    return (0, p - x, x)


rows = [json.loads(x) for x in trace.read_text().splitlines()]
groups = defaultdict(dict)  # (p, s, rep) -> split -> line
for i, r in enumerate(rows):
    if r["tier"]:
        groups[(r["prefix"], r["suffix"], r["rep"])][split_of(r)] = i

cells = defaultdict(lambda: defaultdict(list))  # (p, s, split) -> lists
bad = defaultdict(int)
for (p, s, _), g in sorted(groups.items()):
    base = pf.get(g.get((p, 0, 0)))
    if base is None or base["hit"] < p or base["dram"]:
        bad["[1] baseline"] += 1
        continue
    for sp, i in g.items():
        h, d, x = sp
        if h == p:
            continue
        r = pf.get(i)
        if r is None or r["hit"] != h or r["dram"] < d + x or len(load[i]) != 2 \
                or not ret[i] or any(v != x // BLOCK * LAYERS for v in ret[i]):
            bad[str(sp)] += 1
            continue
        c = cells[(p, s, sp)]
        c["load"].append(max(load[i]))
        c["e2e"].append(r["e2e"])
        c["exp"].append(r["e2e"] - base["e2e"])
        c["comp"].append(base["e2e"])

med = statistics.median
prefixes = sorted({p for p, _, _ in cells})
suffixes = sorted({s for _, s, _ in cells})


def kind(p, sp):
    h, d, x = sp
    return ("HBM head + " if h else "") + ("DRAM+SSD" if d and x else "SSD" if x else "DRAM")


print("== 1. load time (ms), measured vs max form / additive ==")
lt = defaultdict(list)
for s in suffixes:
    fit = {}
    for name, pure in (("dram", lambda p: (0, p, 0)), ("ssd", lambda p: (0, 0, p))):
        xs = [p for p in prefixes if (p, s, pure(p)) in cells]
        fit[name] = np.polyfit(xs, [med(cells[(p, s, pure(p))]["load"]) for p in xs], 1)
    dram = lambda t: fit["dram"][0] * t + fit["dram"][1]  # noqa: E731
    ssd = lambda t: fit["ssd"][0] * t + fit["ssd"][1]  # noqa: E731
    print(f"suffix {s}: pure DRAM {fit['dram'][0] * 1e3:.2f} ms/1K tok + {fit['dram'][1]:.1f}, "
          f"pure SSD {fit['ssd'][0] * 1e3:.2f} ms/1K tok + {fit['ssd'][1]:.1f}")
    for (p, s2, sp), c in cells.items():
        h, d, x = sp
        if s2 != s or (h == 0 and (d == 0 or x == 0)):
            continue
        m = med(c["load"])
        mx = max(dram(d + x), ssd(x)) if x else dram(d)
        ad = dram(d) + ssd(x) - fit["dram"][1]
        lt[(kind(p, sp), s, "max")].append((mx - m, (mx - m) / m))
        lt[(kind(p, sp), s, "add")].append((ad - m, (ad - m) / m))
print("\nmodel - measured: median / max |err| ms, median / max |err| relative, n cells")
for k in sorted({k[0] for k in lt}):
    for s in suffixes:
        for form in ("max", "add"):
            v = lt.get((k, s, form))
            if not v or (form == "add" and "SSD" not in k):
                continue
            a, r = [e for e, _ in v], [e for _, e in v]
            print(f"  {k:22s} s={s:5d} {form}: {med(a):+6.1f} / {max(map(abs, a)):5.1f} ms   "
                  f"{med(r) * 100:+5.1f}% / {max(map(abs, r)) * 100:4.1f}%  n={len(v)}")

model = TimeModel.load(coeffs, BLOCK)
th = model.thresh
print(f"\n== 2. exposed = e2e - e2e[1] (ms), measured vs blockwise model ({Path(coeffs).name}) ==")
print(f"{'prefix':>6} {'sfx':>5} {'H/D/X (K tok)':>15} {'C meas':>7} {'C model':>7} "
      f"{'load':>6} {'exp meas':>8} {'exp model':>9} {'safe m/M':>8}")
ex = defaultdict(list)
conf = defaultdict(int)
for (p, s, sp), c in sorted(cells.items()):
    h, d, x = sp
    cm, em, lm = med(c["comp"]), med(c["exp"]), med(c["load"])
    cmod = model.compute(p, s)
    emod = model.e2e(p, s, d, x) - cmod
    safe_m, safe_mod = em / cm < th, emod / cmod < th
    conf[(safe_mod, safe_m)] += 1
    ratio = lm / cm
    band = "load-bound" if ratio > 1.3 else "near (0.7-1.3)" if ratio > 0.7 else "compute-bound"
    ex[(band, "HBM head" if h else "no HBM")].append(emod - em)
    print(f"{p:6d} {s:5d} {f'{h // 1024}/{d // 1024}/{x // 1024}':>15} {cm:7.1f} {cmod:7.1f} "
          f"{lm:6.1f} {em:8.1f} {emod:9.1f} {('Y' if safe_m else 'n') + '/' + ('Y' if safe_mod else 'n'):>8}")
print("\nexposed: model - measured (ms): median / max |err|; band by measured load / compute")
for k, v in sorted(ex.items()):
    print(f"  {k[0]:15s} {k[1]:8s}: {med(v):+6.1f} / {max(map(abs, v)):5.1f}  n={len(v)}")
print(f"\n== 3. safe (exposed / compute < {th}) cells: model vs measured ==")
print(f"  both safe {conf[(True, True)]}, both unsafe {conf[(False, False)]}, "
      f"model safe but measured unsafe {conf[(True, False)]}, "
      f"model unsafe but measured safe {conf[(False, True)]}")
print("dropped:", dict(bad) or "none")
