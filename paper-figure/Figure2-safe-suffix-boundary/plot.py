"""Per prefix, the shortest suffix at which a prefix loaded from DRAM / SSD
stays hidden behind the suffix compute (delay / compute < 5%), and the HBM /
DRAM / SSD safe regions they split the (prefix, suffix) plane into. Reads only
one run dir, writes the pdf next to this file.

    python3 plot.py              # raw/      -> safe_suffix_boundary.pdf (2P2D)
    python3 plot.py raw/1p1d     # raw/1p1d/ -> safe_suffix_boundary_1p1d.pdf

Every boundary point is a measured suffix: the lowest one judged safe above
the highest one judged unsafe (search.py, then recheck.py for the outliers,
then confirm.py for every point where it ran).
A prefix with no safe suffix above its highest unsafe one is left out.
See README.md.
"""
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paperstyle import *  # noqa: E402,F403

RAW = HERE / (sys.argv[1] if len(sys.argv) > 1 else "raw")
PDF = "safe_suffix_boundary.pdf" if RAW == HERE / "raw" else f"safe_suffix_boundary_{RAW.name}.pdf"
THRESH, BLOCK = 0.05, 512
PRE = re.compile(r"KVPilot prefill req=(\d{8})-\S+ (?:(?!KVPilot).)*?hit=(\d+) dram=(\d+) e2e=([\d.]+)ms")
RET = re.compile(r"KVPilot prelookup retrieve req=(\d{8})-\S+ chunks=(\d+) ssd_values=(\d+)")


def open_log(f):
    return gzip.open(f, "rt", errors="replace") if f.suffix == ".gz" else open(f, errors="replace")


def measurements(d):
    """Every valid pair in run d, recomputed from its prefill logs:
    (prefix, medium) -> {suffix: [delay / compute, ...]}. requests.jsonl says
    what each request was."""
    pre, ret = {}, defaultdict(list)
    for f in sorted(d.glob("prefill-*.log*")):
        for l in open_log(f):
            for m in PRE.finditer(l):
                pre[int(m[1])] = (int(m[2]), int(m[3]), float(m[4]))
            for m in RET.finditer(l):
                ret[int(m[1])].append((int(m[2]), int(m[3])))
    # Infer layer count from SSD retrieves (ssd_values = layers * chunks).
    # TP rank count varies (TP=1/2 → 2 lines historically via duplicate logs;
    # TP=4 → 4 lines); require ≥1 agreeing ranks, not a fixed length.
    layers = None
    for rows in ret.values():
        for c, v in rows:
            if c > 0 and v > 0 and v % c == 0:
                layers = v // c
                break
        if layers is not None:
            break
    pairs = defaultdict(dict)
    for r in map(json.loads, open(d / "requests.jsonl")):
        if r.get("pair"):
            pairs[r["pair"]][r["tier"]] = r
    out, bad = defaultdict(lambda: defaultdict(list)), defaultdict(int)
    for t in pairs.values():
        r1 = t[1]
        rt = next(v for k, v in t.items() if k != 1)
        p, s, medium, tier = r1["prefix"], r1["suffix"], r1["medium"], rt["tier"]
        a, b, rr = pre.get(r1["idx"]), pre.get(rt["idx"]), ret.get(rt["idx"], [])
        if a is None or b is None:
            bad["missing log"] += 1
            continue
        want = 0 if tier == 2 else (None if layers is None else layers * (p // BLOCK))
        ok = (a[0] >= p and a[1] == 0 and b[1] >= p and len(rr) >= 1
              and all(c == p // BLOCK and (want is None or v == want) for c, v in rr)
              and len({v for _, v in rr}) == 1)
        ratio = (b[2] - a[2]) / a[2]
        if not ok or ratio < -0.03:
            bad["wrong source" if not ok else "negative"] += 1
            continue
        out[(p, medium)][s].append(ratio)
    print(f"{d.name}: pairs {len(pairs)}, layers={layers}, anomalies {dict(bad) or 'none'}")
    return out


# The search measures each suffix once; recheck/ re-measures the suffixes
# around the outlier boundaries five times, confirm/ re-judges every boundary
# on five readings per suffix. A (prefix, medium) is judged on the latest
# stage that ran for it alone, each suffix on its median: the earlier single
# readings were what that stage set out to check.
main = measurements(RAW)
stages = {d: measurements(RAW / d) for d in ("recheck", "confirm")
          if (RAW / d / "requests.jsonl").exists()}
logged = {(r["prefix"], r["medium"]): r["boundary"] for r in map(json.loads, open(RAW / "search.jsonl"))
          if "boundary" in r}
meas = {k: {s: v[-1:] for s, v in pts.items()} for k, pts in main.items()}  # a retake replaces its anomaly
for stage in stages.values():
    meas.update(stage)
recheck = stages.get("recheck", {})


def boundary(points):
    """Lowest safe suffix above the highest unsafe one, and that unsafe one."""
    med = {s: statistics.median(v) for s, v in points.items()}
    lo = max((s for s, r in med.items() if r >= THRESH), default=0)
    hi = min((s for s, r in med.items() if s > lo and r < THRESH), default=None)
    return lo, hi


P = sorted({p for p, _ in meas})
bnd = {}
print(f"{'prefix':>6} {'medium':6} {'unsafe':>6} {'safe':>6} {'search':>6} {'rechecked':>9}")
for p in P:
    for medium in ("dram", "ssd"):
        lo, hi = boundary(meas[(p, medium)])
        bnd[(p, medium)] = hi
        rc = sorted(recheck.get((p, medium), {}))
        print(f"{p // 1024:5d}K {medium:6} {lo:6d} {hi or 0:6d} {logged.get((p, medium)) or 0:6d} "
              + (" ".join(map(str, rc)) if rc else "-")
              + ("" if hi else "  NO SAFE SUFFIX ABOVE THE HIGHEST UNSAFE ONE"))

# Regions: below the DRAM line only HBM hides the load; between the lines DRAM
# does; above the SSD line SSD does too.
xs = [p for p in P if bnd[(p, "dram")] and bnd[(p, "ssd")]]
dl = [bnd[(p, "dram")] for p in xs]
sl = [max(bnd[(p, "ssd")], bnd[(p, "dram")]) for p in xs]
lo, hi = min(dl) / 1.25, max(sl) * 1.55
fig, ax = plt.subplots(figsize=(1.9, 1.45))


# Ordered tiers on one blue ramp (paperstyle's sequential hue), darkest for the
# fastest tier. Each line is the lower edge of its region, drawn solid in a
# darker step of that region's blue so it stands out from the fill.
TIERS = (("HBM safe", "#1c5cab", "white"), ("DRAM safe", "#3987e5", "white"),
         ("SSD safe", "#86b6ef", "white"))
LINES = ("#104281", "#256abf")  # DRAM, SSD boundary: two steps darker
edges = [[lo] * len(xs), dl, sl, [hi] * len(xs)]
for (_, c, _), a, b in zip(TIERS, edges, edges[1:]):
    ax.fill_between(xs, a, b, color=c, lw=0)
for c, y, m in zip(LINES, (dl, sl), MARKERS[1:]):
    ax.plot(xs, y, color=c, marker=m, ms=2.5, lw=1.0, clip_on=False, zorder=3)
# Names at the middle of each region. SSD sits further left, where its region
# is tallest, so it can come down from the top edge.
k, j = len(xs) * 2 // 3, len(xs) // 4
for (text, _, ink), x, y in zip(TIERS, (xs[k], xs[k], xs[j]),
                                ((lo * dl[k]) ** 0.5, (dl[k] * sl[k]) ** 0.5, (sl[j] * hi) ** 0.5)):
    ax.text(x, y, text, ha="center", va="center", fontsize=7.5, color=ink)
ax.set_yscale("log", base=2)
ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(lo, hi)
ax.set_xticks(xs, [f"{p // 1024}K" if i % 2 == 0 else "" for i, p in enumerate(xs)],
              rotation=60, ha="right", rotation_mode="anchor", fontsize=5.5)
ax.tick_params(axis="x", pad=1.5)
yt = [s for s in (64, 128, 256, 512, 1024, 2048, 4096) if lo <= s <= hi]
ax.set_yticks(yt, [f"{s // 1024}K" if s >= 1024 else str(s) for s in yt])
ax.minorticks_off()
ax.grid(False)
ax.set_xlabel("Prefix length")
ax.set_ylabel("Suffix length")
fig.savefig(HERE / PDF)
print(f"wrote {PDF}")
