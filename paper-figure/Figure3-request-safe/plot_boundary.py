"""DRAM / SSD safe-suffix boundaries of the fitted time model, and the HBM /
DRAM / SSD safe regions they split the (prefix, suffix) plane into.

    python3 plot_boundary.py

Fits fit_times.py's model on raw/swe/ (the coefficients in fit_times_swe.json,
which utils/placement_ab/dram_miss_sim.py scores e2e with) and, per prefix,
takes the shortest suffix whose predicted all-HBM prefill hides a DRAM / SSD
load of the prefix under 5% (fit_times.bounds). The prefix axis stays inside
the largest prefix of raw/swe/.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paperstyle import *  # noqa: E402,F403
from fit_times import bounds, fit, load_points  # noqa: E402

BLOCK = 512
P_MAX = 72 * 1024
TIERS = (("HBM safe", "#1c5cab"), ("DRAM safe", "#3987e5"), ("SSD safe", "#86b6ef"))
LINES = ("#104281", "#256abf")  # DRAM, SSD boundary: a darker step of the region above

pts = load_points(datasets=["swe"])
k = fit(pts)
ref = json.loads((HERE / "fit_times_swe.json").read_text())
diff = {key: (k[key], ref[key]) for key in ref if k[key] != ref[key]}
print(f"fit on {k['n']} raw/swe points, largest prefix {int(pts[:, 0].max())}; "
      f"vs fit_times_swe.json: {diff or 'identical'}")
print(f"t_hbm = {k['a']:.3e}*s*(p+s/2) + {k['b']:.3e}*s + {k['d']:.2f}")
assert P_MAX <= pts[:, 0].max()

xs = np.arange(BLOCK, P_MAX + 1, BLOCK)
dl, sl = map(np.array, zip(*(bounds(int(p), k) for p in xs)))
print(f"{'prefix':>7} {'dram':>5} {'ssd':>5}")
for p, d, s in zip(xs, dl, sl):
    if p % 8192 == 0 or p == BLOCK:
        print(f"{p if p < 1024 else f'{p // 1024}K':>7} {d:5d} {s:5d}")

lo, hi = 64, 2560
# Same height and font sizes as Eval-req-priority/lru_breakdown.pdf, so the
# two sit side by side at the same scale.
fig, ax = plt.subplots(figsize=(1.43, 1.43))
ax.set_box_aspect(1)
edges = [np.full(len(xs), lo), dl, sl, np.full(len(xs), hi)]
for (_, c), y0, y1 in zip(TIERS, edges, edges[1:]):
    ax.fill_between(xs, y0, y1, color=c, lw=0)
for c, y in zip(LINES, (dl, sl)):
    ax.plot(xs, y, color=c, lw=1.0, zorder=3)
# SSD sits further left, where its region is tallest.
for (text, _), y0, y1, j in zip(TIERS, edges, edges[1:], (len(xs) // 2, len(xs) // 2, len(xs) // 4)):
    ax.text(xs[j], (y0[j] * y1[j]) ** 0.5, text, ha="center", va="center",
            fontsize=7.5, color="white")
ax.set_yscale("log", base=2)
ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(lo, hi)
xt = list(range(16384, P_MAX, 16384))
ax.set_xticks(xt, [f"{p // 1024}K" for p in xt])
yt = [128, 256, 512, 1024, 2048]
ax.set_yticks(yt, [f"{s // 1024}K" if s >= 1024 else str(s) for s in yt])
ax.minorticks_off()
ax.grid(False)
ax.set_xlabel("Prefix length")
ax.set_ylabel("Suffix length")
fig.savefig(HERE / "model_safe_boundary.pdf")
print("wrote model_safe_boundary.pdf")
