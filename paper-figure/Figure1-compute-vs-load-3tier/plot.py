"""Compute vs DRAM / SSD load time over prefix, one panel per suffix.
Reads only raw/, writes compute_vs_load.pdf next to this file.

    python3 plot.py

See README.md for the setup and what each quantity means.
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
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paperstyle import *  # noqa: E402,F403

RAW = HERE / "raw"
SUFFIXES = [64, 1024]
PRE = re.compile(r"KVPilot prefill req=(\d{8})-\S+ .*?hit=(\d+) dram=(\d+) e2e=([\d.]+)ms")
BENCH = re.compile(r"bench tokens=\s*(\d+) .*?load=\s*([\d.]+)ms")
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]

# Load per prefix, both TP ranks at once, no compute: independent of suffix.
load = {tier: {int(m[1]): float(m[2]) for l in open(RAW / f"bench_{tier}.out")
               if (m := BENCH.search(l))} for tier in ("dram", "ssd")}

# Compute = the [1] test: prefix already in this worker's HBM, only the suffix runs.
rows = [json.loads(l) for l in open(RAW / "trace.jsonl")]
e2e = {}
for f in sorted(RAW.glob("prefill-*.log.gz")):
    for l in gzip.open(f, "rt", errors="replace"):
        if m := PRE.search(l):
            e2e[int(m[1])] = (int(m[2]), int(m[3]), float(m[4]))
reps, bad = defaultdict(list), 0
for i, r in enumerate(rows):
    if r["tier"] != 1 or r["suffix"] not in SUFFIXES:
        continue
    hit, dram, e = e2e.get(i, (0, 1, 0))
    if hit >= r["prefix"] and dram == 0:
        reps[(r["prefix"], r["suffix"])].append(e)
    else:
        bad += 1
compute = {k: statistics.median(v) for k, v in reps.items()}
P = sorted({p for p, _ in compute})
print(f"compute cells {len(compute)}, reps per cell {sorted({len(v) for v in reps.values()})}, dropped {bad}")
print(f"{'prefix':>6} " + " ".join(f"{'s=' + str(s):>8}" for s in SUFFIXES) + f" {'dram':>7} {'ssd':>7}  (ms)")
for p in P:
    print(f"{p // 1024:5d}K " + " ".join(f"{compute[(p, s)]:8.1f}" for s in SUFFIXES)
          + f" {load['dram'][p]:7.1f} {load['ssd'][p]:7.1f}")

# Same height and font sizes as Eval-req-priority/lru_breakdown.pdf, so the
# two sit side by side at the same scale.
WSPACE = 0.1
fig, axes = plt.subplots(1, len(SUFFIXES), figsize=(2.6, 1.54), sharey=True)
for ax, s in zip(axes, SUFFIXES):
    ax.plot(P, [compute[(p, s)] for p in P], color=COMPUTE, marker=MARKERS[0], label="Compute")
    for tier, marker in (("dram", MARKERS[1]), ("ssd", MARKERS[2])):
        y = [load[tier][p] for p in P]
        ax.plot(P, y, color=LOAD, marker=marker, label="Load")
        # SSD above its line, DRAM below: the compute line runs just above DRAM.
        up = tier == "ssd"
        ax.annotate(tier.upper(), (P[1], y[1]), xytext=(4, 3 if up else -3),
                    textcoords="offset points", fontsize=6.5, color=LOAD,
                    ha="right" if up else "left", va="bottom" if up else "top",
                    path_effects=HALO)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(P, [f"{p // 1024}K" for p in P])
    ax.minorticks_off()
    ax.text(0.05, 0.95, f"Suffix {s if s < 1024 else f'{s // 1024}K'}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.5)
ax = axes[0]
ax.set_ylabel("Time (ms)")
ax.set_yticks([10, 30, 100, 300], ["10", "30", "100", "300"])
ax.set_ylim(5, 650)
ax.set_xlabel("Prefix length", x=1 + WSPACE / 2)
fig.subplots_adjust(left=0.2, right=0.98, bottom=0.27, top=0.87, wspace=WSPACE)
leg = {}
for h, lab in zip(*axes[0].get_legend_handles_labels()):
    leg.setdefault(lab, h)
x0, x1, top = axes[0].get_position().x0, axes[-1].get_position().x1, axes[0].get_position().y1
fig.legend(leg.values(), leg.keys(), loc="lower center", bbox_to_anchor=((x0 + x1) / 2, top),
           ncol=len(leg), handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2)
fig.savefig(HERE / "compute_vs_load.pdf")
print("wrote compute_vs_load.pdf")
