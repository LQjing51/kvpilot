"""HBM / DRAM / SSD safe shares for ToolAgent + SWE + TraceLab (prefix>0 only).

    python3 plot.py

Reads raw/{toolagent,swe,tracelab}/ (part*/ merged); one grouped bar chart,
x = dataset. Prefix-0 lines were never emitted by make_trace.
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paperstyle import *  # noqa: E402,F403
from tier_logs import dataset, runs  # noqa: E402

THRESH = 0.05
TIERS = (("HBM safe", "#1c5cab"), ("DRAM safe", "#3987e5"), ("SSD safe", "#86b6ef"))
DATASETS = (
    ("ToolAgent", HERE / "raw/toolagent"),
    ("SWE", HERE / "raw/swe"),
    ("TraceLab", HERE / "raw/tracelab"),
)

shares = {}
for name, d in DATASETS:
    if not all((r / "trace.jsonl").exists() for r in runs(d)):
        print(f"{name}: no trace.jsonl under {d}; skip")
        continue
    rows, res, fixed = dataset(d)
    why = Counter(w for _, w in res.values() if w)
    prefix = {r["src_line"]: r["prefix"] for r in rows}
    tier, dropped = Counter(), Counter()
    for line, p in prefix.items():
        if p == 0:
            dropped["prefix=0 (should not appear)"] += 1
            continue
        e = {k: res.get((line, k), (None, "missing"))[0] for k in ("t1", "t2", "t4")}
        if None in e.values():
            dropped["incomplete line"] += 1
            continue
        if (e["t4"] - e["t1"]) / e["t1"] < THRESH:
            tier["SSD safe"] += 1
        elif (e["t2"] - e["t1"]) / e["t1"] < THRESH:
            tier["DRAM safe"] += 1
        else:
            tier["HBM safe"] += 1
    n = sum(tier.values())
    if not n:
        print(f"{name}: no usable lines; dropped {dict(+dropped) or 'none'}")
        continue
    shares[name] = {t: tier[t] / n for t, _ in TIERS}
    print(f"{name}: lines {len(prefix)}, usable {n}, dropped {dict(+dropped) or 'none'}")
    print(f"  failed checks after retry {dict(why) or 'none'}; "
          f"retry fixed {dict(fixed) or 'none'}")
    print("  " + "  ".join(f"{t} {tier[t]} = {tier[t] / n:.1%}" for t, _ in TIERS))

names = [n for n, _ in DATASETS if n in shares]
w = 0.26
fig, ax = plt.subplots(figsize=(1.0 + 0.9 * max(len(names), 1), 1.5))
for i, (t, c) in enumerate(TIERS):
    xs = [j + (i - 1) * w for j in range(len(names))]
    ys = [shares[n][t] for n in names]
    ax.bar(xs, ys, width=w, color=c, label=t)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.0%}", (x, y), xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6)
ax.set_xticks(range(len(names)), names)
ax.set_xlim(-0.6, max(len(names), 1) - 0.4)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_ylabel("Request Distribution")
top_legend(fig, ncol=3)
fig.savefig(HERE / "request_safe.pdf")
print("wrote request_safe.pdf")
