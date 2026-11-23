"""Prefill latency breakdown under LRU eviction.

    python3 plot_breakdown.py

Reads raw/<ds>-lru/prefill-0.log.gz for ds in swereq / tlreq. One bar per
trace: mean prefill latency per request, split into compute and exposed load
(gpu_stall). Prints every number on the figure; writes lru_breakdown.pdf.
"""
import gzip
import re
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paperstyle import *  # noqa: E402,F403

DATASETS = (("SWE", "swereq"), ("TraceLab", "tlreq"))

# One record per prefill; several processes share the log, so a record must not
# run into the next KVPilot line.
NOT_NEXT = r"(?:(?!KVPilot).)*?"
PRE = re.compile(rf"KVPilot prefill req=\S+ line=(\d+) {NOT_NEXT}"
                 rf"e2e=([\d.]+)ms{NOT_NEXT} gpu_stall=([\d.]+)ms")


def load(path):
    """line -> (e2e, gpu_stall) in ms."""
    out = {}
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            for m in PRE.finditer(line):
                out[int(m[1])] = (float(m[2]), float(m[3]))
    return out


res = {}
for name, ds in DATASETS:
    run = load(HERE / "raw" / f"{ds}-lru" / "prefill-0.log.gz")
    e2e = statistics.mean(e for e, _ in run.values())
    stall = statistics.mean(s for _, s in run.values())
    res[name] = dict(e2e=e2e, stall=stall)
    print(f"{name:8s} n={len(run)} prefill latency {e2e:6.1f} ms = compute {e2e - stall:6.1f} + "
          f"exposed load {stall:5.1f} ms (exposed {100 * stall / e2e:4.1f}%)")

fig, ax = plt.subplots(figsize=(1.5, 1.4))
for x, name in enumerate(res):
    r = res[name]
    ax.bar(x, r["e2e"] - r["stall"], 0.8, color=COMPUTE, edgecolor="white")
    ax.bar(x, r["stall"], 0.8, bottom=r["e2e"] - r["stall"], color=WAIT, edgecolor="white")
    ax.annotate(f"{100 * r['stall'] / r['e2e']:.0f}%", (x, r["e2e"]), xytext=(0, 1.5),
                textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color=WAIT)
ax.set_xticks(range(len(res)), list(res))
ax.set_xlim(-0.55, len(res) - 0.45)
ax.set_ylim(0, max(r["e2e"] for r in res.values()) * 1.12)
ax.set_ylabel("Prefill latency (ms)")
ax.legend(handles=[Patch(color=COMPUTE, label="Compute"), Patch(color=WAIT, label="Exposed load")],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.2,
          handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2)
fig.savefig(HERE / "lru_breakdown.pdf")
print("wrote lru_breakdown.pdf")
