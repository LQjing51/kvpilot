"""Plot KV load stall per prefill/forward from KVPILOT_LOAD_STALL logs.

    python3 plot_load_stall.py <log_dir> [out.pdf]

See docs/observability.md for what the panels mean and how to read them.
"""
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Mirrors StallLogger.log(); `line=` optional so pre-2026-09-11 logs still parse.
LINE_RE = re.compile(
    r"KVPilot prefill req=(?P<req>\S+?)\s+(?:line=(?P<line>-?\d+)\s+)?"
    r"prompt=(?P<prompt>\d+)\s+new=(?P<new>\d+)\s+hit=(?P<hit>\d+)\s+"
    r"dram=(?P<dram>\d+)\s+e2e=(?P<e2e>[\d.]+)ms\s+stall=(?P<stall>[\d.]+)ms\s+"
    r"\((?P<pct>[\d.]+)%\)\s+waits=(?P<waits>\d+)\s+steps=(?P<steps>\d+)"
)

SRC = Path(sys.argv[1])
log_files = sorted(SRC.glob("prefill-*.log")) if SRC.is_dir() else [SRC]
log_dir = SRC if SRC.is_dir() else SRC.parent
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else log_dir / "load_stall.pdf"

rows = []
for path in log_files:
    m = re.search(r"prefill-(\d+)", path.name)
    worker = int(m.group(1)) if m else -1
    for text in path.open(errors="replace"):
        hit = LINE_RE.search(text)
        if hit:
            g = hit.groupdict()
            rows.append((worker, float(g["e2e"]), float(g["stall"]),
                         int(g["dram"]), int(g["waits"]), int(g["steps"])))

if not rows:
    sys.exit(f"no 'KVPilot prefill' lines under {SRC} -- was KVPILOT_LOAD_STALL set?")
print(f"parsed {len(rows)} prefill records from {len(log_files)} log(s)")

e2e = np.array([r[1] for r in rows])
stall = np.array([r[2] for r in rows])
dram = np.array([r[3] for r in rows], dtype=float)
waits = np.array([r[4] for r in rows], dtype=float)
steps = np.array([r[5] for r in rows], dtype=float)
pct = 100.0 * stall / e2e
n_workers = len({r[0] for r in rows})

loaded = dram > 0
stall_per_step = stall / steps
ms_per_wait = np.divide(stall, waits, out=np.zeros_like(stall), where=waits > 0)

BLUE, RED = "#2563EB", "#DC2626"

fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))
fig.suptitle(f"KV load stall per prefill  ({len(rows)} prefills, "
             f"{n_workers} prefill workers)", fontsize=13, fontweight="bold")

# No-load requests are all exactly 0 ms; histogramming them would put one bar
# ~9x the tallest loader bin on the axis and flatten the distribution.
ax = axes[0][0]
ax.hist(stall[loaded], bins=60, color=RED, alpha=.85,
        label=f"loaded from DRAM  (n={int(loaded.sum())})")
ax.plot([], [], " ", label=f"no load  (n={int((~loaded).sum())}), all at 0 ms")
ax.set_xlabel("stall per prefill (ms)")
ax.set_ylabel("prefills")
ax.set_title(f"Stall is confined to requests that loaded\n"
             f"no-load max = {stall[~loaded].max():.3f} ms", fontsize=10)
ax.legend(fontsize=8)

ax = axes[0][1]
ax.hist(pct[loaded], bins=60, color=RED, alpha=.85)
med = np.median(pct[loaded])
ax.axvline(med, color="black", ls="--", lw=1.2, label=f"median {med:.1f}%")
ax.set_xlabel("stall / e2e  (%)")
ax.set_ylabel("prefills")
ax.set_title("Fraction of prefill spent blocked\n(loaders only)", fontsize=10)
ax.xaxis.set_major_formatter(PercentFormatter())
ax.legend(fontsize=8)

ax = axes[0][2]
x, y = dram[loaded], stall[loaded]
ax.scatter(x / 1000, y, s=7, alpha=.25, color=RED, edgecolors="none")
# Through the origin: zero tokens loaded must mean zero stall.
k = float((x * y).sum() / (x * x).sum())
xs = np.linspace(0, x.max(), 50)
ax.plot(xs / 1000, k * xs, color="black", lw=1.3,
        label=f"through-origin fit\n{k*1000:.1f} ms / 1k tok")
edges = np.linspace(0, x.max(), 13)
idx = np.digitize(x, edges) - 1
bins = [(b, idx == b) for b in range(len(edges) - 1)]
bins = [(b, s) for b, s in bins if s.sum() >= 10]
if bins:
    bx = [(edges[b] + edges[b + 1]) / 2000 for b, _ in bins]
    ax.plot(bx, [np.median(y[s]) for _, s in bins], color="black", lw=1.6,
            ls=":", marker="o", ms=3.5, label="binned median (p10-p90)")
    ax.fill_between(bx, [np.percentile(y[s], 10) for _, s in bins],
                    [np.percentile(y[s], 90) for _, s in bins],
                    color="black", alpha=.10, linewidth=0)
ax.set_xlabel("DRAM tokens loaded (thousands)")
ax.set_ylabel("stall (ms)")
ax.set_title("Stall vs amount loaded from DRAM", fontsize=10)
ax.legend(fontsize=7.5, loc="upper left")

ax = axes[1][0]
ax.hist(stall_per_step[loaded], bins=60, color=BLUE, alpha=.85)
m = np.median(stall_per_step[loaded])
ax.axvline(m, color="black", ls="--", lw=1.2, label=f"median {m:.0f} ms")
ax.set_xlabel("stall per forward (ms)")
ax.set_ylabel("forwards")
ax.set_title(f"Per-forward stall\ntotal {steps.sum():.0f} forwards, "
             f"{stall.sum()/steps.sum():.0f} ms/forward mean", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1][1]
sel = loaded & (waits > 0)
ax.hist(ms_per_wait[sel], bins=60, color=BLUE, alpha=.85)
m = np.median(ms_per_wait[sel])
ax.axvline(m, color="black", ls="--", lw=1.2, label=f"median {m:.1f} ms")
ax.set_xlabel("stall per blocking call (ms)")
ax.set_ylabel("prefills")
ax.set_title(f"Cost of one layer wait\n{waits.sum()/steps.sum():.0f} waits per "
             "forward (~1 per attention layer)", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1][2]
tot_e2e, tot_stall = e2e.sum(), stall.sum()
share = 100 * tot_stall / tot_e2e
ax.bar(["prefill work"], [(tot_e2e - tot_stall) / 1000], color=BLUE,
       label=f"computing  {100 - share:.1f}%")
ax.bar(["prefill work"], [tot_stall / 1000], bottom=[(tot_e2e - tot_stall) / 1000],
       color=RED, label=f"blocked on KV  {share:.1f}%")
ax.set_ylabel("seconds (sum of per-prefill windows)")
ax.set_title(f"Aggregate: {tot_stall/1000:.0f} s blocked of {tot_e2e/1000:.0f} s\n"
             f"sum(stall)/sum(e2e) over {len(rows)} prefills, {n_workers} workers",
             fontsize=10)
ax.legend(fontsize=8)

for a in axes.flat:
    a.grid(alpha=.25, linewidth=.6)
    a.set_axisbelow(True)

fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
