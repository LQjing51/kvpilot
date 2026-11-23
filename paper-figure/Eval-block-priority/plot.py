"""KVPilot placements on swe_blockprio_k160 at the default HBM (1P1D, -p 1).

    python3 plot.py

Reads raw/pab-sweblock160-<arm>-<tag>-logs/ (benchmark.log, prefill-0.log.gz,
decode-0.log): LRU, Belady and the three KVPilot placements with the oracle
signal (tag r1), and the KVPilot three again with the online-predicted signal
(tag on1). Requests are matched across runs by trace line; the first WARM lines
(the first round, cold mid-joined sessions) are left out. Prints every number
on the figure and the paired differences; writes block_priority_ttft.pdf.

(a) mean TTFT per placement, oracle and online signal; (b) where the prompt
tokens came from with the oracle signal: HBM / DRAM / SSD hits and computed
tokens (the online runs' are printed).
"""
import gzip
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paperstyle import *  # noqa: E402,F403

WARM = 160
LINES = 1300
BATCH = 50  # requests per batch for the paired 95% intervals
ARMS = (("LRU", "lru"), ("Belady", "belady"), ("Incl-pen", "incp"),
        ("Unblock", "exclu"), ("Blockwise", "exclb"))
OURS_ARMS = ("incp", "exclu", "exclb")
SIGNALS = (("Oracle", "r1"), ("Online", "on1"))
TIERS = ("HBM", "DRAM", "SSD", "Computed")
# Computed tokens are a share of the prompt, not compute time: neutral gray.
TIER_COLORS = {"HBM": AQUA, "DRAM": YELLOW, "SSD": MAGENTA, "Computed": "#c8c8c8"}
BASE_STYLE = {"lru": (ORANGE, HATCHES[0]), "belady": (VIOLET, HATCHES[1])}
SIGNAL_HATCH = {"r1": None, "on1": HATCHES[3]}

# One record per prefill; several processes share the log, so a record must not
# run into the next KVPilot line.
NOT_NEXT = r"(?:(?!KVPilot).)*?"
PRE = re.compile(rf"KVPilot prefill req=(\S+) line=(\d+) prompt=(\d+) {NOT_NEXT}hit=(\d+) "
                 rf"dram=(\d+) e2e=([\d.]+)ms{NOT_NEXT} done=([\d.]+)(?: blocked=([\d.]+)ms)?")
RET = re.compile(rf"KVPilot prelookup retrieve req=(\S+) {NOT_NEXT}ssd_tokens=(\d+)")
OK = re.compile(r"^\[ok \] idx=(\d+) .*? sent=([\d.]+)")
MISSES = re.compile(r"label_misses=(\d+)/(\d+)")


def open_log(p):
    return gzip.open(p, "rt", errors="replace") if p.suffix == ".gz" else open(p, errors="replace")


def load(d):
    """line -> dict(prompt, hbm, dram, ssd, e2e, blocked, ttft) for one run."""
    sent = {int(m[1]): float(m[2]) for line in open_log(d / "benchmark.log") if (m := OK.match(line))}
    out, rid_line, ssd = {}, {}, defaultdict(int)
    for line in open_log(d / "prefill-0.log.gz"):
        for m in RET.finditer(line):
            # One line per TP rank, each counting the same tokens.
            ssd[m[1]] = max(ssd[m[1]], int(m[2]))
        for m in PRE.finditer(line):
            k = int(m[2])
            rid_line[m[1]] = k
            out[k] = dict(prompt=int(m[3]), hbm=int(m[4]), loaded=int(m[5]), e2e=float(m[6]),
                          done=float(m[7]), blocked=float(m[8] or 0.0))
    for rid, k in rid_line.items():
        r = out[k]
        r["ssd"] = min(ssd.get(rid, 0), r["loaded"])
        r["dram"] = r["loaded"] - r["ssd"]
        r["ttft"] = (r["done"] - sent[k]) * 1e3
    missing = [k for k in range(LINES) if k not in out]
    if missing:
        raise SystemExit(f"{d.name}: {len(missing)} trace lines without a prefill record")
    return out


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def paired(a, b, keys):
    """Mean of b - a over keys and its 95% half-width from BATCH-request batch means."""
    d = [b[k]["ttft"] - a[k]["ttft"] for k in keys]
    nb = len(d) // BATCH
    means = [statistics.mean(d[i * BATCH:(i + 1) * BATCH]) for i in range(nb)]
    return statistics.mean(d), 1.96 * statistics.stdev(means) / nb ** 0.5


runs = {}
for _, arm in ARMS:
    for _, tag in SIGNALS:
        d = HERE / "raw" / f"pab-sweblock160-{arm}-{tag}-logs"
        if d.exists():
            runs[(arm, tag)] = load(d)
keys = list(range(WARM, LINES))
base = statistics.mean(runs[("lru", "r1")][k]["ttft"] for k in keys)

print(f"swe_blockprio_k160: lines {WARM}-{LINES - 1} ({len(keys)} requests), TTFT = prefill done - sent")
stats = {}
for label, arm in ARMS:
    for sig, tag in SIGNALS:
        if (arm, tag) not in runs:
            continue
        run = runs[(arm, tag)]
        ttft = [run[k]["ttft"] for k in keys]
        e2e = [run[k]["e2e"] for k in keys]
        prompt = sum(run[k]["prompt"] for k in keys)
        tok = {"HBM": sum(run[k]["hbm"] for k in keys), "DRAM": sum(run[k]["dram"] for k in keys),
               "SSD": sum(run[k]["ssd"] for k in keys)}
        tok["Computed"] = prompt - sum(tok.values())
        blocked = [run[k]["blocked"] for k in keys]
        mean = statistics.mean(ttft)
        stats[(arm, tag)] = dict(ttft=mean, share={t: tok[t] / prompt for t in TIERS})
        misses = ""
        if tag == "on1":
            found = MISSES.findall((HERE / "raw" / f"pab-sweblock160-{arm}-{tag}-logs" / "decode-0.log").read_text())
            misses = f" | signal labels missing from the distribution {found[-1][0]}/{found[-1][1]}"
        print(f"  {label:9s} {sig:6s} TTFT mean {mean:6.1f} p50 {pct(ttft, .5):6.1f} p90 {pct(ttft, .9):6.1f} "
              f"p99 {pct(ttft, .99):6.1f} ms ({mean / base * 100 - 100:+5.1f}% vs LRU) | prefill e2e "
              f"{statistics.mean(e2e):6.1f} ms | blocked {sum(blocked) / 1e3:.1f} s on "
              f"{sum(b > 0 for b in blocked)} requests | prompt tokens HBM/DRAM/SSD/computed "
              + "/".join(f"{100 * tok[t] / prompt:.1f}" for t in TIERS) + f"%{misses}")

print("paired TTFT, oracle signal: b - a, 95% interval from batch means")
order = [arm for _, arm in ARMS]
for i, a in enumerate(order):
    for b in order[:i]:
        m, h = paired(runs[(a, "r1")], runs[(b, "r1")], keys)
        print(f"  {b:6s} - {a:6s} {m:+6.1f} +- {h:4.1f} ms")
print("paired TTFT, online - oracle")
for arm in OURS_ARMS:
    m, h = paired(runs[(arm, "r1")], runs[(arm, "on1")], keys)
    print(f"  {arm:6s} {m:+6.1f} +- {h:4.1f} ms")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, PANEL_H + 0.3), gridspec_kw={"wspace": 0.25})
edge = {"edgecolor": "black", "linewidth": 0.4}

# (a) mean TTFT: baselines hatched; KVPilot solid, oracle and online side by side.
width = 0.4
for i, (label, arm) in enumerate(ARMS):
    if arm in OURS_ARMS:
        bars = [(i + (j - 0.5) * width, tag) for j, (_, tag) in enumerate(SIGNALS)]
    else:
        bars = [(i, "r1")]
    for x, tag in bars:
        v = stats[(arm, tag)]["ttft"]
        if arm in OURS_ARMS:
            ax1.bar(x, v, width, color=OURS, hatch=SIGNAL_HATCH[tag],
                    label=dict((t, s) for s, t in SIGNALS)[tag], **edge)
        else:
            color, hatch = BASE_STYLE[arm]
            ax1.bar(x, v, width, color=color, hatch=hatch, **edge)
        if arm != "lru":
            ax1.text(x, v + 4, f"{v / base * 100 - 100:+.0f}%", ha="center", va="bottom", fontsize=5)
ax1.set_xticks(range(len(ARMS)))
ax1.set_xticklabels([label for label, _ in ARMS])
ax1.set_ylabel("TTFT (ms)")
ax1.set_ylim(0, base * 1.15)
panel_label(ax1, "(a) Mean TTFT")

# (b) prompt token sources with the oracle signal, one stacked bar per placement.
bottom = [0.0] * len(ARMS)
for t in TIERS:
    vals = [stats[(arm, "r1")]["share"][t] for _, arm in ARMS]
    ax2.bar(range(len(ARMS)), vals, 0.6, bottom=bottom, color=TIER_COLORS[t], label=t, **edge)
    bottom = [b + v for b, v in zip(bottom, vals)]
ax2.set_xticks(range(len(ARMS)))
ax2.set_xticklabels([label for label, _ in ARMS])
ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax2.set_ylim(0, 1)
ax2.set_ylabel("Prompt tokens")
panel_label(ax2, "(b) Token source")

top_legend(fig)

out = HERE / "block_priority_ttft.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out.name}")
