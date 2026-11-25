"""Eval-EXP1 round 2: one trace per dataset on three models (1P1D, -p 1).

    python3 plot.py

Reads raw/<ds>/<model>/<arm>/ (benchmark.log, prefill-0.log.gz) of the two
random-order traces (swe_rand, tl_rand), blockwise threshold 5%. Requests are
matched across a model's arms by trace line; the first WARM lines (the trace's
K, warm-up) are left out. Prints every number on the figures and the required
pairs' paired differences; writes

  exp1-ttft.pdf          mean TTFT per placement, split into the time before
                         the prefill starts, the prefill's compute, and its
                         exposed load (gpu_stall)
  exp1-exposed-load.pdf  mean exposed load per request
  exp1-token-source.pdf  where the prompt tokens came from: HBM / DRAM / SSD
                         hits and computed tokens
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

BATCH = 50  # requests per batch for the paired 95% intervals
DATASETS = (("swe_rand", "SWE", 144), ("tl_rand", "TraceLab", 160))
MODELS = (("q30", "Qwen3-30B"), ("q36f8", "Qwen3.6-FP8"), ("q38", "Qwen3.8"))
LABEL = {"lru": "LRU", "belady": "Belady", "incp": "Incl-pen", "exclu": "Unblock", "exclb": "Blockwise"}
BASELINE = ("lru", "belady")
# Arm -> run directory. SWE Qwen3-30B's exclusive arms were rerun after the
# exclusive fixes (-wait); the TraceLab ones ran with the fixed code.
ARMS = {
    ("swe_rand", "q30"): {"lru": "lru", "belady": "belady", "incp": "incp", "exclu": "exclu-wait",
                          "exclb": "exclb-wait"},
    ("tl_rand", "q30"): {a: a for a in ("lru", "belady", "incp", "exclu", "exclb")},
}
HYBRID_ARMS = {a: a for a in ("lru", "belady", "exclb")}
# (slower, faster) pairs each model must show.
CHAIN = (("lru", "belady"), ("belady", "exclb"))
PAIRS = {"q30": CHAIN + (("incp", "exclu"), ("exclu", "exclb")), "q36f8": CHAIN, "q38": CHAIN}
TIERS = ("HBM", "DRAM", "SSD", "Computed")
TIER_COLORS = {"HBM": AQUA, "DRAM": YELLOW, "SSD": MAGENTA, "Computed": "#c8c8c8"}
PARTS = (("Before prefill", "#c8c8c8"), ("Prefill compute", COMPUTE), ("Exposed load", WAIT))

# One record per prefill; several processes share the log, so a record must not
# run into the next KVPilot line.
NOT_NEXT = r"(?:(?!KVPilot).)*?"
PRE = re.compile(rf"KVPilot prefill req=(\S+) line=(\d+) prompt=(\d+) new=(\d+) hit=(\d+) dram=(\d+) "
                 rf"e2e=([\d.]+)ms{NOT_NEXT} gpu_stall=([\d.]+)ms done=([\d.]+)")
RET = re.compile(rf"KVPilot prelookup retrieve req=(\S+) {NOT_NEXT}ssd_tokens=(\d+)")
OK = re.compile(r"^\[ok \] idx=(\d+) .*? sent=([\d.]+)")


def load(d):
    """line -> dict(prompt, new, hbm, dram, ssd, e2e, stall, ttft) for one run."""
    sent = {int(m[1]): float(m[2]) for line in open(d / "benchmark.log", errors="replace")
            if (m := OK.match(line))}
    out, rid_line, ssd = {}, {}, defaultdict(int)
    for line in gzip.open(d / "prefill-0.log.gz", "rt", errors="replace"):
        for m in RET.finditer(line):
            # One line per TP rank, each counting the same tokens.
            ssd[m[1]] = max(ssd[m[1]], int(m[2]))
        for m in PRE.finditer(line):
            k = int(m[2])
            if k < 0 or k not in sent:
                continue
            rid_line[m[1]] = k
            out[k] = dict(prompt=int(m[3]), new=int(m[4]), hbm=int(m[5]), loaded=int(m[6]),
                          e2e=float(m[7]), stall=float(m[8]), ttft=(float(m[9]) - sent[k]) * 1e3)
    for rid, k in rid_line.items():
        r = out[k]
        r["ssd"] = min(ssd.get(rid, 0), r["loaded"])
        r["dram"] = r["loaded"] - r["ssd"]
    return out


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def paired(a, b, keys):
    """Mean of a - b over keys and its 95% half-width from BATCH-request batch means."""
    d = [a[k]["ttft"] - b[k]["ttft"] for k in keys]
    nb = len(d) // BATCH
    means = [statistics.mean(d[i * BATCH:(i + 1) * BATCH]) for i in range(nb)]
    return statistics.mean(d), 1.96 * statistics.stdev(means) / nb ** 0.5


stats = {}
for ds, ds_name, warm in DATASETS:
    for model, model_name in MODELS:
        arms = ARMS.get((ds, model), HYBRID_ARMS)
        runs = {a: load(HERE / "raw" / ds / model / d) for a, d in arms.items()
                if (HERE / "raw" / ds / model / d / "prefill-0.log.gz").exists()}
        if set(runs) != set(arms):
            print(f"\n{ds_name} / {model_name}: incomplete, have {sorted(runs)}")
            continue
        keys = sorted(k for k in set.intersection(*(set(r) for r in runs.values())) if k >= warm)
        print(f"\n{ds_name} / {model_name}: lines >= {warm}, {len(keys)} requests")
        base = statistics.mean(runs["lru"][k]["ttft"] for k in keys)
        s = {}
        for a, run in runs.items():
            ttft = [run[k]["ttft"] for k in keys]
            e2e = statistics.mean(run[k]["e2e"] for k in keys)
            stall = statistics.mean(run[k]["stall"] for k in keys)
            prompt = sum(run[k]["prompt"] for k in keys)
            tok = {"HBM": sum(run[k]["hbm"] for k in keys), "DRAM": sum(run[k]["dram"] for k in keys),
                   "SSD": sum(run[k]["ssd"] for k in keys), "Computed": sum(run[k]["new"] for k in keys)}
            mean = statistics.mean(ttft)
            s[a] = dict(ttft=mean, e2e=e2e, stall=stall, share={t: tok[t] / prompt for t in TIERS})
            print(f"  {LABEL[a]:9s} TTFT mean {mean:6.1f} p50 {pct(ttft, .5):6.1f} p90 {pct(ttft, .9):6.1f} ms "
                  f"({mean / base * 100 - 100:+5.1f}% vs LRU) | prefill e2e {e2e:6.1f} ms "
                  f"= compute {e2e - stall:6.1f} + exposed load {stall:5.1f} | before prefill "
                  f"{mean - e2e:5.1f} | HBM/DRAM/SSD/computed "
                  + "/".join(f"{100 * s[a]['share'][t]:.1f}" for t in TIERS) + "%")
        for slow, fast in PAIRS[model]:
            m, h = paired(runs[slow], runs[fast], keys)
            print(f"  {LABEL[fast]} faster than {LABEL[slow]} by {m:+.1f} +- {h:.1f} ms")
        stats[(ds, model)] = s


def grid():
    fig, axes = plt.subplots(len(DATASETS), len(MODELS), figsize=(FULL, 2 * PANEL_H + 0.9),
                             gridspec_kw={"wspace": 0.28, "hspace": 0.75})
    return fig, axes


def arms_of(ds, model):
    return list(ARMS.get((ds, model), HYBRID_ARMS))


def finish(ax, ds_name, model_name, i, arms):
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([LABEL[a] for a in arms], rotation=20, ha="right", rotation_mode="anchor")
    panel_label(ax, f"({'abcdef'[i]}) {ds_name}, {model_name}")


def missing(ax):
    """A model not run on this trace (TraceLab Qwen3-30B): no panel."""
    ax.set_visible(False)


edge = {"edgecolor": "black", "linewidth": 0.4}

# exp1-ttft.pdf: TTFT = before prefill + prefill compute + exposed load.
fig, axes = grid()
i = 0
for r, (ds, ds_name, _) in enumerate(DATASETS):
    for c, (model, model_name) in enumerate(MODELS):
        ax, arms = axes[r][c], arms_of(ds, model)
        s = stats.get((ds, model))
        if s is None:
            missing(ax)
            continue
        else:
            for x, a in enumerate(arms):
                v = s[a]
                hatch = HATCHES[BASELINE.index(a)] if a in BASELINE else None
                parts = (v["ttft"] - v["e2e"], v["e2e"] - v["stall"], v["stall"])
                bottom = 0.0
                for (name, color), h in zip(PARTS, parts):
                    ax.bar(x, h, 0.6, bottom=bottom, color=color, hatch=hatch, label=name, **edge)
                    bottom += h
                if a != "lru":
                    ax.text(x, v["ttft"] * 1.02, f"{v['ttft'] / s['lru']['ttft'] * 100 - 100:+.1f}%",
                            ha="center", va="bottom", fontsize=5)
            ax.set_ylim(0, s["lru"]["ttft"] * 1.15)
            if c == 0 or stats.get((ds, MODELS[c - 1][0])) is None:
                ax.set_ylabel("TTFT (ms)")
        finish(ax, ds_name, model_name, i, arms)
        i += 1
top_legend(fig, [axes[0][0]])
out = HERE / "exp1-ttft.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"\nwrote {out.name}")

# exp1-exposed-load.pdf: mean exposed load per request.
fig, axes = grid()
i = 0
for r, (ds, ds_name, _) in enumerate(DATASETS):
    for c, (model, model_name) in enumerate(MODELS):
        ax, arms = axes[r][c], arms_of(ds, model)
        s = stats.get((ds, model))
        if s is None:
            missing(ax)
            continue
        else:
            for x, a in enumerate(arms):
                v = s[a]["stall"]
                hatch = HATCHES[BASELINE.index(a)] if a in BASELINE else None
                ax.bar(x, v, 0.6, color=WAIT, hatch=hatch, **edge)
                if a != "lru":
                    ax.text(x, v + s["lru"]["stall"] * 0.02, f"{v / s['lru']['stall'] * 100 - 100:+.0f}%",
                            ha="center", va="bottom", fontsize=5)
            ax.set_ylim(0, s["lru"]["stall"] * 1.15)
            if c == 0 or stats.get((ds, MODELS[c - 1][0])) is None:
                ax.set_ylabel("Exposed load (ms)")
        finish(ax, ds_name, model_name, i, arms)
        i += 1
out = HERE / "exp1-exposed-load.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out.name}")

# exp1-token-source.pdf: prompt tokens by where they came from.
fig, axes = grid()
i = 0
for r, (ds, ds_name, _) in enumerate(DATASETS):
    for c, (model, model_name) in enumerate(MODELS):
        ax, arms = axes[r][c], arms_of(ds, model)
        s = stats.get((ds, model))
        if s is None:
            missing(ax)
            continue
        else:
            bottom = [0.0] * len(arms)
            for t in TIERS:
                vals = [s[a]["share"][t] for a in arms]
                ax.bar(range(len(arms)), vals, 0.6, bottom=bottom, color=TIER_COLORS[t], label=t, **edge)
                bottom = [b + v for b, v in zip(bottom, vals)]
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.set_ylim(0, 1)
            if c == 0 or stats.get((ds, MODELS[c - 1][0])) is None:
                ax.set_ylabel("Prompt tokens")
        finish(ax, ds_name, model_name, i, arms)
        i += 1
top_legend(fig, [axes[0][0]])
out = HERE / "exp1-token-source.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out.name}")
