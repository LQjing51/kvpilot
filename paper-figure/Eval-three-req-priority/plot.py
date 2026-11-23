"""Request-level priority eviction vs LRU on the req-priority traces (1P1D, -p 1).

    python3 plot.py

Reads raw/<ds>-<arm>/ (benchmark.log, prefill-0.log.gz) for ds in swereq / tlreq
and arm in lru / reqprio, and raw/req_<trace>.json (each request's own measured
request-safe tier). Requests are matched across the two arms of a dataset by
trace line; only lines completed in both arms count. Prints every number on
the figure; writes req_priority_e2e.pdf.

(a)(b) mean prefill e2e per request, split into exposed load (gpu_stall) and
the rest; (c) where the prefix tokens came from; (d) per-trace share of
input tokens whose measured request-safe tier is HBM / DRAM / SSD.
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
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from paperstyle import *  # noqa: E402,F403

DATASETS = (("SWE", "swereq", "swe_reqprio"), ("TraceLab", "tlreq", "tracelab_reqprio"))
ARMS = (("LRU", "lru"), ("Req-prio", "reqprio"))
TIERS = ("HBM", "DRAM", "SSD")
TIER_COLORS = {"HBM": AQUA, "DRAM": YELLOW, "SSD": MAGENTA}

# One record per prefill; several processes share the log, so a record must not
# run into the next KVPilot line.
NOT_NEXT = r"(?:(?!KVPilot).)*?"
PRE = re.compile(rf"KVPilot prefill req=(\S+) line=(\d+) {NOT_NEXT}hit=(\d+) dram=(\d+) "
                 rf"e2e=([\d.]+)ms{NOT_NEXT} gpu_stall=([\d.]+)ms done=([\d.]+)")
RET = re.compile(rf"KVPilot prelookup retrieve req=(\S+) {NOT_NEXT}ssd_tokens=(\d+)")
OK = re.compile(r"^\[ok \] idx=(\d+) .*? ttft=([\d.]+)ms.*? sent=([\d.]+)")


def open_log(p):
    return gzip.open(p, "rt", errors="replace") if p.suffix == ".gz" else open(p, errors="replace")


def load(d):
    """line -> dict(e2e, stall, hit, dram, ssd, ttft) for one arm."""
    sent = {}
    for line in open_log(d / "benchmark.log"):
        m = OK.match(line)
        if m:
            sent[int(m[1])] = float(m[3])
    plog = d / "prefill-0.log.gz"
    if not plog.exists():
        plog = d / "prefill-0.log"
    out, rid_line, ssd = {}, {}, defaultdict(int)
    for line in open_log(plog):
        for m in RET.finditer(line):
            # One line per TP rank, each counting the same tokens.
            ssd[m[1]] = max(ssd[m[1]], int(m[2]))
        for m in PRE.finditer(line):
            k = int(m[2])
            rid_line[m[1]] = k
            out[k] = dict(hit=int(m[3]), dram=int(m[4]), e2e=float(m[5]), stall=float(m[6]),
                          done=float(m[7]))
    for rid, k in rid_line.items():
        r = out[k]
        r["ssd"] = min(ssd.get(rid, 0), r["dram"])
        r["ttft"] = (r["done"] - sent[k]) * 1e3 if k in sent else None
    return out


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def sources(run, keys):
    tok = {"HBM": sum(run[k]["hit"] for k in keys),
           "SSD": sum(run[k]["ssd"] for k in keys)}
    tok["DRAM"] = sum(run[k]["dram"] for k in keys) - tok["SSD"]
    return tok


def misreads(run, keys, table):
    """Prefix tokens from a slower / faster tier than the request's own."""
    rank = {t: i for i, t in enumerate(TIERS)}
    slow = fast = total = 0
    for k in keys:
        want = (table.get(str(k)) or {}).get("tier")
        if want is None:
            continue
        r = run[k]
        for got, n in (("HBM", r["hit"]), ("DRAM", r["dram"] - r["ssd"]), ("SSD", r["ssd"])):
            total += n
            slow += n * (rank[got] > rank[want])
            fast += n * (rank[got] < rank[want])
    return slow / max(total, 1), fast / max(total, 1)


res = {}
for name, ds, trace in DATASETS:
    dirs = {arm: HERE / "raw" / f"{ds}-{arm}" for _, arm in ARMS}
    if not all((d / "benchmark.log").exists() for d in dirs.values()):
        print(f"{name}: missing {[str(d) for d in dirs.values() if not (d / 'benchmark.log').exists()]}; skip")
        continue
    runs = {arm: load(d) for arm, d in dirs.items()}
    keys = sorted(set.intersection(*(set(r) for r in runs.values())))
    table_path = HERE / "raw" / f"req_{trace}.json"
    table = json.loads(table_path.read_text()) if table_path.exists() else {}
    print(f"== {name}: {len(keys)} requests in both arms "
          f"({', '.join(f'{a} {len(r)}' for a, r in runs.items())})")
    res[name] = {}
    for label, arm in ARMS:
        run = runs[arm]
        e2e = [run[k]["e2e"] for k in keys]
        stall = [run[k]["stall"] for k in keys]
        ttft = [run[k]["ttft"] for k in keys if run[k]["ttft"] is not None]
        tok = sources(run, keys)
        tot = sum(tok.values()) or 1
        slow, fast = misreads(run, keys, table) if table else (float("nan"),) * 2
        res[name][arm] = dict(e2e=statistics.mean(e2e), stall=statistics.mean(stall),
                              share={t: tok[t] / tot for t in TIERS})
        print(f"  {label:13s} prefill e2e mean {statistics.mean(e2e):6.1f} p50 {pct(e2e, .5):6.1f} "
              f"p90 {pct(e2e, .9):6.1f} p99 {pct(e2e, .99):6.1f} ms | exposed load (gpu_stall) "
              f"mean {statistics.mean(stall):5.1f} ms, sum {sum(stall) / 1e3:6.1f} s | rest "
              f"{statistics.mean(e2e) - statistics.mean(stall):6.1f} ms")
        print(f"  {'':13s} TTFT (prefill done - sent) mean {statistics.mean(ttft):6.1f} p50 "
              f"{pct(ttft, .5):6.1f} p90 {pct(ttft, .9):6.1f} p99 {pct(ttft, .99):6.1f} ms | prefix "
              f"tokens {tot / 1e6:.1f}M from HBM/DRAM/SSD "
              + "/".join(f"{100 * tok[t] / tot:.1f}" for t in TIERS)
              + f"% | vs own tier: slower {100 * slow:.1f}%, faster {100 * fast:.1f}%")
    a, b = runs["lru"], runs["reqprio"]
    d_e2e = [b[k]["e2e"] - a[k]["e2e"] for k in keys]
    d_stall = [b[k]["stall"] - a[k]["stall"] for k in keys]
    gain = -statistics.mean(d_e2e) / res[name]["lru"]["e2e"]
    res[name]["gain"] = gain
    both = [k for k in keys if a[k]["ttft"] is not None and b[k]["ttft"] is not None]
    t_gain = -statistics.mean(b[k]["ttft"] - a[k]["ttft"] for k in both) / statistics.mean(
        a[k]["ttft"] for k in both)
    loaded = [k for k in keys if a[k]["hit"] + a[k]["dram"] > 0]
    l_gain = -statistics.mean(b[k]["e2e"] - a[k]["e2e"] for k in loaded) / statistics.mean(
        a[k]["e2e"] for k in loaded)
    print(f"  paired: prefill e2e {statistics.mean(d_e2e):+.1f} ms/req ({-100 * gain:+.1f}%), "
          f"median {statistics.median(d_e2e):+.1f} ms, {sum(x < 0 for x in d_e2e)}/{len(keys)} faster; "
          f"gpu_stall {statistics.mean(d_stall):+.1f} ms/req; TTFT {-100 * t_gain:+.1f}%; "
          f"requests with a prefix ({len(loaded)}): e2e {-100 * l_gain:+.1f}%")

# Request-safe token mix of each trace (independent of A/B arms): weight each
# request by input_length, bucket by its own measured tier in req_*.json.
safe = {}
for name, ds, trace in DATASETS:
    table_path = HERE / "raw" / f"req_{trace}.json"
    trace_path = ROOT / "datasets" / "req-priority-trace" / f"{trace}.jsonl"
    if not table_path.exists() or not trace_path.exists():
        print(f"{name}: skip safe-token mix (missing table or trace)")
        continue
    table = json.loads(table_path.read_text())
    tok = {t: 0 for t in TIERS}
    n_none = 0
    for i, line in enumerate(open(trace_path)):
        r = json.loads(line)
        k = str(int(r.get("line", i)))
        tier = (table.get(k) or {}).get("tier")
        n = int(r["input_length"])
        if tier in tok:
            tok[tier] += n
        else:
            n_none += n
    tot = sum(tok.values()) or 1
    safe[name] = {t: tok[t] / tot for t in TIERS}
    print(f"== {name} safe-token mix (input tokens, exclude unmeasured {n_none}): "
          + "/".join(f"{t} {100 * tok[t] / tot:.1f}" for t in TIERS) + f"% of {tot / 1e6:.1f}M")

if not res:
    sys.exit("no complete dataset under raw/")

def hatch(arm):
    return None if arm == "reqprio" else HATCHES[0]


names = list(res)
n_safe = 1 if safe else 0
fig, axes = plt.subplots(
    1, len(names) + 1 + n_safe,
    figsize=(1.1 * len(names) + 2.2 + 1.1 * n_safe, 1.4),
    gridspec_kw={"width_ratios": [1] * len(names) + [1.9] + ([1.1] if n_safe else [])})
for ax, name in zip(axes, names):
    for x, (_, arm) in enumerate(ARMS):
        r = res[name][arm]
        ax.bar(x, r["e2e"] - r["stall"], 0.6, color=COMPUTE, hatch=hatch(arm), edgecolor="white")
        ax.bar(x, r["stall"], 0.6, bottom=r["e2e"] - r["stall"], color=WAIT, hatch=hatch(arm),
               edgecolor="white")
    ax.annotate(f"{-100 * res[name]['gain']:+.0f}%".replace("-", "\u2212"),
                (1, res[name]["reqprio"]["e2e"]), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=OURS)
    ax.set_xticks(range(len(ARMS)), [label for label, _ in ARMS])
    ax.set_xlim(-0.6, len(ARMS) - 0.4)
    ax.set_ylim(0, res[name]["lru"]["e2e"] * 1.15)
    ax.set_xlabel(name)
axes[0].set_ylabel("Prefill latency (ms)")

ax = axes[len(names)]
w = 0.38
xt, xl = [], []
for j, name in enumerate(names):
    for i, (label, arm) in enumerate(ARMS):
        x = j + (i - 0.5) * w * 1.1
        bottom = 0
        for t in TIERS:
            v = res[name][arm]["share"][t]
            ax.bar(x, v, w, bottom=bottom, color=TIER_COLORS[t], edgecolor="white", hatch=hatch(arm))
            bottom += v
        xt.append(x)
        xl.append(label)
ax.set_xticks(xt, xl)
ax.set_xlim(-0.5, len(names) - 0.5)
for j, name in enumerate(names):
    ax.annotate(name, (j, 0), xycoords=("data", "axes fraction"), xytext=(0, -12),
                textcoords="offset points", ha="center", va="top", fontsize=7.5)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_ylabel("Prefix tokens")

if safe:
    ax = axes[-1]
    safe_names = [n for n in ("SWE", "TraceLab") if n in safe]
    for x, name in enumerate(safe_names):
        bottom = 0
        for t in TIERS:
            v = safe[name][t]
            ax.bar(x, v, 0.6, bottom=bottom, color=TIER_COLORS[t], edgecolor="white")
            bottom += v
    ax.set_xticks(range(len(safe_names)), safe_names)
    ax.set_xlim(-0.6, len(safe_names) - 0.4)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Safe tokens")

fig.tight_layout(w_pad=0.6)
handles = [Patch(color=COMPUTE, label="Compute"), Patch(color=WAIT, label="Exposed load")] + [
    Patch(color=TIER_COLORS[t], label=t) for t in TIERS]
y = max(a.get_position().y1 for a in axes) + 0.02
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(handles),
           handletextpad=0.3, borderaxespad=0.2)
fig.savefig(HERE / "req_priority_e2e.pdf")
print("wrote req_priority_e2e.pdf")
