"""HBM free queue under LRU on HBM + DRAM + SSD, split by where each block
could live without slowing its next use, over time.

    python3 plot.py   # reads raw/, writes free_queue_tiers.pdf

The census (GET_HBM_POOL_DISTRIBUTION) walks each worker's pool once a second
into hbm-pool-worker-<EngineCore pid>.jsonl; only the prefill workers count,
their pids are in raw/prefill-*.pid. A free-queue block is a cached block with
ref_cnt 0 (state 1): an eviction candidate, sitting in HBM. Its row carries
(req_line, block index in that request) of the request that freed it. Its next
consumer is the first later request whose prompt starts with the same block
prefix (raw/toolagent_trace_turn5plus.jsonl, full blocks). That consumer's
measured verdict (raw/tier_safe_per_request.jsonl, from
../toolagent-turn5plus-tier-safe-per-request) decides the block:
  SSD safe       loading it from SSD would not slow its next use (< 5%)
  DRAM safe      from DRAM it would not; from SSD it would
  Needs HBM      neither (or the consumer has no verdict)
  Not reused     no later request uses it
Workers are summed per second of wall time, each carried forward over seconds
it skipped (the census runs from the scheduler step, so idle workers skip).
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paperstyle import *  # noqa: E402,F403

RAW = HERE / "raw"
BLOCK = 512
CACHED_FREE = 1
SSD, DRAM, HBM, NONE = range(4)
NAMES = ["SSD safe", "DRAM safe", "Needs HBM", "Not reused"]
COLORS = [AQUA, ORANGE, WAIT, GRID]

verdict = {}
for l in open(RAW / "tier_safe_per_request.jsonl"):
    r = json.loads(l)
    verdict[r["line"]] = SSD if r["ssd_safe"] else DRAM if r["dram_safe"] else HBM

records = [json.loads(l) for l in open(RAW / "toolagent_trace_turn5plus.jsonl")]
next_by_prefix: dict = {}
klass = [None] * len(records)   # per line, per full block: its class
for idx in range(len(records) - 1, -1, -1):
    d = records[idx]
    full = d["hash_ids"][: d["input_length"] // BLOCK]
    out = []
    for b in range(len(full)):
        key = tuple(full[: b + 1])
        j = next_by_prefix.get(key, -1)
        out.append(NONE if j < 0 else verdict.get(j, HBM))
        next_by_prefix[key] = idx
    klass[idx] = out


def classify(line, b):
    c = klass[line] if 0 <= line < len(klass) else []
    return c[b] if 0 <= b < len(c) else HBM


pids = [p.read_text().strip() for p in sorted(RAW.glob("prefill-*.pid"))]
files = [f for pid in pids for f in RAW.glob(f"hbm-pool-*-{pid}.jsonl*")]
assert files and len(files) == len(pids), (pids, files)
series, pool = [], 0
for f in files:
    opener = gzip.open if f.suffix == ".gz" else open
    s = {}
    for l in opener(f, "rt"):
        r = json.loads(l)
        counts = [0, 0, 0, 0]
        for row in r["blocks"]:
            if row[1] == CACHED_FREE:
                # req_block_idx is the last column in both census formats.
                counts[classify(row[3], row[-1])] += 1
        s[int(r["wall_s"])] = (*counts, r["num_in_use"], r["num_blank"])
    series.append(s)
    pool += r["pool_blocks"]

ts = np.arange(max(min(s) for s in series), min(max(s) for s in series) + 1)
total = np.zeros((len(ts), 6), dtype=np.int64)
for s in series:
    last = (0,) * 6
    for i, t in enumerate(ts):
        last = s.get(t, last)
        total[i] += last
ssd, dram, hbm, none, in_use, blank = total.T
t = (ts - ts[0]) / 60
free = ssd + dram + hbm + none
# Blocks freed without a hash (blank) never drop below ~5% of the pool, so the
# pool counts as full once cached free blocks hold 90% of it: from then on
# every new block evicts one.
full = free >= 0.9 * pool
print(f"workers {len(files)}, samples {len(ts)} ({t[-1]:.1f} min), pool {pool} blocks, "
      f"pool first full at {t[full][0] if full.any() else float('nan'):.1f} min")


def med(x, m):
    v = np.divide(x, free, out=np.full(len(free), np.nan), where=free > 0)
    return np.nanmedian(v[m]) if m.any() else float("nan")


for label, m in (("whole run", np.ones(len(t), bool)), ("pool full", full)):
    print(f"free queue ({label}): median {np.median(free[m]) if m.any() else 0:.0f} blocks; "
          + ", ".join(f"{n} {med(x, m):.1%}" for n, x in zip(NAMES, (ssd, dram, hbm, none))))
    print(f"  could leave HBM without slowing the next use (SSD or DRAM safe): "
          f"{med(ssd + dram, m):.1%}; of those could go to SSD: "
          f"{np.nanmedian(np.divide(ssd, ssd + dram, out=np.full(len(t), np.nan), where=(ssd + dram) > 0)[m]) if m.any() else float('nan'):.1%}")

fig, ax = plt.subplots(figsize=(COL, 1.4))
ax.stackplot(t, ssd, dram, hbm, none, colors=COLORS, alpha=0.9, lw=0, labels=NAMES)
ax.set_xlim(0, t[-1])
ax.set_ylim(0, None)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Free-queue blocks")
top_legend(fig, [ax], pad=0.02)
fig.savefig(HERE / "free_queue_tiers.pdf")
print("wrote free_queue_tiers.pdf")
