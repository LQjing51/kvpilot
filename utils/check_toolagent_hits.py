"""Theoretical vs actual prefix hit, and scheduling order, for the toolagent run.

    python3 utils/check_toolagent_hits.py [raw_dir] [trace]

theory   longest prefix of full 512-token blocks that any earlier trace line put
         in its prompt (infinite pool; partial last block never stored since
         save_unfull_chunk=False; only prompt KV is stored, D has no LMCache)
feasible same, but counting only blocks whose producer (any line) finished
         prefill before this request's lookup -> what ordering allowed
lru      same as theory, through an LRU of the Mooncake capacity in trace order
actual   max(C, H) from the first LMCache lookup line (C = HBM, H is nested)
"""
import gzip
import json
import re
import sys
from bisect import bisect_left
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper-figure/toolagent-hit-rate-and-exposed-load/raw"
TRACE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "datasets/mooncake-trace/toolagent_trace.jsonl"
BLOCK = 512
KV_BYTES = 98_304                      # Qwen3-30B-A3B, all layers, both TP ranks
MOONCAKE_GIB = 800                     # 2 P x 2 TP x 200 GiB global_segment_size
HIGH, EVICT = 0.9, 0.1                 # mooncake_master eviction watermark / ratio

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOOK = re.compile(r"\[(\S+ \S+)\] LMCache INFO: Reqid: (\d{8})-\S+ Total tokens (\d+), "
                  r"Inference Engine computed tokens: (\d+), LMCache hit tokens: (\d+)")
DONE = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d\.\d+) \[kvpilot_load_stall.py:\d+\] KVPilot prefill "
                  r"req=\S+ line=(\d+) ")

tr = [json.loads(l) for l in open(TRACE)]
N = len(tr)
full = [d["hash_ids"][:d["input_length"] // BLOCK] for d in tr]

look, done, nlook = {}, {}, defaultdict(int)
for w in (0, 1):
    for line in gzip.open(RAW / f"prefill-{w}.log.gz", "rt", errors="replace"):
        line = ANSI.sub("", line)
        if m := LOOK.search(line):
            i = int(m[2])
            nlook[i] += 1
            if i not in look:
                t = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S,%f").timestamp()
                look[i] = (t, w, int(m[3]), int(m[4]), int(m[5]))
        elif m := DONE.search(line):
            done.setdefault(int(m[2]), datetime.strptime("2026-" + m[1], "%Y-%m-%d %H:%M:%S.%f").timestamp())
assert len(look) == N and len(done) == N, (len(look), len(done), N)
assert all(look[i][2] == tr[i]["input_length"] for i in range(N))


def lead(blocks, ok):
    k = 0
    while k < len(blocks) and ok(blocks[k]):
        k += 1
    return k


seen, theory = set(), np.zeros(N, int)
for i in range(N):
    theory[i] = lead(full[i], seen.__contains__)
    seen.update(full[i])

cap = int(MOONCAKE_GIB * 2**30 / (BLOCK * KV_BYTES))
lru, lru_hit = OrderedDict(), np.zeros(N, int)
for i in range(N):
    lru_hit[i] = lead(full[i], lru.__contains__)
    for h in full[i]:
        lru[h] = None
        lru.move_to_end(h)
    if len(lru) > HIGH * cap:
        for _ in range(int(EVICT * cap)):
            lru.popitem(last=False)

avail = {}
for i in sorted(range(N), key=done.get):
    for h in full[i]:
        avail.setdefault(h, done[i])
feasible = np.array([lead(full[i], lambda h, t=look[i][0]: avail.get(h, 1e18) < t) for i in range(N)])

T = np.array([d["input_length"] for d in tr])
C = np.array([look[i][3] for i in range(N)])
H = np.array([look[i][4] for i in range(N)])
act = np.maximum(C, H) // BLOCK                 # full blocks actually reused
th, fe, lr = theory, feasible, lru_hit


def pct(a, b):
    return f"{100 * a / b:.1f}%"


print(f"requests {N}, looked up more than once {sum(v > 1 for v in nlook.values())}, "
      f"H < C (Mooncake lost what HBM has) {(H < C).sum()}")
uniq = len(seen)
print(f"unique full blocks {uniq} = {uniq * BLOCK * KV_BYTES / 2**30:.0f} GiB KV vs Mooncake "
      f"{MOONCAKE_GIB} GiB ({cap} blocks); HBM {1_313_792 // BLOCK} blocks/P worker x 2")
tok = lambda x: (x * BLOCK).sum()
print(f"prompt tokens {T.sum():,}")
for name, x in (("theory", th), ("lru(800GiB)", lr), ("feasible", fe), ("actual", act)):
    print(f"  {name:<12} reused tokens {tok(x):>13,}  {pct(tok(x), T.sum())}")
print(f"  actual HBM  {C.sum():>13,}  {pct(C.sum(), T.sum())}   actual DRAM {(H - np.minimum(H, C)).clip(0).sum():,}")

short = act < th
print(f"\nreach theory {(~short).sum()} ({pct((~short).sum(), N)}), short {short.sum()} "
      f"({pct(short.sum(), N)}), above theory {(act > th).sum()}")
lost = (th - act).clip(0)
print(f"shortfall tokens {tok(lost):,} = {pct(tok(lost), tok(th))} of theory reuse; "
      f"per short request median {np.median(lost[short]) * BLOCK:.0f} tok, "
      f"median share of its theory {np.median(lost[short] / th[short]):.0%}")
order = short & (fe < th)
evict = short & (fe >= th)
print(f"  cause: producer not yet finished at lookup (order/concurrency) {order.sum()}, "
      f"tokens {tok((th - np.maximum(fe, act)).clip(0)[order]):,}")
print(f"         prefix was available but not found (evicted from HBM+DRAM) {evict.sum()}, "
      f"tokens {tok(lost[evict]):,}")
print(f"  lru(800GiB) also predicts the miss for {(short & (lr < th) & evict).sum()} of the {evict.sum()} evicted")
hbm_only = (~short) & (C // BLOCK < th)
print(f"  reached theory but partly from DRAM: {hbm_only.sum()}")

ts = np.array([d["timestamp"] for d in tr]) / 1000
print("\nby trace time (min): requests, short, shortfall share of theory tokens")
for a in range(0, 60, 10):
    m = (ts >= a * 60) & (ts < (a + 10) * 60)
    print(f"  {a:>2}-{a + 10:<2} {m.sum():>5} {pct(short[m].sum(), m.sum()):>6} {pct(tok(lost[m]), tok(th[m])):>6}")

# Wall-clock gap since the deepest theoretical block was last looked up by any request.
t_look = np.array([look[i][0] for i in range(N)])
last, gap = {}, np.full(N, np.nan)
for i in np.argsort(t_look, kind="stable"):
    if th[i] and full[i][th[i] - 1] in last:
        gap[i] = t_look[i] - last[full[i][th[i] - 1]]
    for h in full[i]:
        last[h] = t_look[i]
print(f"run wall time {(t_look.max() - t_look.min()) / 60:.0f} min vs trace {ts.max() / 60:.0f} min")
print("by wall-clock reuse gap of deepest matched block: requests, short")
edges = [0, 10, 30, 60, 120, 300, 600, 1e9]
for a, b in zip(edges, edges[1:]):
    m = (gap >= a) & (gap < b)
    if m.sum():
        print(f"  {a:>4.0f}-{b if b < 1e9 else 'inf':>4}s {m.sum():>5} {pct(short[m].sum(), m.sum()):>6}")

seq = np.argsort(t_look, kind="stable")
rank = np.empty(N, int)
rank[seq] = np.arange(N)
disp = rank - np.arange(N)
desc = int((np.diff(seq) < 0).sum())
cross = int(sum(1 for a, b in zip(seq, seq[1:]) if b < a and ts[b] < ts[a]))
srt, inv = [], 0
for i in seq:                                   # Kendall inversions via insertion count
    p = bisect_left(srt, i)
    inv += len(srt) - p
    srt.insert(p, i)
print(f"\norder: strictly by line? {desc == 0}. descents {desc} (of which across different trace "
      f"timestamps {cross}), inversion pairs {inv:,}, |displacement| median "
      f"{np.median(np.abs(disp)):.0f}, p99 {np.percentile(np.abs(disp), 99):.0f}, max {np.abs(disp).max()}")
lag = t_look - (t_look[0] + ts - ts[0])
print(f"lookup lag behind trace arrival (s): median {np.median(lag):.1f}, p90 {np.percentile(lag, 90):.1f}, "
      f"max {lag.max():.1f}")
pre = np.array([any(look[j][0] > t_look[i] for j in range(max(0, i - 64), i)) for i in range(N)])
print(f"requests looked up before some of the 64 lines preceding them: {pre.sum()}; "
      f"of the order-caused short requests {(order & pre).sum()}/{order.sum()}")
