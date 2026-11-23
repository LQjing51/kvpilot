"""Simulate the placements on a dataset's sessions in their original order, K
at a time round-robin (search_trace.build): the first K join at a random turn,
later ones start at turn 0. No session selection.

    python3 sim_original_order.py swe --ks 192,256,320,384 --n 3000
    python3 sim_original_order.py tracelab --ks 24,32,48,64 --n 3000

TraceLab's own timestamps keep only 1-5 sessions active (live context a few
hundred blocks against ~6200 of HBM + DRAM), so replaying its lines as they are
never fills the cache; its sessions are taken in order of their first request.
Prints per K the mean TTFT over lines K..n-1 (the first round is cold), each
placement against LRU, and the paired differences between placements with a
95% interval from BATCH-request batch means.
"""
import argparse
import multiprocessing as mp
import pickle
import random
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import search_trace as st  # noqa: E402

sp, gt = st.sp, st.gt
ROOT = HERE.parents[1]
SRC = {"swe": ROOT / "datasets/agent-traces/swe-smith-trajectories/swe_full.jsonl",
       "tracelab": ROOT / "datasets/agent-traces/tracelab/tracelab_all.jsonl"}
STRATS = ("lru", "belady", "incp", "exclu", "exclb")
PAIRS = (("exclb", "exclu"), ("exclb", "incp"), ("exclb", "belady"), ("exclu", "incp"),
         ("exclu", "belady"), ("incp", "belady"))
BATCH = 50
G = {}


def next_suffix(sess, sid, j, pos):
    """The next turn's suffix after this turn's full blocks, if it is in the
    trace and shares a prefix (what the oracle signal table holds)."""
    rows = sess[sid]
    if (sid, j + 1) not in pos:
        return None
    a, b = rows[j], rows[j + 1]
    full = a["hash_ids"][: a["input_length"] // gt.B]
    k = 0
    while k < len(full) and k < len(b["hash_ids"]) and full[k] == b["hash_ids"][k]:
        k += 1
    return b["input_length"] - k * gt.B if k else None


def evaluate(spec):
    k, seed = spec
    a, sess = G["args"], G["sess"]
    trace = st.build(G["order"], sess, k, True, random.Random(seed), a.n)
    pos = set(trace)
    rows = [sess[sid][j] for sid, j in trace]
    reqs = sp.make_reqs(rows, [next_suffix(sess, sid, j, pos) for sid, j in trace])
    M = sp.tm.TimeModel.load(str(sp.COEFFS), sp.BLOCK)
    dram = int(0.9 * a.dram_gib / (24 / 1024))
    ttft, minutes = {}, 0.0
    for s in STRATS:
        recs = sp.Sim(s, reqs, M, a.hbm_blocks, dram, 16, sp.default_cost(a.tpot)).run()
        ttft[s] = np.array([r["ttft"] for r in recs])
        minutes = max(minutes, recs[-1]["sent"] / 6e4)
    prompt = np.mean([q["n"] for q in reqs[k:]])
    return k, seed, len({sid for sid, _ in trace}), prompt, minutes, ttft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(SRC))
    ap.add_argument("--ks", default="32,48,64")
    ap.add_argument("--seeds", default="0", help="the mid-join turns of the first K")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--hbm-blocks", type=int, default=2361)
    ap.add_argument("--dram-gib", type=float, default=100.0)
    ap.add_argument("--tpot", type=float, default=14.0)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    cache = Path(f"/tmp/sim_original_order_{a.dataset}.pkl")
    if cache.exists():
        sess = pickle.load(open(cache, "rb"))
    else:
        sess = gt.load(SRC[a.dataset])
        pickle.dump(sess, open(cache, "wb"))
    order = [s for s in sess if len(sess[s]) >= 2]
    if a.dataset == "tracelab":
        order.sort(key=lambda s: sess[s][0]["timestamp"])
    G.update(args=a, sess=sess, order=order)

    specs = [(int(k), int(s)) for k in a.ks.split(",") for s in a.seeds.split(",")]
    with mp.get_context("fork").Pool(min(a.jobs, len(specs))) as pool:
        results = pool.map(evaluate, specs, chunksize=1)
    print(f"{a.dataset}: sessions in original order ({len(order)} with 2+ turns), {a.n} requests, "
          f"HBM {a.hbm_blocks} blocks, DRAM {a.dram_gib:g} GiB/rank; lines K..{a.n - 1}")
    print("   K seed sess prompt  min |  LRU ms | belady   incp  exclu  exclb  (vs LRU) | "
          + "  ".join(f"{x}<{y}" for x, y in PAIRS) + "  (ms +- 95%)")
    for k, seed, nsess, prompt, minutes, ttft in sorted(results, key=lambda r: (r[0], r[1])):
        t = {s: v[k:] for s, v in ttft.items()}
        base = t["lru"].mean()
        gaps = []
        for x, y in PAIRS:
            d = t[y] - t[x]
            nb = len(d) // BATCH
            h = 1.96 * statistics.stdev(d[: nb * BATCH].reshape(nb, BATCH).mean(1)) / nb ** 0.5
            gaps.append(f"{d.mean():+6.1f}+-{h:4.1f}")
        print(f"{k:4d} {seed:4d} {nsess:4d} {prompt / 1e3:5.1f}K {minutes:4.0f} | {base:7.1f} | "
              + " ".join(f"{t[s].mean() / base * 100 - 100:+6.1f}" for s in STRATS[1:]) + " | "
              + " ".join(gaps))


if __name__ == "__main__":
    main()
