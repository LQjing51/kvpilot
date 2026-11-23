"""Search swe_full for the shortest trace that separates the placements, by
simulating each candidate (sim_placement.py) at the given capacities.

    python3 search_trace.py                      # grid, default HBM (2361 blocks)
    python3 search_trace.py --write pool=10,k=64,mid=1,seed=0 --out swe_blockprio_short.jsonl

A candidate replays K sessions round-robin (gen_trace.py), a finished one
replaced by the next of a shuffled pool; the pool is the sessions (two turns or
more) in the top --pool % by mean blockwise HBM demand per turn
(gen_trace_capacity.py), or with pool 0 all of them in swe_full's order. With mid=1 the first K sessions join at a random turn,
so the cache fills in their first round instead of over thousands of short
first turns; later sessions start at turn 0.

For every candidate and pair (a, b) of PAIRS, a expected faster: d = TTFT_b -
TTFT_a per request. With the first W requests left out, the steady requests
needed are (Z * sd / mean d)^2, sd from means of BATCH-request batches (the
cache state carries over) scaled back to one request. Needed = min over W of
W + the largest pair's need; a candidate whose need passes its length, or with
a pair under --min-gain of LRU's TTFT (the model's error), does not separate.
"""
import argparse
import importlib.util
import multiprocessing as mp
import pickle
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sp = _load("sim_placement", HERE / "sim_placement.py")
gtc = _load("gen_trace_capacity", HERE / "gen_trace_capacity.py")
gt = gtc.gt

STRATS = ("lru", "belady", "incp", "exclu", "exclb")
PAIRS = (("exclb", "exclu"), ("exclb", "incp"), ("exclb", "belady"), ("exclb", "lru"),
         ("exclu", "incp"), ("exclu", "lru"), ("incp", "lru"), ("belady", "lru"))
BATCH = 50
Z = 3.0

G = {}  # sess, plans, pools, args: set before the workers fork


def build(order, sess, k, mid, rng, cap):
    queue, active, out = deque(order), deque(), []

    def admit(first):
        s = queue.popleft()
        active.append([s, rng.randrange(len(sess[s]) - 1) if first and mid else 0])

    for _ in range(min(k, len(queue))):
        admit(True)
    while active and len(out) < cap:
        cur = active.popleft()
        out.append(tuple(cur))
        cur[1] += 1
        if cur[1] < len(sess[cur[0]]):
            active.append(cur)
        elif queue:
            admit(False)
    return out


def candidate(pool, k, mid, seed, cap):
    """pool 0: every session in swe_full's order; else the top pool % by
    blockwise HBM demand, shuffled by seed."""
    sess = G["sess"]
    rng = random.Random(seed)
    if pool == 0:
        order = list(G["file_order"])
    else:
        order = list(G["pools"][pool])
        rng.shuffle(order)
    return build(order, sess, k, mid, rng, cap)


def next_suffixes(trace):
    pos = set(trace)
    out = []
    for sid, j in trace:
        _, s, pl = G["plans"][sid][j]
        out.append(s if (sid, j + 1) in pos and pl is not None else None)
    return out


def simulate(trace, tpot):
    a = G["args"]
    rows = [G["sess"][sid][j] for sid, j in trace]
    reqs = sp.make_reqs(rows, next_suffixes(trace))
    M = sp.tm.TimeModel.load(str(sp.COEFFS), sp.BLOCK)
    dram = int(0.9 * a.dram_gib / (24 / 1024))
    out = {}
    for s in STRATS:
        recs = sp.Sim(s, reqs, M, a.hbm_blocks, dram, 16, sp.default_cost(tpot)).run()
        out[s] = {k: np.array([r[k] for r in recs]) for k in
                  ("ttft", "e2e", "blocked", "sent", "hbm", "dram", "ssd")}
    return out


def need(sim, min_gain):
    """(requests needed, warm-up W, per pair (mean d, steady need)) or None."""
    n = len(sim["lru"]["ttft"])
    best = None
    for w in range(0, n - 4 * BATCH, BATCH):
        base = sim["lru"]["ttft"][w:].mean()
        worst, per = 0.0, {}
        for a, b in PAIRS:
            d = sim[b]["ttft"][w:] - sim[a]["ttft"][w:]
            nb = len(d) // BATCH
            sd = d[: nb * BATCH].reshape(nb, BATCH).mean(1).std(ddof=1) * BATCH ** 0.5
            m = d.mean()
            s = (Z * sd / m) ** 2 if m > min_gain * base else np.inf
            per[(a, b)] = (m, s)
            worst = max(worst, s)
        tot = w + max(worst, 2 * BATCH)
        if tot <= n and (best is None or tot < best[0]):
            best = (int(tot), w, per)
    return best


def evaluate(spec):
    pool, k, mid, seed = spec
    a = G["args"]
    trace = candidate(pool, k, mid, seed, a.cap)
    sim = simulate(trace, a.tpot)
    res = need(sim, a.min_gain)
    row = {"pool": pool, "k": k, "mid": mid, "seed": seed, "n": len(trace),
           "sim": {s: {x: sim[s][x] for x in ("ttft", "sent")} for s in STRATS}}
    w = res[1] if res else len(trace) // 2
    base = sim["lru"]["ttft"][w:].mean()
    row["gain"] = {s: sim[s]["ttft"][w:].mean() / base - 1 for s in STRATS}
    row["ttft"] = {s: sim[s]["ttft"][w:].mean() for s in STRATS}
    if res:
        tot, w, per = res
        row.update(need=tot, warm=w, per=per,
                   minutes=max(sim[s]["sent"][tot - 1] for s in STRATS) / 6e4)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hbm-blocks", type=int, default=2361)
    ap.add_argument("--dram-gib", type=float, default=100.0)
    ap.add_argument("--tpot", type=float, default=14.0)
    ap.add_argument("--cap", type=int, default=4000, help="requests per candidate")
    ap.add_argument("--pools", default="100,50,20,10,5")
    ap.add_argument("--ks", default="24,32,48,64,96,128,192")
    ap.add_argument("--mids", default="0,1")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--min-gain", type=float, default=0.03,
                    help="smallest mean d a pair may have, as a share of LRU's TTFT")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--write", help="pool=..,k=..,mid=..,seed=..: write this candidate")
    ap.add_argument("--n", type=int, help="with --write: requests to keep")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    sess, plans = pickle.load(open(Path("/tmp/gen_trace_capacity.pkl"), "rb"))
    fp = {s: gtc.foot(plans[s]) for s in sess if len(sess[s]) >= 2}
    order = sorted(fp, key=lambda s: -fp[s][0])
    G.update(sess=sess, plans=plans, args=a)
    G["file_order"] = [s for s in sess if len(sess[s]) >= 2]
    G["pools"] = {int(p): sorted(order[: max(1, len(order) * int(p) // 100)])
                  for p in a.pools.split(",") if int(p)}

    if a.write:
        kv = dict(x.split("=") for x in a.write.split(","))
        trace = candidate(int(kv["pool"]), int(kv["k"]), int(kv["mid"]), int(kv["seed"]),
                          a.n or a.cap)
        gtc.pos = set(trace)
        gt.write(a.out, trace, sess)
        table = ROOT / f"utils/placement_ab/signal_{a.out.stem}.json"
        gtc.write_table(table, trace, sess, plans)
        print(f"wrote {a.out} ({len(trace)} requests, {len({s for s, _ in trace})} sessions) and {table}")
        return

    specs = [(int(p), int(k), int(m), int(s)) for p in a.pools.split(",")
             for k in a.ks.split(",") for m in a.mids.split(",") for s in a.seeds.split(",")]
    with mp.get_context("fork").Pool(a.jobs) as pool:
        rows = pool.map(evaluate, specs, chunksize=1)
    rows.sort(key=lambda r: r.get("need", 1e9))
    print(f"HBM {a.hbm_blocks}, DRAM {a.dram_gib:g} GiB/rank, tpot {a.tpot}; Z {Z}, batch {BATCH}, "
          f"pairs under {a.min_gain:.0%} of LRU fail")
    print("pool  K mid seed | need warm  min | TTFT vs LRU: " + " ".join(f"{s:>6s}" for s in STRATS[1:])
          + " | tightest pair")
    for r in rows:
        g = " ".join(f"{r['gain'][s] * 100:+5.1f}%" for s in STRATS[1:])
        if "need" in r:
            (a_, b_), (m, s) = max(r["per"].items(), key=lambda kv: kv[1][1])
            tail = f"{a_}<{b_} {m:5.1f} ms needs {s:5.0f}"
            head = f"{r['need']:5d} {r['warm']:4d} {r['minutes']:4.1f}"
        else:
            tail, head = "does not separate", "    -    -    -"
        print(f"{r['pool']:4d} {r['k']:3d} {r['mid']:3d} {r['seed']:4d} | {head} | "
              f"LRU {r['ttft']['lru']:5.0f} ms  {g} | {tail}")
    pickle.dump(rows, open(Path(f"/tmp/search_trace_{a.hbm_blocks}_{a.tpot:g}_{a.mids}.pkl"), "wb"))


if __name__ == "__main__":
    main()
