"""One trace for every model: simulate each candidate of search_trace.py on
several models (sim_placement.PROFILES) and show how far the placements
separate on each.

    python3 search_trace_models.py --ks 160,256,384,512 --pools 0,100,50 --cap 4000
    python3 search_trace_models.py --models qwen3-30b,qwen3.6-tp4,qwen3.8 --orders rand \
        --ks 64,128,256 --pools 10,20 --hbm qwen3.6-tp4=3000

A candidate is search_trace's (pool, K, mid, seed, order): K sessions at a
time, round-robin (rr), each request from an active session drawn at random
(rand), or from the first w of the round-robin queue at random (w<w>), a
finished one replaced by the next of the pool; a suffix s<pct> makes pct % of
turns follow their session's previous turn right away (rrs20, w24s20);
h<H>f<pct> has H fast sessions round-robin send pct % of the requests and the
K slow ones the rest. Each model replays
the same requests with its own oracle next suffix (the next turn's prompt past
the blocks this turn leaves: whole blocks of its computed prompt), at its
default capacities unless --hbm / --dram-gib name others.

Per candidate and model: LRU's TTFT over the steady lines (after the first
--warm lines, default K), each placement against it, where LRU's prefix comes
from (HBM/DRAM/SSD), and the model's required pairs (PAIRS) checked one by
one: margin d = TTFT_slower - TTFT_faster less its 95% interval from
BATCH-request batch means (cache state carries over), in ms, each pair's and
the smallest. A positive margin on every pair of every model is a trace that
separates them all.
"""
import argparse
import multiprocessing as mp
import pickle
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import search_trace as st  # noqa: E402

sp, gtc = st.sp, st.gtc
STRATS = ("lru", "belady", "incp", "exclu", "exclb")
# The placements simulated per model, and the (slower, faster) pairs it must
# separate: Qwen3-30B two chains, LRU < belady < blockwise and inclusive req <
# unblocking < blockwise (belady and inclusive req are not compared).
ORDER = {"qwen3-30b": ("lru", "belady", "incp", "exclu", "exclb")}
HYBRID_ORDER = ("lru", "belady", "exclb")
PAIRS = {"qwen3-30b": (("lru", "belady"), ("belady", "exclb"), ("incp", "exclu"), ("exclu", "exclb"))}
HYBRID_PAIRS = (("lru", "belady"), ("belady", "exclb"))
BATCH = 50
G = {}


def parse_order(o):
    """rr | rand | w<w>, optionally s<pct>; or h<H>f<pct>: candidate()'s
    keyword arguments."""
    if o.startswith("h"):
        hot, frac = o[1:].split("f")
        return {"hot": int(hot), "frac": int(frac) / 100}
    base, _, stick = o.partition("s") if not o.startswith("rand") else (o, "", "")
    if o.startswith("rand") and "s" in o[4:]:
        base, stick = "rand", o[o.index("s", 4) + 1:]
    rand = {"rr": 0, "rand": -1}.get(base)
    return {"rand": int(base[1:]) if rand is None else rand, "stick": int(stick or 0) / 100}


def order_of(model):
    return ORDER.get(model, HYBRID_ORDER)


def pairs_of(model):
    return PAIRS.get(model, HYBRID_PAIRS)


def next_suffixes(trace, sess, prof):
    """Per request, the next turn's suffix after this turn's blocks if the
    next turn is in the trace and shares a prefix, else None."""
    pos = set(trace)
    block, out = prof["block"], []
    for sid, j in trace:
        if (sid, j + 1) not in pos:
            out.append(None)
            continue
        a, b = sess[sid][j], sess[sid][j + 1]
        ncomp = a["input_length"] - prof["hybrid"]
        full = a["hash_ids"][: ncomp // sp.BLOCK]
        k = 0
        while k < len(full) and k < len(b["hash_ids"]) and full[k] == b["hash_ids"][k]:
            k += 1
        tokens = min(k * sp.BLOCK, ncomp) // block * block
        out.append(b["input_length"] - tokens if tokens else None)
    return out


def capacities(a, name, k):
    """(HBM blocks, DRAM blocks) of a model. --hbm model=N[+s]: N blocks, +s
    adds a hybrid model's state blocks for K sessions."""
    prof = sp.PROFILES[name]
    hbm = dict(x.split("=") for x in a.hbm.split(",") if x).get(name)
    dram = dict(x.split("=") for x in a.dram_gib.split(",") if "=" in x).get(name)
    dram_gib = float(dram) if dram else float(next(
        (x for x in a.dram_gib.split(",") if "=" not in x), "100"))
    if hbm is None:
        blocks = prof["hbm"]
    elif hbm.endswith("+s"):
        blocks = int(hbm[:-2]) + prof.get("states", 0) * k
    else:
        blocks = int(hbm)
    return blocks, int(sp.WATERMARK * dram_gib / prof["block_gib"])


def evaluate(spec):
    pool, k, mid, seed, o = spec
    a, sess = G["args"], G["sess"]
    trace = st.candidate(pool, k, mid, seed, a.cap, **parse_order(o))
    rows = [sess[sid][j] for sid, j in trace]
    out = {"spec": spec, "n": len(trace), "sessions": len({s for s, _ in trace}),
           "prompt": float(np.mean([r["input_length"] for r in rows[k:]]))}
    for name in a.models.split(","):
        prof = sp.PROFILES[name]
        reqs = sp.make_reqs(rows, next_suffixes(trace, sess, prof), prof)
        M = sp.tm.TimeModel.load(str(prof["coeffs"]), prof["block"])
        hbm, dram = capacities(a, name, k)
        res = {"cap": (hbm, dram)}
        for s in order_of(name):
            recs = sp.Sim(s, reqs, M, hbm, dram, 16, sp.default_cost(a.tpot, prof),
                          prof).run()
            res[s] = {x: np.array([r[x] for r in recs]) for x in ("ttft", "sent", "hbm", "dram", "ssd")}
        out[name] = res
    return out


def margins(res, pairs, w):
    """Per (slower, faster) pair: (mean d, 95% half-width) over lines >= w."""
    out = []
    for slow, fast in pairs:
        d = res[slow]["ttft"][w:] - res[fast]["ttft"][w:]
        nb = len(d) // BATCH
        h = 1.96 * statistics.stdev(d[: nb * BATCH].reshape(nb, BATCH).mean(1)) / nb ** 0.5
        out.append((d.mean(), h))
    return out


def report(r, names, w):
    pool, k, mid, seed, name = r["spec"]
    head = (f"{pool:4d} {k:4d} {mid:3d} {seed:4d} {name:>7s} {r['n']:5d} "
            f"{r['sessions']:4d} {r['prompt'] / 1e3:5.1f}K")
    worst = np.inf
    parts = []
    for m in names:
        res, order = r[m], order_of(m)
        minutes = max(res[s]["sent"][-1] for s in order) / 6e4
        base = res["lru"]["ttft"][w:].mean()
        tot = [res["lru"][x][w:].sum() for x in ("hbm", "dram", "ssd")]
        split = "/".join(f"{t / (sum(tot) or 1) * 100:.0f}" for t in tot)
        mg = margins(res, pairs_of(m), w)
        low = min(m_ - h for m_, h in mg)
        worst = min(worst, low)
        parts.append(f"{base:6.1f} " + " ".join(
            f"{res[s]['ttft'][w:].mean() / base * 100 - 100:+5.1f}" for s in order[1:])
            + f" {split:>8s} {minutes:4.0f}m " + "/".join(f"{m_ - h:+.1f}" for m_, h in mg))
    print(head + " | " + " | ".join(parts) + f" || {worst:+5.1f}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3-30b,qwen3.6")
    ap.add_argument("--dram-gib", default="100",
                    help="GiB per TP rank, all models, and/or model=GiB")
    ap.add_argument("--hbm", default="",
                    help="model=blocks overrides; model=N+s adds K sessions' state blocks")
    ap.add_argument("--tpot", type=float, default=14.0)
    ap.add_argument("--cap", type=int, default=4000, help="requests per candidate")
    ap.add_argument("--warm", type=int, help="steady lines start here (default K)")
    ap.add_argument("--pools", default="0,100,50,20")
    ap.add_argument("--ks", default="160,256,384,512")
    ap.add_argument("--mids", default="1")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--orders", default="rr", help="rr, rand, w<window>, each with s<pct>")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="/tmp/search_trace_models.pkl")
    ap.add_argument("--cache", default="/tmp/gen_trace_capacity.pkl",
                    help="(sessions, plans) of gen_trace_capacity.py (SWE) or make_cache()")
    a = ap.parse_args()

    sess, plans = pickle.load(open(Path(a.cache), "rb"))
    fp = {s: gtc.foot(plans[s]) for s in sess if len(sess[s]) >= 2}
    order = sorted(fp, key=lambda s: -fp[s][0])
    st.G.update(sess=sess, plans=plans, args=a, file_order=[s for s in sess if len(sess[s]) >= 2],
                pools={int(p): sorted(order[: max(1, len(order) * int(p) // 100)])
                       for p in a.pools.split(",") if int(p)})
    G.update(sess=sess, args=a)
    specs = [(int(p), int(k), int(m), int(s), o) for p in a.pools.split(",")
             for k in a.ks.split(",") for m in a.mids.split(",") for s in a.seeds.split(",")
             for o in a.orders.split(",")]
    with mp.get_context("fork").Pool(a.jobs) as pool:
        rows = pool.map(evaluate, specs, chunksize=1)
    names = a.models.split(",")
    print(f"DRAM {a.dram_gib} GiB/rank, HBM {a.hbm or 'default'}, tpot {a.tpot}; steady = after "
          f"{a.warm or 'the first K'} lines; margin = mean d - 95% CI (ms) per required pair, "
          f"|| the smallest")
    print("pool    K mid seed     ord     n sess prompt | " + " | ".join(
        f"{m}: LRU ms " + " ".join(order_of(m)[1:]) + " (%) LRU H/D/S% margins "
        + "/".join(f"{s}>{f}" for s, f in pairs_of(m)) for m in names))
    for r in sorted(rows, key=lambda r: r["spec"]):
        r["worst"] = report(r, names, a.warm or r["spec"][1])
    pickle.dump(rows, open(Path(a.out), "wb"))


if __name__ == "__main__":
    main()
