"""A trace on which HBM and DRAM stay full even under blockwise placement, so
exclusive (one copy) and blockwise (per-block tiers) have capacity to save.

    python3 gen_trace_capacity.py --hbm-blocks 600 --dram-gib 100 --out swe_blockprio.jsonl

Every turn's placement is what the prefill side computes from the oracle
signal (kvpilot_blockwise.place / reqwise with the time model, the next turn's
suffix): blockwise keeps h of its n prompt blocks in HBM and d in DRAM until
the next turn reads them; reqwise keeps all n in one tier. A session holds its
latest turn's placement, so with sessions replayed round-robin the live demand
on a tier is the sum over active sessions.

1. Candidates: sessions whose mean blockwise HBM demand per turn is in the top
   --top %: HBM demand comes from long prefixes with short suffixes, and only
   2.6% of all blocks get it, so HBM fills only with such sessions.
2. Admission (random order, --seed): a finished session is replaced while the
   sessions' mean blockwise demand is below --over x the HBM left for held
   blocks (HBM - one prefill chunk of reserve - the largest prompt) or below
   --over x DRAM at the 0.9 watermark.
3. Cut where the estimated replay time reaches --minutes (gen_trace.py's
   model: send, prefill from HBM, decode-side KV pull, decode).

Writes the trace (gen_trace.py format, hash ids renumbered) and its oracle
signal table: next_suffix, and next_tier / next_time / own_tier from the same
model (the placement strategies only read "is there a next turn" and the
suffix). Prints the live blockwise / reqwise demand against both capacities.
"""
import argparse
import importlib.util
import json
import pickle
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Loading, replay-time estimate and trace writing are req-priority-trace's.
sys.path[:0] = [str(ROOT / "datasets/req-priority-trace")]
import gen_trace as gt  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "kvpilot_time_model", ROOT / "vllm-dev/vllm/v1/core/kvpilot_time_model.py")
tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tm)
COEFFS = ROOT / "paper-figure/Figure3-request-safe/fit_times2_swe.json"
M = tm.TimeModel.load(str(COEFFS), gt.B)
SRC = ROOT / "datasets/agent-traces/swe-smith-trajectories/swe_full.jsonl"
CHUNK_BLOCKS = 16
DRAM_BLOCK_GIB = 24 / 1024  # one 512-token block per TP rank (Qwen3-30B-A3B, TP 2)


def safe_fn(n, s):
    p = n * gt.B
    c = M.compute(p, s)
    return lambda d, x: (M.e2e(p, s, d * gt.B, x * gt.B) - c) / c < M.thresh


def place(n, s):
    """(h, d, x, reqwise tier): kvpilot_blockwise.place / reqwise for a one-point
    suffix distribution, by bisection (both conditions are monotone)."""
    safe = safe_fn(n, s)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if safe(n - mid, 0) else (mid + 1, hi)
    h, m = lo, n - lo
    lo, hi = 0, m
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if safe(m - mid, mid) else (lo, mid - 1)
    tier = "SSD" if safe(0, n) else "DRAM" if safe(n, 0) else "HBM"
    return h, m - lo, lo, tier


def turn_plans(sess):
    """session -> per turn (n, next suffix or None, (h, d, x, tier) or None)."""
    out = {}
    for sid, rows in sess.items():
        plans = []
        for a, b in zip(rows, rows[1:] + [None]):
            n = a["input_length"] // gt.B
            if b is None or n == 0:
                plans.append((n, None, None))
                continue
            full = a["hash_ids"][:n]
            k = 0
            while k < n and k < len(b["hash_ids"]) and full[k] == b["hash_ids"][k]:
                k += 1
            s = b["input_length"] - k * gt.B
            plans.append((n, s, place(n, s) if k else None))
        out[sid] = plans
    return out


def foot(plans):
    """Mean blockwise (HBM, DRAM) demand over a session's turns."""
    h = [p[2][0] if p[2] else 0 for p in plans]
    d = [p[2][1] if p[2] else 0 for p in plans]
    return float(np.mean(h)), float(np.mean(d))


def build(order, sess, fp, h_cap, d_cap, cap):
    queue, active, lh, ld, out = deque(order), deque(), 0.0, 0.0, []

    def admit():
        nonlocal lh, ld
        while queue and (lh < h_cap or ld < d_cap):
            s = queue.popleft()
            active.append([s, 0])
            lh += fp[s][0]
            ld += fp[s][1]

    admit()
    while active and len(out) < cap:
        cur = active.popleft()
        out.append(tuple(cur))
        cur[1] += 1
        if cur[1] < len(sess[cur[0]]):
            active.append(cur)
            continue
        lh -= fp[cur[0]][0]
        ld -= fp[cur[0]][1]
        admit()
    return out


def live_demand(trace, plans):
    """After each request: (blockwise HBM, blockwise DRAM, reqwise HBM,
    reqwise DRAM, all prompt blocks) of the sessions' latest turns."""
    cur, tot, out = {}, np.zeros(5), np.zeros((len(trace), 5))
    for i, (sid, j) in enumerate(trace):
        tot -= cur.get(sid, 0)
        n, _, pl = plans[sid][j]
        v = np.zeros(5)
        v[4] = n
        if pl:
            h, d, x, t = pl
            v[:4] = (h, d, n if t == "HBM" else 0, n if t == "DRAM" else 0)
        cur[sid] = v
        tot += v
        out[i] = tot
    return out


def replay(trace, sess, tpot):
    e2e = np.zeros(len(trace))
    for i, (sid, j) in enumerate(trace):
        r = sess[sid][j]
        p = min(r["prefix_length"], r["input_length"] - 1)
        e2e[i] = M.compute(p, r["input_length"] - p)
    return gt.replay_seconds(trace, sess, e2e, tpot)


def write_table(path, trace, sess, plans):
    """Oracle signal table keyed by replay line."""
    rows = {}
    for i, (sid, j) in enumerate(trace):
        n, s, pl = plans[sid][j]
        nxt = j + 1 < len(sess[sid]) and (sid, j + 1) in pos
        row = {"next_tier": None, "next_time": None, "next_suffix": None}
        if nxt and pl:
            _, _, _, t = pl
            p = n * gt.B
            row = {"next_tier": t,
                   "next_time": [M.compute(p, s), M.e2e(p, s, p, 0), M.e2e(p, s, 0, p)],
                   "next_suffix": s}
        r = sess[sid][j]
        p = min(r["prefix_length"], r["input_length"] - 1) // gt.B * gt.B
        if p:
            own = safe_fn(p // gt.B, r["input_length"] - p)
            row["own_tier"] = ("SSD" if own(0, p // gt.B) else "DRAM"
                               if own(p // gt.B, 0) else "HBM")
        else:
            row["own_tier"] = None
        rows[str(i)] = row
    path.write_text(json.dumps(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=HERE / "swe_blockprio.jsonl")
    ap.add_argument("--table", type=Path,
                    help="default: utils/placement_ab/signal_<out name>.json")
    ap.add_argument("--hbm-blocks", type=int, default=600)
    ap.add_argument("--dram-gib", type=float, default=100.0, help="per TP rank")
    ap.add_argument("--over", type=float, default=1.2,
                    help="admit until mean blockwise demand is this x capacity")
    ap.add_argument("--top", type=float, default=5.0,
                    help="candidates: top N%% sessions by mean blockwise HBM demand")
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tpot", type=float, default=20.0)
    ap.add_argument("--cache", type=Path, default=Path("/tmp/gen_trace_capacity.pkl"))
    a = ap.parse_args()

    if a.cache.exists():
        sess, plans = pickle.load(open(a.cache, "rb"))
    else:
        sess = gt.load(a.src)
        plans = turn_plans(sess)
        pickle.dump((sess, plans), open(a.cache, "wb"))
    fp = {s: foot(plans[s]) for s in sess}
    th = np.percentile([v[0] for v in fp.values()], 100 - a.top)
    cands = sorted(s for s in sess if fp[s][0] >= th)
    random.Random(a.seed).shuffle(cands)
    max_prompt = max(r["input_length"] for s in cands for r in sess[s]) // gt.B + 1
    h_eff = a.hbm_blocks - CHUNK_BLOCKS - max_prompt
    d_cap = int(0.9 * a.dram_gib / DRAM_BLOCK_GIB)
    if h_eff <= 0:
        raise SystemExit(f"--hbm-blocks {a.hbm_blocks} leaves nothing past the largest "
                         f"prompt ({max_prompt} blocks) and the reserve")
    print(f"{len(sess)} sessions; top {a.top:g}% by blockwise HBM demand: {len(cands)} "
          f"(mean HBM {np.mean([fp[s][0] for s in cands]):.1f}, DRAM "
          f"{np.mean([fp[s][1] for s in cands]):.1f} blocks per session); HBM for held "
          f"blocks {h_eff} (of {a.hbm_blocks}, largest prompt {max_prompt}), DRAM {d_cap}")

    cap = 4000
    while True:
        trace = build(cands, sess, fp, a.over * h_eff, a.over * d_cap, cap)
        t = replay(trace, sess, a.tpot)
        if t[-1] >= a.minutes * 60 or len(trace) < cap:
            break
        cap *= 2
    n = int(np.searchsorted(t, a.minutes * 60, side="right"))
    trace = trace[:n]
    global pos
    pos = set(trace)
    dem = live_demand(trace, plans)
    used = {sid for sid, _ in trace}
    print(f"{n} requests, {len(used)} sessions, est. replay {t[n - 1] / 60:.1f} min")
    steady = dem[n // 4:]
    for k, (name, capk) in enumerate((("blockwise HBM", h_eff), ("blockwise DRAM", d_cap),
                                      ("reqwise HBM", h_eff), ("reqwise DRAM", d_cap))):
        col = steady[:, k]
        print(f"  {name:15s} demand p10/p50/p90 {np.percentile(col, 10):6.0f} "
              f"{np.percentile(col, 50):6.0f} {np.percentile(col, 90):6.0f} blocks "
              f"= {np.median(col) / capk:4.2f}x capacity; over capacity in "
              f"{np.mean(col > capk) * 100:3.0f}% of the last 3/4")
    print(f"  live prompt KV (what inclusive keeps a copy of in DRAM) p50 "
          f"{np.median(steady[:, 4]):.0f} blocks = {np.median(steady[:, 4]) / d_cap:.2f}x DRAM")
    both = np.mean((steady[:, 0] > h_eff) & (steady[:, 1] > d_cap)) * 100
    print(f"  blockwise HBM and DRAM both over capacity: {both:.0f}% of the last 3/4")
    gt.write(a.out, trace, sess)
    table = a.table or ROOT / f"utils/placement_ab/signal_{a.out.stem}.json"
    write_table(table, trace, sess, plans)
    print(f"wrote {a.out} and {table}")


if __name__ == "__main__":
    main()
