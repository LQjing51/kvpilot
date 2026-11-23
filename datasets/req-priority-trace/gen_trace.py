"""Round-robin traces that maximize the e2e gain of request-priority eviction over
LRU within a fixed replay time (1P1D, one prefill at a time). See README.md.

    python3 gen_trace.py swe        # -> swe_reqprio.jsonl
    python3 gen_trace.py tracelab   # -> tracelab_reqprio.jsonl

1. Estimate every request's t1 / t2 / t4 (e2e with the prefix in HBM / DRAM /
   SSD) and its request-safe tier from (prefix, suffix): the median over the
   nearest measured points of the SWE request-safe runs, the fitted time model
   outside the measured range.
2. Score each session by its gain ratio: the e2e saved if LRU read every
   prefix from SSD while request priority kept HBM-safe prefixes in HBM and
   DRAM-safe ones in DRAM, over LRU's all-SSD e2e. Keep the top --top share
   (and at most --max-turns turns).
3. Replay them round-robin, in a random admission order: a session finishing
   is replaced while the live KV (each active session counted at its mean
   context over its life) is under --budget blocks.
4. Cut the trace where the LRU arm's estimated replay time reaches --minutes.
5. Repeat for every seed x budget, keep the one with the highest modeled gain.

Cache behavior and both policies are utils/placement_ab/dram_miss_sim.py (SSD
hits written back to DRAM; Mooncake 5 s kv lease by default — BatchEvict skips
unexpired objects, wall clock advances by each request's estimated prefill e2e).
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "utils/placement_ab")]
import dram_miss_sim as dms  # noqa: E402  (puts Figure3-request-safe on sys.path)
from fit_times import THRESH, times  # noqa: E402
from tier_logs import dataset  # noqa: E402

B = 512
K = json.loads((dms.FIG3 / "fit_times_swe.json").read_text())
KV_BYTES = 48 * 4 * 128 * 2 * 2  # Qwen3-30B-A3B bf16 K+V per token, both TP ranks
MEASURED = ("raw/swe", "raw/swe_hbm1to2")
PMAX, SMAX = 65536, 6656  # measured range used for the nearest-point estimate
PRESETS = {
    "swe": dict(src=ROOT / "datasets/agent-traces/swe-smith-trajectories/swe_full.jsonl",
                max_turns=0, budgets="3500,3700,3900,4100,4300", tpot=20.0),
    "tracelab": dict(src=ROOT / "datasets/agent-traces/tracelab/tracelab_all.jsonl",
                     max_turns=50, budgets="3400,3600,3800,4000,4200", tpot=10.0),
}
# Replay time of one request besides its prefill (README.md "回放时间"):
# client send -> prefill start, and the decode side pulling the prompt's KV.
SEND_MS = 17.0


def pull_ms(n):
    return 75.0 + 1.9 * n / 1000


def t_hbm(p, s):
    return float(times(p, s, K)[0])


def tier_of(t1, t2, t4):
    return "SSD" if (t4 - t1) / t1 < THRESH else "DRAM" if (t2 - t1) / t1 < THRESH else "HBM"


class Estimator:
    """(prefix, suffix) -> (tier, t1, t2, t4) in ms."""

    W = 0.125  # bin width, octaves
    NEED = 15  # measured points per estimate

    def __init__(self):
        pts = []
        for raw in MEASURED:
            rows, res, _ = dataset(dms.FIG3 / raw)
            for r in rows:
                e = [res.get((r["src_line"], k), (None, "m"))[0] for k in ("t1", "t2", "t4")]
                if r["kind"] == "t1" and None not in e:
                    pts.append((r["prefix"], r["input_length"] - r["prefix"], *e))
        p, s, t1, t2, t4 = np.array(pts, dtype=float).T
        self.q = np.column_stack([t1 / times(p, s, K)[0], t2 / t1, t4 / t1])
        self.cells = defaultdict(list)
        for i, key in enumerate(zip(self._bin(p), self._bin(s))):
            self.cells[key].append(i)
        self.cache = {}
        print(f"estimator: {len(pts)} measured requests")

    def _bin(self, x):
        return np.floor(np.log2(np.maximum(x, 1)) / self.W).astype(int)

    def _median(self, bp, bs):
        if (bp, bs) not in self.cache:
            r = 0
            while True:
                idx = [i for dp in range(-r, r + 1) for ds in range(-r, r + 1)
                       for i in self.cells.get((bp + dp, bs + ds), ())]
                if len(idx) >= self.NEED:
                    break
                r += 1
            self.cache[(bp, bs)] = np.median(self.q[idx], axis=0)
        return self.cache[(bp, bs)]

    def __call__(self, p, s):
        if p <= PMAX and s <= SMAX:
            a, b, c = self._median(int(self._bin(p)), int(self._bin(s)))
            t1 = t_hbm(p, s) * a
            t2, t4 = t1 * b, t1 * c
        else:
            t1, t2, t4 = (float(x) for x in times(p, s, K))
        return tier_of(t1, t2, t4), t1, t2, t4


def load(path):
    """session_id -> its rows in round order."""
    sess = defaultdict(list)
    for line_no, line in enumerate(open(path)):
        r = json.loads(line)
        r["hash_ids"] = np.asarray(r["hash_ids"], dtype=np.int64)
        r["orig_line"] = line_no
        sess[r["session_id"]].append(r)
    for rows in sess.values():
        rows.sort(key=lambda r: r["round_index"])
    return sess


def session_stats(sess, est):
    """session_id -> (turns, gain ratio, mean context in blocks)."""
    out = {}
    for sid, rows in sess.items():
        seen, gain, lru, ctx = set(), 0.0, 0.0, 0
        for r in rows:
            n = r["input_length"]
            full = r["hash_ids"][: n // B].tolist()
            hit = 0
            while hit < len(full) and full[hit] in seen:
                hit += 1
            seen.update(full)
            ctx += n // B
            p = hit * B
            if not p or p >= n:
                lru += t_hbm(0, n)
                continue
            tier, t1, t2, t4 = est(p, n - p)
            gain += t4 - t1 if tier == "HBM" else t4 - t2 if tier == "DRAM" else 0.0
            lru += t4
        out[sid] = (len(rows), gain / lru, ctx / len(rows))
    return out


def build(order, sess, foot, budget, cap):
    """(session_id, round) in replay order: round-robin, a finished session is
    replaced while the live KV estimate is under budget; stops at cap requests."""
    queue, active, load, out = deque(order), deque(), 0.0, []
    while queue and load < budget:
        s = queue.popleft()
        active.append([s, 0])
        load += foot[s]
    while active and len(out) < cap:
        cur = active.popleft()
        out.append(tuple(cur))
        cur[1] += 1
        if cur[1] < len(sess[cur[0]]):
            active.append(cur)
            continue
        load -= foot[cur[0]]
        while queue and load < budget:
            s = queue.popleft()
            active.append([s, 0])
            load += foot[s]
    return out


def simulate(trace, sess, est, opts, lease_ms=5000):
    """Per-request e2e (ms) under dram_miss_sim, and its KV source counts.

    lease_ms matches Mooncake default_kv_lease_ttl (5 s); BatchEvict skips
    unexpired leases. 0 restores the old no-lease model.
    """
    rows, meas, seen, by = [], {}, set(), defaultdict(list)
    e2e = np.zeros(len(trace))
    for i, (sid, j) in enumerate(trace):
        r = sess[sid][j]
        n = r["input_length"]
        h = r["hash_ids"].tolist()
        full = h[: n // B]
        hit = 0
        while hit < len(full) and full[hit] in seen:
            hit += 1
        seen.update(full)
        rows.append({"input_length": n, "hash_ids": h, "src_line": i})
        by[sid].append(i)
        p = hit * B
        if hit and p < n:
            tier, t1, t2, t4 = est(p, n - p)
            meas[i] = (p, n, tier, {"t1": t1, "t2": t2, "t4": t4})
        else:
            e2e[i] = t_hbm(0, n)
    fut = {}
    for idx in by.values():
        tiers = [meas[i][2] if i in meas else None for i in idx]
        for a, i in enumerate(idx):
            fut[i] = tiers[a + 1:]
    scored = {}
    mis, *_ = dms.run(rows, meas, fut, K, opts, dms.tier_sim.HBM_BLOCKS, dms.tier_sim.DRAM_BLOCKS, 0,
                      e2e=scored, lease_ms=lease_ms)
    for i, v in scored.items():
        e2e[i] = v
    return e2e, mis


def replay_seconds(trace, sess, e2e, tpot):
    """Cumulative replay time after each request under -p 1 --release-at-first-token:
    the next request leaves at this one's first token, but not before its own
    session's previous round has decoded."""
    t, done, out = 0.0, {}, np.zeros(len(trace))
    for i, ((sid, j), e) in enumerate(zip(trace, e2e)):
        r = sess[sid][j]
        start = max(t, done.get(sid, 0.0))
        first = start + SEND_MS + e + pull_ms(r["input_length"]) + tpot
        done[sid] = first + max(0, r["output_length"] - 1) * tpot
        t = first
        out[i] = t / 1e3
    return out


def write(path, trace, sess):
    """Hash ids are renumbered: the replay fills a block with token
    hash_id % vocab, and only first blocks need distinct tokens (a later block's
    cache key chains its parent's), so first blocks take 0..F-1."""
    first = {int(sess[sid][j]["hash_ids"][0]) for sid, j in trace}
    ids = {h: i for i, h in enumerate(sorted(first))}
    seen = set()
    with open(path, "w") as f:
        for i, (sid, j) in enumerate(trace):
            r = sess[sid][j]
            n = r["input_length"]
            h = []
            for x in r["hash_ids"].tolist():
                if x not in ids:
                    ids[x] = len(ids)
                h.append(ids[x])
            full = h[: n // B]
            hit = 0
            while hit < len(full) and full[hit] in seen:
                hit += 1
            seen.update(full)
            out = {"timestamp": 0, "session_id": sid, "round_index": j, "input_length": n,
                   "prefix_length": min(hit, (n - 1) // B) * B, "output_length": r["output_length"],
                   "hash_ids": h}
            if "tool_calls" in r:
                out["tool_calls"] = r["tool_calls"]
            out["src_line"] = i
            out["orig_line"] = r["orig_line"]
            f.write(json.dumps(out) + "\n")


def prepare(src, top, max_turns):
    """Load source, build estimator, pick candidate sessions."""
    est = Estimator()
    sess = load(src)
    stats = session_stats(sess, est)
    th = np.percentile([v[1] for v in stats.values()], 100 - top)
    cands = sorted(s for s, (T, ratio, _) in stats.items()
                   if ratio >= th and (not max_turns or T <= max_turns))
    foot = {s: v[2] for s, v in stats.items()}
    return est, sess, stats, cands, foot


def search(sess, est, cands, foot, budgets, seeds, minutes, tpot, lease_ms,
           keep_trace=True, collect=False):
    """Sweep seed × budget; return the best cell (and all cells if collect=True).

    keep_trace=False drops the per-request arrays from each cell (explore mode).
    """
    best, cells = None, []
    for seed in range(seeds):
        order = list(cands)
        random.Random(seed).shuffle(order)
        for budget in budgets:
            cap = 3000
            while True:
                trace = build(order, sess, foot, budget, cap)
                lru, _ = simulate(trace, sess, est, {"lru"}, lease_ms=lease_ms)
                t = replay_seconds(trace, sess, lru, tpot)
                if t[-1] >= minutes * 60 or len(trace) < cap:
                    break
                cap *= 2
            n = int(np.searchsorted(t, minutes * 60, side="right"))
            trace, lru = trace[:n], lru[:n]
            prio, mis = simulate(trace, sess, est, set(), lease_ms=lease_ms)
            gain = 100 * (lru.sum() - prio.sum()) / lru.sum()
            res = dict(seed=seed, budget=budget, n=n, gain=gain,
                       t_lru=t[n - 1] / 60,
                       t_prio=replay_seconds(trace, sess, prio, tpot)[-1] / 60,
                       lru_mean=float(lru.mean()), prio_mean=float(prio.mean()),
                       mis=mis, lease_ms=lease_ms)
            if keep_trace:
                res.update(trace=trace, lru=lru, prio=prio)
            print(f"  seed {seed} budget {budget}: {n} requests, gain {gain:5.1f}%, "
                  f"replay LRU {res['t_lru']:.1f} / prio {res['t_prio']:.1f} min", flush=True)
            if collect:
                cells.append({k: v for k, v in res.items()
                              if k not in ("trace", "lru", "prio")})
            if best is None or gain > best["gain"]:
                best = res
    return (best, cells) if collect else best


def summarize(best, sess, out_name=None):
    """Print the one-line verdict for a best cell; return (sessions, cut, tokens, src_share)."""
    trace = best["trace"]
    used = Counter(sid for sid, _ in trace)
    cut = sum(1 for sid, k in used.items() if k < len(sess[sid]))
    tokens = sum(sess[sid][j]["input_length"] for sid, j in trace)
    src_share = "/".join(
        f"{100 * sum(best['mis'][(rq, t)] for rq in dms.TIERS) / max(1, sum(best['mis'].values())):.0f}"
        for t in dms.TIERS)
    head = out_name or f"lease {best['lease_ms']:g} ms"
    print(f"\n{head}: seed {best['seed']}, budget {best['budget']}, {len(trace)} requests, "
          f"{len(used)} sessions ({cut} cut at the end)")
    print(f"  modeled prefill e2e mean: LRU {best['lru'].mean():.1f} -> request priority "
          f"{best['prio'].mean():.1f} ms (gain {best['gain']:.1f}%); request priority reads "
          f"prefix blocks from HBM/DRAM/SSD {src_share}%")
    print(f"  estimated replay: LRU {best['t_lru']:.1f} min, request priority {best['t_prio']:.1f} min; "
          f"prompt tokens {tokens / 1e6:.0f}M (a request-safe run writes "
          f"~{tokens * KV_BYTES / 1e12:.0f} TB of KV)")
    return used, cut, tokens, src_share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(PRESETS))
    ap.add_argument("--src", type=Path, help="source trace (default: the full trace of the dataset)")
    ap.add_argument("--out", type=Path, help="default: <dataset>_reqprio.jsonl next to this script")
    ap.add_argument("--minutes", type=float, default=30.0, help="LRU arm replay time to cut at")
    ap.add_argument("--top", type=float, default=25.0, help="keep sessions in the top N%% gain ratio")
    ap.add_argument("--max-turns", type=int, help="drop longer sessions (0 = no cap)")
    ap.add_argument("--budgets", help="live-KV budgets in blocks to try, comma separated")
    ap.add_argument("--seeds", type=int, default=4, help="admission orders to try")
    ap.add_argument("--tpot", type=float, help="decode ms per token in the replay-time estimate")
    ap.add_argument("--lease-ms", type=float, default=5000,
                    help="Mooncake kv lease TTL in ms passed to dram_miss_sim (0 = off)")
    a = ap.parse_args()
    pre = PRESETS[a.dataset]
    src = a.src or pre["src"]
    out = a.out or HERE / f"{a.dataset}_reqprio.jsonl"
    max_turns = pre["max_turns"] if a.max_turns is None else a.max_turns
    budgets = [int(x) for x in (a.budgets or pre["budgets"]).split(",")]
    tpot = pre["tpot"] if a.tpot is None else a.tpot

    est, sess, stats, cands, foot = prepare(src, a.top, max_turns)
    print(f"{src.name}: {len(sess)} sessions; top {a.top:g}% gain ratio"
          f"{f', <= {max_turns} turns' if max_turns else ''}: {len(cands)} sessions, "
          f"{sum(stats[s][0] for s in cands)} requests; lease {a.lease_ms:g} ms")
    best = search(sess, est, cands, foot, budgets, a.seeds, a.minutes, tpot, a.lease_ms)
    write(out, best["trace"], sess)
    summarize(best, sess, out_name=out.name)


if __name__ == "__main__":
    main()
