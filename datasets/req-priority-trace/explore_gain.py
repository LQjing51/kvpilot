"""Sweep lease / budget / seed for the modeled e2e gain of request-priority
eviction over LRU. Does not write a replay trace — use gen_trace.py for that.

    python3 explore_gain.py swe
    python3 explore_gain.py swe --lease-ms 0,1000,5000
    python3 explore_gain.py tracelab --lease-ms 1000 --budgets 3800,4000,4200

Same selection / round-robin / end-cut logic as gen_trace.py (imports it).
For each --lease-ms value, searches seed × budget and prints the best cell,
then a summary table across leases.
"""
import argparse
from pathlib import Path

import numpy as np

import gen_trace as gt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=list(gt.PRESETS))
    ap.add_argument("--src", type=Path, help="source trace (default: the full trace of the dataset)")
    ap.add_argument("--minutes", type=float, default=30.0, help="LRU arm replay time to cut at")
    ap.add_argument("--top", type=float, default=25.0, help="keep sessions in the top N%% gain ratio")
    ap.add_argument("--max-turns", type=int, help="drop longer sessions (0 = no cap)")
    ap.add_argument("--budgets", help="live-KV budgets in blocks to try, comma separated")
    ap.add_argument("--seeds", type=int, default=4, help="admission orders to try")
    ap.add_argument("--tpot", type=float, help="decode ms per token in the replay-time estimate")
    ap.add_argument("--lease-ms", default="0,1000,5000",
                    help="comma-separated Mooncake kv lease TTLs in ms (0 = off). "
                         "Default sweeps 0 / 1s / 5s")
    a = ap.parse_args()
    pre = gt.PRESETS[a.dataset]
    src = a.src or pre["src"]
    max_turns = pre["max_turns"] if a.max_turns is None else a.max_turns
    budgets = [int(x) for x in (a.budgets or pre["budgets"]).split(",")]
    tpot = pre["tpot"] if a.tpot is None else a.tpot
    leases = [float(x) for x in a.lease_ms.split(",")]

    est, sess, stats, cands, foot = gt.prepare(src, a.top, max_turns)
    print(f"{src.name}: {len(sess)} sessions; top {a.top:g}% gain ratio"
          f"{f', <= {max_turns} turns' if max_turns else ''}: {len(cands)} sessions, "
          f"{sum(stats[s][0] for s in cands)} requests")
    print(f"sweep lease_ms={leases}, budgets={budgets}, seeds={a.seeds}, "
          f"cut at {a.minutes:g} min LRU replay\n")

    rows = []
    for lease in leases:
        print(f"=== lease {lease:g} ms ===")
        best, cells = gt.search(sess, est, cands, foot, budgets, a.seeds,
                                a.minutes, tpot, lease, keep_trace=False, collect=True)
        src_share = "/".join(
            f"{100 * sum(best['mis'][(rq, t)] for rq in gt.dms.TIERS) / max(1, sum(best['mis'].values())):.0f}"
            for t in gt.dms.TIERS)
        at_budget = [c["gain"] for c in cells if c["budget"] == best["budget"]]
        rows.append(dict(lease_ms=lease, seed=best["seed"], budget=best["budget"],
                         n=best["n"], gain=best["gain"],
                         mean_at_budget=float(np.mean(at_budget)),
                         lru_mean=best["lru_mean"], prio_mean=best["prio_mean"],
                         src_share=src_share))
        print(f"  best: seed {best['seed']} budget {best['budget']}, {best['n']} requests, "
              f"gain {best['gain']:.1f}% (mean at that budget {np.mean(at_budget):.1f}%), "
              f"HBM/DRAM/SSD {src_share}%\n")

    print(f"{'lease_ms':>10s} {'gain%':>7s} {'mean%':>7s} {'n':>6s} "
          f"{'seed':>5s} {'budget':>7s} {'LRU→prio ms':>14s} {'HBM/DRAM/SSD':>14s}")
    for r in rows:
        print(f"{r['lease_ms']:10g} {r['gain']:6.1f}% {r['mean_at_budget']:6.1f}% "
              f"{r['n']:6d} {r['seed']:5d} {r['budget']:7d} "
              f"{r['lru_mean']:6.1f}→{r['prio_mean']:<5.1f} {r['src_share']:>14s}")


if __name__ == "__main__":
    main()
