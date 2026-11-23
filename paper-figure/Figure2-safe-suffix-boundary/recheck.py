"""Re-measure the boundaries that stand out from their neighbours, several
times each, to tell a one-off reading from the real value.

A boundary (<raw>/search.jsonl) is an outlier when it is more than 20% above
every neighbouring prefix's boundary (the adjacent prefixes the search ran,
whatever their spacing; the end prefixes have one) or more than 20% below all
of them. One the search never bracketed has no boundary. For each outlier, on a fresh service with the prefix warmed once,
measure REPS times, round robin over:
  the suffix the search found unsafe just below the boundary,
  the boundary itself,
  the neighbours' mean boundary, and the midpoint between the two.
Same measurement and anomaly retakes as search.py.

    python3 recheck.py --a <prefill id A> --b <prefill id B> --logs <log dir>   # 2P2D
    python3 recheck.py --a <prefill id> --logs <log dir> --raw raw/1p1d         # 1P1D

Writes <raw>/recheck/{requests,search}.jsonl.
"""
import argparse
import asyncio
import json
import statistics
from collections import defaultdict
from pathlib import Path

from search import BLOCK, THRESH, WARM_S, Search, r4

HERE = Path(__file__).resolve().parent
REPS, TOL = 5, 0.2


def outliers(path):
    recs = [json.loads(l) for l in open(path)]
    bnd = {(r["prefix"], r["medium"]): None if r["note"] else r["boundary"]
           for r in recs if "boundary" in r}
    meas = defaultdict(dict)
    for r in recs:
        if "suffix" in r and r["ratio"] is not None and not r["anomaly"]:
            meas[(r["prefix"], r["medium"])][r["suffix"]] = r["ratio"]
    todo = []
    ps = sorted({p for p, _ in bnd})
    for (p, m), b in sorted(bnd.items()):
        i = ps.index(p)
        adj = ps[max(i - 1, 0):i] + ps[i + 1:i + 2]
        nb = [x for x in (bnd.get((q, m)) for q in adj) if x]
        if b is None or not nb:
            print(f"{p // 1024}K {m}: boundary {b}, neighbours {nb}, not checked")
            continue
        d = [(b - x) / x for x in nb]
        if not (all(v > TOL for v in d) or all(v < -TOL for v in d)):
            continue
        lo = max((s for s, r in meas[(p, m)].items() if r >= THRESH and s < b), default=None)
        ref = r4(sum(nb) / len(nb))
        todo.append((p, m, sorted({s for s in (lo, b, ref, r4((b + ref) / 2)) if s})))
        print(f"{p // 1024}K {m}: boundary {b}, neighbours {nb}, recheck {todo[-1][2]}")
    return todo


async def main(args):
    raw = HERE / args.raw
    todo = outliers(raw / "search.jsonl")
    out = raw / "recheck"
    out.mkdir(exist_ok=True)
    import search as search_mod
    if args.warm_s is not None:
        search_mod.WARM_S = args.warm_s
    s = Search(argparse.Namespace(
        a=args.a, b=args.b, url=args.url, model=args.model, logs=args.logs,
        out=str(out), tp_size=args.tp_size, no_preheat=args.no_preheat))
    for p in sorted({t[0] for t in todo}):
        pre = s.ids(p // BLOCK)
        i = await s.send(pre, 0, 0, "A", {"phase": "warm"})
        got, _ = await s.logs.wait(i, 0)
        print(f"{p // 1024:3d}K warm e2e={got and got[2]}", flush=True)
        await asyncio.sleep(search_mod.WARM_S)
        for _, m, pts in (t for t in todo if t[0] == p):
            res = defaultdict(list)
            for _ in range(REPS):
                for x in pts:
                    r = await s.measure(p, pre, x, m, "recheck")
                    if r is not None:
                        res[x].append(r)
            for x in pts:
                v = res[x]
                unsafe = sum(r >= THRESH for r in v)
                s.note(s.search_log, {"prefix": p, "medium": m, "suffix": x, "ratios": v,
                                      "median": v and statistics.median(v), "unsafe": unsafe})
                print(f"{p // 1024:3d}K {m:4s} s={x:5d} SUMMARY unsafe {unsafe}/{len(v)}  median "
                      f"{statistics.median(v):6.1%}  all " + " ".join(f"{r:.1%}" for r in v), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, required=True)
    ap.add_argument("--b", type=int, default=None, help="2P2D load worker; omit on 1P1D")
    ap.add_argument("--logs", required=True)
    ap.add_argument("--raw", default="raw", help="dir holding the search's search.jsonl")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="/cfs_cloud_code/kiry/model/Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--no-preheat", action="store_true")
    ap.add_argument("--warm-s", type=float, default=None)
    asyncio.run(main(ap.parse_args()))
