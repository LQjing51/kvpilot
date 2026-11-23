"""Confirm every boundary with repeated readings, moving it where the repeats
disagree with what the search (and recheck) saw.

Per (prefix, medium), start from the bracket the earlier readings give (the
highest unsafe suffix lo, the lowest safe one above it hi; recheck medians
replace search readings). On a fresh service with each prefix warmed once,
judge a suffix on the median of REPS pairs:
  hi unsafe -> it becomes lo; the next safe reading above (else hi x 1.25) is hi
  lo safe   -> it becomes hi; the next unsafe reading below (else hi / 1.25) is lo
and once both hold, bisect on medians until hi - lo <= GAP x hi. The boundary
is hi, with lo just below it: both backed by REPS readings each. Same
measurement and anomaly retakes as search.py.

    python3 confirm.py --a <prefill id> --logs <log dir> --raw raw/1p1d   # 1P1D

Writes <raw>/confirm/{requests,search}.jsonl: per judged suffix its readings,
per (prefix, medium) {"boundary", "lo", "note"}; a note means unconfirmed.
"""
import argparse
import asyncio
import json
import statistics
from collections import defaultdict
from pathlib import Path

from search import BLOCK, HI, LO, THRESH, WARM_S, Search, r4

HERE = Path(__file__).resolve().parent
REPS, GAP, MAX_POINTS = 5, 0.1, 10


def readings(raw):
    """(prefix, medium) -> {suffix: ratio} from the search, recheck medians on top."""
    pts = defaultdict(dict)
    for r in map(json.loads, open(raw / "search.jsonl")):
        if "suffix" in r and r["ratio"] is not None and not r["anomaly"]:
            pts[(r["prefix"], r["medium"])][r["suffix"]] = r["ratio"]
    rc = raw / "recheck/search.jsonl"
    if rc.exists():
        for r in map(json.loads, open(rc)):
            if r.get("ratios"):
                pts[(r["prefix"], r["medium"])][r["suffix"]] = r["median"]
    return pts


async def confirm(s, p, pre, m, seen):
    med = {}

    async def safe(x):
        """Median of REPS readings below THRESH; None if none could be taken."""
        if x not in med:
            v = []
            for _ in range(REPS):
                r = await s.measure(p, pre, x, m, "confirm")
                if r is not None:
                    v.append(r)
            med[x] = statistics.median(v) if v else None
            s.note(s.search_log, {"prefix": p, "medium": m, "suffix": x, "ratios": v,
                                  "median": med[x], "unsafe": sum(r >= THRESH for r in v)})
            print(f"{p // 1024:3d}K {m:4s} s={x:5d} CONFIRM "
                  + (f"median {med[x]:6.1%} unsafe {sum(r >= THRESH for r in v)}/{len(v)}"
                     if v else "no reading"), flush=True)
        return None if med[x] is None else med[x] < THRESH

    def up(x):
        cand = [k for k, r in seen.items() if k > x and r < THRESH and k not in med]
        return min(cand + [max(r4(x * 1.25), x + 4)])

    def down(x):
        cand = [k for k, r in seen.items() if k < x and r >= THRESH and k not in med]
        return max(cand, default=None)

    lo = max((k for k, r in seen.items() if r >= THRESH), default=None)
    hi = min((k for k, r in seen.items() if r < THRESH and (lo is None or k > lo)), default=None)
    if hi is None:
        hi = up(lo or LO)
    while True:
        if len(med) >= MAX_POINTS:
            return hi, lo, f"{MAX_POINTS} suffixes judged, still open"
        if hi > HI:
            return None, lo, f"no safe suffix up to {HI}"
        ok = await safe(hi)
        if ok is None:
            return None, lo, f"no reading at {hi}"
        if not ok:
            lo, hi = hi, up(hi)
            continue
        if lo is None:
            lo = r4(hi / 1.25)
            if lo < LO:
                return hi, None, f"safe down to {hi}"
        ok = await safe(lo)
        if ok is None:
            return None, lo, f"no reading at {lo}"
        if ok:
            hi, lo = lo, down(lo)
            continue
        if hi - lo <= max(8, GAP * hi):
            return hi, lo, ""
        mid = r4((lo + hi) / 2)
        ok = await safe(mid)
        if ok is None:
            return None, lo, f"no reading at {mid}"
        if ok:
            hi = mid
        else:
            lo = mid


async def main(args):
    raw = HERE / args.raw
    seen = readings(raw)
    out = raw / "confirm"
    out.mkdir(exist_ok=True)
    import search as search_mod
    if args.warm_s is not None:
        search_mod.WARM_S = args.warm_s
    s = Search(argparse.Namespace(
        a=args.a, b=args.b, url=args.url, model=args.model, logs=args.logs,
        out=str(out), tp_size=args.tp_size, no_preheat=args.no_preheat))
    for p in sorted({k[0] for k in seen}):
        pre = s.ids(p // BLOCK)
        i = await s.send(pre, 0, 0, "A", {"phase": "warm"})
        got, _ = await s.logs.wait(i, 0)
        print(f"{p // 1024:3d}K warm e2e={got and got[2]}", flush=True)
        await asyncio.sleep(search_mod.WARM_S)
        for m in ("dram", "ssd"):
            b, lo, why = await confirm(s, p, pre, m, seen[(p, m)])
            s.note(s.search_log, {"prefix": p, "medium": m, "boundary": b, "lo": lo, "note": why})
            print(f"{p // 1024:3d}K {m:4s} CONFIRMED boundary={b} lo={lo} {why}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, required=True)
    ap.add_argument("--b", type=int, default=None, help="2P2D load worker; omit on 1P1D")
    ap.add_argument("--logs", required=True)
    ap.add_argument("--raw", default="raw", help="dir holding search.jsonl (and recheck/)")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="/cfs_cloud_code/kiry/model/Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--no-preheat", action="store_true")
    ap.add_argument("--warm-s", type=float, default=None)
    asyncio.run(main(ap.parse_args()))
