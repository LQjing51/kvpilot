"""Find, per prefix, the shortest suffix at which loading the prefix from
DRAM / SSD stays hidden behind the suffix compute (delay / compute < 5%), on
one live service.

For each prefix (4K..64K, step 4K): one warm on prefill worker A, wait WARM_S
(default 1 s) for its KV to reach DRAM and SSD, then a search over the suffix
for DRAM and for SSD. One measurement is a pair, each with its own suffix
hash ids (see measure(); --no-preheat skips the throwaway same-shape request
and idles GAP_S before each timed request instead):
    [1]      on A, prefix in A's HBM          -> compute
    [2]/[4]  on B, prefix from DRAM / SSD     -> e2e with the load
B is a second prefill worker on 2P2D (--b); on 1P1D (no --b) it is A itself,
the topology the request-safe runs use, and cache_use_control keeps [2] / [4]
off A's HBM copy. --tp-size N (default 2) is how many rank retrieve lines
the source check expects.
Coarse: start at the boundary the first grid found (raw/v1), step x1.25 until
an unsafe suffix and a safe one bracket it; then bisect down to 4 tokens. The
result is the lowest safe suffix above the highest unsafe one. Each point is
measured once; one with a missing log line, the wrong source or
delay / compute < -3% is retaken, up to 3 times.

Writes <out>/requests.jsonl (every request; plot.py joins it with the prefill
logs) and <out>/search.jsonl (measurements and results as the search saw them).

    python3 search.py --a <prefill id A> --b <prefill id B> --logs <log dir>     # 2P2D
    python3 search.py --a <prefill id> --logs <log dir> --out raw/1p1d           # 1P1D
"""
import argparse
import asyncio
import json
import math
import re
import sys
from pathlib import Path

import aiohttp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "vllm-dev/benchmarks/multi_turn"))
from mooncake_trace_replay import TraceRecord, _make_request_id, build_request, send_one  # noqa: E402

BLOCK, LAYERS = 512, 48
THRESH, NEGATIVE = 0.05, -0.03
PREFIXES = list(range(4096, 65536 + 1, 4096))
TIER = {"dram": 2, "ssd": 4}
# Boundaries of the first grid (raw/v1); for 4K DRAM its first safe suffix.
V1 = {"dram": {4096: 640, 8192: 636, 16384: 443, 32768: 436, 65536: 249},
      "ssd": {4096: 1525, 8192: 1269, 16384: 1513, 32768: 2022, 65536: 2033}}
LO, HI = 32, 8192
PURE_BISECT = False
GAP_S, WARM_S = 1.0, 1.0
VOCAB, OFFSET = 151936, 100
PRE = re.compile(r"KVPilot prefill req=(\d{8})-\S+ (?:(?!KVPilot).)*?hit=(\d+) dram=(\d+) e2e=([\d.]+)ms")
RET = re.compile(r"KVPilot prelookup retrieve req=(\d{8})-\S+ chunks=(\d+) ssd_values=(\d+)")


def r4(x):
    return int(round(x / 4)) * 4


def start(medium, p):
    """The v1 boundary at p, interpolated on log2(prefix)."""
    pts = sorted(V1[medium].items())
    for (p0, s0), (p1, s1) in zip(pts, pts[1:]):
        if p <= p1:
            f = (math.log2(p) - math.log2(p0)) / (math.log2(p1) - math.log2(p0))
            return r4(s0 + f * (s1 - s0))
    return pts[-1][1]


class Logs:
    """Follows the prefill logs from where they end now."""

    def __init__(self, d):
        self.files = [open(f, errors="replace") for f in sorted(Path(d).glob("prefill-*.log"))]
        for f in self.files:
            f.seek(0, 2)
        self.buf = [""] * len(self.files)
        self.pre, self.ret = {}, {}

    def poll(self):
        for k, f in enumerate(self.files):
            *lines, self.buf[k] = (self.buf[k] + f.read()).split("\n")
            for l in lines:  # two ranks' records can share a line
                for m in PRE.finditer(l):
                    self.pre[int(m[1])] = (int(m[2]), int(m[3]), float(m[4]))
                for m in RET.finditer(l):
                    self.ret.setdefault(int(m[1]), []).append((int(m[2]), int(m[3])))

    async def wait(self, idx, nret, timeout=20.0):
        for _ in range(int(timeout / 0.2)):
            self.poll()
            if idx in self.pre and len(self.ret.get(idx, [])) >= nret:
                return self.pre[idx], self.ret.get(idx, [])
            await asyncio.sleep(0.2)
        return None, self.ret.get(idx, [])


class Search:
    def __init__(self, args):
        self.wid = {"A": args.a, "B": args.a if args.b is None else args.b}
        self.url = args.url.rstrip("/") + "/v1/completions"
        self.model = args.model
        self.logs = Logs(args.logs)
        self.tp_size = getattr(args, "tp_size", 2)
        self.no_preheat = getattr(args, "no_preheat", False)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        self.req_log = open(out / "requests.jsonl", "a")
        self.search_log = open(out / "search.jsonl", "a")
        self.idx, self.h, self.pair = 0, 1, 0

    def ids(self, n):
        self.h += n
        return list(range(self.h - n, self.h))

    def note(self, f, rec):
        print(json.dumps(rec), file=f, flush=True)

    async def send(self, pre, s, tier, worker, meta, gap=True):
        idx, p = self.idx, len(pre) * BLOCK
        self.idx += 1
        extra = {"nvext": {"prefill_worker_id": self.wid[worker]}}
        if tier:
            extra |= {"cache_use_control": [tier], "cache_use_max_tokens": p}
        rec = TraceRecord(0, p + s, 1, pre + self.ids(-(-s // BLOCK)))
        prompt = build_request(rec, vocab_range=VOCAB - OFFSET, token_offset=OFFSET)
        async with aiohttp.ClientSession() as session:
            _, e2e, _, _, err = await send_one(session, self.url, self.model, prompt, 1,
                                               _make_request_id(idx), 600.0, "[ -~]*", extra)
        self.note(self.req_log, {"idx": idx, "prefix": p, "suffix": s, "tier": tier,
                                 "worker": worker, **meta, "client_e2e": e2e, "err": err})
        if gap:
            await asyncio.sleep(GAP_S)
        return idx

    async def measure(self, p, pre, s, medium, phase):
        """delay / compute of one pair, retaken on an anomaly; None after 3 tries.

        Default: throwaway same-shape pre-heat, then [1] and [2]/[4] back to
        back (busy-hot). --no-preheat: idle GAP_S before each timed request
        instead (both cold; matches raw/1p1d-idle)."""
        tier = TIER[medium]
        for attempt in range(3):
            self.pair += 1
            meta = {"medium": medium, "phase": phase, "pair": self.pair, "attempt": attempt}
            if self.no_preheat:
                await asyncio.sleep(GAP_S)
            else:
                for w in sorted({"A", "B"} if self.wid["A"] != self.wid["B"] else {"A"}):
                    await self.send(pre, s, 1, w, {"medium": medium, "phase": "preheat"}, gap=False)
            i1 = await self.send(pre, s, 1, "A", meta, gap=False)
            if self.no_preheat:
                await asyncio.sleep(GAP_S)
            it = await self.send(pre, s, tier, "B", meta, gap=not self.no_preheat)
            (r1, _), (rt, ret) = (
                await self.logs.wait(i1, 0),
                await self.logs.wait(it, self.tp_size),
            )
            why, ratio = None, None
            if r1 is None or rt is None:
                why = "missing log"
            elif not (r1[0] >= p and r1[1] == 0):
                why = "[1] source"
            elif not (rt[1] >= p and len(ret) == self.tp_size and all(
                    c == p // BLOCK and v == (0 if tier == 2 else LAYERS * c) for c, v in ret)):
                why = f"[{tier}] source"
            else:
                ratio = (rt[2] - r1[2]) / r1[2]
                if ratio < NEGATIVE:
                    why = "negative"
            self.note(self.search_log, {"prefix": p, "medium": medium, "suffix": s,
                                        "phase": phase, "pair": self.pair,
                                        "compute": r1 and r1[2], "e2e": rt and rt[2],
                                        "ratio": ratio, "anomaly": why})
            print(f"{p // 1024:3d}K {medium:4s} s={s:5d} {phase:7s} "
                  + (f"ANOMALY {why}" if why else f"{ratio:7.1%} {'safe' if ratio < THRESH else 'UNSAFE'}"),
                  flush=True)
            if why is None:
                return ratio
        return None

    async def search(self, p, pre, medium):
        """Lowest measured-safe suffix above the highest measured-unsafe one."""
        seen = {}

        async def m(s, phase):
            r = await self.measure(p, pre, s, medium, phase)
            if r is not None:
                seen[s] = r
            return r is not None

        def bracket():
            lo = max((s for s, r in seen.items() if r >= THRESH), default=0)
            return lo, min((s for s, r in seen.items() if s > lo and r < THRESH), default=None)

        if PURE_BISECT:
            # Endpoints then midpoint bisect within [LO, HI] (token multiples of 4).
            lo_s = max(r4(LO), 4)
            hi_s = r4(HI)
            for s in (lo_s, hi_s):
                if not await m(s, "end"):
                    return None, "measurement failed"
            lo, hi = bracket()
            if hi is None:
                return None, f"no safe suffix within [{lo_s}, {hi_s}]"
            if lo == 0:
                return hi, ""
            while hi - lo > 4:
                mid = r4((lo + hi) / 2)
                if mid <= lo:
                    mid = lo + 4
                if mid >= hi:
                    mid = hi - 4
                if mid in seen or not (lo_s <= mid <= hi_s):
                    mid = lo + max((hi - lo) // 8, 1) * 4
                    if mid in seen or mid <= lo or mid >= hi:
                        break
                if not await m(mid, "bisect"):
                    return None, "measurement failed"
                lo, hi = bracket()
                if hi is None:
                    return None, f"no safe suffix within [{lo_s}, {hi_s}]"
            return hi, ""

        if not await m(start(medium, p), "coarse"):
            return None, "measurement failed"
        while True:
            lo, hi = bracket()
            if hi is not None and lo:
                break
            s = max(r4(lo * 1.25), lo + 4) if hi is None else min(r4(hi / 1.25), hi - 4)
            if not LO <= s <= HI:
                return hi, f"no bracket within [{LO}, {HI}]"
            if not await m(s, "coarse"):
                return None, "measurement failed"
        while hi - lo > 4:
            if not await m(lo + max((hi - lo) // 8, 1) * 4, "bisect"):
                return None, "measurement failed"
            lo, hi = bracket()
        return hi, ""

    async def run(self):
        for p in PREFIXES:
            pre = self.ids(p // BLOCK)
            i = await self.send(pre, 0, 0, "A", {"phase": "warm"})
            got, _ = await self.logs.wait(i, 0)
            print(f"{p // 1024:3d}K warm e2e={got and got[2]}", flush=True)
            await asyncio.sleep(WARM_S)
            for medium in ("dram", "ssd"):
                b, why = await self.search(p, pre, medium)
                self.note(self.search_log, {"prefix": p, "medium": medium, "boundary": b, "note": why})
                print(f"{p // 1024:3d}K {medium:4s} RESULT boundary={b} {why}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, required=True)
    ap.add_argument("--b", type=int, default=None, help="2P2D load worker; omit on 1P1D")
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", default=str(HERE / "raw"))
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="/cfs_cloud_code/kiry/model/Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--prefixes", default=None, help="comma-separated, instead of 4K..64K")
    ap.add_argument("--tp-size", type=int, default=2,
                    help="TP ranks; source check expects this many retrieve lines")
    ap.add_argument("--no-preheat", action="store_true",
                    help="idle GAP_S before each timed request; no throwaway pre-heat")
    ap.add_argument("--warm-s", type=float, default=None,
                    help="seconds to wait after warm (default WARM_S)")
    ap.add_argument("--layers", type=int, default=None,
                    help="transformer layers; SSD retrieve check uses LAYERS * chunks "
                         "(48 for Qwen3-30B, 94 for Qwen3-235B)")
    ap.add_argument("--suffix-lo", type=int, default=None,
                    help="search lower bound (default LO)")
    ap.add_argument("--suffix-hi", type=int, default=None,
                    help="search upper bound (default HI)")
    ap.add_argument("--pure-bisect", action="store_true",
                    help="bisect [suffix-lo, suffix-hi] from endpoints; ignore V1 start")
    args = ap.parse_args()
    if args.prefixes:
        PREFIXES[:] = [int(p) for p in args.prefixes.split(",")]
    if args.warm_s is not None:
        globals()["WARM_S"] = args.warm_s
    if args.layers is not None:
        globals()["LAYERS"] = args.layers
    if args.suffix_lo is not None:
        globals()["LO"] = args.suffix_lo
    if args.suffix_hi is not None:
        globals()["HI"] = args.suffix_hi
    if args.pure_bisect:
        globals()["PURE_BISECT"] = True
    asyncio.run(Search(args).run())
