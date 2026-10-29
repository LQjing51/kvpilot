#!/usr/bin/env python3
"""Derive measured dram-safe verdicts from a run's load-stall logs.

    python3 build_measured_safe.py <log_dir> -o req_measured_safe.json

A turn is dram-safe if keeping its KV in DRAM costs the next prefill nothing --
that is, if the load hid behind the compute. The load-stall log measures exactly
that residue: `stall` is the part of a prefill that went to waiting for KV
rather than computing.

Keyed by the trace line in the request id, which is the join key the cost-model
table (`req_ratio.json`) and the HBM census already use, so the three can be
compared row by row.

Only requests that actually loaded from DRAM carry evidence: one with `dram=0`
never exercised the load path, so its stall says nothing about whether a load
would have hidden, and it is left out rather than recorded as trivially safe.

The label compares measured stall against the compute it had to hide behind,
which is what "dram-safe" means. Two corrections make that comparison honest:

`e2e` is NOT the denominator. It spans scheduled-until-done and so absorbs time
the request spent batched behind other work: measured across these runs,
apparent compute per 1k recomputed tokens came out 2076 ms without the H2D path
and 100 ms with it, a 21x spread on statistically identical token counts.
Compute cannot vary that much, so most of `e2e` is queueing, and a ratio against
it grades congestion rather than overlap.

Stall carries a fixed per-request cost that no placement decision can avoid.
Fitting `stall = fixed + slope * dram_tokens` over a run gives ~11 ms of setup
plus ~2.2 ms per 1k tokens, the two balancing around 5k tokens. Judging on raw
ms-per-1k therefore just asks "did this request load enough to amortize the
setup": small loads look catastrophic (p50 19.7 ms/1k under 2k tokens) and large
ones look free (2.8 ms/1k), which is why the naive version labelled 98% of the
largest prefixes safe and 14% of the smallest -- exactly backwards from the cost
model. Subtracting the fitted fixed cost flattens the per-token rate to ~2.07
across every size, and what remains is the part that placement can actually
change.
"""
import argparse
import json
import statistics
import re
from collections import defaultdict
from pathlib import Path

LINE_RE = re.compile(
    r"KVPilot prefill req=(?P<req>\S+?)\s+(?:line=(?P<line>-?\d+)\s+)?"
    r"prompt=(?P<prompt>\d+)\s+new=(?P<new>\d+)\s+hit=(?P<hit>\d+)\s+"
    r"dram=(?P<dram>\d+)\s+e2e=(?P<e2e>[\d.]+)ms\s+stall=(?P<stall>[\d.]+)ms\s+"
    r"\((?P<pct>[\d.]+)%\)\s+waits=(?P<waits>\d+)\s+steps=(?P<steps>\d+)"
)


def parse(log_dir):
    by_line = defaultdict(list)
    for path in sorted(Path(log_dir).glob("prefill-*.log")):
        for text in path.open(errors="replace"):
            m = LINE_RE.search(text)
            if not m:
                continue
            g = m.groupdict()
            line = int(g["line"] or -1)
            if line <= 0:
                continue
            by_line[line].append({
                "dram": int(g["dram"]),
                "new": int(g["new"]),
                "e2e": float(g["e2e"]),
                "stall": float(g["stall"]),
                "pct": float(g["pct"]),
            })
    return by_line


def fit_fixed_cost(samples):
    """Least-squares `stall = fixed + slope * dram_tokens` over the run.

    The fixed term is the per-request setup a load pays regardless of size;
    placement cannot remove it, so it is subtracted before judging.
    """
    n = len(samples)
    sx = sum(t for t, _ in samples)
    sy = sum(s for _, s in samples)
    sxx = sum(t * t for t, _ in samples)
    sxy = sum(t * s for t, s in samples)
    den = n * sxx - sx * sx
    if not den:
        return 0.0, 0.0
    slope = (n * sxy - sx * sy) / den
    return max((sy - slope * sx) / n, 0.0), slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument(
        "--max-ms-per-1k", type=float, default=1.0,
        help="A load counts as hidden when its size-dependent stall -- the part "
             "left after the fitted fixed cost -- is at most this many ms per "
             "1k DRAM tokens.",
    )
    ap.add_argument(
        "--fixed-ms", type=float, default=None,
        help="Override the fitted per-request fixed cost (ms).",
    )
    args = ap.parse_args()

    by_line = parse(args.log_dir)
    samples = [
        (r["dram"], r["stall"])
        for recs in by_line.values() for r in recs if r["dram"] > 0
    ]
    if not samples:
        raise SystemExit("no DRAM-loading prefills found in that log dir")
    fitted_fixed, slope = fit_fixed_cost(samples)
    fixed = args.fixed_ms if args.fixed_ms is not None else fitted_fixed

    entries = {}
    n_noload = 0
    for line, recs in by_line.items():
        loaded = [r for r in recs if r["dram"] > 0]
        if not loaded:
            n_noload += 1
            continue
        # Several attempts of one turn are averaged: the same KV either hides or
        # does not, and a single scheduling accident should not flip the label.
        stall = statistics.fmean(r["stall"] for r in loaded)
        dram = statistics.fmean(r["dram"] for r in loaded)
        per_1k = 1000.0 * stall / dram if dram else 0.0
        adj_1k = 1000.0 * max(stall - fixed, 0.0) / dram if dram else 0.0
        entries[str(line)] = {
            "dram_safe": adj_1k <= args.max_ms_per_1k,
            "adj_ms_per_1k": round(adj_1k, 3),
            "ms_per_1k": round(per_1k, 3),
            "stall_ms": round(stall, 3),
            "stall_pct": round(statistics.fmean(r["pct"] for r in loaded), 3),
            "dram_tokens": int(dram),
            "n": len(loaded),
        }

    safe = sum(1 for v in entries.values() if v["dram_safe"])
    payload = {
        "_meta": {
            "source_log_dir": str(Path(args.log_dir).resolve()),
            "max_ms_per_1k": args.max_ms_per_1k,
            "fixed_ms": round(fixed, 3),
            "fitted_fixed_ms": round(fitted_fixed, 3),
            "fitted_ms_per_1k": round(1000 * slope, 3),
            "requests": len(entries),
            "dram_safe": safe,
            "lines_without_load": n_noload,
            "definition": "dram_safe = (mean stall - fixed) / mean dram tokens "
                          "<= max_ms_per_1k, per 1k tokens",
        },
        "requests": entries,
    }
    with open(args.output, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"-> {args.output}: requests={len(entries):,} "
          f"dram_safe={safe:,} ({100*safe/max(len(entries),1):.1f}%) "
          f"skipped_no_load={n_noload:,}")
    print(f"   fitted fixed={fitted_fixed:.2f} ms, "
          f"slope={1000*slope:.3f} ms/1k tokens, using fixed={fixed:.2f} ms")


if __name__ == "__main__":
    main()
