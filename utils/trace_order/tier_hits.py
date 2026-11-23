"""Measured HBM / DRAM / SSD hit shares of one replay, from the prefill logs.

    python3 tier_hits.py <log dir or prefill-0.log[.gz]> [trace.jsonl]
                         [--hbm BLOCKS] [--dram-gib GIB_PER_TP_RANK]

Per request (the 8-digit request-id prefix is the trace line):
  HBM   `KVPilot prefill ... hit=`               local prefix cache
  SSD   `KVPilot prelookup retrieve ... ssd_tokens=`, larger of the two TP ranks
  DRAM  `KVPilot prefill ... dram=` minus SSD    connector-loaded, from DRAM
  new   `KVPilot prefill ... new=`
A request prefilled more than once (preempted) counts by its first prefill.

With a trace, also prints tier_sim's prediction for the same order (at the
given capacities, tier_sim's defaults otherwise), and the shares over
trace-order windows so a run can be read while it is going.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PRE = re.compile(r"KVPilot prefill req=(\d{8})-\S+ line=-?\d+ prompt=(\d+) new=(\d+) "
                 r"hit=(\d+) dram=(\d+) e2e=([\d.]+)ms")
RET = re.compile(r"KVPilot prelookup retrieve req=(\d{8})-\S+ .*?ssd_tokens=(\d+)")


def read_logs(src: Path) -> tuple[dict, dict]:
    files = sorted(src.glob("prefill-*.log*")) if src.is_dir() else [src]
    files = [f for f in files if not (f.suffix == ".log" and f.with_suffix(".log.gz") in files)]
    pre, ssd = {}, {}
    for f in files:
        op = gzip.open if f.suffix == ".gz" else open
        with op(f, "rt", errors="replace") as fh:
            for line in fh:
                if "KVPilot pre" not in line:
                    continue
                line = ANSI.sub("", line)
                if m := PRE.search(line):
                    pre.setdefault(int(m[1]), tuple(int(m[i]) for i in range(2, 6)) + (float(m[6]),))
                elif m := RET.search(line):
                    i = int(m[1])
                    ssd[i] = max(ssd.get(i, 0), int(m[2]))
    return pre, ssd


def tiers(pre: dict, ssd: dict) -> dict:
    """line -> (hbm, dram, ssd, new) tokens."""
    out = {}
    for i, (prompt, new, hit, conn, _) in pre.items():
        s = min(ssd.get(i, 0), conn)
        out[i] = (hit, conn - s, s, new)
    return out


def share(rows) -> str:
    tot = sum(sum(x) for x in rows) or 1
    s = [sum(x[i] for x in rows) for i in range(4)]
    n = len(rows) or 1
    return (f"HBM {s[0] / tot:5.1%}  DRAM {s[1] / tot:5.1%}  SSD {s[2] / tot:5.1%}  "
            f"new {s[3] / tot:5.1%}  | req w/ DRAM {sum(x[1] > 0 for x in rows) / n:5.1%}"
            f"  w/ SSD {sum(x[2] > 0 for x in rows) / n:5.1%}")


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tier_sim import DRAM_GIB, HBM_BLOCKS, dram_blocks, simulate

    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--hbm", type=int, default=HBM_BLOCKS)
    ap.add_argument("--dram-gib", type=float, default=DRAM_GIB)
    a = ap.parse_args()
    pre, ssd = read_logs(Path(a.logs))
    t = tiers(pre, ssd)
    lines = sorted(t)
    print(f"{len(t)} prefills, prompt tokens {sum(sum(x) for x in t.values()):,}")
    print(f"measured   {share([t[i] for i in lines])}")
    if a.trace:
        dram = dram_blocks(a.dram_gib)
        rows = [json.loads(x) for x in open(a.trace)]
        sim = simulate(rows, a.hbm, dram)
        print(f"simulation at hbm {a.hbm} blocks, dram {dram} blocks")
        done = [i for i in lines if i < len(rows)]
        print(f"simulated  {share([sim[i] for i in done])}   (same {len(done)} lines)")
        print(f"simulated  {share(sim)}   (whole trace)")
        step = max(1, len(rows) // 10)
        print("by trace window: measured | simulated")
        for a in range(0, len(rows), step):
            w = [i for i in done if a <= i < a + step]
            if w:
                print(f"  {a:>6}-{a + step - 1:<6} {share([t[i] for i in w])}\n"
                      f"  {'':13} {share([sim[i] for i in w])}")


if __name__ == "__main__":
    main()
