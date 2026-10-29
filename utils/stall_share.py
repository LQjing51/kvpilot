#!/usr/bin/env python3
"""Aggregate KVPILOT_LOAD_STALL log lines into the exposed-KV-load share.

    python3 stall_share.py <log_dir> [<log_dir> ...]

The headline number is sum(stall)/sum(e2e) over every prefill in the run: how
much of a unit of prefill work went to waiting for KV instead of computing. It
is the same quantity nsys reports as kv_load_exposed_pct, measured independently
from the engine thread's wall clock, and the target demo_test.md sets thresholds
against.

Also prints the loaders-only share, because the two answer different questions
and only the first is comparable across variants: KVPilot changes both which
requests need to load and how long each waits, so a denominator of all prefills
is what keeps an A/B honest.
"""
import re
import sys
from pathlib import Path

LINE_RE = re.compile(
    r"KVPilot prefill req=(?P<req>\S+?)\s+(?:line=(?P<line>-?\d+)\s+)?"
    r"prompt=(?P<prompt>\d+)\s+new=(?P<new>\d+)\s+hit=(?P<hit>\d+)\s+"
    r"dram=(?P<dram>\d+)\s+e2e=(?P<e2e>[\d.]+)ms\s+stall=(?P<stall>[\d.]+)ms\s+"
    r"\((?P<pct>[\d.]+)%\)\s+waits=(?P<waits>\d+)\s+steps=(?P<steps>\d+)"
)


def rows_for(log_dir):
    rows = []
    for path in sorted(Path(log_dir).glob("prefill-*.log")):
        for text in path.open(errors="replace"):
            m = LINE_RE.search(text)
            if m:
                g = m.groupdict()
                rows.append({
                    "dram": int(g["dram"]),
                    "e2e": float(g["e2e"]),
                    "stall": float(g["stall"]),
                    "line": int(g["line"] or 0),
                })
    return rows


def report(log_dir):
    rows = rows_for(log_dir)
    name = Path(log_dir).name
    if not rows:
        print(f"{name:28s} no records (KVPILOT_LOAD_STALL unset?)")
        return
    e2e = sum(r["e2e"] for r in rows)
    stall = sum(r["stall"] for r in rows)
    ld = [r for r in rows if r["dram"] > 0]
    ld_e2e = sum(r["e2e"] for r in ld)
    ld_stall = sum(r["stall"] for r in ld)
    # A request that loaded nothing cannot have waited for a load; a nonzero
    # figure here would mean the scheduler's split leaked another request's wait.
    bad = [r for r in rows if r["dram"] == 0 and r["stall"] > 0]
    print(f"{name:28s} n={len(rows):5d} loaders={len(ld):5d} "
          f"({100*len(ld)/len(rows):4.1f}%)  "
          f"EXPOSED={100*stall/e2e:5.2f}%  "
          f"loaders_only={100*ld_stall/ld_e2e if ld_e2e else 0:5.1f}%  "
          f"sum_e2e={e2e/1000:7.0f}s sum_stall={stall/1000:6.0f}s  "
          f"selfcheck={'OK' if not bad else f'{len(bad)} BAD'}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        report(d)
