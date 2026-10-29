#!/usr/bin/env python3
"""Side-by-side token and stall breakdown for two or more runs.

Answers why one configuration's exposed-load share differs from another's, by
splitting the ratio into its parts: how many requests loaded, how much each
loaded, and how long a unit of loading cost.
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

FLOATS = {"e2e", "stall", "pct"}


def rows_for(log_dir):
    rows = []
    for path in sorted(Path(log_dir).glob("prefill-*.log")):
        for text in path.open(errors="replace"):
            m = LINE_RE.search(text)
            if not m:
                continue
            g = m.groupdict()
            rows.append({
                k: (float(g[k]) if k in FLOATS else int(g[k]))
                for k in ("prompt", "new", "hit", "dram", "e2e", "stall",
                          "waits", "steps")
            })
    return rows


def report(log_dir):
    rows = rows_for(log_dir)
    name = Path(log_dir).name
    if not rows:
        print(f"== {name}: no records ==")
        return
    n = len(rows)
    ld = [r for r in rows if r["dram"] > 0]
    tot = lambda k, rs=rows: sum(r[k] for r in rs)  # noqa: E731

    print(f"== {name}  n={n} ==")
    print(f"  sum_stall={tot('stall')/1000:8.0f}s   sum_e2e={tot('e2e')/1000:8.0f}s"
          f"   -> EXPOSED {100*tot('stall')/tot('e2e'):.2f}%")
    print(f"  tokens  prompt={tot('prompt'):>12,}  new={tot('new'):>12,}"
          f"  hit={tot('hit'):>12,}  dram={tot('dram'):>12,}")
    print(f"  /req    prompt={tot('prompt')/n:>12,.0f}  new={tot('new')/n:>12,.0f}"
          f"  hit={tot('hit')/n:>12,.0f}  dram={tot('dram')/n:>12,.0f}")
    print(f"  /req    e2e={tot('e2e')/n:.0f} ms   stall={tot('stall')/n:.0f} ms")
    if ld:
        m = len(ld)
        print(f"  loaders n={m} ({100*m/n:.1f}%)  dram/req={tot('dram', ld)/m:,.0f}"
              f"  stall/req={tot('stall', ld)/m:.0f} ms"
              f"  e2e/req={tot('e2e', ld)/m:.0f} ms")
        print(f"  cost of loading: {1000*tot('stall', ld)/tot('dram', ld):.1f}"
              f" ms per 1k DRAM tokens")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        report(d)
