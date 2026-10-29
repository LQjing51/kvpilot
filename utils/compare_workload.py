#!/usr/bin/env python3
"""Did two runs actually replay the same workload?

    python3 compare_workload.py <run_a.out> <run_b.out> [...]

The first file is the baseline. Reads the client logs (`run-<name>.out`), not
the server logs, because the question is what was *sent*.

Worth checking before trusting any A/B, and easy to assume without checking.
The schedule is seeded, so sessions start at fixed times -- but `-n` truncates
on completion count, and `--tool-time` is a closed loop: the next turn is
dispatched once the previous one *returns*, plus a pre-drawn Z. So a
configuration that serves faster walks further into the trace before hitting the
cap, and every turn after the first drifts earlier. Two runs can therefore share
a seed and still not share a request set.

Three things are reported, and they fail independently:

  set      which requests were served. A few percent of difference is the
           truncation boundary; a large one means the runs are not comparable.
  order    the sequence they were sent in. Near-zero agreement is normal and
           mostly harmless: with hundreds of concurrent session coroutines the
           log order reflects which coroutine won the event loop, not the
           schedule.
  timing   when each request was dispatched. This is the one that matters. If
           the same request goes out at very different times, the runs faced
           different pool states, and a per-request label from one does not
           transfer to the other.

Early requests (before the closed loop has accumulated drift) are reported
separately, since they are the part that genuinely is identical.
"""
import re
import sys
from pathlib import Path

REQ_RE = re.compile(r"^\[req\] idx=(\d+) ts=(\d+)ms in=(\d+) out=(\d+)")

# Before the feedback loop has had time to diverge, dispatch times should match
# exactly. Past it, differences are expected rather than suspicious.
EARLY_MS = 60_000


def parse(path):
    rows = []
    for line in Path(path).open(errors="replace"):
        m = REQ_RE.match(line)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return rows


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    runs = {Path(p).stem.replace("run-", ""): parse(p) for p in sys.argv[1:]}
    base_name = list(runs)[0]
    base = runs[base_name]
    b_ts = {i: t for i, t, _ in base}
    b_order = [i for i, _, _ in base]

    print(f"baseline = {base_name}: {len(base):,} requests\n")
    print(f"{'run':<26} {'共有':>7} {'独有':>6} {'集合重合':>9} "
          f"{'顺序相同位':>11} {'早期时刻相同':>13} {'时刻中位差':>11}")
    for name, rows in runs.items():
        ts = {i: t for i, t, _ in rows}
        order = [i for i, _, _ in rows]
        common = sorted(set(ts) & set(b_ts))
        same_pos = sum(1 for a, b in zip(order, b_order) if a == b)
        diffs = sorted(abs(ts[i] - b_ts[i]) for i in common)
        early = [i for i in common if b_ts[i] < EARLY_MS]
        early_same = sum(1 for i in early if ts[i] == b_ts[i])
        print(f"{name:<26} {len(common):>7,} {len(set(ts) - set(b_ts)):>6,} "
              f"{100*len(common)/max(len(b_ts),1):>8.1f}% "
              f"{100*same_pos/max(len(b_order),1):>10.1f}% "
              f"{100*early_same/max(len(early),1):>12.1f}% "
              f"{diffs[len(diffs)//2] if diffs else 0:>9,}ms")

    print(f"\n（早期 = 发送时刻 < {EARLY_MS//1000}s，此时闭环反馈还没积累出偏差）\n")
    print(f"{'run':<26} {'总输入 token':>14} {'平均 prompt':>13} "
          f"{'跑进 trace 多深':>16} {'span':>9}")
    for name, rows in runs.items():
        tot = sum(r[2] for r in rows)
        print(f"{name:<26} {tot/1e6:>12.2f}M {tot/max(len(rows),1):>13,.0f} "
              f"{max(r[0] for r in rows):>16,} "
              f"{rows[-1][1]/1000:>8,.0f}s")
    print("\n总 token 相近 => 负载总量可比，尽管具体请求集合和时刻不同。")


if __name__ == "__main__":
    main()
