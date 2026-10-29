#!/usr/bin/env python3
"""Is a measured dram-safe label worth predicting at all?

Four checks that belong together, because each one rules out a different reason
a predictor might fail and they are only conclusive as a set. Run them before
tuning constants or percentiles: if the label carries no signal, no amount of
sweeping will find one, and the sweep will instead land on whichever threshold
happens to match the majority class.

    python3 label_diagnostics.py signal    <measured.json> <req_ratio.json>
    python3 label_diagnostics.py overhead  <measured.json>
    python3 label_diagnostics.py noise     <log_dir_a> <log_dir_b>
    python3 label_diagnostics.py denominator <log_dir> [<log_dir> ...]

What each answers, and what a bad answer means:

  signal       Does the label track the cost model's ratio? Prints the measured
               safe-rate per decile of that ratio. It should fall monotonically;
               if it rises, the label is measuring something the model orders the
               other way round, and the fit is not merely imprecise but inverted.

  overhead     Is a per-request fixed cost dominating? Fits
               `stall = fixed + slope * dram_tokens`. When the fixed term is
               large, ms-per-1k-tokens is really asking "did this request load
               enough to amortize the setup", which is a property of the
               transfer path rather than of whether compute hid the load.

  noise        Does the label agree with itself across runs? A predictor cannot
               beat the label's own reproducibility, so this is the hard ceiling
               that no feature engineering moves.

  denominator  Is `e2e` a fair denominator? Prints apparent compute per 1k
               recomputed tokens. Token counts being equal, that rate should be
               too; a large spread means `e2e` is mostly queueing, and any ratio
               against it grades congestion rather than overlap.
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LINE_RE = re.compile(
    r"KVPilot prefill req=(?P<req>\S+?)\s+(?:line=(?P<line>-?\d+)\s+)?"
    r"prompt=(?P<prompt>\d+)\s+new=(?P<new>\d+)\s+hit=(?P<hit>\d+)\s+"
    r"dram=(?P<dram>\d+)\s+e2e=(?P<e2e>[\d.]+)ms\s+stall=(?P<stall>[\d.]+)ms\s+"
    r"\((?P<pct>[\d.]+)%\)\s+waits=(?P<waits>\d+)\s+steps=(?P<steps>\d+)"
)


def _rows(log_dir):
    out = []
    for path in sorted(Path(log_dir).glob("prefill-*.log")):
        for text in path.open(errors="replace"):
            m = LINE_RE.search(text)
            if m:
                g = m.groupdict()
                out.append({
                    "line": int(g["line"] or -1),
                    "dram": int(g["dram"]),
                    "new": int(g["new"]),
                    "e2e": float(g["e2e"]),
                    "stall": float(g["stall"]),
                })
    return out


def _by_line(log_dir):
    """Per trace line, the mean stall and DRAM tokens over its loading attempts."""
    grouped = defaultdict(list)
    for r in _rows(log_dir):
        if r["line"] > 0 and r["dram"] > 0:
            grouped[r["line"]].append(r)
    return {
        line: (statistics.fmean(r["stall"] for r in v),
               statistics.fmean(r["dram"] for r in v))
        for line, v in grouped.items()
    }


def _measured_value(entry):
    """The per-token figure, preferring the fixed-cost-adjusted one when present."""
    return entry.get("adj_ms_per_1k", entry.get("ms_per_1k", 0.0))


def cmd_signal(measured_path, ratio_path):
    measured = json.load(open(measured_path))["requests"]
    ratio = json.load(open(ratio_path))["requests"]
    rows = [
        (ratio[k]["ratio"], bool(measured[k]["dram_safe"]),
         _measured_value(measured[k]), ratio[k]["prefix"])
        for k in sorted(set(measured) & set(ratio), key=int)
        if ratio[k].get("ratio") is not None
    ]
    rows.sort(key=lambda r: r[0])
    n = len(rows)
    print(f"{n:,} rows with a model ratio\n")
    print(f"{'decile':>7s} {'ratio range':>22s} {'n':>6s} {'measured safe':>14s} "
          f"{'median ms/1k':>13s} {'median prefix':>14s}")
    for d in range(10):
        chunk = rows[n * d // 10:n * (d + 1) // 10]
        if not chunk:
            continue
        safe = sum(1 for r in chunk if r[1])
        mid = sorted(r[2] for r in chunk)[len(chunk) // 2]
        pmid = sorted(r[3] for r in chunk)[len(chunk) // 2]
        print(f"{d+1:>7d} {chunk[0][0]:>10.3f}-{chunk[-1][0]:<11.3f} "
              f"{len(chunk):>6d} {100*safe/len(chunk):>13.1f}% {mid:>13.2f} "
              f"{pmid:>14,d}")
    print("\n安全率应随 ratio 单调下降；上升说明标签与模型的排序相反。")


def cmd_overhead(measured_path):
    measured = json.load(open(measured_path))["requests"]
    pts = [(v["dram_tokens"], v["stall_ms"]) for v in measured.values()
           if v["dram_tokens"] > 0]
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    fixed = (sy - slope * sx) / n
    print(f"n={n:,}")
    print(f"  stall_ms ~ {fixed:.2f} ms fixed + {1000*slope:.3f} ms per 1k tokens")
    print(f"  两项在 {fixed/slope:,.0f} 个 DRAM token 处相等\n")
    print(f"{'dram tokens':>20s} {'n':>6s} {'median ms/1k':>13s} "
          f"{'扣固定项后':>14s}")
    pts.sort()
    for d in range(6):
        chunk = pts[n * d // 6:n * (d + 1) // 6]
        if not chunk:
            continue
        raw = sorted(1000 * s / t for t, s in chunk)
        adj = sorted(1000 * max(s - fixed, 0) / t for t, s in chunk)
        print(f"{chunk[0][0]:>9,}-{chunk[-1][0]:<10,} {len(chunk):>6d} "
              f"{raw[len(raw)//2]:>13.2f} {adj[len(adj)//2]:>14.2f}")
    print("\n扣掉固定项后各档应齐平；不齐平说明还有别的尺寸相关项。")


def cmd_noise(dir_a, dir_b, fixed=11.03, thresh=2.0):
    a, b = _by_line(dir_a), _by_line(dir_b)
    shared = sorted(set(a) & set(b))
    label = lambda s, d: (1000.0 * max(s - fixed, 0.0) / d) <= thresh  # noqa: E731
    agree = sum(1 for k in shared if label(*a[k]) == label(*b[k]))
    xs = [1000.0 * max(a[k][0] - fixed, 0) / a[k][1] for k in shared]
    ys = [1000.0 * max(b[k][0] - fixed, 0) / b[k][1] for k in shared]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    print(f"{len(shared):,} 条共有行  (fixed={fixed} ms, 阈值={thresh} ms/1k)")
    print(f"  verdict 一致率 = {100*agree/max(len(shared),1):.1f}%")
    print(f"  连续值 Pearson r = {cov/(sx*sy) if sx and sy else 0:.3f}")
    both = sorted(xs + ys)
    near = sum(1 for v in both if thresh - 0.5 <= v <= thresh + 0.5)
    print(f"  {100*near/len(both):.1f}% 的测量值落在阈值 ±0.5 ms/1k 内"
          f"（越集中，阈值越容易被噪声翻转）")
    print("\n一致率就是任何预测器的天花板：标签与自己都不一致的部分无法预测。")


def cmd_denominator(*log_dirs):
    print(f"{'run':<26} {'new/req':>9} {'e2e/req':>10} {'compute/1k new':>16}")
    for d in log_dirs:
        rows = _rows(d)
        if not rows:
            print(f"{Path(d).name:<26} (no records)")
            continue
        nz = [r for r in rows if r["new"] > 0]
        compute = sum(r["e2e"] - r["stall"] for r in nz)
        print(f"{Path(d).name:<26} "
              f"{statistics.fmean(r['new'] for r in rows):>9,.0f} "
              f"{statistics.fmean(r['e2e'] for r in rows):>9,.0f}ms "
              f"{1000*compute/sum(r['new'] for r in nz):>14,.1f}ms")
    print("\ntoken 数相近时最后一列也应相近；差几倍说明 e2e 主要是排队，"
          "拿它当分母量的是拥塞不是掩盖。")


COMMANDS = {
    "signal": cmd_signal,
    "overhead": cmd_overhead,
    "noise": cmd_noise,
    "denominator": cmd_denominator,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    COMMANDS[sys.argv[1]](*sys.argv[2:])
