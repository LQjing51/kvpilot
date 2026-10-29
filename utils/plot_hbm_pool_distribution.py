#!/usr/bin/env python3
"""HBM block placement: what LRU orders by, against what KVPilot orders by.

Reads an HBM pool census (hbm-pool-*.jsonl, written by the serving workers under
$GET_HBM_POOL_DISTRIBUTION) and the ratio table validate_premises.py writes for
the replayed trace, and bins the blocks onto two axes:

    x  how long the block has been idle -- the only thing LRU orders by
    y  log2 of its load/compute ratio -- what KVPilot orders by

A cell's value is how many blocks fell in it, so the plot is a census, not a
sample. The two axes carve it into four quadrants, and the two off-diagonal ones
are the whole argument:

    upper left   old, and must stay in HBM -> LRU evicts these first, and the
                 GPU then stalls waiting for the reload
    lower right  fresh, and safe in DRAM   -> LRU keeps these, occupying HBM
                 that buys nothing

Every request has a real ratio: `append` is `decode_len + tool_resp` throughout,
and a terminal turn simply measures `tool_resp = 0`, so there is no session-end
case to place by hand. That definition is shared with the tables in
validate_premises.py, which is what keeps this y axis and their verdict talking
about the same thing.

Usage:
  python3 plot_hbm_pool_distribution.py --census dynamo-log/run-logs/hbm-pool-*.jsonl \\
      --ratios datasets/agent-traces/swe-smith-trajectories/req_ratio.json
"""

import argparse
import glob
import json
import math
import os
from collections import Counter

# Block states, as written by kvpilot_hbm_snapshot.
_IN_USE, _CACHED_FREE = 0, 1

# Idle-time bin edges in seconds. Logarithmic because idle time spans from a
# single scheduling step to a whole run, and because LRU cares about relative
# age, not absolute differences.
IDLE_EDGES_S = [0.01, 0.05, 0.2, 1.0, 5.0, 20.0, 60.0, 300.0]

# log2(ratio) bin edges. Symmetric about 0, which is ratio == 1 -- the dram-safe
# boundary itself. Blocks past either end are clamped into the edge bin rather
# than dropped, so the census stays whole.
Y_LIM = 4.0
Y_STEP = 0.5


def idle_bin(idle_s):
    for i, edge in enumerate(IDLE_EDGES_S):
        if idle_s < edge:
            return i
    return len(IDLE_EDGES_S)


def idle_labels():
    def fmt(v):
        if v < 1:
            return f"{int(v * 1000)}ms"
        return f"{v:g}s" if v < 60 else f"{int(v / 60)}m"

    out = [f"<{fmt(IDLE_EDGES_S[0])}"]
    out += [
        f"{fmt(IDLE_EDGES_S[i])}-{fmt(IDLE_EDGES_S[i + 1])}"
        for i in range(len(IDLE_EDGES_S) - 1)
    ]
    out.append(f">{fmt(IDLE_EDGES_S[-1])}")
    return out


def y_bins():
    n = int(2 * Y_LIM / Y_STEP)
    return [(-Y_LIM + i * Y_STEP, -Y_LIM + (i + 1) * Y_STEP) for i in range(n)]


def y_bin(log_ratio):
    idx = int((log_ratio + Y_LIM) / Y_STEP)
    return max(0, min(int(2 * Y_LIM / Y_STEP) - 1, idx))


def load_ratios(path):
    with open(path) as fh:
        payload = json.load(fh)
    return payload.get("requests", {}), payload.get("_meta", {})


def load_census(patterns, last_n):
    """Census lines from every matching file, newest `last_n` per file.

    Defaults to the final line of each: the plot is a snapshot of the pool at
    steady state, and averaging over startup would blend an empty pool into it.
    """
    lines = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as fh:
                records = [json.loads(ln) for ln in fh if ln.strip()]
            if not records:
                continue
            tag = os.path.basename(path)
            for record in records[-last_n:]:
                lines.append((tag, record))
    return lines


def build_grid(census, ratios, include_in_use):
    """Count blocks into (idle bin, y bin), plus what could not be placed."""
    grid = Counter()
    stats = Counter()
    for _tag, record in census:
        stats["blank"] += record.get("num_blank", 0)
        for row in record["blocks"]:
            _block_id, state, idle_ns, req_line, _tier = row
            stats["rows"] += 1
            if state == _IN_USE and not include_in_use:
                stats["in_use_skipped"] += 1
                continue
            if idle_ns < 0:
                # Never freed or touched since startup: no age to place it at.
                stats["no_timestamp"] += 1
                continue
            entry = ratios.get(str(req_line))
            if entry is None:
                # The census saw a request the table does not describe: warmup
                # traffic, or a table built from a different trace.
                stats["unmatched"] += 1
                continue
            ratio = entry.get("ratio")
            if not ratio or ratio <= 0:
                # No compute to hide behind, so the quotient says nothing. Does
                # not arise while append is decode_len + tool_resp, but a table
                # from an older or hand-edited run could still carry it.
                stats["no_ratio"] += 1
                continue
            grid[(idle_bin(idle_ns / 1e9), y_bin(math.log2(ratio)))] += 1
            stats["placed"] += 1
    return grid, stats


def quadrants(grid):
    """Block counts per quadrant, split at ratio == 1 and the middle idle bin.

    Bin indices run newest-first (bin 0 is the freshest), so a high index is an
    old block -- the opposite of how the grid is drawn. Keeping the comparison
    here tied to the index rather than to the drawn position is what stops the
    two from disagreeing.
    """
    mid_x = (len(IDLE_EDGES_S) + 1) / 2.0
    mid_y = len(y_bins()) / 2.0
    out = Counter()
    for (xi, yi), n in grid.items():
        old = xi >= mid_x
        must_stay = yi >= mid_y
        if old and must_stay:
            out["lru_evicts_needed"] += n
        elif old:
            out["agree_evict"] += n
        elif must_stay:
            out["agree_keep"] += n
        else:
            out["lru_keeps_useless"] += n
    return out


def render_text(grid, stats, meta, out):
    """ASCII heatmap, for reading the result without a plotting library."""
    xs = idle_labels()
    ys = y_bins()
    peak = max(grid.values()) if grid else 0
    ramp = " .:-=+*#%@"

    out.append("=" * 100)
    out.append("HBM block placement: idle time (LRU 的依据) x log2 载入/计算比 (KVPilot 的依据)")
    out.append("=" * 100)
    out.append(
        f"trace={meta.get('trace', '?')} machine={meta.get('machine', '?')} "
        f"placed={stats['placed']:,d} blocks, peak cell={peak:,d}"
    )
    out.append("")
    # Drawn oldest-on-the-left, so LRU's eviction order reads left to right and
    # the quadrant a cell sits in is visible from its position. Bin indices run
    # the other way (0 = freshest), hence the reversal here.
    order = list(range(len(xs) - 1, -1, -1))
    for yi in range(len(ys) - 1, -1, -1):
        lo, hi = ys[yi]
        row = "".join(
            ramp[min(len(ramp) - 1,
                     int(grid.get((xi, yi), 0) / peak * (len(ramp) - 1)))]
            if peak
            else " "
            for xi in order
        )
        marker = " <- ratio=1" if lo <= 0 < hi else ""
        out.append(f"{lo:>+5.1f} |{row}|{marker}")
    out.append(f"      +{'-' * len(xs)}+")
    out.append(f"       {''.join(str(i % 10) for i in order)}")
    out.append("       老 <-- 上次访问距今 --> 新   (LRU 从左往右淘汰)")
    out.append("")
    for i in order:
        out.append(f"       {i % 10} = {xs[i]}")

    out.append("")
    q = quadrants(grid)
    total = sum(q.values()) or 1
    out.append("象限占比（左 = 更老，上 = 载入掩不住必须留 HBM）")
    for key, desc in (
        ("lru_evicts_needed", "老 & 必须留 HBM  -> LRU 先淘汰它，GPU 会空等"),
        ("lru_keeps_useless", "新 & 放 DRAM 无损 -> LRU 留着它，白占 HBM"),
        ("agree_keep", "新 & 必须留 HBM  -> 两者一致保留"),
        ("agree_evict", "老 & 放 DRAM 无损 -> 两者一致淘汰"),
    ):
        n = q.get(key, 0)
        out.append(f"  {desc:52s} {n:>9,d}  {100.0 * n / total:5.1f}%")

    out.append("")
    out.append("未计入的块")
    for key, desc in (
        ("blank", "空白块（无 KV，恒最先淘汰）"),
        ("in_use_skipped", "正被请求占用（此刻不参与淘汰）"),
        ("no_timestamp", "启动后从未被 free/touch 过"),
        ("unmatched", "req_line 在 ratio 表里查不到"),
        ("no_ratio", "ratio 缺失或 <= 0（取不了 log）"),
    ):
        if stats.get(key):
            out.append(f"  {desc:52s} {stats[key]:>9,d}")
    if stats.get("unmatched", 0) > stats.get("placed", 0):
        out.append("")
        out.append(
            "  warning: 查不到的比查到的还多。ratio 表可能来自另一个 trace，"
            "或压测用的输入文件与建表用的不是同一个"
        )
    return out


def write_csv(grid, path):
    """The grid as tidy rows, for plotting elsewhere."""
    xs = idle_labels()
    ys = y_bins()
    with open(path, "w") as fh:
        fh.write("idle_bin,idle_label,log2_ratio_lo,log2_ratio_hi,blocks\n")
        for xi, xlabel in enumerate(xs):
            for yi, (lo, hi) in enumerate(ys):
                fh.write(f"{xi},{xlabel},{lo},{hi},{grid.get((xi, yi), 0)}\n")


def write_pdf(grid, stats, meta, path):
    """The census as a 2-D histogram: LRU's axis against the verdict's.

    Imported here rather than at module scope so the text and CSV outputs keep
    working on a machine without matplotlib, which is the case locally.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    xs, ys = idle_labels(), y_bins()
    counts = np.array(
        [[grid.get((xi, yi), 0) for xi in range(len(xs))] for yi in range(len(ys))],
        dtype=float,
    )
    q = quadrants(grid)
    total = sum(q.values()) or 1

    fig, ax = plt.subplots(figsize=(11, 7.5))
    # Log colour scale: the corner bins run three orders of magnitude above the
    # sparse ones, and a linear ramp renders everything but the mode as white.
    shown = np.ma.masked_where(counts == 0, counts)
    mesh = ax.pcolormesh(
        np.arange(len(xs) + 1),
        np.arange(len(ys) + 1),
        shown,
        cmap="viridis",
        norm=LogNorm(vmin=max(shown.min(), 1) if shown.count() else 1,
                     vmax=max(counts.max(), 1)),
    )
    fig.colorbar(mesh, ax=ax, label="blocks in bin (log scale)")

    ax.set_xticks(np.arange(len(xs)) + 0.5)
    ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(ys) + 1))
    ax.set_yticklabels([f"{lo:+.1f}" for lo, _ in ys] + [f"{ys[-1][1]:+.1f}"],
                       fontsize=8)
    ax.set_xlabel("block idle time  (what LRU orders by)  ->  newer")
    ax.set_ylabel("log2(load / compute)  (what KVPilot orders by)")

    # ratio == 1 is the dram-safe boundary: above it the load cannot hide.
    y0 = next((i for i, (lo, hi) in enumerate(ys) if lo <= 0.0 < hi), None)
    if y0 is not None:
        yline = y0 + (0.0 - ys[y0][0]) / (ys[y0][1] - ys[y0][0])
        ax.axhline(yline, color="white", lw=1.4, ls="--")
        # Left-anchored: at the right edge the label runs off the axes.
        ax.text(0.15, yline + 0.12, "load == compute", color="white",
                fontsize=8, ha="left", va="bottom")

    pct = lambda k: 100.0 * q.get(k, 0) / total  # noqa: E731
    ax.set_title(
        f"HBM pool census: {int(counts.sum()):,} cached-free blocks\n"
        f"upper-left (LRU evicts, GPU then stalls) {pct('lru_evicts_needed'):.1f}%   |   "
        f"lower-right (LRU keeps, buys nothing) {pct('lru_keeps_useless'):.1f}%",
        fontsize=11,
    )
    sub = meta.get("trace") or ""
    if sub:
        fig.text(0.01, 0.01, f"trace={sub}  machine={meta.get('machine', '?')}",
                 fontsize=7, color="#666")
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--census",
        nargs="+",
        required=True,
        help="hbm-pool-*.jsonl 路径（可给多个或用通配）",
    )
    ap.add_argument("--ratios", required=True,
                    help="validate_premises.py 写的 req_ratio.json")
    ap.add_argument(
        "--last-n",
        type=int,
        default=1,
        help="每个文件取最后 N 条快照（默认 1，即稳态那一张）",
    )
    ap.add_argument(
        "--include-in-use",
        action="store_true",
        help="把正被请求占用的块也计入（默认只看淘汰候选）",
    )
    ap.add_argument("--csv-output", help="同时把网格写成 CSV")
    ap.add_argument("--pdf-output", help="二维直方图 PDF 的落点")
    ap.add_argument("--output-file", default="hbm_pool_distribution.log")
    args = ap.parse_args()

    ratios, meta = load_ratios(args.ratios)
    census = load_census(args.census, args.last_n)
    if not census:
        raise SystemExit(f"no census lines matched {args.census}")

    grid, stats = build_grid(census, ratios, args.include_in_use)
    out = []
    render_text(grid, stats, meta, out)
    text = "\n".join(out)
    print(text)
    with open(args.output_file, "w") as fh:
        fh.write(text + "\n")
    if args.csv_output:
        write_csv(grid, args.csv_output)
        print(f"\nwrote {args.csv_output}")
    if args.pdf_output:
        write_pdf(grid, stats, meta, args.pdf_output)


if __name__ == "__main__":
    main()
