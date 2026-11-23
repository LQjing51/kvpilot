"""Tool-response length distributions for SWE and TraceLab (bucket width 10).

Within a session, consecutive requests satisfy

    input[j+1] - input[j] = decode_len[j] + tool_resp[j]

so tool_resp = Δinput - output_length. The label is the joined tool_calls of
request j (same `+` join as kvpilot_signal's online predictor). Empty
tool_calls are kept under the label \"\" (no tool).

For each label, emit a probability vector over buckets
    [0, 10), [10, 20), …, [max-10, max)
where max is ceil(max_resp / 10) * 10 for that label (at least one bucket).

One file per dataset, resp_length_dist_<dataset>.json (label -> hist): the
file KVPILOT_TOOL_RESP_DISTRIBUTION_PATH points at.

    python3 resp_length_distribution.py              # both datasets
    python3 resp_length_distribution.py --dataset swe
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUCKET = 10
TRACES = {
    "swe": ROOT / "datasets/agent-traces/swe-smith-trajectories/swe_mooncake.jsonl",
    "tracelab": ROOT / "datasets/agent-traces/tracelab/tracelab_mooncake.jsonl",
}


def label_of(tool_calls):
    """One string for the turn's call set; '' when there is no tool call."""
    if not tool_calls:
        return ""
    return "+".join(tool_calls)


def consecutive_tool_resps(trace_path):
    """Yield (label, tool_resp) for every consecutive pair in each session."""
    by_sess = defaultdict(list)
    with open(trace_path) as f:
        for line in f:
            r = json.loads(line)
            by_sess[r["session_id"]].append(r)
    for rows in by_sess.values():
        rows.sort(key=lambda r: r["round_index"])
        for a, b in zip(rows, rows[1:]):
            delta = b["input_length"] - a["input_length"]
            tool_resp = delta - a["output_length"]
            if tool_resp < 0:
                # Context compression / reset; drop the pair.
                continue
            yield label_of(a.get("tool_calls") or []), int(tool_resp)


def _bucket_edges(max_resp, width=BUCKET):
    top = max(width, ((max(max_resp, 0) + width - 1) // width) * width)
    return list(range(0, top + width, width))


def hist_for_label(resps, width=BUCKET):
    """Return sparse hist for one label: only non-zero buckets.

    `buckets` is [[lo, hi, prob], ...] covering [0, max_edge); sum(prob) = 1.
    """
    if not resps:
        return None
    max_resp = max(resps)
    edges = _bucket_edges(max_resp, width)
    counts = [0] * (len(edges) - 1)
    for x in resps:
        # Last bucket is closed on the right so max_resp lands inside.
        i = min(x // width, len(counts) - 1)
        counts[i] += 1
    n = len(resps)
    buckets = [[edges[i], edges[i + 1], counts[i] / n]
               for i in range(len(counts)) if counts[i]]
    return {
        "bucket_width": width,
        "buckets": buckets,
        "n": n,
        "max_resp": max_resp,
    }


def build_distribution(trace_path, width=BUCKET, min_n=1):
    """label -> hist dict. Labels with fewer than min_n samples are dropped."""
    by = defaultdict(list)
    for label, resp in consecutive_tool_resps(trace_path):
        by[label].append(resp)
    out = {}
    for label, resps in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(resps) < min_n:
            continue
        out[label] = hist_for_label(resps, width)
    return out


def build_all(datasets=None, width=BUCKET, min_n=1):
    datasets = list(datasets or TRACES)
    return {name: build_distribution(TRACES[name], width=width, min_n=min_n)
            for name in datasets}


def dist_path(dataset, out_dir=HERE):
    return Path(out_dir) / f"resp_length_dist_{dataset}.json"


def save(dist, path):
    Path(path).write_text(json.dumps(dist, indent=2) + "\n")


def load(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(TRACES), action="append", default=None)
    ap.add_argument("--bucket", type=int, default=BUCKET)
    ap.add_argument("--min-n", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--show", type=int, default=5, help="top labels to print per dataset")
    args = ap.parse_args()

    dist = build_all(args.dataset, width=args.bucket, min_n=args.min_n)
    for name, labels in dist.items():
        out = dist_path(name, args.out_dir)
        save(labels, out)
        n_pairs = sum(h["n"] for h in labels.values())
        print(f"{name}: {len(labels)} labels, {n_pairs} pairs -> {out}")
        top = sorted(labels.items(), key=lambda kv: -kv[1]["n"])[: args.show]
        for lab, h in top:
            nz = h["buckets"]
            head = ", ".join(f"[{lo},{hi})={p:.3f}" for lo, hi, p in nz[:6])
            more = f" … +{len(nz)-6} buckets" if len(nz) > 6 else ""
            print(f"  {lab!r:40s} n={h['n']:4d} max={h['max_resp']:5d}  {head}{more}")


if __name__ == "__main__":
    main()
