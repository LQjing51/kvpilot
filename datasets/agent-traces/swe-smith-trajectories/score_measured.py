#!/usr/bin/env python3
"""Score the dram-safe predictors against what a run actually measured.

    python3 score_measured.py --measured req_measured_safe.json \
                              --ratio req_ratio.json [--trace swe_trace.json]

Three columns, all judged on the same set of trace lines:

  theory     the cost model's verdict for that turn, read from req_ratio.json.
             It uses the *measured* tool response, so it is the ceiling the
             prediction could reach if it guessed response length perfectly.
  predicted  the same cost model but with the response length the online judge
             would have substituted from the lookup table (tool_resp_len.json),
             which is what actually runs in serving.
  majority   always answer with whichever verdict is more common. The floor.

The gap between theory and predicted is the cost of not knowing the response
length; the gap between theory and 100% is the cost model's own error.
"""
import argparse
import json
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../vllm-dev/vllm/v1/core"
    ),
)
from kvpilot_predict import derive_constants, is_dram_safe  # noqa: E402


def load_resp_table(path):
    with open(path) as fh:
        raw = json.load(fh)
    return raw.get("_default", 0), {
        k: v["resp_len"] for k, v in raw.get("labels", {}).items()
    }


def score(name, pairs, out):
    """pairs: list of (truth, pred). Same five columns as validate_premises.

    Reports balanced accuracy too -- the mean of the two per-class recalls.
    When one verdict dominates the truth, plain accuracy is maximized by
    answering that verdict everywhere, which predicts nothing while scoring
    well; balanced accuracy pins any such constant answer at 50%.
    """
    n = len(pairs)
    if not n:
        out.append(f"{name:22s} (no rows)")
        return 0.0
    hit = sum(t == p for t, p in pairs)
    fp = sum(p and not t for t, p in pairs)
    fn = sum(t and not p for t, p in pairs)
    truth = sum(t for t, _ in pairs)
    pred = sum(p for _, p in pairs)
    n_safe = truth
    n_unsafe = n - truth
    tp = sum(t and p for t, p in pairs)
    tn = sum((not t) and (not p) for t, p in pairs)
    bal = 50.0 * ((tp / n_safe if n_safe else 0) +
                  (tn / n_unsafe if n_unsafe else 0))
    out.append(
        f"{name:22s} {n:>7,d} {100*truth/n:>9.1f}% {100*pred/n:>9.1f}% "
        f"{100*fp/n:>7.1f}% {100*fn/n:>7.1f}% {100*hit/n:>8.1f}% {bal:>10.1f}%"
    )
    return 100.0 * hit / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", required=True)
    ap.add_argument("--ratio", required=True)
    ap.add_argument("--resp-len", default=None,
                    help="tool_resp_len.json; enables the 'predicted' column")
    ap.add_argument("--machine", default="h20")
    ap.add_argument("--compute-scale", type=float, default=1.0)
    ap.add_argument("--dram-scale", type=float, default=1.0)
    ap.add_argument("--output-file", default=None)
    args = ap.parse_args()

    measured = json.load(open(args.measured))["requests"]
    ratio = json.load(open(args.ratio))["requests"]
    k = derive_constants(args.machine, args.compute_scale, args.dram_scale)

    shared = sorted(set(measured) & set(ratio), key=int)
    out = [
        f"机器 {args.machine} 算力x{args.compute_scale} DRAMx{args.dram_scale}",
        f"常数 a={k['a_us']:.4e} b={k['b_us']:.4f} c_dram={k['c_dram_us']:.4f} us/token",
        f"实测表 {len(measured):,} 行 / ratio 表 {len(ratio):,} 行 / 交集 {len(shared):,} 行",
        "",
        f"{'预测方式':22s} {'轮数':>7s} {'真实safe':>10s} {'预测safe':>10s} "
        f"{'假阳':>8s} {'假阴':>8s} {'正确率':>9s} {'平衡':>11s}",
    ]

    truths = [bool(measured[x]["dram_safe"]) for x in shared]

    # Theory: recompute rather than reading ratio["dram_safe"], so --compute-scale
    # and --dram-scale can be swept without regenerating the ratio table.
    theory = [
        is_dram_safe(ratio[x]["prefix"], ratio[x]["append"], k) for x in shared
    ]
    score("理论(实测resp)", list(zip(truths, theory)), out)

    if args.resp_len:
        default, labels = load_resp_table(args.resp_len)
        pred = []
        for x in shared:
            r = ratio[x]
            est = labels.get(r["label"], default)
            pred.append(is_dram_safe(r["prefix"], r["decode_len"] + est, k))
        score("按类型预测resp", list(zip(truths, pred)), out)

    n_safe = sum(truths)
    always = n_safe >= len(truths) - n_safe
    score("全猜多数类", [(t, always) for t in truths], out)

    meta = json.load(open(args.measured))["_meta"]
    out += [
        "",
        f"实测 dram_safe 占比 {100*n_safe/max(len(truths),1):.1f}%"
        f"（判据：{meta.get('definition', '?')}，"
        f"阈值 {meta.get('max_ms_per_1k', meta.get('stall_pct_threshold'))}）",
    ]
    text = "\n".join(out)
    print(text)
    if args.output_file:
        with open(args.output_file, "w") as fh:
            fh.write(text + "\n")
        print(f"\n-> {args.output_file}")


if __name__ == "__main__":
    main()
