#!/usr/bin/env python3
"""Sweep the cost-model constants to fit the measured dram-safe verdicts.

    python3 fit_constants.py --measured req_measured_safe.json \
                             --ratio req_ratio.json --resp-len tool_resp_len.json

demo_test.md asks for >=80% prediction accuracy, and prescribes the order: fit
the theoretical verdict first by scaling DRAM bandwidth and FLOPS, then tune the
predicted one by choosing a response-length percentile. This script does the
first half -- it searches (dram_scale, compute_scale) for the best agreement
between the cost model and what the run measured.

Only the ratio of the two scales matters to the verdict: `t_load < t_comp`
compares c*prefix against (a*..+b)*append, so scaling compute by x and DRAM by x
leaves every verdict unchanged. The sweep therefore reports the ratio, and a fit
is a statement about how much slower DRAM is relative to compute than the
datasheet figures imply -- which is what docs/pitfalls.md already records
measuring (c came out ~600x the model's prediction).
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


def accuracy(rows, k, est=None):
    """Agreement with measured truth; `est` overrides append's response part.

    Returns `(acc, fp, fn, pred_safe, balanced)`. `balanced` is the mean of the
    two per-class recalls and is the number to judge a fit by: when one verdict
    dominates the truth, plain accuracy is maximized by answering that verdict
    everywhere, which scores well while predicting nothing. Balanced accuracy
    sits at 50% for any such constant answer.
    """
    hit = fp = fn = pred_safe = 0
    tp = tn = n_safe = n_unsafe = 0
    for truth, prefix, decode_len, append, label in rows:
        a = append if est is None else decode_len + est(label)
        p = is_dram_safe(prefix, a, k)
        hit += p == truth
        fp += p and not truth
        fn += truth and not p
        pred_safe += p
        if truth:
            n_safe += 1
            tp += p
        else:
            n_unsafe += 1
            tn += not p
    n = len(rows)
    recall_safe = tp / n_safe if n_safe else 0.0
    recall_unsafe = tn / n_unsafe if n_unsafe else 0.0
    return (100.0 * hit / n, 100.0 * fp / n, 100.0 * fn / n,
            100.0 * pred_safe / n, 50.0 * (recall_safe + recall_unsafe))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", required=True)
    ap.add_argument("--ratio", required=True)
    ap.add_argument("--resp-len", default=None)
    ap.add_argument("--machine", default="h20")
    ap.add_argument("--output-file", default=None)
    args = ap.parse_args()

    measured = json.load(open(args.measured))["requests"]
    ratio = json.load(open(args.ratio))["requests"]
    shared = sorted(set(measured) & set(ratio), key=int)
    rows = [
        (
            bool(measured[x]["dram_safe"]),
            ratio[x]["prefix"],
            ratio[x]["decode_len"],
            ratio[x]["append"],
            ratio[x]["label"],
        )
        for x in shared
    ]

    out = [
        f"拟合样本 {len(rows):,} 行（实测表 ∩ ratio 表）",
        f"实测 safe 占比 {100*sum(r[0] for r in rows)/len(rows):.1f}%",
        "",
        "只有 compute/dram 的比值影响判据（两边同乘不改变不等式），故按比值扫描：",
        "",
        f"{'dram_scale':>11s} {'compute_scale':>14s} {'ratio':>9s} "
        f"{'预测safe':>9s} {'假阳':>7s} {'假阴':>7s} {'正确率':>8s} {'平衡正确率':>11s}",
    ]

    # Both directions. Scaling DRAM *up* models a load path faster than the
    # datasheet -- which is what the H2D shared-memory route is: it skips the
    # host round trip the model prices in, so its effective bandwidth is higher
    # than the DRAM figure. Ranked by balanced accuracy, because plain accuracy
    # peaks wherever every turn is called by the majority verdict, predicting
    # nothing.
    scales = (0.002, 0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0,
              100.0, 200.0, 500.0, 1000.0)
    best = None
    for ds in scales:
        k = derive_constants(args.machine, 1.0, ds)
        acc, fp, fn, ps, bal = accuracy(rows, k)
        out.append(
            f"{ds:>11.4f} {1.0:>14.2f} {1/ds:>9.3f} "
            f"{ps:>8.1f}% {fp:>6.1f}% {fn:>6.1f}% {acc:>7.1f}% {bal:>10.1f}%"
        )
        if best is None or bal > best[5]:
            best = (acc, ds, fp, fn, ps, bal)

    out += [
        "",
        f"最佳（按平衡正确率）：dram_scale={best[1]:.4f}"
        f"（DRAM 相对 datasheet {'快' if best[1] > 1 else '慢'} "
        f"{best[1] if best[1] > 1 else 1/best[1]:.0f}x）"
        f" 正确率 {best[0]:.1f}% 平衡 {best[5]:.1f}%"
        f" 假阳 {best[2]:.1f}% 假阴 {best[3]:.1f}%",
    ]

    if args.resp_len:
        raw = json.load(open(args.resp_len))
        default = raw.get("_default", 0)
        labels = {k2: v["resp_len"] for k2, v in raw.get("labels", {}).items()}
        k = derive_constants(args.machine, 1.0, best[1])
        acc, fp, fn, ps, bal = accuracy(
            rows, k, est=lambda lb: labels.get(lb, default)
        )
        out += [
            "",
            f"在最佳常数下，用查表预测 resp 长度（当前表 "
            f"_percentile={raw.get('_percentile')}）：",
            f"  正确率 {acc:.1f}% 平衡 {bal:.1f}% 假阳 {fp:.1f}% "
            f"假阴 {fn:.1f}% 预测safe {ps:.1f}%",
        ]

    text = "\n".join(out)
    print(text)
    if args.output_file:
        with open(args.output_file, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
