#!/usr/bin/env python3
"""Does the tool name predict KV placement in the TraceLab trace?

Same three tables as the SWE-smith `validate_premises.py`, on real Claude Code
sessions instead of SWE-smith trajectories:

Table 1  incoming length per call type: the tool response, or the user's
         next message when the round made no call.
Table 2  accuracy of predicting the dram-safe verdict from that type.
Table 3  the same verdict from one global median, ignoring the type — the
         control that says how much of table 2's score the type actually earned.
Summary  the three predictors side by side.

The cost model and `is_dram_safe` are imported from the SWE-smith directory
rather than restated: they decide P-side behaviour online, and a second copy of
the constants would drift without anyone noticing.

Unlike the SWE-smith path, append is measured rather than tokenized — TraceLab
records each round's newly_append_tokens directly — so nothing here needs a
tokenizer, and `--input-file` is the output of prepare_tracelab_trace.py.

Usage:
  python3 validate_tracelab.py [--input-file tracelab_turns.jsonl]
                               [--machine h20]
"""

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

# The shared predictor lives with the serving code it feeds.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../vllm-dev/vllm/v1/core"
    ),
)

from kvpilot_predict import (  # noqa: E402
    MACHINES,
    derive_constants,
    is_dram_safe,
    write_resp_len_table,
)


def load(path, max_turns=0, provider=None):
    """Read the per-round records and pair each round with its successor.

    The trace file holds one record per round, each carrying only its own
    numbers. The prediction question spans two rounds — "this round emits these
    tool calls; how long is the response that lands next round?" — so the pairing
    is done here, explicitly, where it can be seen.

    Every round participates, tool call or not. A round that made no call still
    has to be placed, and what arrives next is a user message rather than a tool
    response — measured at a median of ~1,000 tokens, so it is emphatically not
    zero. Both cases answer the same question, "how much lands in the next
    append", and the `null` label is the honest name for the one the predictor
    cannot see coming. Excluding those rounds would drop 13.7% of the trace.

    The label is every call this round made, joined by `+`, matching the
    SWE-smith path: the calls' responses all land in the same append, so
    `exec_command+exec_command+exec_command` is its own category rather than a
    variant of `exec_command`. A round with no calls is labelled `null`.
    """
    by_session = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if provider and r.get("provider") != provider:
                continue
            by_session[r["session_id"]].append(r)

    turns = []
    for rs in by_session.values():
        rs.sort(key=lambda r: r["round_index"])
        nxt = {r["round_index"]: r for r in rs}
        last = rs[-1]["round_index"]
        for r in rs:
            if r["prefix"] <= 0:
                continue          # nothing placed yet; the ratio is undefined
            follower = nxt.get(r["round_index"] + 1)
            if follower is None and r["round_index"] != last:
                continue          # gap in the session: no ground truth
            labels = r.get("labels") or []
            turns.append(
                {
                    # Whole call set as one category, `null` when there was none.
                    "label": "+".join(labels) if labels else "null",
                    "n_tools": len(labels),
                    "prefix": r["prefix"],
                    "decode_len": r["output"],
                    # This turn's own decode plus the response it provokes. The
                    # response arrives as the next round's append, which already
                    # includes this round's output, so the tool response alone is
                    # that append minus this output.
                    "next_append": follower["append"] if follower else 0,
                    "provider": r.get("provider"),
                    "session_id": r["session_id"],
                }
            )
            if max_turns and len(turns) >= max_turns:
                return turns
    return turns


def percentile(values, p):
    """The p-th percentile of an already-materialized list (p in [0, 1])."""
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p))]


def quantiles(values, extra=None):
    """Summary stats, plus the percentile the predictor is configured to use.

    `extra` is that percentile as a fraction; it lands under the "pred" key so
    table 2 can read one name regardless of which percentile was chosen.
    """
    values = sorted(values)
    n = len(values)
    pick = lambda p: values[min(n - 1, int(n * p))]  # noqa: E731
    p10, p90 = pick(0.10), pick(0.90)
    q = {
        "n": n,
        "p10": p10,
        "p50": pick(0.50),
        "p90": p90,
        "p90_over_p10": None if p90 == 0 else round(p90 / max(p10, 1), 2),
        "mean": round(statistics.fmean(values), 1),
    }
    q["pred"] = pick(extra) if extra is not None else q["p50"]
    return q


def tool_resp(turn):
    """The part of append the predictor cannot see when the call is emitted.

    The next round's append is this round's decode output plus whatever arrives
    to answer it, so that arrival alone is the append minus this decode. What
    arrives is a tool response when the round made calls, and the user's next
    message when it made none. Clamped at 0: a few rounds report an append
    smaller than the preceding decode because the agent's context was compacted
    in between.
    """
    return max(0, turn["next_append"] - turn["decode_len"])


def table1(turns, out, pct=0.50):
    """Incoming length by call type, over every turn of that type.

    The `null` row is not zero: those turns made no call, so what follows is the
    user typing again, which this trace measures like anything else. Only
    terminal turns contribute 0, and they belong in the distribution because the
    predictor sees only the call type and cannot tell a terminal turn from a
    continuing one.
    """
    by = defaultdict(list)
    for t in turns:
        by[t["label"]].append(tool_resp(t))

    out.append("=" * 100)
    out.append(f"（预测值取 p{pct*100:.0f}，可用 --predict-percentile 调整）")
    out.append("表 1  调用类型 -> tool resp / user input 长度（token）")
    out.append("=" * 100)
    out.append("append = 本轮 output + (tool resp / user input)，本表统计括号里那部分")
    out.append("  = 下一轮 append - 本轮 output")
    out.append("label=null 是本轮没发 tool call 的轮，其后到达的是【用户新消息】而非 "
               "tool 响应（实测 p50 约 1,000 token，不是 0）")
    out.append("终止轮没有下一轮，按 0 计入")
    out.append("一轮发多个 call 时，整个调用集合用 + 连接成独立类型（同 SWE-smith 口径）")
    out.append(
        f"{'调用类型':46s} {'轮数':>8s} {'p10':>8s} {'p50':>8s} "
        f"{'p90':>8s} {'mean':>10s} {'p90/p10':>9s} {'预测值':>9s}"
    )
    stats = {}
    for label, vals in sorted(by.items(), key=lambda kv: -len(kv[1])):
        q = quantiles(vals, pct)
        stats[label] = q
        spread = "n/a" if q["p90_over_p10"] is None else f"{q['p90_over_p10']:.1f}x"
        out.append(
            f"{label:46s} {q['n']:>8,d} {q['p10']:>8,d} {q['p50']:>8,d} "
            f"{q['p90']:>8,d} {q['mean']:>10,.1f} {spread:>9s} "
            f"{q['pred']:>9,d}"
        )

    flat = [v for vs in by.values() for v in vs]
    grand = statistics.fmean(flat)
    var = statistics.pvariance(flat)
    between = sum(
        len(vs) * (statistics.fmean(vs) - grand) ** 2 for vs in by.values()
    ) / len(flat)
    out.append("")
    out.append(
        f"类别数 {len(by):,}   组间方差/总方差 = {between / var if var else 0:.4f}"
    )
    return stats


VERDICT_HEADER = (
    f"{'调用类型':46s} {'轮数':>8s} {'真实safe':>10s} {'预测safe':>10s} "
    f"{'假阳':>8s} {'假阴':>8s} {'正确率':>9s}"
)


def tally_verdicts(turns, estimate, k):
    """Four-cell counts per label for one tool-response-length estimator.

    Ground truth uses the measured response; both sides use the real prefix and
    decode_len, so the estimator is the only thing under test.
    """
    by = defaultdict(
        lambda: {"n": 0, "hit": 0, "truth": 0, "pred": 0, "fp": 0, "fn": 0}
    )
    for t in turns:
        truth = is_dram_safe(t["prefix"], t["decode_len"] + tool_resp(t), k)
        pred = is_dram_safe(t["prefix"], t["decode_len"] + estimate(t), k)
        row = by[t["label"]]
        row["n"] += 1
        row["hit"] += truth == pred
        row["truth"] += truth
        row["pred"] += pred
        row["fp"] += pred and not truth
        row["fn"] += truth and not pred
    return by


def verdict_row(label, r):
    n = r["n"]
    return (
        f"{label:46s} {n:>8,d} "
        f"{100 * r['truth'] / n:>9.1f}% {100 * r['pred'] / n:>9.1f}% "
        f"{100 * r['fp'] / n:>7.1f}% {100 * r['fn'] / n:>7.1f}% "
        f"{100 * r['hit'] / n:>8.1f}%"
    )


def verdict_total(by):
    keys = ("n", "hit", "truth", "pred", "fp", "fn")
    return {key: sum(r[key] for r in by.values()) for key in keys}


def table2(turns, stats, k, out):
    """Accuracy of deciding dram-safe from the call type alone.

    fp / fn are both reported over all turns, not over the truth-unsafe and
    truth-safe subsets, so the two identities hold row by row:
        pred_safe = truth_safe + fp - fn
        accuracy  = 1 - fp - fn
    """
    by = tally_verdicts(
        turns, lambda t: stats.get(t["label"], {}).get("pred", 0), k
    )

    out.append("")
    out.append("=" * 100)
    out.append("表 2  按类型预测 dram-safe 的正确率")
    out.append("=" * 100)
    out.append("真实 append = 本轮 output + 实测的 tool resp / user input")
    out.append("预测 append = 本轮 output + 该类型的预测值（表 1 末列）")
    out.append("label=null 一行代表无 tool call 的轮，它预测的是用户下一条消息的长度——"
               "线上无从得知，这一行的正确率就是这个局限的量化")
    out.append("两者都用真实的 prefix 与 decode_len")
    out.append("假阳 = 真实 unsafe 但预测 safe，假阴 = 真实 safe 但预测 unsafe")
    out.append("五栏均以该行轮数为分母，故 预测safe = 真实safe + 假阳 - 假阴，"
               "正确率 = 1 - 假阳 - 假阴")
    out.append("")
    out.append(VERDICT_HEADER)
    for label, r in sorted(by.items(), key=lambda kv: -kv[1]["n"]):
        out.append(verdict_row(label, r))

    tot = verdict_total(by)
    out.append("")
    out.append(verdict_row("合计", tot))
    return tot


def table3(turns, k, out, pct=0.50):
    """Same verdict, but from one global median that ignores the call type."""
    global_pred = quantiles([tool_resp(t) for t in turns], pct)["pred"]
    by = tally_verdicts(turns, lambda t: global_pred, k)

    out.append("")
    out.append("=" * 100)
    out.append("表 3  不区分类型：全体 tool resp / user input 的单一分位点作为预测值")
    out.append("=" * 100)
    out.append(f"预测 append = decode_len + {global_pred:,}"
               f"（全部 {len(turns):,} 轮的 p{pct*100:.0f}，与类型无关）")
    out.append("与表 2 的唯一差别就是这个估计量")
    out.append("")
    out.append(VERDICT_HEADER)
    for label, r in sorted(by.items(), key=lambda kv: -kv[1]["n"]):
        out.append(verdict_row(label, r))

    tot = verdict_total(by)
    out.append("")
    out.append(verdict_row("合计", tot))
    return tot


def majority_baseline(turns, k):
    """Four-cell counts for always answering with the more common verdict."""
    truths = [
        is_dram_safe(t["prefix"], t["decode_len"] + tool_resp(t), k)
        for t in turns
    ]
    n, safe = len(truths), sum(truths)
    always_safe = safe >= n - safe
    return {
        "n": n,
        "truth": safe,
        "pred": n if always_safe else 0,
        "fp": (n - safe) if always_safe else 0,
        "fn": 0 if always_safe else safe,
        "hit": safe if always_safe else n - safe,
    }


def summary(turns, k, t2, t3, out):
    """The three predictors side by side, same five columns as tables 2 and 3."""
    rows = (
        ("全猜多数类", majority_baseline(turns, k)),
        ("按类型预测", t2),
        ("统一类型预测", t3),
    )
    out.append("")
    out.append("=" * 100)
    out.append("总结")
    out.append("=" * 100)
    out.append(
        f"{'预测方式':46s} {'轮数':>8s} {'真实safe':>10s} {'预测safe':>10s} "
        f"{'假阳':>8s} {'假阴':>8s} {'正确率':>9s}"
    )
    for name, r in rows:
        out.append(verdict_row(name, r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", default="tracelab_turns.jsonl")
    ap.add_argument("--max-turns", type=int, default=0, help="0 = 全部")
    ap.add_argument("--machine", choices=sorted(MACHINES), default="h20")
    ap.add_argument("--compute-scale", type=float, default=1.0)
    ap.add_argument("--dram-scale", type=float, default=1.0)
    ap.add_argument("--provider", default=None,
                    help="只统计某个 provider（claude / codex）；两者的 tool 名与 "
                         "prefix 语义不同，混在一起会掩盖各自规律")
    ap.add_argument("--predict-percentile", type=float, default=50.0,
                    help="预测值取该分位点（0-100，默认 50 即中位数）。"
                         "假阴恒为 0 说明预测偏高，调低此值可把误差移向假阴")
    ap.add_argument("--output-file", default="validation_tracelab.log")
    ap.add_argument(
        "--resp-len-output",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "tool_resp_len.json",
        ),
        help="预测查表用的 JSON，默认写在本目录。表里的值取 --predict-percentile "
             "指定的分位点（未必是中位数），文件里会记下用的是哪个分位点。"
             "TraceLab 的 label 命名空间（Read / Bash:cd）与 SWE-smith"
             "（bash:python）零重叠，两份表不可互换；"
             "跑服务时用 KVPILOT_TOOL_RESP_LEN_PATH 指向本文件",
    )
    args = ap.parse_args()

    k = derive_constants(args.machine, args.compute_scale, args.dram_scale)
    turns = load(args.input_file, args.max_turns, args.provider)
    if not turns:
        sys.exit(f"{args.input_file} 没有数据，先跑 prepare_tracelab_trace.py")

    sessions = len({t["session_id"] for t in turns})
    out = [
        f"数据 TraceLab (Claude Code 真实会话) / 机器 {args.machine.upper()}"
        f"（算力 x{args.compute_scale}, DRAM x{args.dram_scale}）",
        f"常数 a={k['a_us']:.4e} b={k['b_us']:.4f} "
        f"c_dram={k['c_dram_us']:.4f} us/token",
        f"样本 {sessions:,} 个 session / {len(turns):,} 个有后继的轮"
        + f"（其中无 tool call 的 {sum(1 for t in turns if t['label']=='null'):,}）"
        + (f" / provider={args.provider}" if args.provider
           else f" / provider {dict(Counter(t['provider'] for t in turns))}"),
        "",
    ]
    pct = args.predict_percentile / 100.0
    stats = table1(turns, out, pct)
    t2 = table2(turns, stats, k, out)
    t3 = table3(turns, k, out, pct)
    summary(turns, k, t2, t3, out)

    text = "\n".join(out) + "\n"
    with open(args.output_file, "w") as fh:
        fh.write(text)
    print(f"-> {args.output_file}")
    write_resp_len_table(stats, args.resp_len_output,
                         args.predict_percentile)


if __name__ == "__main__":
    main()
