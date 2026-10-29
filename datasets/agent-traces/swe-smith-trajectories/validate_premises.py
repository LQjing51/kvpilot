#!/usr/bin/env python3
"""Three tables on whether tool call type predicts KV placement, plus the
per-request ratio table the HBM pool distribution plot joins against.

Table 1  tool resp length per call type.
Table 2  accuracy of predicting the dram-safe verdict from that type.
Table 3  the same verdict from one global median, ignoring the type — the
         control that says how much of table 2's score the type actually earned.

The predictor only gets what is knowable when decode emits the tool call: the
call type, this turn's decode length, and the running prefix. It has not seen
the tool response, so it substitutes that type's median response from table 1.
The median, not the mean: response length is heavily right-skewed (p90/p10 runs
past 25x on the head types), which drags the mean into the upper third of the
distribution and makes the verdict fire safe for nearly every turn. It also
cannot know whether a response will arrive at all, so terminal turns get the
same estimate as any other and count 0 toward it. Ground truth uses the measured
response.

The ratio table (--ratio-output) comes out of the same pass, keyed by the trace
line the replay scripts encode in each request id. It exists because the HBM pool
census records which request last held each block but not what that block was
worth keeping — the ratio is a property of the request, and is not even known
until the turn ends, which for a session's last turn is after the blocks are
gone. Sharing this walk is the point: prefix, decode_len and the cost-model
constants are then computed once, so what the tables measure and what the plot
draws cannot describe two different worlds. It also halves the cost of a full
run, which is dominated by the tokenizer over a 2.5 GB trace.

Usage:
  python3 validate_premises.py [--input-file swe_trace.json] [--machine h20]
"""

import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict

# The shared predictor now lives with the serving code it feeds, so the offline
# validator and the online path cannot drift.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../vllm-dev/vllm/v1/core"
    ),
)

from kvpilot_predict import (  # noqa: E402
    MACHINES,
    derive_constants,
    dram_safe_times,
    is_dram_safe,
    turn_labels,
    write_resp_len_table as _write_resp_len_table,
)


_DECODER = json.JSONDecoder()


def stream_records(path, max_traj):
    """Yield trajectories one at a time from a pretty-printed JSON array.

    json.load would materialize all 23k trajectories first — 17 s and ~13 GB for
    the 2.4 GB file, paid in full even when --max-trajectories asks for 300, and
    close enough to this machine's RAM to start swapping. Decoding
    incrementally keeps only the current record alive and makes a bounded run
    bounded in cost too.
    """
    buf, started, n = "", False, 0
    with open(path) as fh:
        while not max_traj or n < max_traj:
            chunk = fh.read(1 << 22)
            if not chunk:
                break
            buf += chunk
            while not max_traj or n < max_traj:
                if not started:
                    i = buf.find("[")
                    if i < 0:
                        break
                    buf, started = buf[i + 1:], True
                buf = buf.lstrip().lstrip(",").lstrip()
                if not buf or buf[0] == "]":
                    break
                try:
                    obj, end = _DECODER.raw_decode(buf)
                except ValueError:
                    break  # record straddles the chunk boundary; read more
                buf = buf[end:]
                n += 1
                yield obj


ID_LINE_RE = re.compile(r'"id"\s*:\s*"([^"\\]*)"')
USER_ROLE_RE = re.compile(r'"role"\s*:\s*"user"')


def scan_user_lines(path):
    """Map each trajectory id to the line of each of its user messages.

    Mirrors the scan in benchmark_serving_multi_turn.py: the file is
    pretty-printed one `"role":` per line, so every `"role": "user"` after an
    `"id"` belongs to that trajectory, in order. That script embeds these same
    line numbers in the request ids it sends, which is what lets the ratio table
    be joined against an HBM census — so the two scans have to agree, and
    reading the same file the same way is what keeps them agreeing.

    Cheap next to the rest of the run: plain text, no tokenizer.
    """
    by_id = {}
    with open(path) as fh:
        current = None
        for lineno, line in enumerate(fh, start=1):
            m = ID_LINE_RE.search(line)
            if m:
                current = []
                by_id[m.group(1)] = current
                continue
            if current is not None and USER_ROLE_RE.search(line):
                current.append(lineno)
    return by_id


def collect(path, max_traj, ntok, user_lines=None):
    """One record per agent turn that made a call.

    prefix counts prompt history only: the decode instance runs without
    LMCache/Mooncake, so its output KV is never reused by the next prefill and
    belongs to append instead.

    `user_lines` (from scan_user_lines) adds the trace line of the request the
    replay sends for each turn, which the ratio table keys on. The turns
    themselves are the same either way; the tables ignore the extra fields.
    """
    turns = []
    n_traj = 0
    for record in stream_records(path, max_traj):
        msgs = record["messages"]
        n_traj += 1
        # One batched call per trajectory: the fast tokenizer amortizes its
        # Rust-side setup over the batch, which measures ~2.6x faster than
        # calling it per message.
        lens = ntok([m.get("content") or "" for m in msgs])
        lines = (user_lines or {}).get(record["id"]) or []
        prefix = 0
        n_user = 0
        turn_idx = 0
        for i, msg in enumerate(msgs):
            if msg.get("role") != "assistant":
                prefix += lens[i]
                if msg.get("role") == "user":
                    n_user += 1
                continue
            labels = turn_labels(msg)
            if not labels or prefix <= 0:
                prefix += lens[i]
                continue
            has_next = i + 1 < len(msgs) and msgs[i + 1].get("role") == "user"
            decode_len = lens[i]
            # The replay sends one request per user message, carrying that
            # message's line; this turn answers the one just consumed.
            ordinal = n_user - 1
            turns.append(
                {
                    # Whole call set as one category, joined by `+`: the
                    # responses all land in the same next append, and this is
                    # the key the online judge looks up (see
                    # kvpilot_predict.explain_dram_safe).
                    "label": "+".join(labels),
                    "prefix": prefix,
                    "decode_len": decode_len,
                    "tool_resp": lens[i + 1] if has_next else 0,
                    "line": lines[ordinal] if 0 <= ordinal < len(lines) else 0,
                    "traj_id": record["id"],
                    "turn_idx": turn_idx,
                }
            )
            turn_idx += 1
            prefix += decode_len
    return turns, n_traj


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


def table1(turns, out, pct=0.50):
    """tool resp length by call type, over every turn of that type.

    A terminal turn contributes 0: no response ever arrives. Those zeros belong
    in this distribution because the predictor sees only the call type and cannot
    tell a terminal turn from a continuing one, so the statistic it substitutes
    has to be taken over both kinds.
    """
    by = defaultdict(list)
    for t in turns:
        by[t["label"]].append(t["tool_resp"])

    out.append("=" * 100)
    out.append(f"（预测值取 p{pct*100:.0f}，可用 --predict-percentile 调整）")
    out.append("表 1  tool call 类型 -> tool resp 长度")
    out.append("=" * 100)
    out.append("终止轮（没有下一个请求）按 tool_resp = 0 计入")
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

    all_vals = [v for vs in by.values() for v in vs]
    grand = statistics.fmean(all_vals)
    var = statistics.pvariance(all_vals)
    between = sum(
        len(vs) * (statistics.fmean(vs) - grand) ** 2 for vs in by.values()
    ) / len(all_vals)
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

    `estimate(turn)` returns the response length the predictor would assume.
    Ground truth always uses the measured response, and both sides use the real
    prefix and decode_len, so the estimator is the only thing under test.
    """
    by = defaultdict(
        lambda: {"n": 0, "hit": 0, "truth": 0, "pred": 0, "fp": 0, "fn": 0}
    )
    for t in turns:
        truth = is_dram_safe(t["prefix"], t["decode_len"] + t["tool_resp"], k)
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
    # Every turn gets the type's median, terminal ones included: online the
    # predictor sees only the call type and cannot know whether a response
    # follows, so exempting terminal turns would credit it with a distinction it
    # cannot make.
    by = tally_verdicts(
        turns, lambda t: stats.get(t["label"], {}).get("pred", 0), k
    )

    out.append("")
    out.append("=" * 100)
    out.append("表 2  按类型预测 dram-safe 的正确率")
    out.append("=" * 100)
    out.append("真实 append = decode_len + 实测 tool_resp")
    out.append("预测 append = decode_len + 该类型 tool_resp 的预测值（表 1 末列）")
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
    """Same verdict, but from one global median that ignores the call type.

    This is the control for table 2: if knowing the tool call type buys nothing,
    a single number applied to every turn scores the same. The rows are the same
    call types so the two tables can be read side by side, but every row here is
    predicted with the identical estimate.
    """
    global_pred = quantiles([t["tool_resp"] for t in turns], pct)["pred"]
    by = tally_verdicts(turns, lambda t: global_pred, k)

    out.append("")
    out.append("=" * 100)
    out.append("表 3  不区分类型：全体 tool_resp 的单一分位点作为预测值")
    out.append("=" * 100)
    out.append(f"预测 append = decode_len + {global_pred:,}"
               f"（全部 {len(turns):,} 轮 tool_resp 的 p{pct*100:.0f}，与类型无关）")
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
    """Four-cell counts for always answering with the more common verdict.

    The floor any predictor has to clear: it reads neither the call type nor the
    response, so whichever of safe/unsafe is more common becomes every answer.
    """
    truths = [
        is_dram_safe(t["prefix"], t["decode_len"] + t["tool_resp"], k)
        for t in turns
    ]
    n, safe = len(truths), sum(truths)
    always_safe = safe >= n - safe
    return {
        "n": n,
        "truth": safe,
        # Predicting safe everywhere makes every truth-unsafe turn a false
        # positive and leaves no false negatives; predicting unsafe inverts it.
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


def write_resp_len_table(stats, path, percentile=None):
    """Emit the label -> response-length table the online predictor reads.

    The implementation is shared with the TraceLab validator (and with the
    serving code that reads the file) so the format cannot drift; see
    kvpilot_predict.write_resp_len_table.
    """
    return _write_resp_len_table(stats, path, percentile)


def write_ratio_table(turns, k, path, meta):
    """Emit the per-request load/compute ratios the HBM pool plot joins against.

    Keyed by the trace line the replay encodes in the leading UUID group of each
    request id, so a census row's `req_line` looks the ratio up directly.

    `append` is `decode_len + tool_resp`, exactly as the tables use it: a
    terminal turn measures `tool_resp = 0` and so lands at `append = decode_len`,
    not at `append = 0`. Both numbers are defensible and they mean different
    things -- 0 says "nothing will ever reload this KV", decode_len says "this
    much was computed on top of it" -- but they must not both be in play at
    once, or the y axis of the plot would stop matching the verdict the tables
    score. One definition, shared, is the whole reason this lives here.

    The consequence worth knowing: a terminal turn gets a real ratio rather than
    a null, so there is no session-end special case for the plotter to place.

    The times already look one turn ahead and must not be shifted again:
    `dram_safe_times(prefix, append)` weighs loading the KV this request leaves
    behind against the compute that follows it, which is what the next request
    experiences on these very blocks. `next_line` is recorded to check the join,
    never to source the ratio from.
    """
    entries = {}
    collisions = 0
    by_traj = defaultdict(list)
    for t in turns:
        by_traj[t["traj_id"]].append(t)

    for traj_turns in by_traj.values():
        ordered = sorted(traj_turns, key=lambda t: t["turn_idx"])
        for idx, t in enumerate(ordered):
            if not t["line"]:
                continue
            append = t["decode_len"] + t["tool_resp"]
            t_load, t_comp = dram_safe_times(t["prefix"], append, k)
            nxt = ordered[idx + 1] if idx + 1 < len(ordered) else None
            key = str(t["line"])
            if key in entries:
                collisions += 1
            entries[key] = {
                "prefix": t["prefix"],
                "decode_len": t["decode_len"],
                "tool_resp": t["tool_resp"],
                "append": append,
                "label": t["label"],
                "t_load_us": round(t_load, 1),
                "t_comp_us": round(t_comp, 1),
                "ratio": round(t_load / t_comp, 4) if t_comp > 0 else None,
                "dram_safe": is_dram_safe(t["prefix"], append, k),
                "traj_id": t["traj_id"],
                "turn_idx": t["turn_idx"],
                "next_line": nxt["line"] if nxt and nxt["line"] else None,
            }

    n_missing = sum(1 for t in turns if not t["line"])
    payload = {
        "_meta": {**meta, "requests": len(entries)},
        "requests": entries,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(
        f"-> {path}: requests={len(entries):,} "
        f"missing_line={n_missing:,} collisions={collisions:,}"
    )
    if n_missing:
        print(
            "   warning: 有轮次拿不到行号，HBM census 的 req_line 将 join 不上。"
            "确认 --input-file 与压测时用的是同一个文件"
        )
    if collisions:
        print(
            "   warning: 行号重复，同一行被多轮覆盖。"
            "输入文件可能不是每条消息一行的 pretty-print 格式"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", default="swe_trace.json")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--max-trajectories", type=int, default=0, help="0 = 全部")
    ap.add_argument("--machine", choices=sorted(MACHINES), default="h20")
    ap.add_argument("--compute-scale", type=float, default=1.0)
    ap.add_argument("--dram-scale", type=float, default=1.0)
    ap.add_argument("--predict-percentile", type=float, default=50.0,
                    help="预测值取该分位点（0-100，默认 50 即中位数）。"
                         "假阴恒为 0 说明预测偏高，调低此值可把误差移向假阴")
    ap.add_argument("--output-file", default="validation.log")
    ap.add_argument(
        "--resp-len-output",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "tool_resp_len.json",
        ),
        help="预测查表用的 JSON，默认写在本目录（与产出它的 trace 放在一起）。"
             "表里的值取 --predict-percentile 指定的分位点（未必是中位数），"
             "文件里会记下用的是哪个分位点；"
             "跑服务时用 KVPILOT_TOOL_RESP_LEN_PATH 指向它",
    )
    ap.add_argument(
        "--ratio-output",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "req_ratio.json",
        ),
        help="每个请求的 load/comp 比，HBM block 分布图用它做 y 轴。"
             "按 request_id 里编码的源文件行号索引，故建表与压测必须用同一个"
             "输入文件；见 plot_hbm_pool_distribution.py",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def ntok(texts):
        """Token counts for a list of texts, in order.

        Empty strings are dropped before the call and filled back in as 0: the
        tokenizer is happy to encode them but they are the single most common
        input here, and skipping them keeps the batch to real work.
        """
        counts = [0] * len(texts)
        idx = [i for i, t in enumerate(texts) if t]
        if not idx:
            return counts
        enc = tok([texts[i] for i in idx], add_special_tokens=False)
        for i, ids in zip(idx, enc.input_ids):
            counts[i] = len(ids)
        return counts

    k = derive_constants(args.machine, args.compute_scale, args.dram_scale)
    user_lines = scan_user_lines(args.input_file)
    turns, n_traj = collect(args.input_file, args.max_trajectories, ntok,
                            user_lines)

    out = [
        f"模型 Qwen3-30B-A3B bf16 / 机器 {args.machine.upper()}"
        f"（算力 x{args.compute_scale}, DRAM x{args.dram_scale}）",
        f"常数 a={k['a_us']:.4e} b={k['b_us']:.4f} "
        f"c_dram={k['c_dram_us']:.4f} us/token",
        f"样本 {n_traj:,} 条轨迹 / {len(turns):,} 个 agent 轮",
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
    write_ratio_table(
        turns,
        k,
        args.ratio_output,
        {
            "machine": args.machine,
            "trace": os.path.basename(args.input_file),
            "tokenizer": args.tokenizer,
            "compute_scale": args.compute_scale,
            "dram_scale": args.dram_scale,
            "constants_us": k,
            "trajectories": n_traj,
            "append": "decode_len + tool_resp (terminal turns: tool_resp = 0)",
        },
    )


if __name__ == "__main__":
    main()
