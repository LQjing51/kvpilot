#!/usr/bin/env python3
"""Preprocess SWE-smith trajectories into input for benchmark_serving_multi_turn.py.

SWE-smith ships two incompatible styles in one dataset:

  structured   assistant turns carry a `tool_calls` array, and tool results are
               their own `role:"tool"` messages   (9,116 of 24,100 trajectories)
  inline       the call is written into the assistant's text as
               `<function=NAME><parameter=k>v</parameter></function>`, and the
               observation comes back as a plain `user` turn  (the other 14,984)

The inline style already matches what benchmark_serving_multi_turn.py accepts —
two keys per message, user/assistant strictly alternating — so this script
normalizes the structured ones INTO that style rather than the other way round.
The benchmark then needs no modification at all, and both halves of the dataset
become usable together.

Rendering the same trajectory both ways through Qwen3's chat template differs by
1.5% in tokens (12,182 vs 11,997), because that template serializes tool turns
into a `user` message with `<tool_response>` anyway.

Dropped, because no chat template reads them and one actively breaks:
  - `agent`, `message_type`, `cache_control`  — SWE-agent bookkeeping
  - `thought`, `action`                       — duplicates of content/tool_calls
  - `tool_call_id(s)`, `tool_calls[].id/type/index` — Qwen3 renders none of them
  - list-shaped `content`  — `{%- if message.content is string %}` in the
    template silently substitutes '' for a list, which would erase every tool
    observation (measured: 64% of tokens lost). Flattened to text instead.

Usage:
  python3 prepare_swe_trace.py --input-file swe_smith_all.jsonl \\
      --output-file swe_trace.json --max-trajectories 200

  python3 benchmark_serving_multi_turn.py -i swe_trace.json \\
      -m <MODEL_PATH> -u http://localhost:8000 -p 32
"""

import argparse
import json
import re
from collections import Counter


def flatten_content(content):
    """Return message text, collapsing the `[{"type":"text","text":...}]` form.

    Tool observations and first user turns use the list form. Qwen3's template
    tests `message.content is string` and falls back to '' otherwise, so a list
    left as-is renders empty.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def inline_tool_calls(tool_calls):
    """Serialize a `tool_calls` array into the dataset's own inline syntax.

    Mirrors exactly what the inline half of the dataset already contains, so
    both styles end up byte-comparable:

        <function=bash>
        <parameter=command>
        find /testbed -name widgets.py
        </parameter>
        </function>
    """
    blocks = []
    for call in tool_calls or []:
        fn = call.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        params = "".join(
            f"\n<parameter={key}>\n{value}\n</parameter>" for key, value in args.items()
        )
        blocks.append(f"<function={name}>{params}\n</function>")
    return "\n".join(blocks)


# SWE-agent writes its own termination note as the final assistant message when
# a run is killed rather than submitted: cost limit, wall-clock limit, repeated
# syntax errors, or an upstream API error. Those turns carry no tool call, so
# they are not agent behaviour and their length (5-10 tokens for the templated
# ones) would distort the last request's output budget.
RX_ABORTED = re.compile(r"^\s*Exit due to ", re.I)


def aborted_reason(messages):
    """The harness's termination note, if this trajectory was killed."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        text = flatten_content(msg.get("content")).strip()
        return text if RX_ABORTED.match(text) else None
    return None


def normalize(messages):
    """Convert one trajectory to alternating user/assistant, two keys each.

    Returns None when the result cannot satisfy the benchmark's invariants,
    so the caller can skip it rather than fail mid-run.
    """
    out = []
    for msg in messages:
        role = msg.get("role")
        text = flatten_content(msg.get("content"))

        if role == "assistant" and msg.get("tool_calls"):
            calls = inline_tool_calls(msg["tool_calls"])
            if calls:
                text = f"{text}\n\n{calls}" if text.strip() else calls

        # A tool observation is context the model must answer, which is the
        # role a `user` turn plays here — and the same thing the inline half of
        # the dataset already does with its observations.
        if role == "tool":
            role = "user"

        if role not in ("system", "user", "assistant"):
            continue

        # The benchmark asserts the first message it sends is a user turn, so a
        # leading system prompt has to merge into it.
        if role == "system":
            if out:
                continue
            out.append({"role": "user", "content": text})
            continue

        # Consecutive same-role turns would break the alternation assertion.
        # They arise whenever an assistant turn held only tool_calls.
        if out and out[-1]["role"] == role:
            merged = out[-1]["content"].rstrip()
            out[-1]["content"] = f"{merged}\n{text}" if text else merged
            continue

        out.append({"role": role, "content": text})

    # First turn must be user, and the transcript must end on an assistant turn
    # because the benchmark reads messages[messages_to_use] as the answer.
    while out and out[0]["role"] != "user":
        out.pop(0)
    while out and out[-1]["role"] != "assistant":
        out.pop()
    if len(out) < 2:
        return None

    # An empty content would make the server generate from nothing.
    for msg in out:
        if not msg["content"].strip():
            msg["content"] = " "
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-file",
        default="swe_smith_all.jsonl",
        help="8 个 tool-*.parquet 合并出的原始 JSONL，一行一条轨迹。",
    )
    ap.add_argument(
        "--output-file",
        default="swe_trace.json",
        help="给 benchmark_serving_multi_turn.py 用的 JSON 数组。",
    )
    ap.add_argument("--max-trajectories", type=int, default=0, help="0 = 全部")
    ap.add_argument(
        "--resolved-only",
        action="store_true",
        help="只保留补丁真正修好任务的轨迹（全量里约 39%%）。",
    )
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="模型路径或 HF 名称。给了才能统计 token 分布并启用 "
        "--max-prompt-tokens（需要 transformers）。",
    )
    ap.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=0,
        help="整段对话超过这个 token 数就跳过，避免撞 --max-model-len。0 = 不限。",
    )
    ap.add_argument("--min-turns", type=int, default=2)
    ap.add_argument(
        "--keep-aborted",
        action="store_true",
        help="保留被 harness 强制终止的轨迹（末条是 'Exit due to ...'）。"
        "默认剔除：它们没有 tool call，末条只有 5-10 token，"
        "会让最后一次请求的输出预算失真。",
    )
    args = ap.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    out = []
    skipped = too_long = aborted = 0
    aborted_reasons = Counter()
    n_structured = n_inline = 0
    turn_counts = []
    token_counts = []

    with open(args.input_file) as fh:
        for line in fh:
            record = json.loads(line)
            if args.resolved_only and not record.get("resolved"):
                continue

            reason = aborted_reason(record["messages"])
            if reason is not None and not args.keep_aborted:
                aborted += 1
                aborted_reasons[reason.split(":")[0][:60]] += 1
                continue

            was_structured = any(m.get("tool_calls") for m in record["messages"])
            messages = normalize(record["messages"])
            if messages is None or len(messages) < args.min_turns:
                skipped += 1
                continue

            if tokenizer is not None:
                total = sum(
                    len(tokenizer(m["content"], add_special_tokens=False).input_ids)
                    for m in messages
                )
                if args.max_prompt_tokens and total > args.max_prompt_tokens:
                    too_long += 1
                    continue
                token_counts.append(total)

            out.append({"id": record["traj_id"], "messages": messages})
            turn_counts.append(len(messages))
            if was_structured:
                n_structured += 1
            else:
                n_inline += 1

            if args.max_trajectories and len(out) >= args.max_trajectories:
                break

    # indent=1 keeps one "id" per line: the benchmark regex-scans the raw file
    # to map each conversation back to its source line number.
    with open(args.output_file, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    print(f"轨迹数:       {len(out):,}  (原结构化 {n_structured:,} + 原内联 {n_inline:,})")
    if skipped:
        print(f"不合规跳过:   {skipped:,}")
    if too_long:
        print(f"超长跳过:     {too_long:,}")
    if aborted:
        print(f"异常终止剔除: {aborted:,}")
        for reason, cnt in aborted_reasons.most_common():
            print(f"   {cnt:6,d}  {reason}")
    if turn_counts:
        turn_counts.sort()
        print(
            f"每轨迹轮数:   p50={turn_counts[len(turn_counts) // 2]}"
            f" min={turn_counts[0]} max={turn_counts[-1]}"
        )
    if token_counts:
        token_counts.sort()
        q = lambda p: token_counts[int(len(token_counts) * p)]  # noqa: E731
        print(
            f"整段 token:   p50={q(0.5):,} p90={q(0.9):,} max={token_counts[-1]:,}"
        )
    print(f"已写入:       {args.output_file}")

    verify(args.output_file)


def verify(path):
    """Re-read the output and assert exactly what the benchmark asserts."""
    data = json.load(open(path))
    problems = 0
    for item in data:
        messages = item["messages"]
        for msg in messages:
            if set(msg.keys()) != {"role", "content"}:
                problems += 1
                break
            if not isinstance(msg["content"], str):
                problems += 1
                break
        for i in range(len(messages) - 1):
            if messages[i]["role"] == messages[i + 1]["role"]:
                problems += 1
                break
        if messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
            problems += 1
        if len(messages) % 2:
            problems += 1
    print(
        f"校验:         {len(data):,} 条轨迹，{problems} 处问题"
        f"（键名/类型/交替/首末/偶数轮）"
    )


if __name__ == "__main__":
    main()
