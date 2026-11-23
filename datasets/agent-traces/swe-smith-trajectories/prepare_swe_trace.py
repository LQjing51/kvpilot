#!/usr/bin/env python3
"""Turn raw SWE-smith trajectories into a Mooncake-style trace, one request per line.

Each trajectory is normalized into alternating user/assistant messages. Round k
is the request whose prompt is messages[:2k+1] rendered through the Qwen3 chat
template, and whose output is assistant message 2k+1.

Normalization. SWE-smith ships two styles in one dataset:

  structured   assistant turns carry a `tool_calls` array, and tool results are
               their own `role:"tool"` messages   (9,116 of 24,100 trajectories)
  inline       the call is written into the assistant's text as
               `<function=NAME><parameter=k>v</parameter></function>`, and the
               observation comes back as a plain `user` turn  (the other 14,984)

Structured ones are rewritten into the inline style, which is what
`kvpilot_signal.turn_labels` parses. The two render within 1.5% of each other in
tokens, because Qwen3's template serializes tool turns into a `user` message
anyway. List-shaped `content` is flattened to text: the template's
`{%- if message.content is string %}` silently substitutes '' for a list, which
would erase every tool observation (measured: 64% of tokens lost). Trajectories
the harness killed (last assistant message `Exit due to ...`) are dropped: they
are not agent behaviour, and their 5-10 token last turn distorts the output
budget.

A trajectory is cut before its first round with input + output above
--max-model-len, as in prepare_tracelab_trace.py.

hash_ids: `hash_ids[i]` names the content of blocks 0..i, BLOCK tokens each, the
last one possibly partial -- nodes of a prefix tree over the real token stream,
so whatever sharing exists, in-session or across trajectories, appears by
itself. `prefix_length` is the longest token prefix shared with any earlier
prompt, read off the same tree (matched full blocks, then the closest sibling
of the first unmatched one), so the full dataset costs one pass. `verify`
replays the file through an unbounded cache to check every request hits
floor(prefix_length / BLOCK) blocks.

Each line, in trajectory order:
    timestamp      null -- SWE-smith records no times
    session_id round_index input_length prefix_length output_length hash_ids
    tool_calls     call types this round's output made, in order

Size: either the first --num-requests requests (0 = every trajectory), or, with
--min-unique-kv-gib, whole trajectories until the distinct full blocks hold
that much KV at --kv-bytes-per-token (98,304 = Qwen3-30B-A3B BF16: 48 layers x
4 KV heads x 128 x K/V x 2 B). The committed file is --min-unique-kv-gib 400;
swe_full.jsonl is --num-requests 0. --sessions keeps only the listed
trajectories, as if the others were not in the input.

Usage:
  python3 prepare_swe_trace.py [--input swe_smith_all.jsonl]
                               [--out swe_mooncake.jsonl]
                               [--num-requests 2000 | --min-unique-kv-gib 400]
                               [--sessions FILE] [--workers N]
                               [--max-model-len 262144] [--tokenizer PATH]
"""

import argparse
import bisect
import json
import os
import re
import sys
from collections import Counter
from itertools import islice
from multiprocessing import Pool

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../vllm-dev/vllm/v1/core"
    ),
)
from kvpilot_signal import turn_labels  # noqa: E402

# mooncake_trace_replay.py expands every hash id into this many tokens.
BLOCK = 512
TOKEN = np.int32


def flatten_content(content):
    """Return message text, collapsing the `[{"type":"text","text":...}]` form."""
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
# syntax errors, or an upstream API error.
RX_ABORTED = re.compile(r"^\s*Exit due to ", re.I)


def aborted(messages):
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return bool(RX_ABORTED.match(flatten_content(msg.get("content"))))
    return False


def normalize(messages):
    """Alternating user/assistant, starting with user and ending with assistant.

    Returns None when not even one user -> assistant round is left.
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
        # role a `user` turn plays here -- and the same thing the inline half of
        # the dataset already does with its observations.
        if role == "tool":
            role = "user"

        if role not in ("system", "user", "assistant"):
            continue

        # A leading system prompt merges into the first user turn.
        if role == "system":
            if out:
                continue
            out.append({"role": "user", "content": text})
            continue

        # Consecutive same-role turns arise whenever an assistant turn held
        # only tool_calls; merging them keeps one round per user -> assistant.
        if out and out[-1]["role"] == role:
            merged = out[-1]["content"].rstrip()
            out[-1]["content"] = f"{merged}\n{text}" if text else merged
            continue

        out.append({"role": role, "content": text})

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


class Ids:
    """Prefix-tree node ids: one per (parent id, block content).

    `children[parent]` keeps each node's child blocks sorted as bytes. Tokens
    are fixed-width, so the child sharing the most leading tokens with a new
    block is one of its two neighbours in that order.
    """

    def __init__(self):
        self.table = {}
        self.children = {}

    def get(self, parent, content):
        node = self.table.get((parent, content))
        if node is None:
            node = self.table[(parent, content)] = len(self.table)
            bisect.insort(self.children.setdefault(parent, []), content)
        return node

    def shared(self, parent, content):
        """Leading tokens `content` has in common with some child of `parent`."""
        kids = self.children.get(parent, [])
        i = bisect.bisect_left(kids, content)
        a, best = np.frombuffer(content, TOKEN), 0
        for other in kids[max(0, i - 1): i + 1]:
            b = np.frombuffer(other, TOKEN)
            m = min(len(a), len(b))
            diff = np.flatnonzero(a[:m] != b[:m])
            best = max(best, int(diff[0]) if len(diff) else m)
        return best

    def place(self, p):
        """(hash_ids, longest token prefix shared with any earlier prompt)."""
        h, parent, prefix = [], None, None
        for s in range(0, len(p), BLOCK):
            content = p[s: s + BLOCK].tobytes()
            if prefix is None and (parent, content) not in self.table:
                prefix = s + self.shared(parent, content)
            parent = self.get(parent, content)
            h.append(parent)
        return h, len(p) if prefix is None else prefix


_tok = _max_model_len = _sessions = None


def _init_worker(tokenizer, max_model_len, sessions):
    global _tok, _max_model_len, _sessions
    from transformers import AutoTokenizer

    _tok = AutoTokenizer.from_pretrained(tokenizer)
    _max_model_len, _sessions = max_model_len, sessions


def tokenize(line):
    """One raw trajectory -> (traj_id, kind, rounds, cut_rounds), or None when
    --sessions leaves it out.

    kind is "aborted" / "malformed" (dropped) or "structured" / "inline".
    rounds holds (prompt tokens, output length, tool calls) up to the
    --max-model-len cut; cut_rounds is how many rounds that cut dropped.
    """
    record = json.loads(line)
    if _sessions is not None and record["traj_id"] not in _sessions:
        return None
    if aborted(record["messages"]):
        return record["traj_id"], "aborted", [], 0
    msgs = normalize(record["messages"])
    if msgs is None:
        return record["traj_id"], "malformed", [], 0
    structured = any(m.get("tool_calls") for m in record["messages"])

    rounds = []
    n_rounds = len(msgs) // 2
    for rnd in range(n_rounds):
        p = np.array(_tok.apply_chat_template(
            msgs[: 2 * rnd + 1], tokenize=True, add_generation_prompt=True), dtype=TOKEN)
        out_len = len(_tok(msgs[2 * rnd + 1]["content"],
                           add_special_tokens=False).input_ids)
        if len(p) + out_len > _max_model_len:
            return (record["traj_id"], "structured" if structured else "inline",
                    rounds, n_rounds - rnd)
        rounds.append((p, out_len, turn_labels(msgs[2 * rnd + 1])))
    return record["traj_id"], "structured" if structured else "inline", rounds, 0


def tokenized(args):
    """tokenize() over the input in file order, `--workers` processes at a time."""
    sessions = set(open(args.sessions).read().split()) if args.sessions else None
    with open(args.input) as fh, Pool(
            args.workers, _init_worker,
            (args.tokenizer, args.max_model_len, sessions)) as pool:
        while batch := list(islice(fh, 8 * args.workers)):
            yield from pool.map(tokenize, batch, chunksize=1)


def convert(args, stats):
    ids = Ids()
    rows = []
    full_blocks = set()             # ids of distinct full blocks written so far
    kv_target = args.min_unique_kv_gib * 2**30 / (BLOCK * args.kv_bytes_per_token)
    limit = None if args.min_unique_kv_gib else (args.num_requests or None)
    for traj in tokenized(args):
        if limit is not None and len(rows) >= limit:
            break
        if args.min_unique_kv_gib and len(full_blocks) >= kv_target:
            break
        if traj is None:
            continue
        traj_id, kind, rounds, cut_rounds = traj
        stats["trajectories_seen"] += 1
        if stats["trajectories_seen"] % 1000 == 0:
            print(f"  ... 轨迹 {stats['trajectories_seen']:,}，请求 {len(rows):,}",
                  file=sys.stderr, flush=True)
        if kind in ("aborted", "malformed"):
            stats[f"drop_{kind}"] += 1
            continue
        stats[kind] += 1

        for rnd, (p, out_len, tool_calls) in enumerate(rounds):
            h, prefix = ids.place(p)
            full_blocks.update(h[: len(p) // BLOCK])
            rows.append({
                "timestamp": None,
                "session_id": traj_id,
                "round_index": rnd,
                "input_length": len(p),
                "prefix_length": prefix,
                "output_length": out_len,
                "hash_ids": h,
                "tool_calls": tool_calls,
            })
            if limit is not None and len(rows) >= limit:
                break
        else:
            if cut_rounds:
                stats["cut_sessions"] += 1
                stats["cut_rounds"] += cut_rounds
    stats["unique_full_blocks"] = len(full_blocks)
    return rows


def renumber(rows):
    """Ids from 0 in order of first appearance in the output."""
    new = {}
    for r in rows:
        r["hash_ids"] = [new.setdefault(h, len(new)) for h in r["hash_ids"]]


def verify(rows):
    """Unbounded-cache replay in file order; returns (mismatches, hit rate)."""
    seen = set()
    bad = hit = total = 0
    for r in rows:
        full = r["input_length"] // BLOCK
        h = r["hash_ids"]
        k = 0
        while k < full and h[k] in seen:
            k += 1
        bad += k != r["prefix_length"] // BLOCK
        hit += k * BLOCK
        total += r["input_length"]
        seen.update(h[:full])
    return bad, hit / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="swe_smith_all.jsonl",
                    help="8 个 tool-*.parquet 合并出的原始 JSONL，一行一条轨迹")
    ap.add_argument("--out", default="swe_mooncake.jsonl")
    ap.add_argument("--num-requests", type=int, default=2000,
                    help="按文件顺序写满 N 条请求为止；0 = 全部轨迹")
    ap.add_argument("--min-unique-kv-gib", type=float, default=0,
                    help="非 0 时取代 --num-requests：按整条轨迹写，直到不同的整块 "
                         "合计 KV 达到这么多 GiB")
    ap.add_argument("--kv-bytes-per-token", type=int, default=98_304,
                    help="每 token 的 KV 字节（全部层、K+V）；默认 Qwen3-30B-A3B BF16")
    ap.add_argument("--max-model-len", type=int, default=262144,
                    help="服务端 --max-model-len；轨迹在第一次 input+output "
                         "超过它的那一轮截断")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                    help="须与服务端模型一致；集群上传本地路径")
    ap.add_argument("--sessions",
                    help="只处理这个文件里列出的 traj_id（空白分隔），其余轨迹当作不存在："
                         "不计数、不进前缀树")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="tokenize 的进程数")
    args = ap.parse_args()

    s = Counter()
    rows = convert(args, s)
    renumber(rows)
    bad, hit_rate = verify(rows)

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    sessions = len({r["session_id"] for r in rows})
    cross = sum(r["round_index"] == 0 and r["prefix_length"] >= BLOCK for r in rows)
    print(f"读入轨迹 {s['trajectories_seen']:,}")
    print(f"  剔除 harness 强制终止的    {s['drop_aborted']:,}")
    print(f"  剔除 规整后不足一轮的      {s['drop_malformed']:,}")
    print(f"  超过 --max-model-len {args.max_model_len:,} 截断 "
          f"{s['cut_sessions']:,} 条轨迹，丢尾部 {s['cut_rounds']:,} 轮")
    print(f"保留轨迹 原结构化 {s['structured']:,} + 原内联 {s['inline']:,}")
    print(f"写出 {len(rows):,} 条请求 / {sessions:,} 条轨迹，"
          f"round 0 跨轨迹命中 ≥1 块的 {cross:,} 条")
    kv = s["unique_full_blocks"] * BLOCK * args.kv_bytes_per_token / 2**30
    print(f"不同的整块 {s['unique_full_blocks']:,} 个 = {kv:.1f} GiB KV")
    print(f"无限缓存回放：命中率 {hit_rate:.1%}，"
          f"命中块数 != floor(prefix/{BLOCK}) 的 {bad} 条")
    print(f"-> {args.out}")
    if bad:
        sys.exit("hash_ids 与 prefix_length 不一致")


if __name__ == "__main__":
    main()
