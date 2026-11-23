#!/usr/bin/env python3
"""Turn the raw TraceLab round log into a Mooncake-style trace, one request per line.

TraceLab ships one JSON object per agent round, sanitized: token counts and tool
metadata survive, prompt text does not.

Filters, each discarding the whole session:
  * provider is not claude (codex quantizes prefix to 128);
  * round 0 has prefix > 0 -- its cache hit came from another session;
  * input[N] <= input[N-1] anywhere -- context was trimmed or reset.

A surviving session is cut before its first round with input + output above
--max-model-len. The server rejects that request, every later round is longer
still, and skipping just that round would leave its successor's prefix cold.

Rounds with output_tokens == 0 are dropped on their own (the server rejects
max_tokens=0); the round after one continues from the last kept round, so its
round_index skips a number.

Conversion: prefix[N] = input of the previous kept round (0 for the first).
What gets reused is exactly the previous prompt: more than that means Anthropic
also cached the previous decode output, which our decode instance never stores;
less is a cache breakpoint or expiry, which a prefix cache would not miss.

hash_ids: `hash_ids[i]` names the content of blocks 0..i, BLOCK tokens each, the
last one possibly partial. Round N keeps the first input[N-1] // BLOCK ids of
round N-1 and takes fresh ids for the rest, including the slot of N-1's partial
tail block, which round N has filled with different content. No id is shared
across sessions: the round-0 filter removed every cross-session hit. Ids are
renumbered from 0 in order of first appearance, and `verify` replays the file
through an unbounded cache to check every request hits floor(prefix / BLOCK).

Each line, the earliest --num-requests by timestamp:
    timestamp      ms since the first request in the file
    session_id round_index input_length prefix_length output_length hash_ids
    tool_calls     call types this round's output made, in order

Usage:
  python3 prepare_tracelab_trace.py [--input syfi_coding_trace.jsonl.gz]
                                    [--out tracelab_mooncake.jsonl]
                                    [--max-model-len 262144] [--num-requests 2000]
"""

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime

# mooncake_trace_replay.py expands every hash id into this many tokens.
BLOCK = 512


def tool_label(tool):
    """The call type; Bash is split by its leading executable.

    `<assign>` becomes `(assign)`: the label is rendered into `<function=NAME>`
    and read back with `<function=([^>\\s]+)>`, which would cut at `>`.
    """
    name = tool.get("tool_name") or "?"
    if name != "Bash":
        return _wire_safe(name)
    words = (tool.get("command_skeleton") or "").split()
    head = words[0].strip(";|&") if words else ""
    return f"Bash:{_wire_safe(head)}" if head else "Bash"


def _wire_safe(name):
    return name.replace("<", "(").replace(">", ")")


def _arrival_ms(r):
    """Epoch ms of the round's first timing event (user_message / tool_result)."""
    events = r.get("timing_events") or []
    stamp = events[0].get("timestamp") if events else None
    if not stamp:
        return None
    return int(datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() * 1000)


def load_sessions(path, stats):
    by_session = defaultdict(list)
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as fh:
        for line in fh:
            r = json.loads(line)
            stats["rounds_seen"] += 1
            if r.get("provider") == "claude":
                by_session[r["session_id"]].append(r)
    stats["claude_sessions"] = len(by_session)
    for rs in by_session.values():
        rs.sort(key=lambda r: r["round_index"])
    return by_session


def keep_rounds(rs, max_model_len, stats):
    """The rounds of one session that survive the filters, or [] to drop it."""
    if rs[0]["prefix_tokens"] > 0:
        stats["drop_round0_prefix"] += 1
        return []
    if any(b["input_tokens_total"] <= a["input_tokens_total"]
           for a, b in zip(rs, rs[1:])):
        stats["drop_not_increasing"] += 1
        return []
    if any(_arrival_ms(r) is None for r in rs):
        stats["drop_no_arrival"] += 1
        return []
    cut = next((i for i, r in enumerate(rs)
                if r["input_tokens_total"] + r["output_tokens"] > max_model_len),
               len(rs))
    if cut < len(rs):
        stats["cut_sessions"] += 1
        stats["cut_rounds"] += len(rs) - cut
    if cut == 0:
        stats["drop_round0_too_long"] += 1
    return rs[:cut]


def convert(by_session, max_model_len, stats):
    reqs = []
    next_id = 0
    for sid, rs in by_session.items():
        prev_input, prev_ids = 0, []
        for r in keep_rounds(rs, max_model_len, stats):
            if r["output_tokens"] == 0:
                stats["drop_output0"] += 1
                continue
            inp = r["input_tokens_total"]
            if r["round_index"] > 0:
                stats["prefix_raised" if r["prefix_tokens"] < prev_input else
                      "prefix_lowered" if r["prefix_tokens"] > prev_input else
                      "prefix_kept"] += 1
            ids = prev_ids[: prev_input // BLOCK]
            fresh = -(-inp // BLOCK) - len(ids)
            ids = ids + list(range(next_id, next_id + fresh))
            next_id += fresh
            reqs.append({
                "timestamp": _arrival_ms(r),
                "session_id": sid,
                "round_index": r["round_index"],
                "input_length": inp,
                "prefix_length": prev_input,
                "output_length": r["output_tokens"],
                "hash_ids": ids,
                "tool_calls": [tool_label(t) for t in r.get("tools") or []],
            })
            prev_input, prev_ids = inp, ids
    reqs.sort(key=lambda q: (q["timestamp"], q["session_id"], q["round_index"]))
    return reqs


def finalize(reqs):
    """Shift timestamps to start at 0 and renumber ids by first appearance."""
    t0 = reqs[0]["timestamp"] if reqs else 0
    new = {}
    for q in reqs:
        q["timestamp"] -= t0
        q["hash_ids"] = [new.setdefault(h, len(new)) for h in q["hash_ids"]]


def verify(reqs):
    """Unbounded-cache replay in file order; returns (mismatches, hit rate)."""
    seen = set()
    bad = hit = total = 0
    for q in reqs:
        full = q["input_length"] // BLOCK
        h = q["hash_ids"]
        k = 0
        while k < full and h[k] in seen:
            k += 1
        bad += k != q["prefix_length"] // BLOCK
        hit += k * BLOCK
        total += q["input_length"]
        seen.update(h[:full])
    return bad, hit / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="syfi_coding_trace.jsonl.gz")
    ap.add_argument("--out", default="tracelab_mooncake.jsonl")
    ap.add_argument("--max-model-len", type=int, default=262144,
                    help="服务端 --max-model-len；session 在第一次 input+output "
                         "超过它的那一轮截断（默认 262144，即未显式设置时的推导值）")
    ap.add_argument("--num-requests", type=int, default=2000,
                    help="只写按时间最早的 N 条，0 = 全部")
    args = ap.parse_args()

    s = Counter()
    reqs = convert(load_sessions(args.input, s), args.max_model_len, s)
    if args.num_requests:
        reqs = reqs[: args.num_requests]
    finalize(reqs)
    bad, hit_rate = verify(reqs)

    with open(args.out, "w") as fh:
        for q in reqs:
            fh.write(json.dumps(q) + "\n")

    sessions = len({q["session_id"] for q in reqs})
    span_h = reqs[-1]["timestamp"] / 3.6e6 if reqs else 0
    print(f"读入 round {s['rounds_seen']:,}   claude session {s['claude_sessions']:,}")
    print(f"  剔除 round 0 prefix>0 的 session  {s['drop_round0_prefix']:,}")
    print(f"  剔除 input 未严格递增的 session   {s['drop_not_increasing']:,}")
    print(f"  剔除 缺到达时间的 session         {s['drop_no_arrival']:,}")
    print(f"  超过 --max-model-len {args.max_model_len:,} 截断 "
          f"{s['cut_sessions']:,} 个 session，丢尾部 {s['cut_rounds']:,} 轮"
          f"（其中 round 0 就超长、整个丢掉的 {s['drop_round0_too_long']:,} 个）")
    print(f"  删掉 output=0 的请求              {s['drop_output0']:,}")
    print(f"N≥1 的 prefix 改为上轮 input：上调 {s['prefix_raised']:,} / "
          f"下调 {s['prefix_lowered']:,} / 本就相等 {s['prefix_kept']:,}")
    print(f"写出 {len(reqs):,} 条请求 / {sessions:,} 个 session，"
          f"时间跨度 {span_h:,.1f} 小时")
    print(f"无限缓存回放：命中率 {hit_rate:.1%}，"
          f"命中块数 != floor(prefix/{BLOCK}) 的 {bad} 条")
    print(f"-> {args.out}")
    if bad:
        sys.exit("hash_ids 与 prefix_length 不一致")


if __name__ == "__main__":
    main()
