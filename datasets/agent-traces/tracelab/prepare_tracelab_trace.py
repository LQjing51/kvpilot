#!/usr/bin/env python3
"""Flatten the TraceLab round log into one record per round.

TraceLab ships one JSON object per agent round, sanitized: token counts and tool
metadata survive, prompt text does not. That is enough for placement work, which
only ever needed lengths, and it means no tokenizer runs here.

Every round is kept, tool call or not. The output is meant to be a faithful
projection of the source trace, so a round that only carried text still occupies
its slot: dropping those would break `round_index` continuity and hide the shape
of the session from any analysis reading this file.

Each record holds only that round's own quantities:

    input  = input_tokens_total      the whole prompt
    prefix = prefix_tokens           the part served from cache
    append = newly_append_tokens     the part that must be prefilled
    output = output_tokens           what the model generated
    label  = the tool call type, or null when the round made no call

`input = prefix + append` holds exactly across the trace, so the three length
fields are consistent by construction. Nothing is copied between rounds:
relating round N's tool call to round N+1's append is a question about the
sequence, and it belongs to whoever asks it, not to this file. Mixing the two
here previously produced records pairing `prefix[N]` with `append[N+1]`, which
are quantities of different rounds and differ by `append[N]`.

Usage:
  python3 prepare_tracelab_trace.py [--input syfi_coding_trace.jsonl.gz]
                                    [--out tracelab_turns.jsonl]
                                    [--provider claude|codex]
"""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime


def rounds(path, limit=0):
    """Stream the trace. `.gz` is read as-is; no need to decompress first."""
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        for n, line in enumerate(fh):
            if limit and n >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def tool_label(tool):
    """The call type, at the granularity that predicts response length.

    TraceLab names the tool outright, so unlike the SWE-smith path there is no
    command line to parse. Bash is the exception worth splitting: it covers a
    large share of calls and the trace hands us a pre-parsed `command_skeleton`,
    so the leading executable comes for free.

    Angle brackets are rewritten to parentheses. TraceLab writes syntactic
    categories as `<assign>` / `<loops>` (5.5% of calls with a tool), and the
    label doubles as a wire token: the replay renders it into `<function=NAME>`
    and the online judge reads it back with `<function=([^>\\s]+)>`, whose `>`
    terminator would cut `Bash:<assign>` down to `Bash:<assign`. That silently
    yields a label absent from the lookup table, so the verdict falls back to
    the global median with nothing in the logs to show for it. Parentheses keep
    the "category, not a program" signal and survive the round trip.
    """
    name = tool.get("tool_name") or "?"
    if name != "Bash":
        return _wire_safe(name)
    skeleton = tool.get("command_skeleton") or ""
    head = skeleton.split()[0].strip(";|&") if skeleton.split() else ""
    return f"Bash:{_wire_safe(head)}" if head else "Bash"


def _wire_safe(name):
    """Make a label survive `<function=NAME>` rendering and re-parsing."""
    return name.replace("<", "(").replace(">", ")")


def _arrival_ms(r):
    """When this round's request reached the server, as epoch milliseconds.

    `timing_events` is ordered, and its first entry is what starts the round:
    `user_message` on round 0, `tool_result` afterwards. That instant is the
    round's arrival time — everything later in the list is the model answering.
    None when a round carries no events; the replay treats those as "no arrival
    time known" and falls back to issuing them in order.
    """
    events = r.get("timing_events") or []
    if not events:
        return None
    stamp = events[0].get("timestamp")
    if not stamp:
        return None
    # Stored as ISO-8601 with a trailing Z, which fromisoformat rejects before
    # Python 3.11.
    dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def convert(path, limit=0, span_percentile=90.0):
    """One record per round for claude sessions, inconsistent sessions dropped.

    Only claude: it is the sole provider carrying the cache breakdown these
    records are about (codex quantizes prefix to 128, which makes
    `prefix`/`append` describe its bucketing rather than the workload).

    Three filters, each discarding the **whole session**, not just the offending
    round or the tail after it. A session that breaks the prefix relation
    somewhere is not a growing conversation the way the replay assumes, and
    keeping its head would silently mix two populations — the clean sessions and
    the prefixes of broken ones — in every distribution measured from this file:

      * `input[N] < input[N-1]` — the prompt shrank, so context was trimmed or
        reset partway through.
      * `prefix[N] > input[N-1]` — more cache hit than the previous prompt even
        contained, which cannot describe a prefix of it.
      * session span above `--span-percentile` — the long tail is someone
        leaving the terminal open for days; those gaps say nothing about serving
        and would dominate any replay schedule.
    """
    kept_rounds = []
    stats = Counter()
    by_session = defaultdict(list)

    for r in rounds(path, limit):
        stats["rounds_seen"] += 1
        if r.get("provider") != "claude":
            stats["dropped_not_claude"] += 1
            continue
        by_session[r["session_id"]].append(r)
    stats["sessions_seen"] = len(by_session)

    # Pass 1: drop any session with an inconsistent round anywhere in it.
    consistent = {}
    for sid, rs in by_session.items():
        rs.sort(key=lambda r: r["round_index"])
        reason = None
        for i in range(1, len(rs)):
            prev_input = rs[i - 1]["input_tokens_total"]
            if rs[i]["input_tokens_total"] < prev_input:
                reason = "input_shrank"
                break
            if rs[i]["prefix_tokens"] > prev_input:
                reason = "prefix_exceeds_prev_input"
                break
        if reason is not None:
            stats[f"dropped_rounds_{reason}"] += len(rs)
            stats[f"dropped_sessions_{reason}"] += 1
            continue
        consistent[sid] = rs

    # Pass 2: drop the long-span tail, measured over the sessions that survived
    # pass 1 so the percentile describes the population actually being kept.
    spans = {}
    for sid, rs in consistent.items():
        stamps = [s for s in (_arrival_ms(r) for r in rs) if s is not None]
        spans[sid] = (max(stamps) - min(stamps)) / 1000.0 if stamps else 0.0
    ordered = sorted(spans.values())
    if ordered:
        idx = min(len(ordered) - 1, int(span_percentile / 100.0 * (len(ordered) - 1)))
        span_cut = ordered[idx]
    else:
        span_cut = 0.0

    for sid, rs in consistent.items():
        if spans[sid] > span_cut:
            stats["dropped_rounds_long_span"] += len(rs)
            stats["dropped_sessions_long_span"] += 1
            continue
        for r in rs:
            tools = r.get("tools") or []
            labels = [tool_label(t) for t in tools]
            stats["kept_rounds"] += 1
            stats["with_tools" if tools else "without_tools"] += 1
            if len(tools) > 1:
                stats["multi_tool"] += 1
            arrival_ms = _arrival_ms(r)
            if arrival_ms is None:
                stats["no_arrival_ts"] += 1
            kept_rounds.append(
                {
                    "session_id": r["session_id"],
                    "round_index": r["round_index"],
                    "provider": r.get("provider"),
                    "model": r.get("model"),
                    "input": r["input_tokens_total"],
                    "prefix": r["prefix_tokens"],
                    "append": r["newly_append_tokens"],
                    "output": r["output_tokens"],
                    # Absolute epoch ms of this round's arrival. The replay turns
                    # it into a relative offset itself.
                    "arrival_ms": arrival_ms,
                    # Every call this round made, in order. There is no separate
                    # first-call field: it would be labels[0] by definition, and
                    # naming one call "the" label hides that 16% of rounds make
                    # several — whose responses all land in the next append, so
                    # the same leading tool reads very differently at one call
                    # versus three (exec_command: 530 vs 3,588 at the median).
                    "labels": labels,
                    "n_tools": len(labels),
                }
            )
    kept_rounds.sort(key=lambda t: (t["session_id"], t["round_index"]))
    stats["span_cut_sec"] = int(span_cut)
    return kept_rounds, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="syfi_coding_trace.jsonl.gz")
    ap.add_argument("--out", default="tracelab_turns.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部行")
    ap.add_argument("--span-percentile", type=float, default=90.0,
                    help="剔除跨度超过该分位的 session（默认 90）")
    args = ap.parse_args()

    turns, stats = convert(args.input, args.limit, args.span_percentile)
    with open(args.out, "w") as fh:
        for t in turns:
            fh.write(json.dumps(t) + "\n")

    sessions = len({t["session_id"] for t in turns})
    labels = len({l for t in turns for l in t["labels"]})
    by_prov = Counter(t["provider"] for t in turns)
    print(f"读入 round {stats['rounds_seen']:,}"
          f"   claude session {stats['sessions_seen']:,}")
    print(f"  剔除 非 claude                {stats['dropped_not_claude']:,} round")
    print(f"  剔除 input 缩减的 session      "
          f"{stats['dropped_sessions_input_shrank']:,} 个"
          f"（{stats['dropped_rounds_input_shrank']:,} round）")
    print(f"  剔除 prefix>上轮 input 的       "
          f"{stats['dropped_sessions_prefix_exceeds_prev_input']:,} 个"
          f"（{stats['dropped_rounds_prefix_exceeds_prev_input']:,} round）")
    print(f"  剔除 跨度 > p{args.span_percentile:g}（{stats['span_cut_sec']:,}s）的  "
          f"{stats['dropped_sessions_long_span']:,} 个"
          f"（{stats['dropped_rounds_long_span']:,} round）")
    print(f"保留 session {sessions:,}   round {stats['kept_rounds']:,}"
          f"（带 tool call {stats['with_tools']:,} / "
          f"无 tool call {stats['without_tools']:,}）")
    print(f"provider {dict(by_prov)}   标签种类 {labels:,}")
    print(f"其中一轮发多个 call 的 {stats['multi_tool']:,}")
    print(f"每条记录: session_id round_index provider model "
          f"input prefix append output arrival_ms labels n_tools")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
