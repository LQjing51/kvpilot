"""Add next_suffix to a signal table: the session's next turn's suffix in
tokens, its prompt past the full blocks it shares with this turn (the prefix
the blocks this turn leaves behind can serve). Blockwise placement uses it as
a one-point suffix distribution (vllm/v1/core/kvpilot_signal.py).

    python3 add_next_suffix.py <trace.jsonl> <signal table.json> [--block 1536 --drops-last]

--block is the vLLM block size (512 for Qwen3-30B-A3B, 1536 for Qwen3.6 at
TP 2); a trace hash id is always 512 tokens. --drops-last: the prefill side
computes a turn but its last token (hybrid models under P/D), so the turn
leaves blocks, and its state, up to its prompt less one. The table is
rewritten in place; rows are keyed by replay line ("line" field, else row
index), as make_req_table.py writes them.
"""
import argparse
import json
from collections import defaultdict

TRACE_BLOCK = 512


def main(trace: str, table_path: str, block: int, drops_last: bool) -> None:
    if block % TRACE_BLOCK:
        raise SystemExit(f"--block {block} is not whole trace hashes ({TRACE_BLOCK})")
    rows = [json.loads(x) for x in open(trace)]
    table = json.load(open(table_path))
    by_sess = defaultdict(list)
    for i, r in enumerate(rows):
        by_sess[r["session_id"]].append((r["round_index"], int(r.get("line", i)), r))
    added = 0
    for turns in by_sess.values():
        turns.sort(key=lambda t: t[0])
        for (_, line, cur), (_, _, nxt) in zip(turns, turns[1:]):
            full = cur["hash_ids"][: cur["input_length"] // TRACE_BLOCK]
            shared = 0
            while shared < len(full) and shared < len(nxt["hash_ids"]) \
                    and full[shared] == nxt["hash_ids"][shared]:
                shared += 1
            computed = cur["input_length"] - drops_last
            tokens = min(shared * TRACE_BLOCK, computed) // block * block
            table[str(line)]["next_suffix"] = nxt["input_length"] - tokens
            added += 1
        table[str(turns[-1][1])]["next_suffix"] = None
    json.dump(table, open(table_path, "w"))
    print(f"{table_path}: next_suffix on {added} rows (block {block})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("table")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--drops-last", action="store_true")
    a = ap.parse_args()
    main(a.trace, a.table, a.block, a.drops_last)
