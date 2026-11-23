"""Add next_suffix to a signal table: the session's next turn's suffix in
tokens, its prompt past the full blocks it shares with this turn (the prefix
the blocks this turn leaves behind can serve). Blockwise placement uses it as
a one-point suffix distribution (vllm/v1/core/kvpilot_signal.py).

    python3 add_next_suffix.py <trace.jsonl> <signal table.json>

The table is rewritten in place; rows are keyed by replay line ("line" field,
else row index), as make_req_table.py writes them.
"""
import json
import sys
from collections import defaultdict

BLOCK = 512


def main(trace: str, table_path: str) -> None:
    rows = [json.loads(x) for x in open(trace)]
    table = json.load(open(table_path))
    by_sess = defaultdict(list)
    for i, r in enumerate(rows):
        by_sess[r["session_id"]].append((r["round_index"], int(r.get("line", i)), r))
    added = 0
    for turns in by_sess.values():
        turns.sort(key=lambda t: t[0])
        for (_, line, cur), (_, _, nxt) in zip(turns, turns[1:]):
            full = cur["hash_ids"][: cur["input_length"] // BLOCK]
            shared = 0
            while shared < len(full) and shared < len(nxt["hash_ids"]) \
                    and full[shared] == nxt["hash_ids"][shared]:
                shared += 1
            table[str(line)]["next_suffix"] = nxt["input_length"] - shared * BLOCK
            added += 1
        table[str(turns[-1][1])]["next_suffix"] = None
    json.dump(table, open(table_path, "w"))
    print(f"{table_path}: next_suffix on {added} rows")


if __name__ == "__main__":
    main(*sys.argv[1:3])
