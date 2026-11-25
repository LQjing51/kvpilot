"""Signal table for KVPILOT_OFFLINE_TABLE from the time model instead of a
request-safe run (make_req_table.py): per line, the session's next turn's
suffix past the blocks this turn leaves (as add_next_suffix.py), and its tier
and [t_hbm, t_dram, t_ssd] with that prefix in each tier
(kvpilot_blockwise.reqwise, the rule the P side applies to a signal).

    python3 make_model_table.py <trace.jsonl> <fit_times2 json> <out.json> [--block 1024 --drops-last]

Same keys as make_req_table.py (own_tier: the tier this line gets as its
previous turn's next turn) plus next_suffix. A next turn sharing no full
block with this one has nothing to place: all None, as at a session's end.
"""
import argparse
import importlib.util
import json
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

CORE = Path(__file__).resolve().parents[2] / "vllm-dev/vllm/v1/core"
TRACE_BLOCK = 512


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for _pkg in ("vllm", "vllm.v1", "vllm.v1.core"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
tm = _load("vllm.v1.core.kvpilot_time_model", CORE / "kvpilot_time_model.py")
bw = _load("vllm.v1.core.kvpilot_blockwise", CORE / "kvpilot_blockwise.py")


def main(trace: str, coeffs: str, out: str, block: int, drops_last: bool) -> None:
    if block % TRACE_BLOCK:
        raise SystemExit(f"--block {block} is not whole trace hashes ({TRACE_BLOCK})")
    model = tm.TimeModel.load(coeffs, block)
    rows = [json.loads(x) for x in open(trace)]
    by_sess = defaultdict(list)
    for i, r in enumerate(rows):
        by_sess[r["session_id"]].append((r["round_index"], int(r.get("line", i)), r))
    table = {}
    for turns in by_sess.values():
        turns.sort(key=lambda t: t[0])
        own = None
        for (_, line, cur), later in zip(turns, turns[1:] + [None]):
            row = {"own_tier": own, "next_tier": None, "next_time": None, "next_suffix": None}
            own = None
            if later is not None:
                nxt = later[2]
                full = cur["hash_ids"][: cur["input_length"] // TRACE_BLOCK]
                shared = 0
                while shared < len(full) and shared < len(nxt["hash_ids"]) \
                        and full[shared] == nxt["hash_ids"][shared]:
                    shared += 1
                tokens = min(shared * TRACE_BLOCK, cur["input_length"] - drops_last) // block * block
                if tokens:
                    suffix = nxt["input_length"] - tokens
                    tier, time = bw.reqwise(model, tokens // block, [[1.0, suffix]])
                    row["next_tier"], row["next_time"] = tier, [round(t, 3) for t in time]
                    row["next_suffix"], own = suffix, tier
            table[str(line)] = row
    Path(out).write_text(json.dumps(table, sort_keys=True) + "\n")
    print(f"{len(table)} lines -> {out}")
    print("next tier:", dict(Counter(v["next_tier"] for v in table.values())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("coeffs")
    ap.add_argument("out")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--drops-last", action="store_true")
    a = ap.parse_args()
    main(a.trace, a.coeffs, a.out, a.block, a.drops_last)
