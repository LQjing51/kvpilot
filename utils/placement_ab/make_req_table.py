"""Offline signal table for KVPILOT_OFFLINE_TABLE (vllm/v1/core/kvpilot_signal.py).

    python3 make_req_table.py <trace.jsonl> <request-safe raw dir> <out.json>
    python3 make_req_table.py \
        ../../datasets/agent-traces/swe-smith-trajectories/swe_hbm1to2_budget220.jsonl \
        raw/swe_hbm1to2 signal_swe_hbm1to2_budget220.json

Named signal_<trace basename>.json (run_ab.sh finds it so). Keyed by replay
line: the trace's "line" field, else the row index, which is how
mooncake_trace_replay numbers request ids. Per line:

    own_tier   the request's own measured request-safe tier (HBM / DRAM /
               SSD), for scoring where its KV should have come from
    next_tier  the measured request-safe tier of the next turn of the same
               session, null when the session ends there: the signal's tier
    next_time  that next turn's measured prefill e2e [t1, t2, t4] (ms) with
               its prefix in HBM / DRAM / SSD: the signal's time The raw dir is the
request-safe run (under paper-figure/Figure3-request-safe) whose numbering
the trace's src_line follows.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dram_miss_sim import FIG3, measured


def main(trace: str, raw: str, out: str) -> None:
    rows = [json.loads(x) for x in open(trace)]
    meas = measured(FIG3 / raw)
    tier, times = {}, {}
    by_sess = defaultdict(list)
    for i, r in enumerate(rows):
        line = int(r.get("line", i))
        m = meas.get(r["src_line"])
        tier[line] = m[2] if m else None
        if m:
            times[line] = [round(m[3][k], 3) for k in ("t1", "t2", "t4")]
        by_sess[r["session_id"]].append((r["round_index"], line))
    table = {}
    for turns in by_sess.values():
        turns.sort()
        for (_, line), later in zip(turns, turns[1:] + [None]):
            table[str(line)] = {"own_tier": tier[line],
                                "next_tier": tier[later[1]] if later else None,
                                "next_time": times.get(later[1]) if later else None}
    Path(out).write_text(json.dumps(table, sort_keys=True) + "\n")
    print(f"{len(table)} lines -> {out}")
    print("own tier:", dict(Counter(v["own_tier"] for v in table.values())))
    print("next tier:", dict(Counter(v["next_tier"] for v in table.values())))


if __name__ == "__main__":
    main(*sys.argv[1:4])
