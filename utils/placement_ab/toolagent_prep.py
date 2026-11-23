"""Mooncake toolagent trace -> replayable A/B trace + request-level tier table.

    python3 toolagent_prep.py [--lines N] [--out-trace ...] [--out-table ...]

The toolagent trace has no session ids. A request that starts with every full
block of an earlier request extends it: that is the earlier request's next turn
(each request has at most one). Chains of such turns are sessions. Other prefix
sharing (a common system prompt, or part of a prompt) stays in hash_ids and is
reused by the cache as usual, but is not a next turn.

Each request's request-safe tier and its t1 / t2 / t4 (prefill e2e with the
prefix in HBM / DRAM / SSD) are estimated from (prefix, suffix) with
datasets/req-priority-trace/gen_trace.py's Estimator: the nearest SWE
request-safe measurements (the times depend on hardware and model, not the
dataset), the fitted time model outside their range. prefix is the full blocks
an earlier request of the file already had.

Writes the first --lines requests in file (= timestamp) order with session_id,
round_index, prefix_length and src_line added, and the signal table run_ab.sh /
analyze_excl.py read, as make_req_table.py writes it: line -> {"own_tier",
"next_tier", "next_time": next turn's [t1, t2, t4] in ms} (None = no next turn
within the file, or no prefix).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "datasets/req-priority-trace")]
import gen_trace  # noqa: E402

B = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "datasets/mooncake-trace/toolagent_trace.jsonl")
    ap.add_argument("--lines", type=int, default=0, help="keep the first N requests (0 = all)")
    ap.add_argument("--out-trace", type=Path,
                    default=ROOT / "datasets/mooncake-trace/toolagent_sessions.jsonl")
    ap.add_argument("--out-table", type=Path,
                    default=Path(__file__).resolve().parent / "signal_toolagent_sessions.json")
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(a.src)]
    if a.lines:
        rows = rows[: a.lines]
    est = gen_trace.Estimator()
    trie, last_full = {}, {}
    sess, rnd, nxt = [None] * len(rows), [0] * len(rows), [None] * len(rows)
    prefix, tier, times, n_sess = [0] * len(rows), [None] * len(rows), [None] * len(rows), 0
    for i, r in enumerate(rows):
        full = tuple(r["hash_ids"][: r["input_length"] // B])
        node, hit = (), 0
        while hit < len(full) and node + (full[hit],) in trie:
            node += (full[hit],)
            hit += 1
        parent = last_full.get(full[:hit]) if hit > 1 else None
        if parent is not None and nxt[parent] is None:
            nxt[parent] = i
            sess[i], rnd[i] = sess[parent], rnd[parent] + 1
        else:
            sess[i], n_sess = f"ta{n_sess:05d}", n_sess + 1
        node = ()
        for b in full:
            node += (b,)
            trie[node] = True
        if full:
            last_full[full] = i
        n = r["input_length"]
        prefix[i] = min(hit, (n - 1) // B) * B
        if prefix[i]:
            tier[i], t1, t2, t4 = est(prefix[i], n - prefix[i])
            times[i] = [round(t1, 3), round(t2, 3), round(t4, 3)]
    table = {
        str(i): {"own_tier": tier[i],
                 "next_tier": tier[nxt[i]] if nxt[i] is not None else None,
                 "next_time": times[nxt[i]] if nxt[i] is not None else None}
        for i in range(len(rows))
    }
    with open(a.out_trace, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({
                "timestamp": r["timestamp"], "session_id": sess[i], "round_index": rnd[i],
                "input_length": r["input_length"], "prefix_length": prefix[i],
                "output_length": r["output_length"], "hash_ids": r["hash_ids"],
                "src_line": i,
            }) + "\n")
    a.out_table.write_text(json.dumps(table))
    from collections import Counter
    print(f"{len(rows)} requests, {n_sess} sessions; own tier "
          f"{dict(Counter(tier))}; next-turn tier {dict(Counter(v['next_tier'] for v in table.values()))}")
    print(f"wrote {a.out_trace} and {a.out_table}")


if __name__ == "__main__":
    main()
