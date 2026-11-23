"""Trace for the end-to-end SSD load check (Mooncake replay format).

Per (prefix, suffix, rep), on one prefill worker:
  warm   prompt = prefix            stores its KV: HBM, Mooncake DRAM, and SSD
                                    one heartbeat later (write-through)
  test   prompt = prefix + suffix,  cache_use_max_tokens = prefix, three times:
         cache_use_control [1]      prefix from HBM: suffix compute only
                           [2]      prefix from Mooncake DRAM only
                           [4]      prefix from SSD only, though DRAM has it
Each test has its own suffix hash ids of the same length, so all three
compute and store their suffix KV like a real request; with a shared suffix
only the first test would store it and the others would skip the write.
The tests start GAP_MS after the warm, so the SSD write (heartbeat + bucket
write + fdatasync) has finished. Each (prefix, rep) has its own hash ids.

    python3 make_trace.py > trace.jsonl                          # suffix 64/256/1024
    python3 make_trace.py 1536,2048,4096,8192 > trace_long.jsonl  # longer suffixes
"""
import json
import sys

BLOCK = 512
PREFIX = [4096, 8192, 16384, 32768, 65536]
SUFFIX = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [64, 256, 1024]
TIERS = [1, 2, 4]
REPS = 3
GAP_MS = 10000
TEST_GAP_MS = 2000

t, h = 0, 1


def ids(n):
    global h
    h += n
    return list(range(h - n, h))


for rep in range(REPS):
    for p in PREFIX:
        for s in SUFFIX:
            pre = ids(p // BLOCK)
            row = {"output_length": 1, "prefix": p, "suffix": s, "rep": rep}
            print(json.dumps({**row, "timestamp": t, "input_length": p, "hash_ids": pre,
                              "tier": 0}))
            t += GAP_MS
            for tier in TIERS:
                print(json.dumps({**row, "timestamp": t, "input_length": p + s,
                                  "hash_ids": pre + ids(-(-s // BLOCK)), "tier": tier,
                                  "extra_body": {"cache_use_control": [tier],
                                                 "cache_use_max_tokens": p}}))
                t += TEST_GAP_MS
