"""Trace for gpu_stall vs e2e delay under mixed DRAM + SSD hits.

Per (prefix, suffix, rep), on one prefill worker:
  warm   prompt = prefix            stores its KV: HBM, Mooncake DRAM, and SSD
                                    one heartbeat later (write-through)
  test   prompt = prefix + suffix:
         [1]        prefix from HBM, cache_use_max_tokens = prefix: compute only
         f = 0      [2], prefix from DRAM
         0 < f < 1  [2, 4] with spans [prefix * (1 - f), prefix * f]: the
                    first part from DRAM, the rest from SSD
         f = 1      [4], prefix from SSD, though DRAM has it
Each test has its own suffix hash ids of the same length, so every test
computes and stores its suffix KV like a real request. The tests start GAP_MS
after the warm, so the SSD write has finished.

    python3 make_trace_mixed.py > trace_mixed.jsonl
"""
import json
import sys

BLOCK = 512
PREFIX = [4096, 8192, 16384, 32768, 65536]
SUFFIX = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [64, 256, 1024, 4096]
SSD_FRAC = [0, 0.25, 0.5, 0.75, 1]
REPS = 3
GAP_MS = 10000
TEST_GAP_MS = 2000

t, h = 0, 1


def ids(n):
    global h
    h += n
    return list(range(h - n, h))


def control(p, f):
    if f == 0:
        return [2], p
    if f == 1:
        return [4], p
    ssd = int(p * f)
    return [2, 4], [p - ssd, ssd]


for rep in range(REPS):
    for p in PREFIX:
        for s in SUFFIX:
            pre = ids(p // BLOCK)
            row = {"output_length": 1, "prefix": p, "suffix": s, "rep": rep}
            print(json.dumps({**row, "timestamp": t, "input_length": p, "hash_ids": pre,
                              "tier": 0}))
            t += GAP_MS
            tests = [([1], p, None)] + [(*control(p, f), f) for f in SSD_FRAC]
            for tiers, cap, f in tests:
                print(json.dumps({**row, "timestamp": t, "input_length": p + s,
                                  "hash_ids": pre + ids(-(-s // BLOCK)), "tier": tiers,
                                  "ssd_frac": f,
                                  "extra_body": {"cache_use_control": tiers,
                                                 "cache_use_max_tokens": cap}}))
                t += TEST_GAP_MS
