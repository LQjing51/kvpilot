"""Trace for blockwise layouts: an HBM head, then DRAM, then SSD.

Per (prefix p, suffix s, rep), on one prefill worker:
  warm   prompt = prefix            stores its KV: HBM, Mooncake DRAM, and SSD
                                    one heartbeat later (write-through)
  test   prompt = prefix + suffix:
         [1]   prefix from HBM: compute only
         split (H, D, X) of the prefix in quarters: the first H tokens from
               HBM, the next D from DRAM, the last X from SSD
               (cache_use_control over the non-empty spans)
Each test has its own suffix hash ids of the same length, so every test
computes and stores its suffix KV like a real request. The tests start GAP_MS
after the warm, so the SSD write has finished. mixed_model.py reads it.

    python3 make_trace_hbmmix.py > trace_hbmmix.jsonl
"""
import json

BLOCK = 512
PREFIX = [16384, 32768, 65536]
SUFFIX = [256, 1024, 2048]
# (HBM, DRAM, SSD) quarters of the prefix.
SPLITS = [
    (0, 4, 0), (0, 3, 1), (0, 2, 2), (0, 0, 4),
    (1, 3, 0), (1, 2, 1), (1, 1, 2), (1, 0, 3),
    (2, 2, 0), (2, 1, 1), (2, 0, 2),
    (3, 1, 0), (3, 0, 1),
]
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
            tests = [((p, 0, 0), [1], p)]
            for q in SPLITS:
                split = tuple(p // 4 * k for k in q)
                spans = [(tier, n) for tier, n in zip((1, 2, 4), split) if n]
                cap = [n for _, n in spans] if len(spans) > 1 else spans[0][1]
                tests.append((split, [a for a, _ in spans], cap))
            for split, tiers, cap in tests:
                print(json.dumps({**row, "timestamp": t, "input_length": p + s,
                                  "hash_ids": pre + ids(-(-s // BLOCK)), "tier": tiers,
                                  "split": list(split),
                                  "extra_body": {"cache_use_control": tiers,
                                                 "cache_use_max_tokens": cap}}))
                t += TEST_GAP_MS
