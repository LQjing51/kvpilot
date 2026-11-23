"""Trace for the cross-worker SSD check: 2 prefill workers, A and B.

Per (prefix, suffix, rep):
  warm   prompt = prefix, on A          stores its KV in A's HBM, DRAM and SSD
  test   prompt = prefix + suffix,      cache_use_max_tokens = prefix:
         [1] on A    prefix from A's HBM: compute only
         [4] on A    prefix from A's own SSD files
         [4] on B    prefix from A's SSD files, read by B
         [2] on B    prefix from A's DRAM, mapped into B (shm)
Each test has its own suffix hash ids of the same length, so every test
computes and stores its suffix KV like a real request.
Rows carry "worker": "A" / "B"; run_xworker.sh fills in the prefill worker ids
(nvext.prefill_worker_id) once the service is up.

    python3 make_trace_xworker.py > trace_xworker.jsonl
"""
import json

BLOCK = 512
PREFIX = [4096, 16384, 65536]
SUFFIX = [64, 1024, 4096]
TESTS = [(1, "A"), (4, "A"), (4, "B"), (2, "B")]
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
                              "tier": 0, "worker": "A"}))
            t += GAP_MS
            for tier, worker in TESTS:
                print(json.dumps({**row, "timestamp": t, "input_length": p + s,
                                  "hash_ids": pre + ids(-(-s // BLOCK)), "tier": tier,
                                  "worker": worker,
                                  "extra_body": {"cache_use_control": [tier],
                                                 "cache_use_max_tokens": p}}))
                t += TEST_GAP_MS
