"""Prefix x suffix grid on 2P2D for the DRAM-safe / SSD-safe suffix boundary.

Per (prefix, suffix, rep):
  warm  prompt = prefix, on A               KV into A's HBM, DRAM and SSD
  test  prompt = prefix + suffix,           cache_use_max_tokens = prefix:
        [1] on A   prefix from A's HBM      -> compute
        [2] on B   prefix from DRAM         -> DRAM e2e
        [4] on B   prefix from SSD          -> SSD e2e
Each test has its own suffix hash ids, so all three store their suffix KV like
a real request. Rows carry "worker"; raw/run.sh fills in
nvext.prefill_worker_id once the service is up.

The prefixes are split into groups, one fresh service per group, so A's
Mooncake DRAM segment (200 GiB per TP rank) never fills: once it does, A's KV
spills into B's segments and part of the prefix is not on SSD when [4] runs.
A stores about 49 KB per token per rank, so a group holds at most ~84K of
prefix x len(SUFFIX) x REPS.

    python3 make_trace.py        # writes raw/g<k>/trace.jsonl
"""
import json
from pathlib import Path

BLOCK = 512
GROUPS = [[65536, 12288, 8192], [61440, 16384], [57344, 20480], [53248, 24576],
          [49152, 28672], [45056, 32768], [40960, 36864]]
SUFFIX = [128, 192, 256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072]
TESTS = [(1, "A"), (2, "B"), (4, "B")]
REPS = 3
GAP_MS = 10000
TEST_GAP_MS = 2000

RAW = Path(__file__).resolve().parent / "raw"
h = 1


def ids(n):
    global h
    h += n
    return list(range(h - n, h))


for g, prefixes in enumerate(GROUPS):
    (RAW / f"g{g}").mkdir(parents=True, exist_ok=True)
    t = 0
    with open(RAW / f"g{g}" / "trace.jsonl", "w") as out:
        for rep in range(REPS):
            for p in prefixes:
                for s in SUFFIX:
                    pre = ids(p // BLOCK)
                    row = {"output_length": 1, "prefix": p, "suffix": s, "rep": rep}
                    print(json.dumps({**row, "timestamp": t, "input_length": p,
                                      "hash_ids": pre, "tier": 0, "worker": "A"}), file=out)
                    t += GAP_MS
                    for tier, worker in TESTS:
                        print(json.dumps({**row, "timestamp": t, "input_length": p + s,
                                          "hash_ids": pre + ids(-(-s // BLOCK)),
                                          "tier": tier, "worker": worker,
                                          "extra_body": {"cache_use_control": [tier],
                                                         "cache_use_max_tokens": p}}),
                              file=out)
                        t += TEST_GAP_MS
    print(f"g{g}: prefixes {[p // 1024 for p in prefixes]}K, {t / 60000:.0f} min")
