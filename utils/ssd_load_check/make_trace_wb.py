"""Trace for the SSD-hit write-back interference check (Mooncake replay format,
timestamps 0, replayed with -p 1 so each request starts when the last is done).

    python3 make_trace_wb.py [rounds] > trace_wb.jsonl

  probe-warm  Q (16K) once, so the probes find it in HBM
  per round i (ROUNDS):
    warm      P_i (32K): computed, stored to DRAM and SSD
    wb        round >= LAG: P_{i-LAG} + 1K suffix from SSD only ([4]). LAG warms
              later P_{i-LAG} is out of an 8 GiB-per-rank DRAM, so the SSD hit is
              written back (64 chunks x 48 layers, ~1.5 GiB per rank)
    probe 1-4 Q + 1K fresh suffix from HBM only ([1]): fixed shape, run while the
              write-back of the round's wb is in flight; rounds < LAG have no wb
              and give the baseline

Every prompt has its own hash ids (a fresh first block too), so nothing is shared
by accident.
"""
import json
import sys

BLOCK = 512
PREFIX, PROBE_PREFIX, SUFFIX = 32768, 16384, 1024
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
LAG, PROBES = 8, 4
_next = 1


def ids(n):
    global _next
    _next += n
    return list(range(_next - n, _next))


def row(kind, tokens, hids, rnd, k=0, tier=None, prefix=0):
    r = {"timestamp": 0, "input_length": tokens, "output_length": 1, "hash_ids": hids,
         "kind": kind, "round": rnd, "k": k}
    if tier:
        r["extra_body"] = {"cache_use_control": [tier], "cache_use_max_tokens": prefix}
    return r


def main():
    q = ids(PROBE_PREFIX // BLOCK)
    out = [row("probe-warm", PROBE_PREFIX, q, -1)]
    warm = []
    for i in range(ROUNDS):
        p = ids(PREFIX // BLOCK)
        warm.append(p)
        out.append(row("warm", PREFIX, p, i))
        if i >= LAG:
            out.append(row("wb", PREFIX + SUFFIX, warm[i - LAG] + ids(SUFFIX // BLOCK), i,
                           tier=4, prefix=PREFIX))
        for k in range(1, PROBES + 1):
            out.append(row("probe", PROBE_PREFIX + SUFFIX, q + ids(SUFFIX // BLOCK), i, k,
                           tier=1, prefix=PROBE_PREFIX))
    for r in out:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
