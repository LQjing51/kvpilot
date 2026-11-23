"""Offline HBM / DRAM / SSD hit model for a Mooncake-format trace in send order.

    python3 tier_sim.py <trace.jsonl> [--hbm 2361] [--dram-gib 100]

One 1P worker with --max-num-seqs 1 runs prefills one at a time, so the cache
state evolves in send order. Per request, over its full 512-token blocks:

  HBM   vLLM prefix cache: the leading blocks still in the free-block LRU. A
        finished request frees its blocks last-block-first, so a prefix falls
        out of HBM from its tail.
  DRAM  Mooncake: LRU on lease time. PutEnd grants a lease and so does every
        LMCache lookup (`_lookup_views` queries every chunk of the prompt,
        including the part HBM covers), so each request refreshes its whole
        prefix. The master evicts the oldest 10% of objects once usage passes
        90% (dynamo_script.sh: -eviction_high_watermark_ratio=0.9,
        -eviction_ratio=0.1).
  SSD   write-through of every put, never evicted here (4 TiB per TP rank).
        A chunk read from SSD is not written back to DRAM.

The LMCache hit is the leading run of chunks found in DRAM or SSD; the part
beyond HBM loads from DRAM where it can and from SSD otherwise. Newly computed
full chunks are put into DRAM and SSD.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict

BLOCK = 512
# 1P1D TP2, GPU_MEM_UTIL=0.9: "GPU KV cache size: 1,208,832 tokens" = 2361
# blocks. The whole pool fits the measured HBM hits (order-swe-end, first 1360
# lines: 88.2% measured, 89.3% at 2361, 84.5% at 2200).
HBM_BLOCKS = 2361
# Mooncake global_segment_size per TP rank (run.sh DRAM_GIB).
DRAM_GIB = 100
HIGH, EVICT = 0.9, 0.1


def dram_blocks(gib_per_rank: float) -> int:
    """Two TP ranks each hold half of a 512-token block (48 MiB for Qwen3-30B-A3B)."""
    return int(2 * gib_per_rank * 1024 // 48)


DRAM_BLOCKS = dram_blocks(DRAM_GIB)


def simulate(rows: list[dict], hbm_cap: int = HBM_BLOCKS, dram_cap: int = DRAM_BLOCKS):
    """Per-request (hbm, dram, ssd, new) token counts, in the order given."""
    hbm: OrderedDict = OrderedDict()
    dram: OrderedDict = OrderedDict()
    ssd: set = set()
    out = []
    for r in rows:
        n_in = r["input_length"]
        full = r["hash_ids"][: n_in // BLOCK]
        k = 0
        while k < len(full) and full[k] in hbm:
            k += 1
        # vLLM always computes at least the last token.
        if k * BLOCK >= n_in:
            k -= 1
        hit = 0
        while hit < len(full) and (full[hit] in dram or full[hit] in ssd):
            hit += 1
        n_dram = n_ssd = 0
        for b in full[k:hit]:
            if b in dram:
                n_dram += 1
            else:
                n_ssd += 1
        for b in full:
            if b in dram:
                dram.move_to_end(b)
        for b in full[max(k, hit):]:
            dram[b] = None
            ssd.add(b)
        if len(dram) > HIGH * dram_cap:
            for _ in range(int(EVICT * len(dram))):
                dram.popitem(last=False)
        for b in reversed(full):
            hbm[b] = None
            hbm.move_to_end(b)
        while len(hbm) > hbm_cap:
            hbm.popitem(last=False)
        n_hbm, n_dram, n_ssd = k * BLOCK, n_dram * BLOCK, n_ssd * BLOCK
        out.append((n_hbm, n_dram, n_ssd, n_in - n_hbm - n_dram - n_ssd))
    return out


def summarize(res) -> dict:
    tot = sum(sum(x) for x in res)
    s = [sum(x[i] for x in res) for i in range(4)]
    reqs = len(res)
    return {
        "requests": reqs,
        "tokens": tot,
        "hbm": s[0] / tot,
        "dram": s[1] / tot,
        "ssd": s[2] / tot,
        "new": s[3] / tot,
        "req_with_dram": sum(x[1] > 0 for x in res) / reqs,
        "req_with_ssd": sum(x[2] > 0 for x in res) / reqs,
    }


def fmt(s: dict) -> str:
    return (f"tokens {s['tokens']:,}: HBM {s['hbm']:.1%} DRAM {s['dram']:.1%} "
            f"SSD {s['ssd']:.1%} new {s['new']:.1%} | requests with DRAM load "
            f"{s['req_with_dram']:.1%}, with SSD load {s['req_with_ssd']:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--hbm", type=int, default=HBM_BLOCKS)
    ap.add_argument("--dram-gib", type=float, default=DRAM_GIB)
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(a.trace)]
    print(fmt(summarize(simulate(rows, a.hbm, dram_blocks(a.dram_gib)))))


if __name__ == "__main__":
    main()
