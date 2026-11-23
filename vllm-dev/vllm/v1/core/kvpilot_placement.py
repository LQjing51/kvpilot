# SPDX-License-Identifier: Apache-2.0
"""KVPilot placement strategies (KVPILOT_PLACEMENT) and the signal they use.

    ""                          LRU baseline: upstream eviction, no signal.
    belady                      Copies and writes as LRU, no signal; HBM and
                                DRAM evict the block whose next use in the
                                replayed trace is farthest (Belady's MIN,
                                kvpilot_belady).
    inclusive-safe-reqwise      Write-through (HBM, DRAM and SSD each hold a
                                copy). HBM and DRAM evict by the safe tier of
                                the next turn that needs a block: no next turn
                                first, then SSD, DRAM, HBM (= signal pending);
                                FIFO / LRU within a class.
    inclusive-penalty-reqwise   Write-through, but HBM and DRAM evict by the
                                request's penalty in that tier (smallest
                                first; no next turn 0, pending last).
    exclusive-penalty-reqwise   One copy per block. A prefill writes nothing
                                below HBM; a signal moves the blocks to their
                                safe tier. A tier that fills up demotes the
                                blocks whose move down adds the least exposed
                                load (kvpilot_exclusive).
    exclusive-penalty-reqwise-unblocking
                                Exclusive, without waiting for HBM: one
                                prefill chunk of HBM is kept free for the
                                computed suffix, and the prefix an admission
                                loads borrows held blocks rather than eat
                                into it, their KV written to DRAM within
                                that load.
    exclusive-penalty-blockwise-unblocking
                                Unblocking, but a turn's blocks are split
                                across tiers: the time model gives the fewest
                                HBM, then DRAM blocks that keep the next turn
                                safe (head HBM, then DRAM, tail SSD), and each
                                block's penalty is the expected TTFT its move
                                down adds (kvpilot_blockwise).

Every strategy evicts a request's blocks from its tail in HBM and DRAM, so a
hit reads HBM, then DRAM, then SSD.

The decode side sends one signal per finished turn: the safe tier of the
session's next turn (None = no next turn), that turn's prefill time with its
prefix read from HBM / DRAM / SSD in ms (None with no next turn), and its
suffix length distribution [[probability, tokens], ...].

Every strategy but the baselines (MODEL) takes only "is there a next turn" from the
signal's tier and derives the rest on the prefill side from the suffix
distribution and the time model (kvpilot_time_model, kvpilot_blockwise), the
same way at either granularity: a request's tier is the lowest one its whole
prefix is safe in at the suffix 95% of the distribution reaches, a penalty
the expected prefill time a move one tier down adds.

Shared by the prefill and decode sides, so the two never disagree.
"""

from __future__ import annotations

import os

LRU = ""
INCLUSIVE = "inclusive-safe-reqwise"
EXCLUSIVE = "exclusive-penalty-reqwise"
EXCLUSIVE_UNBLOCKING = "exclusive-penalty-reqwise-unblocking"
EXCLUSIVE_BLOCKWISE = "exclusive-penalty-blockwise-unblocking"
INCLUSIVE_PENALTY = "inclusive-penalty-reqwise"
BELADY = "belady"
# Signalled by the decode side, tier and penalties from the time model
# (KVPILOT_TIME_COEFFS_PATH).
MODEL = (INCLUSIVE, INCLUSIVE_PENALTY, EXCLUSIVE, EXCLUSIVE_UNBLOCKING,
         EXCLUSIVE_BLOCKWISE)
ALL = (LRU, BELADY) + MODEL

TIERS = ("HBM", "DRAM", "SSD")

# Inclusive eviction class of a block by the fastest tier any live session
# still needs it in, smaller evicted first. A pending signal counts as HBM.
CLASS = {None: 1, "SSD": 2, "DRAM": 3, "HBM": 4}
PENDING_CLASS = CLASS["HBM"]
# inclusive-penalty-reqwise keys free HBM blocks by penalty (ms): a pending
# signal sorts above any.
PENDING_PENALTY = 1e8


def strategy() -> str:
    """The configured strategy; an unknown value is a configuration error."""
    name = os.environ.get("KVPILOT_PLACEMENT", "")
    if name not in ALL:
        raise ValueError(
            f"KVPILOT_PLACEMENT={name!r}: use '' (LRU) or one of {ALL[1:]}"
        )
    return name


def pending_key(name: str) -> float:
    """Free-queue key of a freed block still waiting for its signal."""
    return PENDING_PENALTY if name == INCLUSIVE_PENALTY else float(PENDING_CLASS)


def penalties(time: list[float]) -> tuple[float, float]:
    """Exposed load (ms) the next turn gains when its prefix moves down one
    tier: (HBM -> DRAM, DRAM -> SSD)."""
    t_hbm, t_dram, t_ssd = time
    return max(0.0, t_dram - t_hbm), max(0.0, t_ssd - t_dram)
