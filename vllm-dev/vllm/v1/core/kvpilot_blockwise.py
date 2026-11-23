# SPDX-License-Identifier: Apache-2.0
"""KVPilot exclusive-penalty-blockwise-unblocking: where each block of a turn
goes, and what moving one down costs, from the time model.

A turn's n prompt blocks are the next turn's prefix: its head h blocks in HBM,
then d in DRAM, then x in SSD (a hit reads a prefix of HBM, then DRAM, then
SSD, as every tier evicts a request from its tail). Its prefill time for a
next-turn suffix s is the time model's (kvpilot_time_model).

  * Placement: safe at the suffix that 95% of the distribution reaches (the
    largest of its values s with P(S >= s) >= 95%): exposed load under
    `thresh` of C. The fewest HBM blocks with all the rest in DRAM, then the
    most SSD blocks with that HBM.
  * Penalty: a block's move one tier down adds this much expected e2e (over
    the whole distribution), with the blocks behind it in that tier already
    moved: tiers give up a request from its tail.
  * reqwise: the same for the reqwise penalty strategies, a whole request in
    one tier.
"""

from __future__ import annotations

from vllm.v1.core.kvpilot_time_model import TimeModel

# Placement must be safe for this share of the next turn's suffixes.
QUANTILE = 0.95


def safe_suffix(dist: list) -> float:
    """The largest suffix of `dist` ([[probability, tokens], ...]) that at
    least QUANTILE of it reaches."""
    pts = sorted(dist, key=lambda x: x[1])
    total = sum(p for p, _ in pts)
    above, best = total, pts[0][1]
    for p, s in pts:
        if above >= QUANTILE * total - 1e-12:
            best = s
        above -= p
    return best


def expected(model: TimeModel, n: int, dist: list, d: int, x: int) -> float:
    """Expected prefill ms of the next turn with n prefix blocks, d of them
    in DRAM and x in SSD."""
    b = model.block
    total = sum(prob for prob, _ in dist)
    return sum(prob * model.e2e(n * b, s, d * b, x * b) for prob, s in dist) / total


def _safe(model: TimeModel, n: int, dist: list):
    """safe(d, x): the next turn stays safe with d / x of its n prefix blocks
    in DRAM / SSD, at the suffix QUANTILE of `dist` reaches."""
    b = model.block
    s = safe_suffix(dist)
    c = model.compute(n * b, s)
    return lambda d, x: (model.e2e(n * b, s, d * b, x * b) - c) / c < model.thresh


def place(model: TimeModel, n: int, dist: list) -> tuple[int, int, int]:
    """(HBM, DRAM, SSD) block counts of a turn's n prompt blocks."""
    safe = _safe(model, n, dist)
    h = next(h for h in range(n + 1) if safe(n - h, 0))
    m = n - h
    x = max(x for x in range(m + 1) if safe(m - x, x))
    return h, m - x, x


def reqwise(model: TimeModel, n: int, dist: list) -> tuple[str, list[float]]:
    """The same placement for a whole request: the lowest tier its n prefix
    blocks are safe in together, and its expected prefill time with all of
    them in HBM / DRAM / SSD (kvpilot_placement.penalties of it: the expected
    time a move one tier down adds)."""
    safe = _safe(model, n, dist)
    tier = "SSD" if safe(0, n) else "DRAM" if safe(n, 0) else "HBM"
    return tier, [expected(model, n, dist, 0, 0), expected(model, n, dist, n, 0),
                  expected(model, n, dist, 0, n)]


def hbm_penalty(model: TimeModel, n: int, dist: list, d: int, x: int) -> float:
    """Expected ms the move of the last HBM block to DRAM adds."""
    return expected(model, n, dist, d + 1, x) - expected(model, n, dist, d, x)


def dram_penalties(model: TimeModel, n: int, dist: list, d: int, x: int) -> list:
    """Per DRAM block, head first: the expected ms its move to SSD adds once
    the DRAM blocks behind it are on SSD."""
    tail_first = [
        expected(model, n, dist, d - j, x + j)
        - expected(model, n, dist, d - j + 1, x + j - 1)
        for j in range(1, d + 1)
    ]
    return tail_first[::-1]
