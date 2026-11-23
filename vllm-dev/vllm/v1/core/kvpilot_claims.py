# SPDX-License-Identifier: Apache-2.0
"""Per-session claims on prompt blocks, shared by the KVPilot strategies.

A request that frees its blocks becomes its session's head and claims its full
prompt blocks for the tier its next turn needs, HBM while that turn's signal
is pending. The next turn's prompt contains this one, so a newly freed request
holding a head's last block supersedes that head, which gives up its claims. A
block several sessions share (a common system prompt) is required in the
fastest tier any head claims it for.

The replay sends a session's next turn only after this one has finished, so a
turn's signal always arrives after its request was freed and before the next
turn is: a signal finds its head pending, or the run is broken. A pending head
is therefore never superseded: a request holding its last block then belongs
to another session whose prompt starts the same way (two trajectories of one
task can even share a whole first prompt).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Claim slot per head tier; "P" = signal pending, claimed as HBM meanwhile.
_SLOT = {"P": 0, "HBM": 0, "DRAM": 1, "SSD": 2}
_SLOT_TIER = ("HBM", "DRAM", "SSD")


@dataclass
class Head:
    request_id: str
    blocks: list[Any]
    hashes: list[Any]
    # LMCache chunk hash of each block (block size == chunk size).
    lmc: list[int]
    # "P" until the signal arrives, then the next turn's tier (None = none),
    # or "B" when each block has its own (block_tiers).
    tier: str | None = "P"
    # Next turn's prefill time (ms) with its prefix in HBM / DRAM / SSD.
    time: list[float] | None = None
    # Blockwise: each block's tier, and what the placement keeps per block
    # (kvpilot_exclusive): the suffix distribution, the current (HBM, DRAM,
    # SSD) counts and each DRAM block's penalty.
    block_tiers: list[str] | None = None
    dist: list | None = None
    layout: list[int] | None = None
    dram_pen: list[float | None] | None = None


class Claims:
    def __init__(self) -> None:
        self.heads: dict[str, Head] = {}
        # Block hash -> LMCache chunk hash, for every block ever tracked.
        self.lmc: dict[Any, int] = {}
        # Last block hash -> request ids of the heads ending there.
        self._tail: dict[Any, set[str]] = {}
        # Block hash -> claims per slot (HBM incl. pending, DRAM, SSD).
        self._counts: dict[Any, list[int]] = {}
        # Block hash -> {request id: head} for every head claiming it.
        self._claimants: dict[Any, dict[str, Head]] = {}

    def track(self, request_id: str, blocks: list[Any], lmc: list[int]) -> Head:
        """`request_id` freed `blocks` (its full prompt blocks, head first):
        it becomes its session's head, superseding the previous one."""
        hashes = [b.block_hash for b in blocks]
        for h in hashes:
            owners = self._tail.get(h)
            if not owners:
                continue
            for owner in [o for o in owners if self.heads[o].tier != "P"]:
                owners.discard(owner)
                self._claim(self.heads.pop(owner), -1)
            if not owners:
                del self._tail[h]
        head = Head(request_id, blocks, hashes, lmc)
        self.lmc.update(zip(hashes, lmc))
        self.heads[request_id] = head
        self._tail.setdefault(hashes[-1], set()).add(request_id)
        self._claim(head, +1)
        return head

    def pending(self, request_id: str) -> Head:
        """The head waiting for `request_id`'s signal."""
        head = self.heads.get(request_id)
        if head is None or head.tier != "P":
            raise RuntimeError(
                f"KVPilot signal for {request_id}: no pending head "
                f"({'none' if head is None else head.tier})"
            )
        return head

    def signal(
        self, request_id: str, tier: str | None, time: list[float] | None,
        block_tiers: list[str] | None = None,
    ) -> Head:
        """The next turn's tier for a pending head; None ends the head.
        With `block_tiers` (tier "B") each block is claimed for its own."""
        head = self.pending(request_id)
        self._claim(head, -1)
        head.tier, head.time, head.block_tiers = tier, time, block_tiers
        if tier is None:
            del self.heads[request_id]
            owners = self._tail[head.hashes[-1]]
            owners.discard(request_id)
            if not owners:
                del self._tail[head.hashes[-1]]
        else:
            self._claim(head, +1)
        return head

    def move(self, head: Head, i: int, tier: str) -> None:
        """Blockwise: block i of a signalled head now claimed for `tier`."""
        c = self._counts[head.hashes[i]]
        c[_SLOT[head.block_tiers[i]]] -= 1
        c[_SLOT[tier]] += 1
        head.block_tiers[i] = tier

    def required(self, h: Any) -> str | None:
        """Fastest tier any head claims block `h` for; None if none does."""
        c = self._counts.get(h)
        if c is None:
            return None
        for slot, n in enumerate(c):
            if n:
                return _SLOT_TIER[slot]
        return None

    def claimants(self, h: Any) -> list[Head]:
        return list(self._claimants.get(h, {}).values())

    def _claim(self, head: Head, sign: int) -> None:
        if head.tier is None:
            return
        for i, h in enumerate(head.hashes):
            slot = _SLOT[head.block_tiers[i] if head.tier == "B" else head.tier]
            c = self._counts.setdefault(h, [0, 0, 0])
            c[slot] += sign
            heads = self._claimants.setdefault(h, {})
            if sign > 0:
                heads[head.request_id] = head
            else:
                heads.pop(head.request_id, None)
            if not any(c):
                del self._counts[h]
                del self._claimants[h]
