# SPDX-License-Identifier: Apache-2.0
"""KVPilot inclusive placements, scheduler side.

Writes stay write-through (HBM, DRAM and SSD each hold a copy); the signal
only orders eviction. A freed block enters the HBM free queue at the pending
key, and the signal rekeys it there and sets its chunk group on the Mooncake
master:

  inclusive-safe-reqwise     the class of the fastest tier any session still
                             needs it in (kvpilot_claims; CLASS in
                             kvpilot_placement).
  inclusive-penalty-reqwise  HBM keys by the request's HBM -> DRAM penalty,
                             the master by its DRAM -> SSD one, the most over
                             the sessions claiming a block (0 with none).

Both take the request's tier and times from the time model
(kvpilot_blockwise.reqwise) when given one.

Ties go in free order (HBM) and release order (master).
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.core import kvpilot_blockwise, kvpilot_placement
from vllm.v1.core.kvpilot_claims import Claims

logger = init_logger(__name__)


def full_prompt_blocks(block_groups, num_prompt_tokens: int, block_size: int):
    """The request's full blocks of prompt tokens, head first."""
    blocks = []
    for b in block_groups[0][: num_prompt_tokens // block_size]:
        if b.is_null or b.block_hash is None:
            break
        blocks.append(b)
    return blocks


class InclusivePlacement:
    def __init__(self, block_pool, block_size: int,
                 model: kvpilot_blockwise.TimeModel | None = None,
                 by_penalty: bool = False) -> None:
        """`model`: tier and times from this time model and the signal's
        suffix distribution instead of the signal's. `by_penalty`:
        inclusive-penalty-reqwise (needs `model`)."""
        if by_penalty and model is None:
            raise ValueError("KVPilot inclusive-penalty placement needs a time model")
        self._queue = block_pool.free_block_queue
        self._block_size = block_size
        self._model = model
        self._by_penalty = by_penalty
        self._connector = None
        self._claims = Claims()

    def bind(self, connector) -> None:
        """The scheduler-side KV connector: names chunks, sets their classes."""
        self._connector = connector

    def track(self, request_id: str, block_groups, token_ids) -> None:
        """A request freed its blocks (just queued at the pending class)."""
        blocks = full_prompt_blocks(block_groups, len(token_ids), self._block_size)
        if not blocks:
            return
        lmc = self._connector.kvpilot_chunk_hashes(request_id, token_ids, len(blocks))
        if lmc is None:
            raise RuntimeError(f"KVPilot: no LMCache chunk hashes for {request_id}")
        self._connector.kvpilot_group_order(lmc)
        self._claims.track(request_id, blocks, lmc)

    def apply(self, request_id: str, tier: str | None, time, suffix=None) -> dict:
        """The signal for a freed request: rekey its blocks in both tiers."""
        if self._model is not None and tier is not None:
            if not suffix:
                raise RuntimeError(f"KVPilot signal for {request_id}: no suffix")
            n = len(self._claims.pending(request_id).hashes)
            tier, time = kvpilot_blockwise.reqwise(self._model, n, suffix)
        head = self._claims.signal(request_id, tier, time)
        tiers = [self._claims.required(h) for h in head.hashes]
        if not self._by_penalty:
            keys = [float(kvpilot_placement.CLASS[t]) for t in tiers]
            pens = [0.0] * len(tiers)
        else:
            keys, pens = map(list, zip(*map(self._penalties, head.hashes)))
        live = [
            (b, k) for b, h, k in zip(head.blocks, head.hashes, keys)
            if b.block_hash == h
        ]
        moved = self._queue.rekey([b for b, _ in live], [k for _, k in live])
        groups = self._connector.kvpilot_set_group_placement(head.lmc, tiers, pens)
        return {
            "tier": tier,
            "blocks": len(head.blocks),
            "hbm_rekeyed": moved,
            "dram_groups": groups,
        }

    def _penalties(self, h) -> tuple[float, float]:
        """inclusive-penalty-reqwise: block `h`'s (HBM -> DRAM, DRAM -> SSD)
        penalty, the most over the sessions claiming it; pending while any
        signal is still to come."""
        heads = self._claims.claimants(h)
        if any(g.tier == "P" for g in heads):
            pending = kvpilot_placement.PENDING_PENALTY
            return pending, pending
        pens = [kvpilot_placement.penalties(g.time) for g in heads]
        return max((a for a, _ in pens), default=0.0), max((b for _, b in pens), default=0.0)
