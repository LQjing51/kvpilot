# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVPilot online dram-safe feedback (prefill side).

A turn's dram-safe verdict is only computable once decode has emitted its tool
call, by which point the prefill worker has already queued the blocks. The
decode worker sends the verdict back keyed by the prefill request id, and the
blocks that request left behind are moved between tiers here.

Blocks are held as `(block, hash at free time)`: the hash pins which KV the
verdict was computed for, so a block that has since been evicted and refilled
is left alone. Nothing about the verdict is written onto the block — it applies
to one turn only, and the tier is the whole of its effect.
"""

from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (
    FreeKVCacheOnlineDramSafeQueue,
    KVCacheBlock,
)

logger = init_logger(__name__)

# Caps bookkeeping left behind by requests whose verdict never arrives. Entries
# hold a few block references and are dropped as soon as the verdict lands.
_PENDING_LIMIT = 4096


class OnlineDramSafe:
    """Holds a freed request's blocks until its verdict arrives."""

    def __init__(self, queue: FreeKVCacheOnlineDramSafeQueue):
        self._queue = queue
        self._pending: dict[str, list[tuple[KVCacheBlock, Any]]] = {}
        logger.info("KVPilot online dram-safe feedback enabled")

    def track(self, request_id: str, block_groups) -> None:
        """Record a freed request's cached blocks, pending its verdict."""
        pending = [
            (block, block.block_hash)
            for blocks in block_groups
            for block in blocks
            if block.block_hash is not None
        ]
        if not pending:
            return
        if len(self._pending) >= _PENDING_LIMIT:
            self._pending.pop(next(iter(self._pending)))
        self._pending[request_id] = pending

    def apply(self, request_id: str, dram_safe: bool) -> int:
        """Apply a verdict, returning how many blocks changed tier."""
        pending = self._pending.pop(request_id, None)
        if pending is None:
            return 0
        moved = 0
        for block, expected_hash in pending:
            # A different hash means the block was evicted and refilled since,
            # so the verdict no longer describes the KV it holds.
            if block.block_hash == expected_hash:
                moved += self._queue.retier(block, dram_safe)
        return moved
