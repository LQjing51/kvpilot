# SPDX-License-Identifier: Apache-2.0
"""KVPilot LRU baseline, scheduler side: no signal, no placement. A freed
request's full prompt chunks go to the Mooncake master in prompt order, so
DRAM, like HBM, evicts a request from its tail (a hit reads HBM, then DRAM,
then SSD)."""

from __future__ import annotations

from vllm.v1.core.kvpilot_inclusive import full_prompt_blocks


class LruPlacement:
    def __init__(self, block_size: int) -> None:
        self._block_size = block_size
        self._connector = None

    def bind(self, connector) -> None:
        """The scheduler-side KV connector: names chunks, carries the order."""
        self._connector = connector

    def track(self, request_id: str, block_groups, token_ids) -> None:
        """A request freed its blocks: its chunks' release order."""
        blocks = full_prompt_blocks(block_groups, len(token_ids), self._block_size)
        if not blocks:
            return
        lmc = self._connector.kvpilot_chunk_hashes(request_id, token_ids, len(blocks))
        if lmc is None:
            raise RuntimeError(f"KVPilot: no LMCache chunk hashes for {request_id}")
        self._connector.kvpilot_group_order(lmc)

    def apply(self, request_id: str, tier, time_ms, suffix=None) -> dict:
        raise RuntimeError(f"KVPilot signal for {request_id} under LRU")
