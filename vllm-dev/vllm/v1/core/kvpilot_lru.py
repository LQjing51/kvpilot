# SPDX-License-Identifier: Apache-2.0
"""KVPilot LRU baseline, scheduler side: no signal, no placement. A freed
request's full prompt chunks go to the Mooncake master in prompt order, so
DRAM, like HBM, evicts a request from its tail (a hit reads HBM, then DRAM,
then SSD).

A hybrid model's state at the end of a prompt: given the replayed trace
(KVPILOT_BELADY_TRACE, run_ab.sh on a hybrid model), held as Belady holds it,
while a later line reuses the prompt, so every arm keeps the same states and
the arms differ only in where the KV goes. Without a trace (request-safe
runs) it is not held: freed after the request's KV, so the free queue's order
evicts the KV first."""

from __future__ import annotations

from vllm.v1.core.kvpilot_belady import next_uses
from vllm.v1.core.kvpilot_inclusive import full_prompt_blocks
from vllm.v1.core.kvpilot_signal import trace_line
from vllm.v1.core.kvpilot_states import StatePins


class LruPlacement:
    def __init__(self, block_pool, block_size: int, trace_path: str | None = None) -> None:
        self._block_size = block_size
        self._connector = None
        self.states = StatePins(block_pool)
        self._next = None if trace_path is None else next_uses(trace_path, block_size)

    def bind(self, connector) -> None:
        """The scheduler-side KV connector: names chunks, carries the order."""
        self._connector = connector

    def track(self, request_id: str, block_groups, token_ids, states=None) -> None:
        """A request freed its blocks: its chunks' release order. `states`:
        the state at its prompt's end (hybrid models, kvpilot_states)."""
        blocks = full_prompt_blocks(block_groups, len(token_ids), self._block_size)
        if not blocks:
            return
        self.states.track(blocks, states, hold=self._reused(request_id, len(blocks)))
        lmc = self._connector.kvpilot_chunk_hashes(request_id, token_ids, len(blocks))
        if lmc is None:
            raise RuntimeError(f"KVPilot: no LMCache chunk hashes for {request_id}")
        self._connector.kvpilot_group_order(lmc)

    def _reused(self, request_id: str, n: int) -> bool:
        """Whether a later trace line reuses the request's last full block."""
        if self._next is None:
            return False
        line = trace_line(request_id)
        if line is None:
            return False
        uses = self._next[line]
        if len(uses) < n:
            raise RuntimeError(
                f"KVPilot LRU: {request_id} has {n} full blocks, its trace line {line} {len(uses)}"
            )
        return uses[n - 1] is not None

    def apply(self, request_id: str, tier, time_ms, suffix=None) -> dict:
        raise RuntimeError(f"KVPilot signal for {request_id} under LRU")
