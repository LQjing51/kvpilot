# SPDX-License-Identifier: Apache-2.0
"""KVPilot Belady baseline (KVPILOT_PLACEMENT=belady), scheduler side.

Copies as under LRU (write-through; SSD hits load through into DRAM) and no
signal: only eviction differs. HBM and DRAM evict the block whose next use is
farthest away, never-again first -- Belady's MIN, what reuse-probability
eviction tries to approach. Next uses come from the replayed trace
(KVPILOT_BELADY_TRACE): the replay sends its lines in file order, a request id
carries its line (kvpilot_signal.trace_line), and a block of a line is used
next by the first later line whose full blocks contain the same content with
the same prefix (the chain of hash ids up to it, as vLLM chains block hashes).

A freed request's full prompt blocks are keyed by their next use: -line in
the HBM free queue (smallest leaves first), penalty FAR - line on the Mooncake
master, which drops the smallest first in the inclusive-penalty order.
"""

from __future__ import annotations

import json

from vllm.v1.core.kvpilot_inclusive import full_prompt_blocks
from vllm.v1.core.kvpilot_signal import trace_line

# Above every trace line.
FAR = 1e9


def next_uses(path: str, block_size: int) -> list[list[int | None]]:
    """For every trace line, each full block's next using line (None: never)."""
    chains = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            chain, out = None, []
            for h in r["hash_ids"][: r["input_length"] // block_size]:
                chain = hash((chain, h))
                out.append(chain)
            chains.append(out)
    nxt: list[list[int | None]] = [[] for _ in chains]
    last: dict[int, int] = {}
    for i in range(len(chains) - 1, -1, -1):
        nxt[i] = [last.get(c) for c in chains[i]]
        for c in chains[i]:
            last[c] = i
    return nxt


class BeladyPlacement:
    def __init__(self, block_pool, block_size: int, trace_path: str) -> None:
        self._queue = block_pool.free_block_queue
        self._block_size = block_size
        self._next = next_uses(trace_path, block_size)
        self._connector = None

    def bind(self, connector) -> None:
        """The scheduler-side KV connector: names chunks, sets their order."""
        self._connector = connector

    def track(self, request_id: str, block_groups, token_ids) -> None:
        """A request freed its blocks: key them by their next use."""
        blocks = full_prompt_blocks(block_groups, len(token_ids), self._block_size)
        line = trace_line(request_id)
        if not blocks or line is None:
            return
        uses = self._next[line]
        if len(uses) < len(blocks):
            raise RuntimeError(
                f"KVPilot Belady: {request_id} has {len(blocks)} full blocks, "
                f"its trace line {line} {len(uses)}"
            )
        uses = uses[: len(blocks)]
        lmc = self._connector.kvpilot_chunk_hashes(request_id, token_ids, len(blocks))
        if lmc is None:
            raise RuntimeError(f"KVPilot: no LMCache chunk hashes for {request_id}")
        self._connector.kvpilot_group_order(lmc)
        self._queue.rekey(blocks, [-FAR if u is None else -float(u) for u in uses])
        self._connector.kvpilot_set_group_placement(
            lmc, [None] * len(uses), [0.0 if u is None else FAR - u for u in uses]
        )

    def apply(self, request_id: str, tier, time_ms, suffix=None) -> dict:
        raise RuntimeError(f"KVPilot signal for {request_id} under Belady")
