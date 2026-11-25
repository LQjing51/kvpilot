# SPDX-License-Identifier: Apache-2.0
"""Hybrid models (Qwen3.5 / 3.6): the linear-attention states KVPilot keeps.

A linear-attention layer has no per-token KV, only a recurrent state after
the last token, and vLLM caches one per prompt in HBM, at its last full block
(mamba "align" mode). KVPilot never offloads states: a prefix is reusable up
to a state in HBM, its KV loaded from wherever it went
(KVCacheManager.get_computed_blocks), and lost with that state.

So a placement that knows a session goes on holds the state at the end of
its latest turn in HBM, outside the free queue, until the next turn
supersedes it (its prompt contains the boundary) or the signal says there is
none: Belady when the trace uses the prompt again, inclusive and exclusive
until their signal. Held states are not free: they take HBM from the KV.
LRU holds as Belady does when given the trace (the A/B: every arm keeps the
same states), else nothing: a turn's end state is freed after its KV.
A state that is not held (superseded, mid-prefill) goes to the free queue,
first in line where it is ordered by key (FreeKVCacheClassQueue).

vLLM drops a finished request's state blocks before handing its blocks to the
connector (the decode side gets only the running one), so the state is taken
then (KVCacheManager.kvpilot_take_end_state) and handed to the placement when
the request frees.
"""

from __future__ import annotations

from typing import Any

from vllm.v1.core.kv_cache_utils import get_block_hash


def boundary_states(block_groups, blocks) -> list:
    """Each state group's block holding the state after `blocks` (a request's
    full prompt blocks; `block_groups`: the full-attention group's blocks,
    then each state group's). The state after token T sits in block
    (T - 1) // block_size, cached under that block's hash."""
    k, h = len(blocks), get_block_hash(blocks[-1].block_hash)
    states = []
    for g in block_groups[1:]:
        s = g[k - 1] if len(g) >= k else None
        if s is None or s.is_null or s.block_hash is None or get_block_hash(s.block_hash) != h:
            raise RuntimeError(
                f"KVPilot: no linear-attention state cached at the end of block {k - 1}"
            )
        states.append(s)
    return states


class StatePins:
    """The held states, by the hash of the block they end."""

    def __init__(self, block_pool) -> None:
        self._pool = block_pool
        # Block hash -> [state blocks, holders]: sessions whose prompts end
        # alike (two trajectories of one task) share one.
        self._held: dict[Any, list] = {}

    def __len__(self) -> int:
        return len(self._held)

    def track(self, blocks, states: list | None, hold: bool = True) -> None:
        """A request freed `blocks`, its full prompt blocks, with `states` at
        their end taken (one reference; None: not a hybrid model). It
        supersedes the states held inside its prompt and, with `hold`, keeps
        its own."""
        if states is None:
            return
        hashes = [get_block_hash(b.block_hash) for b in blocks]
        for h in hashes[:-1]:
            if h in self._held:
                self.release(h)
        held = self._held.get(hashes[-1])
        if hold and held is None:
            self._held[hashes[-1]] = [states, 1]
            return
        if hold:
            held[1] += 1
        self._pool.free_blocks(states)

    def release(self, h) -> None:
        """One holder of the state ending block hash `h` lets go: no next
        turn, or superseded. A state already let go has nothing to release."""
        held = self._held.get(h)
        if held is None:
            return
        held[1] -= 1
        if held[1] == 0:
            del self._held[h]
            self._pool.free_blocks(held[0])
