# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVPilot exclusive-penalty-reqwise placement, scheduler side.

Every prompt block keeps one copy, in the tier its session's next turn needs
(kvpilot_claims: the fastest tier any head claims it for; HBM while a signal
is pending):

  * A prefill writes nothing below HBM. A freed request's blocks stay pinned
    in HBM, out of the free queue, until its signal arrives: a pending block
    is never evicted.
  * HBM: the blocks stay; a copy an earlier turn left below is removed. They
    go back to the free queue as held blocks (kv_cache_utils), the only copy
    of their KV: last in line, by penalty, never handed out as they are.
  * DRAM / SSD: blocks not yet in that tier are written there (a copy in the
    other lower tier is removed first), then their HBM copies are dropped. A
    block already there only loses its HBM copy.
  * No next turn: a block keeps the lower copy it has, or is written to SSD;
    then its HBM copy goes.

The writes run on the prefill workers (LMCache, kvpilot_exclusive.py there),
from HBM blocks that stay pinned until the job id comes back as a finished
send; the forward never waits for them.

A request that cannot be admitted for lack of free blocks waits while held
blocks are written to DRAM: the smallest exposed load added by the move first,
tail blocks of a request first; once written they are blank and handed out.
Pending blocks are not candidates, so with no held block left the request
waits for signals. DRAM demotes into SSD the same way (Mooncake master).

Unblocking (exclusive-penalty-reqwise-unblocking, `reserve` > 0) removes that
wait. An admission only needs its loaded prefix and its first prefill chunk
to fit (not the whole prompt), taking blocks from the back:
  * Suffix chunk (computed, at most one prefill chunk): takes free blocks
    first, the reserve included. After every signal and every allocation,
    held blocks are demoted in the background until `reserve` blocks (one
    prefill chunk) are free again, so a long prompt's next chunk finds
    them; if it does not yet, it skips the step instead of being preempted.
  * Prefix (loaded): takes only free blocks above the reserve and borrows
    held blocks for the rest (`lend`). They are handed out first, so they
    land in the loaded prefix; each worker copies their KV to DRAM on its
    own stream before the load, and a layer's scatter waits for that
    layer's copy (LMCache kvpilot_exclusive.Evictor).
  An admission that cannot fit even so (its chunk past the free blocks
  before the reserve is refilled, too few held blocks) waits as above.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core import kvpilot_blockwise, kvpilot_placement
from vllm.v1.core.kvpilot_claims import Claims
from vllm.v1.core.kvpilot_inclusive import full_prompt_blocks

logger = init_logger(__name__)

_LOG_EVERY_S = 30.0
# Blocks per demotion job.
_DEMOTE_BATCH = 64


@dataclass
class _Job:
    demote: bool
    pinned: list[Any]  # job pins: the blocks the write reads
    stored: list[tuple[Any, Any]]  # (block, hash) written below
    dropped: list[Any]  # hashes whose lower copy the job removes


class ExclusivePlacement:
    """Moves freed prompt blocks to their single tier as signals arrive."""

    def __init__(self, block_pool, block_size: int, reserve: int = 0,
                 model: kvpilot_blockwise.TimeModel | None = None,
                 blockwise: bool = False) -> None:
        """`reserve` > 0: unblocking, keeping that many blocks free. `model`:
        tiers and penalties from this time model and the signal's suffix
        distribution instead of the signal's tier and times; `blockwise`:
        per block (needs `model`)."""
        if blockwise and model is None:
            raise ValueError("KVPilot blockwise placement needs a time model")
        self._pool = block_pool
        self._queue = block_pool.free_block_queue
        self._block_size = block_size
        self._reserve = reserve
        self._model = model
        self._blockwise = blockwise
        # Blockwise HBM candidates: each signalled head's last HBM block,
        # (key, seq, request id, block index); stale entries are skipped.
        self._cand: list[tuple[float, int, str, int]] = []
        self._cand_seq = itertools.count()
        self._connector = None
        self._claims = Claims()
        # Block hash -> the lower tier holding its copy (DRAM / SSD).
        self._lower: dict[Any, str] = {}
        # block_id -> (block, hash): tracked blocks pinned in HBM.
        self._pinned: dict[int, tuple[Any, Any]] = {}
        self._jobs: dict[str, _Job] = {}
        self._demoting = 0
        # Hash -> jobs in flight removing its lower copy: not lent until they
        # finish (the copy a lend writes would be removed after it).
        self._dropping: Counter = Counter()
        self._ids = itertools.count()
        self.stats: Counter = Counter()
        self._last_log = time.monotonic()
        if reserve:
            logger.info("KVPilot exclusive unblocking: reserve=%d blocks", reserve)

    def bind(self, connector) -> None:
        """The scheduler-side KV connector that names chunks and carries jobs
        to the workers."""
        self._connector = connector

    def busy(self) -> bool:
        """Whether jobs are in flight: the engine must keep stepping to hear
        them finish."""
        return bool(self._jobs)

    # ---- events -------------------------------------------------------------

    def track(self, request_id: str, block_groups, token_ids) -> None:
        """A request freed its blocks: it becomes its session's head, and its
        blocks stay pinned in HBM until the signal arrives."""
        blocks = full_prompt_blocks(block_groups, len(token_ids), self._block_size)
        if not blocks:
            return
        lmc = self._connector.kvpilot_chunk_hashes(request_id, token_ids, len(blocks))
        if lmc is None:
            raise RuntimeError(f"KVPilot: no LMCache chunk hashes for {request_id}")
        self._connector.kvpilot_group_order(lmc)
        head = self._claims.track(request_id, blocks, lmc)
        for b, h in zip(blocks, head.hashes):
            self._pin(b, h)

    def apply(self, request_id: str, tier: str | None, time_ms, suffix=None) -> dict:
        """The signal for a freed request: move its blocks to their tier.
        With a time model the tier and times come from the next turn's
        `suffix` distribution (blockwise: each block to its own)."""
        if self._model is None or tier is None:
            head = self._claims.signal(request_id, tier, time_ms)
        elif not suffix:
            raise RuntimeError(f"KVPilot signal for {request_id}: no suffix")
        elif self._blockwise:
            n = len(self._claims.pending(request_id).hashes)
            h, d, x = kvpilot_blockwise.place(self._model, n, suffix)
            head = self._claims.signal(
                request_id, "B", time_ms, ["HBM"] * h + ["DRAM"] * d + ["SSD"] * x
            )
            head.dist, head.layout = suffix, [h, d, x]
            self._set_dram_pen(head)
            self.stats["placed_hbm"] += h
            self.stats["placed_dram"] += d
            self.stats["placed_ssd"] += x
        else:
            n = len(self._claims.pending(request_id).hashes)
            signalled = tier
            tier, time_ms = kvpilot_blockwise.reqwise(self._model, n, suffix)
            # How the model's tier relates to the signal's (oracle: measured).
            self.stats[f"model_{signalled}_to_{tier}"] += 1
            head = self._claims.signal(request_id, tier, time_ms)
        out = self._reconcile(head)
        if head.tier == "B":
            self._push_tail(head)
            out["layout"] = list(head.layout)
        self.stats[f"signal_{tier}"] += 1
        self.keep_reserve()
        self._maybe_log()
        return out

    @property
    def unblocking(self) -> bool:
        return self._reserve > 0

    def lend(self, request_id: str, need: int, loaded_blocks: int,
             computed) -> int:
        """Unblocking, before allocating an admission that needs `need` free
        blocks, `loaded_blocks` of them for the prefix its load fills: what
        the free blocks (the reserve included, refilled after the allocation)
        lack is borrowed from held blocks, cheapest to expose first as
        `demote` picks. Lent blocks become free now and are handed out first,
        so they land in the loaded prefix, whose load copies their KV to DRAM
        before overwriting them. They may not reach the computed suffix: its
        KV is written by the forward, which does not wait for that copy.

        Lends nothing unless the admission then fits: one short of more than
        its loaded prefix, or of more than the held blocks it may borrow,
        waits as under exclusive.

        Args:
            request_id: The request being admitted.
            need: Free blocks its allocation needs.
            loaded_blocks: Whole blocks its connector load fills.
            computed: Its HBM prefix hits (held, but about to be its own).

        Returns:
            How many blocks were lent.
        """
        short = need - self._pool.get_num_free_blocks()
        if short <= 0:
            return 0
        plan = self._plan_held(short, computed, lend=True) if short <= loaded_blocks else []
        if len(plan) < short:
            self.stats["lend_no_fit"] += 1
            return 0
        take = [b for b, _ in self._commit_held(plan)]
        hashes = [b.block_hash for b in take]
        self._connector.kvpilot_exclusive_evict(request_id, {
            "hashes": [self._claims.lmc[h] for h in hashes],
            "block_ids": [b.block_id for b in take],
            "dram_pen": [self._dram_penalty(h) for h in hashes],
        })
        for b, h in zip(take, hashes):
            self._lower[h] = "DRAM"
            if not self._pool.kvpilot_uncache(b):
                raise RuntimeError(f"KVPilot: lent block {b.block_id} was not cached")
        self._queue.lend(take)
        self.stats["lend_reqs"] += 1
        self.stats["lent"] += len(take)
        return len(take)

    def keep_reserve(self) -> None:
        """Unblocking: after a signal or an allocation, demote held blocks in
        the background until `reserve` blocks are free (in flight counts)."""
        if self._reserve:
            self.stats["reserve_demote"] += self.demote(
                self._reserve - self._pool.get_num_free_blocks()
            )

    def chunk_short(self) -> None:
        """Unblocking: a running prefill's next chunk does not fit yet. It
        skips this step (no preemption, which would drop what it computed)
        while the reserve is refilled."""
        self.stats["chunk_wait"] += 1
        self.keep_reserve()

    def on_job_done(self, job_id: str) -> None:
        """Every TP rank finished `job_id`: unpin and drop HBM copies now safe below."""
        job = self._jobs.pop(job_id)
        for h in job.dropped:
            self._dropping[h] -= 1
            if not self._dropping[h]:
                del self._dropping[h]
        if job.pinned:
            self._pool.free_blocks(list(reversed(job.pinned)))
        dropped = 0
        for b, h in job.stored:
            if self._lower.get(h) is None or b.block_hash != h:
                continue
            if self._pending(h):
                # The session's next turn tracked it while it was written:
                # pinned for that turn's signal, which sees the copy below.
                continue
            self._unpin(b)
            dropped += self._release(b, h, force=job.demote)
        if job.demote:
            self._demoting -= len(job.stored)
        self.stats["jobs_done"] += 1
        self.stats["hbm_dropped_after_write"] += dropped

    def demote(self, shortfall: int, keep=()) -> int:
        """A waiting request needs `shortfall` more free blocks: write held
        blocks to DRAM, cheapest to expose first, except its own prefix hits
        (`keep`). Returns how many."""
        short = shortfall - self._demoting
        if short <= 0:
            return 0
        write = self._commit_held(self._plan_held(short, keep, lend=False))
        for start in range(0, len(write), _DEMOTE_BATCH):
            part = write[start : start + _DEMOTE_BATCH]
            self._send(
                demote=True,
                lmc=[self._claims.lmc[h] for _, h in part],
                blocks=[b for b, _ in part],
                drop=[], dram=list(range(len(part))), ssd=[],
                dram_pen=[self._dram_penalty(h) for _, h in part],
                req="demote", tier="DRAM",
            )
            for _, h in part:
                self._lower[h] = "DRAM"
            self._demoting += len(part)
            self.stats["demote_written"] += len(part)
        return len(write)

    # ---- placement ----------------------------------------------------------

    def _reconcile(self, head) -> dict:
        drop: list[int] = []
        store: dict[str, list[int]] = {"DRAM": [], "SSD": []}
        zero: list[int] = []
        now_release = []
        hold = []
        for i, (b, h) in enumerate(zip(head.blocks, head.hashes)):
            if b.block_hash != h:
                raise RuntimeError(f"KVPilot: block {h} left HBM before its signal")
            lower = self._lower.get(h)
            want = self._claims.required(h)
            if want == "HBM":
                if lower is not None:
                    drop.append(i)
                    del self._lower[h]
                if not self._pending(h):
                    hold.append((b, h))
                continue
            if lower is not None and (want is None or lower == want):
                # Nobody needs it soon, or it already sits where it should.
                if want is None and lower == "DRAM":
                    # First to demote to SSD when DRAM fills up.
                    zero.append(i)
                now_release.append((b, h))
                continue
            if lower is not None:
                drop.append(i)
            dest = want or "SSD"
            store[dest].append(i)
            self._lower[h] = dest
        released = 0
        for b, h in now_release:
            self._unpin(b)
            released += self._release(b, h)
        self.stats["hbm_dropped_now"] += released
        # Tail first, as a request frees: within a penalty the tail goes first.
        for b, h in reversed(hold):
            self._unpin(b)
            # Still in use by a running request: it pins the block again
            # when it frees it (its prompt holds it).
            if b.ref_cnt == 0:
                # Blockwise picks from its own candidates (_plan_held).
                pen = 0.0 if self._blockwise else self._hbm_penalty(h)
                self._queue.hold([b], [pen])
                self.stats["held"] += 1
        job_id = None
        if drop or store["DRAM"] or store["SSD"] or zero:
            n = 1 + max(drop + store["DRAM"] + store["SSD"] + zero)
            job_id = self._send(
                demote=False,
                lmc=head.lmc[:n],
                blocks=head.blocks[:n],
                drop=drop, dram=store["DRAM"], ssd=store["SSD"],
                dram_pen=[self._dram_penalty(head.hashes[i]) for i in store["DRAM"]],
                req=head.request_id, tier=head.tier, zero=zero,
            )
        self.stats["write_dram"] += len(store["DRAM"])
        self.stats["write_ssd"] += len(store["SSD"])
        self.stats["drop"] += len(drop)
        return {
            "tier": head.tier,
            "blocks": len(head.blocks),
            "write_dram": len(store["DRAM"]),
            "write_ssd": len(store["SSD"]),
            "drop": len(drop),
            "hbm_released": released,
            "job": job_id,
        }

    def _plan_held(self, n: int, own, lend: bool) -> list:
        """Up to n held blocks to move to DRAM, cheapest to expose first,
        never one of the admitted request's own prefix hits (`own`) and, for a
        lend, none whose lower copy a job still removes. Blockwise takes only
        a head's last HBM block (HBM stays a prefix), keyed by what its move
        adds, the next one after it. Plans only; _commit_held applies.
        Returns (block, hash, head, index) tuples."""
        own_ids = {b.block_id for b in own}
        if not self._blockwise:
            return [
                (b, b.block_hash, None, -1) for b in self._queue.held()
                if b.block_id not in own_ids
                and not (lend and b.block_hash in self._dropping)
            ][:n]
        heap = list(self._cand)
        layouts: dict[str, list[int]] = {}
        out, picked = [], set()
        while len(out) < n and heap:
            _, _, rid, idx = heapq.heappop(heap)
            head = self._claims.heads.get(rid)
            if head is None or head.tier != "B":
                continue
            lay = layouts.get(rid, head.layout)
            if lay[0] - 1 != idx:
                continue
            b, h = head.blocks[idx], head.hashes[idx]
            if (b.block_hash != h or not b.kvp_held or b.block_id in own_ids
                    or b.block_id in picked or (lend and h in self._dropping)
                    or not self._hbm_tail_everywhere(h, head, layouts)):
                continue
            out.append((b, h, head, idx))
            picked.add(b.block_id)
            lay = [lay[0] - 1, lay[1] + 1, lay[2]]
            layouts[rid] = lay
            if lay[0] > 0:
                heapq.heappush(heap, (self._tail_key(head, lay), next(self._cand_seq),
                                      rid, lay[0] - 1))
        return out

    def _commit_held(self, plan: list) -> list:
        """Apply a _plan_held plan: blockwise heads giving up a block get its
        claim moved to DRAM, new DRAM penalties (sent to the master for their
        DRAM blocks) and their next HBM candidate. Returns (block, hash)."""
        out = [(b, h) for b, h, _, _ in plan]
        if not self._blockwise:
            return out
        touched = {}
        for _, h, _, _ in plan:
            # The block leaves HBM for every head keeping it there (all have
            # it as their last HBM block, _hbm_tail_everywhere).
            for g in self._claims.claimants(h):
                if g.tier != "B":
                    continue
                i = g.hashes.index(h)
                if g.block_tiers[i] != "HBM":
                    continue
                g.layout = [g.layout[0] - 1, g.layout[1] + 1, g.layout[2]]
                self._claims.move(g, i, "DRAM")
                touched[g.request_id] = g
        moved = {h for _, h in out}
        for g in touched.values():
            self._set_dram_pen(g)
            self._push_tail(g)
            repen = [
                (i, g.dram_pen[i]) for i, h in enumerate(g.hashes)
                if g.dram_pen[i] is not None and h not in moved
                and self._lower.get(h) == "DRAM"
            ]
            if repen:
                n = 1 + max(i for i, _ in repen)
                self._send(demote=False, lmc=g.lmc[:n], blocks=g.blocks[:n],
                           drop=[], dram=[], ssd=[], dram_pen=[],
                           req=g.request_id, tier="B", repen=repen)
                self.stats["repen"] += len(repen)
        return out

    def _hbm_tail_everywhere(self, h, head, layouts) -> bool:
        """Blockwise: every other head keeping block `h` in HBM has it as its
        last HBM block, so moving it breaks no head's HBM prefix."""
        for g in self._claims.claimants(h):
            if g is head or g.tier != "B":
                continue
            i = g.hashes.index(h)
            if g.block_tiers[i] == "HBM" and layouts.get(g.request_id, g.layout)[0] - 1 != i:
                return False
        return True

    def _tail_key(self, head, lay) -> float:
        """Blockwise: what moving head's last HBM block (under layout `lay`)
        to DRAM adds to its next turn's expected prefill; for a shared block,
        the most over the heads keeping it in HBM."""
        model = self._model
        key = kvpilot_blockwise.hbm_penalty(model, len(head.hashes), head.dist, lay[1], lay[2])
        h = head.hashes[lay[0] - 1]
        for g in self._claims.claimants(h):
            if g is not head and g.tier == "B" and g.block_tiers[g.hashes.index(h)] == "HBM":
                key = max(key, kvpilot_blockwise.hbm_penalty(
                    model, len(g.hashes), g.dist, g.layout[1], g.layout[2]))
        return key

    def _push_tail(self, head) -> None:
        """Blockwise: head's last HBM block becomes its HBM candidate."""
        if head.layout[0] > 0:
            heapq.heappush(self._cand, (self._tail_key(head, head.layout),
                                        next(self._cand_seq), head.request_id,
                                        head.layout[0] - 1))
        if len(self._cand) > 4 * len(self._claims.heads) + 64:
            live = []
            for e in self._cand:
                g = self._claims.heads.get(e[2])
                if g is not None and g.tier == "B" and g.layout[0] - 1 == e[3]:
                    live.append(e)
            heapq.heapify(live)
            self._cand = live

    def _set_dram_pen(self, head) -> None:
        """Blockwise: each DRAM block's penalty under head's current layout."""
        h, d, x = head.layout
        pens = kvpilot_blockwise.dram_penalties(self._model, len(head.hashes), head.dist, d, x)
        head.dram_pen = [None] * h + pens + [None] * x

    def _pending(self, h) -> bool:
        """Whether a head still waiting for its signal claims block `h`."""
        return any(head.tier == "P" for head in self._claims.claimants(h))

    def _hbm_penalty(self, h) -> float:
        """Exposed load added by moving block `h` from HBM to DRAM: the most
        any head needing it in HBM would gain (all signalled)."""
        return max(
            kvpilot_placement.penalties(head.time)[0]
            for head in self._claims.claimants(h)
            if head.tier == "HBM"
        )

    def _dram_penalty(self, h) -> float:
        """Exposed load added by moving block `h` from DRAM to SSD, over the
        heads needing it in DRAM or faster (all signalled)."""
        if self._blockwise:
            return max(
                (g.dram_pen[g.hashes.index(h)] or 0.0
                 for g in self._claims.claimants(h) if g.tier == "B"),
                default=0.0,
            )
        return max(
            kvpilot_placement.penalties(head.time)[1]
            for head in self._claims.claimants(h)
            if head.tier in ("HBM", "DRAM")
        )

    def _send(self, *, demote, lmc, blocks, drop, dram, ssd, dram_pen, req, tier, zero=(),
              repen=()):
        job_id = f"kvx-{next(self._ids)}"
        written = [blocks[i] for i in dram + ssd]
        if written:
            self._pool.touch(written)
        dropped = [blocks[i].block_hash for i in drop]
        self._dropping.update(dropped)
        self._jobs[job_id] = _Job(
            demote, written, [(b, b.block_hash) for b in written], dropped
        )
        self._connector.kvpilot_exclusive_submit({
            "id": job_id,
            "req": req,
            "tier": tier,
            "hashes": list(lmc),
            "block_ids": [b.block_id for b in blocks],
            "drop": drop,
            "dram": dram,
            "ssd": ssd,
            "dram_pen": dram_pen,
            "zero": list(zero),
            "repen": [list(r) for r in repen],
        })
        self.stats["jobs"] += 1
        return job_id

    def _release(self, block, h, force: bool = False) -> bool:
        """Drop a free block's HBM copy once a copy below holds it, unless a
        head still wants it in HBM (a demotion drops it anyway: that is the
        eviction)."""
        if block.ref_cnt != 0 or block.block_hash != h or h not in self._lower:
            return False
        if not force and self._claims.required(h) == "HBM":
            return False
        if not self._pool.kvpilot_uncache(block):
            return False
        # Blank now: reused before anything that still holds KV.
        self._queue.rekey([block], [-math.inf])
        return True

    def _pin(self, block, h) -> None:
        if block.block_id in self._pinned:
            return
        self._pool.touch([block])
        self._pinned[block.block_id] = (block, h)

    def _unpin(self, block) -> None:
        if self._pinned.pop(block.block_id, None) is not None:
            self._pool.free_blocks([block])

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log < _LOG_EVERY_S:
            return
        self._last_log = now
        lower = Counter(self._lower.values())
        logger.info(
            "KVPilot exclusive heads=%d jobs_in_flight=%d pinned=%d held=%d "
            "demoting=%d lower DRAM/SSD=%d/%d free=%d hbm_candidates=%d stats=%s",
            len(self._claims.heads), len(self._jobs), len(self._pinned),
            len(self._queue.held()), self._demoting, lower["DRAM"], lower["SSD"],
            self._pool.get_num_free_blocks(), len(self._cand), dict(self.stats),
        )
