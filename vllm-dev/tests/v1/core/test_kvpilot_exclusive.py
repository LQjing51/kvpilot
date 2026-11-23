# SPDX-License-Identifier: Apache-2.0
"""KVPilot exclusive placement: which blocks are written, dropped, kept, demoted."""

import pytest

from vllm.v1.core import kvpilot_heap
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kvpilot_exclusive import ExclusivePlacement

BS = 16


class FakeConnector:
    def __init__(self):
        self.jobs = []
        self.evicts = []
        self.orders = []

    def kvpilot_chunk_hashes(self, request_id, token_ids, n):
        return [1000 + i for i in range(n)]

    def kvpilot_group_order(self, lmc):
        self.orders.append(list(lmc))

    def kvpilot_exclusive_submit(self, job):
        self.jobs.append(job)

    def kvpilot_exclusive_evict(self, request_id, evict):
        self.evicts.append((request_id, evict))


def _time(hbm_pen, dram_pen):
    """A signal's [t_hbm, t_dram, t_ssd] with these two penalties."""
    return [10.0, 10.0 + hbm_pen, 10.0 + hbm_pen + dram_pen]


def _make(monkeypatch, blocks=64, reserve=0):
    monkeypatch.setenv("KVPILOT_PLACEMENT", "exclusive-penalty-reqwise")
    monkeypatch.setattr(kvpilot_heap, "_CLS", kvpilot_heap.PyIndexedHeap)
    pool = BlockPool(num_gpu_blocks=blocks, enable_caching=True, hash_block_size=BS)
    excl = ExclusivePlacement(pool, BS, reserve=reserve)
    conn = FakeConnector()
    excl.bind(conn)
    return pool, excl, conn


@pytest.fixture
def env(monkeypatch):
    return _make(monkeypatch)


def _run(pool, excl, rid, hashes):
    """Prefill `rid` over `hashes` (reusing cached blocks) and free it."""
    blocks = []
    for h in hashes:
        hit = pool.cached_block_hash_to_block.get_one_block(h)
        if hit is not None:
            pool.touch([hit])
            blocks.append(hit)
            continue
        b = pool.get_new_blocks(1)[0]
        b._block_hash = h
        pool.cached_block_hash_to_block.insert(h, b)
        blocks.append(b)
    pool.free_blocks(list(reversed(blocks)))
    excl.track(rid, [blocks], list(range(BS * len(hashes))))
    return blocks


def _done(excl, conn):
    for job in conn.jobs:
        excl.on_job_done(job["id"])
    conn.jobs.clear()


def test_blocks_pinned_until_signal(env):
    pool, excl, conn = env
    free0 = pool.get_num_free_blocks()
    a = _run(pool, excl, "a", [b"h1", b"h2", b"h3"])
    # The only copy is in HBM: out of the free queue, so never evicted.
    assert all(b.ref_cnt == 1 for b in a)
    assert pool.get_num_free_blocks() == free0 - 3
    assert conn.jobs == []


def test_dram_then_same_tier_writes_only_new(env):
    pool, excl, conn = env
    free0 = pool.get_num_free_blocks()
    a = _run(pool, excl, "a", [b"h1", b"h2", b"h3"])
    excl.apply("a", "DRAM", _time(2.0, 9.0))
    (job,) = conn.jobs
    assert job["dram"] == [0, 1, 2] and job["drop"] == [] and job["ssd"] == []
    assert job["block_ids"] == [b.block_id for b in a]
    assert job["dram_pen"] == [9.0] * 3
    assert all(b.ref_cnt == 2 and b.block_hash is not None for b in a)
    assert excl.busy()
    _done(excl, conn)
    assert not excl.busy()
    assert all(b.ref_cnt == 0 and b.block_hash is None for b in a)
    assert pool.get_num_free_blocks() == free0

    # The next turn loads the prefix from DRAM into new blocks and adds one;
    # all four stay pinned until its signal.
    b = _run(pool, excl, "b", [b"h1", b"h2", b"h3", b"h4"])
    assert [x.ref_cnt for x in b] == [1, 1, 1, 1]
    excl.apply("b", "DRAM", _time(2.0, 9.0))
    (job,) = conn.jobs
    assert job["dram"] == [3] and job["drop"] == []
    # The prefix is already in DRAM: its HBM copy goes at once.
    assert all(x.block_hash is None and x.ref_cnt == 0 for x in b[:3])
    assert b[3].block_hash == b"h4"
    _done(excl, conn)
    assert b[3].block_hash is None and b[3].ref_cnt == 0


def test_tier_changes_move_blocks(env):
    pool, excl, conn = env
    _run(pool, excl, "a", [b"h1", b"h2"])
    excl.apply("a", "DRAM", _time(1.0, 4.0))
    _done(excl, conn)
    # HBM safe: stays resident, DRAM copies removed; held in the free queue
    # (the only copy), not counted free.
    c = _run(pool, excl, "c", [b"h1", b"h2", b"h3"])
    free = pool.get_num_free_blocks()
    excl.apply("c", "HBM", _time(8.0, 22.0))
    (job,) = conn.jobs
    assert job["drop"] == [0, 1] and job["dram"] == [] and job["ssd"] == []
    _done(excl, conn)
    assert all(x.block_hash is not None and x.ref_cnt == 0 for x in c)
    assert pool.get_num_free_blocks() == free
    # SSD safe next: everything written to SSD (nothing left below to drop).
    d = _run(pool, excl, "d", [b"h1", b"h2", b"h3", b"h4"])
    excl.apply("d", "SSD", _time(1.0, 1.0))
    (job,) = conn.jobs
    assert job["ssd"] == [0, 1, 2, 3] and job["drop"] == []
    _done(excl, conn)
    assert all(x.block_hash is None and x.ref_cnt == 0 for x in d)
    # Back to DRAM: removed from SSD and written to DRAM.
    _run(pool, excl, "e", [b"h1", b"h2", b"h3", b"h4", b"h5"])
    excl.apply("e", "DRAM", _time(1.0, 4.0))
    (job,) = conn.jobs
    assert job["drop"] == [0, 1, 2, 3] and job["dram"] == [0, 1, 2, 3, 4]


def test_no_next_turn_keeps_data_below(env):
    pool, excl, conn = env
    _run(pool, excl, "a", [b"h1", b"h2"])
    excl.apply("a", "SSD", _time(1.0, 1.0))
    _done(excl, conn)
    b = _run(pool, excl, "b", [b"h1", b"h2", b"h3"])
    excl.apply("b", None, None)
    (job,) = conn.jobs
    # Nothing is deleted: h1, h2 keep their SSD copy, h3 goes to SSD.
    assert job["drop"] == [] and job["dram"] == [] and job["ssd"] == [2]
    assert b[0].block_hash is None and b[1].block_hash is None
    assert b[2].block_hash == b"h3"
    _done(excl, conn)
    assert all(x.block_hash is None and x.ref_cnt == 0 for x in b)
    # The signal ended the head: another one for it means a broken run.
    with pytest.raises(RuntimeError):
        excl.apply("b", "DRAM", _time(1.0, 1.0))


def test_no_next_turn_marks_dram_copies_first_to_demote(env):
    pool, excl, conn = env
    _run(pool, excl, "a", [b"h1", b"h2"])
    excl.apply("a", "DRAM", _time(1.0, 4.0))
    _done(excl, conn)
    _run(pool, excl, "b", [b"h1", b"h2", b"h3"])
    excl.apply("b", None, None)
    (job,) = conn.jobs
    assert job["zero"] == [0, 1] and job["ssd"] == [2] and job["drop"] == []


def test_shared_prefix_follows_fastest_claim(env):
    pool, excl, conn = env
    # Two sessions share block s; session 2's signal is still pending.
    _run(pool, excl, "a1", [b"s", b"a2"])
    shared = _run(pool, excl, "b1", [b"s", b"b2"])
    excl.apply("a1", "DRAM", _time(1.0, 4.0))
    (job,) = conn.jobs
    # s is held in HBM for b1 (pending counts as HBM); only a2 moves.
    assert job["dram"] == [1]
    _done(excl, conn)
    assert shared[0].block_hash == b"s"
    excl.apply("b1", "SSD", _time(1.0, 1.0))
    (job,) = conn.jobs
    # a1 still needs s in DRAM, the faster of the two remaining claims.
    assert job["dram"] == [0] and job["ssd"] == [1]
    _done(excl, conn)
    assert shared[0].block_hash is None


def test_demote_cheapest_exposure_first(monkeypatch):
    pool, excl, conn = _make(monkeypatch, blocks=16)
    a = _run(pool, excl, "a", [b"a1", b"a2", b"a3"])
    b = _run(pool, excl, "b", [b"b1", b"b2", b"b3"])
    _run(pool, excl, "c", [b"c1", b"c2", b"c3"])
    excl.apply("a", "HBM", _time(9.0, 40.0))
    excl.apply("b", "HBM", _time(2.0, 30.0))
    assert conn.jobs == []  # nothing below to drop, nothing to write
    # a and b are held (only copy), c pinned (pending): none of them free.
    assert pool.get_num_free_blocks() == 15 - 9
    excl.demote(4)
    (job,) = conn.jobs
    # b (2 ms exposed from DRAM) before a (9 ms), tail blocks first.
    assert job["block_ids"] == [
        b[2].block_id, b[1].block_id, b[0].block_id, a[2].block_id
    ]
    assert job["dram"] == [0, 1, 2, 3] and job["drop"] == []
    assert job["dram_pen"] == [30.0] * 3 + [40.0]
    # Not counted twice while in flight.
    excl.demote(4)
    assert len(conn.jobs) == 1
    _done(excl, conn)
    assert pool.get_num_free_blocks() == 15 - 9 + 4
    assert all(x.block_hash is None for x in b) and a[2].block_hash is None
    # Forced out of HBM though claimed there: b's next turn reloads the
    # prefix, and its signal finds the DRAM copy to drop.
    _run(pool, excl, "b2", [b"b1", b"b2", b"b3", b"b4"])
    excl.apply("b2", "HBM", _time(2.0, 30.0))
    (job,) = conn.jobs
    assert job["drop"] == [0, 1, 2] and job["dram"] == []


def test_pending_blocks_are_never_demoted(monkeypatch):
    pool, excl, conn = _make(monkeypatch, blocks=16)
    a = _run(pool, excl, "a", [b"a1", b"a2"])
    c = _run(pool, excl, "c", [b"c1", b"c2", b"c3"])
    excl.apply("a", "HBM", _time(3.0, 10.0))
    # Short by far more than the two held blocks: only those are written.
    excl.demote(10)
    (job,) = conn.jobs
    assert job["block_ids"] == [a[1].block_id, a[0].block_id]
    assert all(x.block_hash is not None and x.ref_cnt == 1 for x in c)
    _done(excl, conn)
    assert all(x.block_hash is None for x in a)
    assert all(x.block_hash is not None and x.ref_cnt == 1 for x in c)
    # Nothing left to write until c's signal arrives.
    excl.demote(10)
    assert conn.jobs == []


def test_write_finishing_after_next_turn_keeps_block_pinned(env):
    pool, excl, conn = env
    a = _run(pool, excl, "a", [b"h1", b"h2"])
    excl.apply("a", "DRAM", _time(1.0, 4.0))
    (job,) = conn.jobs
    # The next turn hits both blocks in HBM and is freed before the write
    # job comes back: they now wait for its signal.
    b = _run(pool, excl, "b", [b"h1", b"h2", b"h3"])
    assert b[:2] == a
    _done(excl, conn)
    assert all(x.block_hash is not None and x.ref_cnt == 1 for x in b)
    excl.apply("b", "DRAM", _time(1.0, 4.0))
    (job,) = conn.jobs
    # h1, h2 already sit in DRAM: their HBM copy goes; only h3 is written.
    assert job["dram"] == [2] and job["drop"] == []
    assert all(x.block_hash is None and x.ref_cnt == 0 for x in b[:2])


def test_signal_needs_a_pending_head(env):
    pool, excl, conn = env
    with pytest.raises(RuntimeError):
        excl.apply("a", "DRAM", _time(1.0, 4.0))
    _run(pool, excl, "a", [b"h1", b"h2"])
    excl.apply("a", None, None)
    _done(excl, conn)
    with pytest.raises(RuntimeError):
        excl.apply("a", "DRAM", _time(1.0, 4.0))


def test_other_session_sharing_the_prompt_keeps_pending_head(env):
    pool, excl, conn = env
    a = _run(pool, excl, "a", [b"h1", b"h2"])
    # Another session's turn starts with a's whole prompt and is freed before
    # a's signal: a is not superseded, both get their signals.
    b = _run(pool, excl, "b", [b"h1", b"h2", b"h3"])
    excl.apply("b", "DRAM", _time(1.0, 4.0))
    (job,) = conn.jobs
    # h1, h2 stay in HBM for a (pending); only h3 moves.
    assert job["dram"] == [2]
    _done(excl, conn)
    assert all(x.block_hash is not None and x.ref_cnt == 1 for x in a)
    excl.apply("a", "SSD", _time(1.0, 1.0))
    (job,) = conn.jobs
    # b still needs them in DRAM, the faster claim.
    assert job["dram"] == [0, 1]
    _done(excl, conn)
    assert all(x.block_hash is None for x in b)


def test_unblocking_signal_keeps_a_chunk_free(monkeypatch):
    pool, excl, conn = _make(monkeypatch, blocks=16, reserve=8)
    a = _run(pool, excl, "a", [b"a1", b"a2", b"a3"])
    _run(pool, excl, "b", [b"b1", b"b2", b"b3"])
    _run(pool, excl, "c", [b"c1", b"c2"])
    assert pool.get_num_free_blocks() == 15 - 8
    # a stays in HBM (held, not free): one block short of the reserve, so
    # the cheapest held block goes to DRAM in the background.
    excl.apply("a", "HBM", _time(9.0, 40.0))
    (job,) = conn.jobs
    assert job["req"] == "demote" and job["block_ids"] == [a[2].block_id]
    # In flight: not demoted again.
    excl.apply("b", "HBM", _time(2.0, 30.0))
    assert len(conn.jobs) == 1
    _done(excl, conn)
    assert pool.get_num_free_blocks() == 8
    excl.apply("c", "HBM", _time(2.0, 30.0))
    assert conn.jobs == []


def test_exclusive_signal_keeps_no_reserve(monkeypatch):
    pool, excl, conn = _make(monkeypatch, blocks=16)
    _run(pool, excl, "a", [b"a1", b"a2", b"a3"])
    _run(pool, excl, "b", [b"b1", b"b2", b"b3", b"b4", b"b5"])
    excl.apply("a", "HBM", _time(9.0, 40.0))
    assert conn.jobs == [] and not excl.unblocking


def _held_pool(monkeypatch, reserve):
    """a and b held in HBM (b cheaper to expose), 9 blocks free."""
    pool, excl, conn = _make(monkeypatch, blocks=16, reserve=reserve)
    a = _run(pool, excl, "a", [b"a1", b"a2", b"a3"])
    b = _run(pool, excl, "b", [b"b1", b"b2", b"b3"])
    excl.apply("a", "HBM", _time(9.0, 40.0))
    excl.apply("b", "HBM", _time(2.0, 30.0))
    assert conn.jobs == [] and pool.get_num_free_blocks() == 9
    return pool, excl, conn, a, b


def test_lend_borrows_what_the_free_blocks_lack(monkeypatch):
    pool, excl, conn, a, b = _held_pool(monkeypatch, reserve=4)
    # The 9 free blocks hold it, the reserve included: nothing borrowed.
    assert excl.lend("r0", 9, 7, []) == 0
    # 11 blocks with a 9-block prefix: the 2 the free blocks lack are
    # borrowed (b's tail is cheapest to expose; b1 is the request's own hit).
    assert excl.lend("r", 11, 9, [b[0]]) == 2
    ((rid, ev),) = conn.evicts
    assert rid == "r" and ev["block_ids"] == [b[2].block_id, b[1].block_id]
    assert ev["hashes"] == [1002, 1001] and ev["dram_pen"] == [30.0, 30.0]
    assert b[2].block_hash is None and b[0].block_hash == b"b1"
    assert pool.get_num_free_blocks() == 11
    # Handed out before every other free block: the load's blocks are
    # allocated first.
    assert pool.get_new_blocks(2) == [b[2], b[1]]
    # Their next turn finds them in DRAM: its signal drops that copy.
    _run(pool, excl, "b2", [b"b1", b"b2", b"b3", b"b4"])
    excl.apply("b2", "HBM", _time(2.0, 30.0))
    (job,) = conn.jobs
    assert job["drop"] == [1, 2]


def test_lend_nothing_when_the_admission_cannot_fit(monkeypatch):
    pool, excl, conn, a, b = _held_pool(monkeypatch, reserve=4)
    # Short of 3 with a 2-block prefix: a lent block would land in the
    # computed suffix, which does not wait for its copy to DRAM.
    assert excl.lend("r", 12, 2, []) == 0
    # Short of 9, past all 6 held blocks.
    assert excl.lend("r", 18, 16, []) == 0
    assert conn.evicts == [] and pool.get_num_free_blocks() == 9
    assert excl.stats["lend_no_fit"] == 2


def test_lend_nothing_when_no_held_block_can_go(monkeypatch):
    pool, excl, conn, a, b = _held_pool(monkeypatch, reserve=4)
    # Short of 2, and every held block is the request's own hit: it waits,
    # and no empty eviction goes to the workers (2026-11-22 sweblock: one
    # did and LayerEvictor raised).
    own = list(a) + list(b)
    assert excl.lend("r", 11, 9, own) == 0
    assert conn.evicts == [] and pool.get_num_free_blocks() == 9
    assert excl.stats["lend_no_fit"] == 1


def test_admission_refills_the_reserve(monkeypatch):
    pool, excl, conn, a, b = _held_pool(monkeypatch, reserve=4)
    pool.get_new_blocks(7)  # an admission leaves 2 free
    excl.keep_reserve()
    (job,) = conn.jobs
    assert job["req"] == "demote"
    assert job["block_ids"] == [b[2].block_id, b[1].block_id]
    # In flight counts: not demoted twice.
    excl.keep_reserve()
    assert len(conn.jobs) == 1


def test_next_chunk_short_refills_the_reserve(monkeypatch):
    pool, excl, conn, a, b = _held_pool(monkeypatch, reserve=4)
    pool.get_new_blocks(9)  # a long prompt's first chunk took every free block
    excl.chunk_short()
    (job,) = conn.jobs
    assert job["req"] == "demote" and len(job["block_ids"]) == 4
    assert excl.stats["chunk_wait"] == 1
    # Still in flight on the next step: nothing more to demote.
    excl.chunk_short()
    assert len(conn.jobs) == 1 and excl.stats["chunk_wait"] == 2


def test_lend_skips_blocks_whose_lower_copy_is_being_dropped(monkeypatch):
    pool, excl, conn = _make(monkeypatch, blocks=16, reserve=2)
    _run(pool, excl, "a", [b"h1", b"h2"])
    excl.apply("a", "DRAM", _time(1.0, 4.0))
    _done(excl, conn)
    c = _run(pool, excl, "c", [b"h1", b"h2", b"h3"])
    excl.apply("c", "HBM", _time(1.0, 4.0))
    (job,) = conn.jobs
    assert job["drop"] == [0, 1]
    assert pool.get_num_free_blocks() == 12
    # The drop runs after anything written now: h1, h2 cannot be lent yet,
    # and h3 alone does not make a 14-block admission fit.
    assert excl.lend("r", 14, 4, []) == 0
    assert excl.lend("r", 13, 4, []) == 1
    assert conn.evicts[-1][1]["block_ids"] == [c[2].block_id]
    _done(excl, conn)
    assert excl.lend("r2", 15, 4, []) == 2
    assert sorted(conn.evicts[-1][1]["block_ids"]) == sorted(x.block_id for x in c[:2])


def test_demote_keeps_the_waiting_requests_own_prefix(monkeypatch):
    pool, excl, conn = _make(monkeypatch, blocks=16)
    a = _run(pool, excl, "a", [b"a1", b"a2", b"a3"])
    b = _run(pool, excl, "b", [b"b1", b"b2"])
    excl.apply("a", "HBM", _time(1.0, 40.0))
    excl.apply("b", "HBM", _time(9.0, 40.0))
    # Held: flagged, so a prefix hit on them takes no free capacity.
    assert all(x.kvp_held for x in a + b)
    # a's next turn waits for room; its prefix is the cheapest to expose but
    # it is about to use it.
    excl.demote(2, a)
    (job,) = conn.jobs
    assert job["block_ids"] == [b[1].block_id, b[0].block_id]
    pool.touch(a)
    assert not any(x.kvp_held for x in a)
