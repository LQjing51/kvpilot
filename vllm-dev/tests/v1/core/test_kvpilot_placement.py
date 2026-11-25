# SPDX-License-Identifier: Apache-2.0
"""KVPilot placement pieces: heap, class queue, claims, inclusive, signal."""

import json
import math
import random

import pytest

from vllm.v1.core import kvpilot_blockwise, kvpilot_heap, kvpilot_placement, kvpilot_signal
from vllm.v1.core.kv_cache_utils import (
    FreeKVCacheClassQueue,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.kvpilot_claims import Claims
from vllm.v1.core.kvpilot_inclusive import InclusivePlacement
from vllm.v1.core.kvpilot_states import boundary_states

RID = "{:08d}-aaaa-4bbb-8ccc-dddddddddddd"
WARMUP = "ffffffff-aaaa-4bbb-8ccc-dddddddddddd"


@pytest.mark.parametrize("seed", range(5))
def test_heap_matches_reference(seed):
    """Random op sequences: the C++ heap pops the same ids in the same order."""
    cpp = kvpilot_heap.heap_class()
    if cpp is kvpilot_heap.PyIndexedHeap:
        pytest.skip("C++ heap unavailable")
    rng = random.Random(seed)
    n = 300
    a, b = kvpilot_heap.PyIndexedHeap(n), cpp(n)
    inside = set()
    keys = [-math.inf, 1.0, 2.0, 3.0, 4.0]
    for _ in range(20000):
        op = rng.random()
        if op < 0.35 and len(inside) < n:
            i = rng.choice([x for x in range(n) if x not in inside])
            k = rng.choice(keys)
            a.push(i, k)
            b.push(i, k)
            inside.add(i)
        elif op < 0.5 and inside:
            i = rng.choice(sorted(inside))
            assert a.remove(i) and b.remove(i)
            inside.discard(i)
        elif op < 0.7 and inside:
            i = rng.choice(sorted(inside))
            k = rng.choice(keys)
            assert a.update(i, k) and b.update(i, k)
        elif inside:
            m = rng.randint(1, min(5, len(inside)))
            pa, pb = a.pop_n(m), b.pop_n(m)
            assert pa == pb
            inside.difference_update(pa)
        assert a.size() == b.size() == len(inside)
    assert sorted(a.ids()) == sorted(b.ids())


@pytest.fixture
def py_heap(monkeypatch):
    monkeypatch.setattr(kvpilot_heap, "_CLS", kvpilot_heap.PyIndexedHeap)


def _queue(n):
    blocks = [KVCacheBlock(i) for i in range(n)]
    q = FreeKVCacheClassQueue(blocks)
    q.popleft_n(n)
    return blocks, q


def _hash(b, v):
    b._block_hash = v  # bypass the set-once setter for the test


def test_class_queue_order_and_rekey(py_heap):
    blocks, q = _queue(8)
    for b in blocks[1:7]:
        _hash(b, b"x%d" % b.block_id)
    q.append_n(blocks[1:7])  # free order 1..6, all at the pending class
    q.append_n([blocks[7]])  # no KV: before every class
    ssd = float(kvpilot_placement.CLASS["SSD"])
    assert q.rekey([blocks[2], blocks[5]], [ssd, ssd]) == 2
    assert q.rekey([blocks[3]], [float(kvpilot_placement.CLASS[None])]) == 1
    assert q.rekey([blocks[0]], [1.0]) == 0  # not free
    assert [b.block_id for b in q.popleft_n(7)] == [7, 3, 2, 5, 1, 4, 6]


def test_class_queue_state_blocks_first(py_heap):
    """A freed linear-attention state goes before every class, after blocks
    without KV."""
    blocks, q = _queue(4)
    q.set_state_groups([0, 1])
    _hash(blocks[0], make_block_hash_with_group_id(b"a", 2))
    _hash(blocks[1], make_block_hash_with_group_id(b"a", 1))
    _hash(blocks[2], make_block_hash_with_group_id(b"a", 0))
    q.append_n(blocks)
    assert q.rekey([blocks[0]], [float(kvpilot_placement.CLASS[None])]) == 1
    assert [b.block_id for b in q.popleft_n(4)] == [3, 1, 2, 0]


def test_class_queue_prepend_and_remove(py_heap):
    blocks, q = _queue(4)
    for b in blocks:
        _hash(b, bytes([b.block_id]))
    q.append_n(blocks[:3])
    q.prepend_n([blocks[3]])
    q.remove(blocks[1])
    assert sorted(b.block_id for b in q.get_all_free_blocks()) == [0, 2, 3]
    assert [b.block_id for b in q.popleft_n(3)] == [3, 0, 2]
    with pytest.raises(RuntimeError):
        q.remove(blocks[1])


def test_class_queue_held_blocks(py_heap):
    blocks, q = _queue(6)
    for b in blocks:
        _hash(b, bytes([b.block_id]))
    q.append_n(blocks[:5])
    q.hold([blocks[1], blocks[2], blocks[3]], [7.0, 2.0, 7.0])
    # Held blocks are not free: the allocator never gets them.
    assert q.num_free_blocks == 2
    assert [b.block_id for b in q.held()] == [2, 1, 3]
    with pytest.raises(ValueError):
        q.popleft_n(3)
    q.remove(blocks[2])  # a prefix hit takes it back
    q.rekey([blocks[1]], [-math.inf])  # written below: blank again
    assert [b.block_id for b in q.held()] == [3]
    assert q.num_free_blocks == 3
    assert sorted(b.block_id for b in q.popleft_n(3)) == [0, 1, 4]


class _B:
    def __init__(self, h):
        self.block_hash = h


def _head(claims, rid, hashes):
    return claims.track(rid, [_B(h) for h in hashes], list(range(len(hashes))))


def test_claims_supersede_and_fastest_tier():
    c = Claims()
    _head(c, "a1", ["s", "a"])
    _head(c, "b1", ["s", "b"])
    assert c.required("s") == "HBM" and c.required("a") == "HBM"  # pending
    c.signal("a1", "SSD", [1.0, 2.0, 3.0])
    assert c.required("a") == "SSD" and c.required("s") == "HBM"
    c.signal("b1", "DRAM", [1.0, 2.0, 3.0])
    assert c.required("s") == "DRAM"
    # a's next turn holds a1's last block: a1 gives up its claims.
    _head(c, "a2", ["s", "a", "a+"])
    assert "a1" not in c.heads
    assert c.required("a") == "HBM" and c.required("s") == "HBM"
    c.signal("a2", None, None)
    assert "a2" not in c.heads
    assert c.required("a") is None and c.required("s") == "DRAM"
    assert [h.request_id for h in c.claimants("s")] == ["b1"]


def test_claims_pending_head_survives_other_sessions():
    c = Claims()
    # Two trajectories of one task: identical first prompts, both pending.
    _head(c, "x0", ["s", "t"])
    _head(c, "y0", ["s", "t"])
    # A longer turn of a third session holding the same start, also pending.
    _head(c, "z5", ["s", "t", "z"])
    assert set(c.heads) == {"x0", "y0", "z5"}
    c.signal("x0", "DRAM", [1.0, 2.0, 3.0])
    c.signal("y0", "SSD", [1.0, 2.0, 3.0])
    assert c.required("t") == "HBM"  # z5 still pending
    # x's next turn supersedes every signalled head ending at t, y0 too: the
    # prefill side cannot tell sessions apart. The blocks stay claimed by x1
    # and z5.
    _head(c, "x1", ["s", "t", "x"])
    assert set(c.heads) == {"z5", "x1"}
    c.signal("x1", None, None)
    c.signal("z5", None, None)
    assert c.required("s") is None and not c.heads


def test_claims_signal_needs_a_pending_head():
    c = Claims()
    with pytest.raises(RuntimeError):
        c.signal("x", "HBM", [1.0, 2.0, 3.0])
    _head(c, "x", ["h"])
    c.signal("x", "HBM", [1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError):
        c.signal("x", "HBM", [1.0, 2.0, 3.0])


class _Conn:
    def __init__(self):
        self.groups = []
        self.orders = []

    def kvpilot_chunk_hashes(self, request_id, token_ids, n):
        return [500 + i for i in range(n)]

    def kvpilot_group_order(self, lmc):
        self.orders.append(list(lmc))

    def kvpilot_set_group_placement(self, lmc, tiers, penalties):
        self.groups.append((list(lmc), list(tiers), list(penalties)))
        return len(lmc)


class _Pool:
    def __init__(self, queue):
        self.free_block_queue = queue

    def touch(self, blocks):
        for b in blocks:
            if b.ref_cnt == 0:
                self.free_block_queue.remove(b)
            b.ref_cnt += 1

    def free_blocks(self, blocks):
        for b in blocks:
            b.ref_cnt -= 1
        self.free_block_queue.append_n([b for b in blocks if b.ref_cnt == 0])


def test_inclusive_reclasses_hbm_and_dram(py_heap):
    blocks, q = _queue(6)
    for i, b in enumerate(blocks[:4]):
        _hash(b, b"h%d" % i)
    inc = InclusivePlacement(_Pool(q), block_size=16)
    conn = _Conn()
    inc.bind(conn)
    q.append_n([blocks[1], blocks[0]])
    # 40 prompt tokens: two full blocks tracked, the partial third is not.
    inc.track("a", [[blocks[0], blocks[1], blocks[4]]], list(range(40)))
    q.append_n([blocks[3], blocks[2]])
    inc.track("b", [[blocks[2], blocks[3]]], list(range(32)))
    out = inc.apply("a", "SSD", [1.0, 2.0, 3.0])
    assert out["hbm_rekeyed"] == 2 and out["blocks"] == 2
    assert conn.groups == [([500, 501], ["SSD", "SSD"], [0.0, 0.0])]
    inc.apply("b", None, None)
    assert conn.groups[-1] == ([500, 501], [None, None], [0.0, 0.0])
    # No next turn (b) before SSD (a); free order within each.
    assert [x.block_id for x in q.popleft_n(4)] == [3, 2, 1, 0]



def test_inclusive_hybrid_holds_the_end_state(py_heap):
    """Hybrid model (state groups 0 and 1, attention 2): the states at the
    end of a head's prompt stay out of the free queue until its signal says
    there is no next turn, then go first with the other freed states."""
    blocks, q = _queue(10)
    q.set_state_groups([0, 1])
    null = KVCacheBlock(99, is_null=True)
    attn = blocks[0:3]  # 2 full blocks + a partial one
    _hash(attn[0], make_block_hash_with_group_id(b"h0", 2))
    _hash(attn[1], make_block_hash_with_group_id(b"h1", 2))
    g0 = [null, blocks[3], blocks[4]]  # state at the end of block 1, running
    g1 = [null, blocks[5], blocks[6]]
    _hash(g0[1], make_block_hash_with_group_id(b"h1", 0))
    _hash(g1[1], make_block_hash_with_group_id(b"h1", 1))
    mid = blocks[7]  # the state after block 0, freed during the prefill
    _hash(mid, make_block_hash_with_group_id(b"h0", 0))
    inc = InclusivePlacement(_Pool(q), block_size=16)
    inc.bind(_Conn())
    q.append_n([mid])
    # The end states were taken when the request finished; the rest frees,
    # each group tail first.
    end = boundary_states([attn, g0, g1], attn[:2])
    assert end == [blocks[3], blocks[5]]
    for s in end:
        s.ref_cnt = 1
    q.append_n([g0[2], g1[2]])
    q.append_n(attn[::-1])
    inc.track("a", [attn], list(range(40)), end)
    assert not {3, 5} & {b.block_id for b in q.get_all_free_blocks()}
    assert len(inc.states) == 1
    inc.apply("a", None, None)
    assert len(inc.states) == 0
    # Running states and the partial block, then the states (mid prefill
    # first), then the prefix with no next turn.
    assert [b.block_id for b in q.popleft_n(8)] == [4, 6, 2, 7, 3, 5, 1, 0]
    with pytest.raises(RuntimeError):
        boundary_states([attn, [null, null, blocks[4]]], attn[:2])


def test_inclusive_penalty_keys_by_the_model(py_heap, monkeypatch):
    monkeypatch.setenv("KVPILOT_PLACEMENT", kvpilot_placement.INCLUSIVE_PENALTY)
    monkeypatch.setattr(kvpilot_blockwise, "reqwise",
                        lambda model, n, dist: ("DRAM", [10.0, 12.0, 40.0]))
    blocks, q = _queue(6)
    for i, b in enumerate(blocks[:5]):
        _hash(b, b"h%d" % i)
    inc = InclusivePlacement(_Pool(q), block_size=16, model=object(), by_penalty=True)
    conn = _Conn()
    inc.bind(conn)
    q.append_n([blocks[1], blocks[0]])
    inc.track("a", [[blocks[0], blocks[1]]], list(range(32)))
    q.append_n([blocks[3], blocks[2]])
    inc.track("b", [[blocks[2], blocks[3]]], list(range(32)))
    # c shares a's first block and is still waiting for its signal.
    inc.track("c", [[blocks[0], blocks[4]]], list(range(32)))
    with pytest.raises(RuntimeError):
        inc.apply("a", "HBM", [1.0, 2.0, 3.0])  # no suffix
    # The model's tier and times replace the signal's.
    out = inc.apply("a", "HBM", [1.0, 2.0, 3.0], [[1.0, 300]])
    assert out["tier"] == "DRAM"
    pending = kvpilot_placement.PENDING_PENALTY
    assert conn.groups[-1] == ([500, 501], ["HBM", "DRAM"], [pending, 28.0])
    inc.apply("b", None, None)
    assert conn.groups[-1] == ([500, 501], [None, None], [0.0, 0.0])
    # b (no next turn: 0) before a's block 1 (penalty 2); block 0 waits for c.
    assert [x.block_id for x in q.popleft_n(4)] == [3, 2, 1, 0]



def test_inclusive_safe_classes_by_the_model_tier(py_heap, monkeypatch):
    monkeypatch.setattr(kvpilot_blockwise, "reqwise",
                        lambda model, n, dist: ("SSD", [10.0, 10.0, 10.0]))
    blocks, q = _queue(4)
    for i, b in enumerate(blocks[:2]):
        _hash(b, b"h%d" % i)
    inc = InclusivePlacement(_Pool(q), block_size=16, model=object())
    conn = _Conn()
    inc.bind(conn)
    q.append_n([blocks[1], blocks[0]])
    inc.track("a", [[blocks[0], blocks[1]]], list(range(32)))
    out = inc.apply("a", "HBM", [1.0, 2.0, 3.0], [[1.0, 2000]])
    assert out["tier"] == "SSD"
    assert conn.groups == [([500, 501], ["SSD", "SSD"], [0.0, 0.0])]
    with pytest.raises(ValueError):
        InclusivePlacement(_Pool(q), block_size=16, by_penalty=True)


def test_penalties():
    assert kvpilot_placement.penalties([10.0, 12.5, 30.0]) == (2.5, 17.5)
    # Measurement noise can put a slower tier ahead: never negative.
    assert kvpilot_placement.penalties([10.0, 9.0, 8.0]) == (0.0, 0.0)


def test_strategy_rejects_unknown(monkeypatch):
    monkeypatch.setenv("KVPILOT_PLACEMENT", "penalty")
    with pytest.raises(ValueError):
        kvpilot_placement.strategy()


def test_trace_line():
    assert kvpilot_signal.trace_line(RID.format(42)) == 42
    assert kvpilot_signal.trace_line(WARMUP) is None
    with pytest.raises(ValueError):
        kvpilot_signal.trace_line("3f2a9c10-aaaa-4bbb-8ccc-dddddddddddd")


def test_offline_table(tmp_path):
    path = tmp_path / "table.json"
    path.write_text(json.dumps({
        "7": {"own_tier": "HBM", "next_tier": "DRAM", "next_time": [10.0, 12.0, 30.0],
              "next_suffix": 640},
        "8": {"own_tier": "DRAM", "next_tier": None, "next_time": None,
              "next_suffix": None},
    }))
    table = kvpilot_signal.OfflineTable(str(path))
    text = lambda: ""  # noqa: E731
    none = {"tier": None, "time": None, "suffix": None}
    assert table.signal(RID.format(7), 1024, 10, text) == {
        "tier": "DRAM", "time": [10.0, 12.0, 30.0], "suffix": [[1.0, 640]]
    }
    assert table.signal(RID.format(8), 1024, 10, text) == none
    assert table.signal(WARMUP, 4096, 1, text) == none
    with pytest.raises(KeyError):
        table.signal(RID.format(9), 1024, 10, text)
