# SPDX-License-Identifier: Apache-2.0
"""KVPilot Belady baseline: next uses from the trace, keys on release."""

import json

import pytest

from vllm.v1.core import kvpilot_belady, kvpilot_heap
from vllm.v1.core.kv_cache_utils import (
    FreeKVCacheClassQueue,
    KVCacheBlock,
    make_block_hash_with_group_id,
)

BS = 16
RID = "{:08d}-aaaa-4bbb-8ccc-dddddddddddd"


def _trace(tmp_path, rows):
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


@pytest.fixture(autouse=True)
def trace_block(monkeypatch):
    """The trace's tokens per hash id: one test block."""
    monkeypatch.setattr(kvpilot_belady, "TRACE_BLOCK", BS)


def test_next_uses_groups_trace_hashes(tmp_path, monkeypatch):
    """A block of several trace hash ids (Qwen3.6: 1536 / 512) is named by the
    chain up to its last one."""
    monkeypatch.setattr(kvpilot_belady, "TRACE_BLOCK", BS // 4)
    path = _trace(tmp_path, [
        {"input_length": 2 * BS, "hash_ids": list(range(8))},
        {"input_length": BS + 3, "hash_ids": [0, 1, 2, 3, 9]},       # block 0 again
        {"input_length": 2 * BS, "hash_ids": [0, 1, 2, 5, 4, 5, 6, 7]},  # other prefix
    ])
    assert kvpilot_belady.next_uses(path, BS) == [[1, None], [None], [None, None]]
    with pytest.raises(ValueError):
        kvpilot_belady.next_uses(path, BS + 1)


def test_next_uses(tmp_path):
    path = _trace(tmp_path, [
        {"input_length": 3 * BS, "hash_ids": [1, 2, 3]},
        {"input_length": 2 * BS + 5, "hash_ids": [7, 8, 9]},   # another session
        {"input_length": 4 * BS, "hash_ids": [1, 2, 3, 4]},    # line 0's next turn
        {"input_length": 2 * BS, "hash_ids": [1, 5]},          # shares block 0 only
        {"input_length": 2 * BS, "hash_ids": [9, 2]},          # same id, other prefix
    ])
    nxt = kvpilot_belady.next_uses(path, BS)
    assert nxt[0] == [2, 2, 2]
    assert nxt[1] == [None, None]          # its partial third block is not counted
    assert nxt[2] == [3, None, None, None]
    assert nxt[3] == [None, None]
    assert nxt[4] == [None, None]


class _Conn:
    def __init__(self):
        self.groups, self.orders = [], []

    def kvpilot_chunk_hashes(self, request_id, token_ids, n):
        return [500 + i for i in range(n)]

    def kvpilot_group_order(self, lmc):
        self.orders.append(list(lmc))

    def kvpilot_set_group_placement(self, lmc, tiers, penalties):
        self.groups.append((list(lmc), list(tiers), list(penalties)))


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


def test_release_keys_by_next_use(tmp_path, monkeypatch):
    monkeypatch.setenv("KVPILOT_PLACEMENT", "belady")
    monkeypatch.setattr(kvpilot_heap, "_CLS", kvpilot_heap.PyIndexedHeap)
    path = _trace(tmp_path, [
        {"input_length": 2 * BS, "hash_ids": [1, 2]},
        {"input_length": 2 * BS, "hash_ids": [7, 8]},
        {"input_length": 2 * BS, "hash_ids": [7, 8]},          # reuses line 1
        {"input_length": 3 * BS, "hash_ids": [1, 2, 3]},       # reuses line 0
    ])
    blocks = [KVCacheBlock(i) for i in range(6)]
    q = FreeKVCacheClassQueue(blocks)
    q.popleft_n(6)
    for i, b in enumerate(blocks[:4]):
        b._block_hash = b"h%d" % i
    bel = kvpilot_belady.BeladyPlacement(_Pool(q), BS, path)
    conn = _Conn()
    bel.bind(conn)
    q.append_n([blocks[1], blocks[0]])
    bel.track(RID.format(0), [[blocks[0], blocks[1]]], list(range(2 * BS)))
    q.append_n([blocks[3], blocks[2]])
    bel.track(RID.format(1), [[blocks[2], blocks[3]]], list(range(2 * BS)))
    far = kvpilot_belady.FAR
    assert conn.groups == [([500, 501], [None, None], [far - 3, far - 3]),
                           ([500, 501], [None, None], [far - 2, far - 2])]
    # Line 0's blocks come back later (line 3) than line 1's (line 2): first out.
    assert [b.block_id for b in q.popleft_n(4)] == [1, 0, 3, 2]
    with pytest.raises(RuntimeError):
        bel.apply(RID.format(0), "HBM", [1.0, 2.0, 3.0])


def test_hybrid_state_held_while_reused(tmp_path, monkeypatch):
    """Hybrid model: the state at a prompt's end stays in HBM when a later
    line reuses the prompt; a state no prompt reuses, or from mid prefill,
    goes before everything."""
    monkeypatch.setenv("KVPILOT_PLACEMENT", "belady")
    monkeypatch.setattr(kvpilot_heap, "_CLS", kvpilot_heap.PyIndexedHeap)
    path = _trace(tmp_path, [
        {"input_length": 2 * BS, "hash_ids": [1, 2]},
        {"input_length": BS, "hash_ids": [7]},                 # never again
        {"input_length": 3 * BS, "hash_ids": [1, 2, 3]},       # reuses line 0
    ])
    blocks = [KVCacheBlock(i) for i in range(8)]
    q = FreeKVCacheClassQueue(blocks)
    q.popleft_n(8)
    q.set_state_groups([0])
    h = [b"h%d" % i for i in range(3)]
    attn, state = 1, 0
    blocks[0]._block_hash = make_block_hash_with_group_id(h[0], attn)
    blocks[1]._block_hash = make_block_hash_with_group_id(h[1], attn)
    blocks[2]._block_hash = make_block_hash_with_group_id(h[0], state)  # mid-prefill
    blocks[3]._block_hash = make_block_hash_with_group_id(h[1], state)  # at the end
    blocks[4]._block_hash = make_block_hash_with_group_id(h[2], attn)
    blocks[5]._block_hash = make_block_hash_with_group_id(h[2], state)
    bel = kvpilot_belady.BeladyPlacement(_Pool(q), BS, path)
    bel.bind(_Conn())
    # The end states were taken when the requests finished (one reference).
    blocks[3].ref_cnt = blocks[5].ref_cnt = 1
    q.append_n([blocks[2]])
    q.append_n([blocks[1], blocks[0]])
    bel.track(RID.format(0), [[blocks[0], blocks[1]]], list(range(2 * BS)), [blocks[3]])
    q.append_n([blocks[4]])
    bel.track(RID.format(1), [[blocks[4]]], list(range(BS)), [blocks[5]])
    assert len(bel.states) == 1
    # States first (mid prefill, then line 1's), then line 1's block (never
    # again), then line 0's; line 0's end state is held.
    assert [b.block_id for b in q.popleft_n(5)] == [2, 5, 4, 1, 0]
    assert q.num_free_blocks == 0
