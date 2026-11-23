# SPDX-License-Identifier: Apache-2.0
"""KVPilot Belady baseline: next uses from the trace, keys on release."""

import json

import pytest

from vllm.v1.core import kvpilot_belady, kvpilot_heap
from vllm.v1.core.kv_cache_utils import FreeKVCacheClassQueue, KVCacheBlock

BS = 16
RID = "{:08d}-aaaa-4bbb-8ccc-dddddddddddd"


def _trace(tmp_path, rows):
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


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
