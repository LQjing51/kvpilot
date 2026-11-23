# SPDX-License-Identifier: Apache-2.0
"""Indexed min-heap for the KVPilot class free block queue.

The C++ version (csrc/kvpilot_heap.cpp) is built on first use with torch's JIT
extension loader and cached under the torch extensions dir; concurrent workers
serialize on its file lock. `PyIndexedHeap` has the same interface and exists
for hosts without a compiler (and as the reference the C++ one is tested
against).
"""

from __future__ import annotations

import heapq
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

_SRC = Path(__file__).resolve().parent / "csrc" / "kvpilot_heap.cpp"
_CLS = None


class PyIndexedHeap:
    """heapq with lazy deletion; same interface as the C++ IndexedHeap."""

    def __init__(self, capacity: int):
        self._entry: list[tuple[float, int, int] | None] = [None] * capacity
        self._heap: list[tuple[float, int, int]] = []
        self._seq = 0
        self._n = 0

    def size(self) -> int:
        return self._n

    def contains(self, block_id: int) -> bool:
        return self._entry[block_id] is not None

    def key(self, block_id: int) -> float:
        e = self._entry[block_id]
        if e is None:
            raise KeyError(block_id)
        return e[0]

    def push(self, block_id: int, key: float) -> None:
        self.push_seq(block_id, key, self._seq)
        self._seq += 1

    def push_seq(self, block_id: int, key: float, seq: int) -> None:
        if self._entry[block_id] is not None:
            raise RuntimeError("id already in heap")
        e = (key, seq, block_id)
        self._entry[block_id] = e
        heapq.heappush(self._heap, e)
        self._n += 1

    def push_many(self, ids, keys) -> None:
        for i, k in zip(ids, keys):
            self.push(i, k)

    def push_many_seq(self, ids, keys, seqs) -> None:
        for i, k, s in zip(ids, keys, seqs):
            self.push_seq(i, k, s)

    def remove(self, block_id: int) -> bool:
        if self._entry[block_id] is None:
            return False
        self._entry[block_id] = None
        self._n -= 1
        self._maybe_compact()
        return True

    def update(self, block_id: int, key: float) -> bool:
        e = self._entry[block_id]
        if e is None:
            return False
        ne = (key, e[1], block_id)
        self._entry[block_id] = ne
        heapq.heappush(self._heap, ne)
        self._maybe_compact()
        return True

    def update_many(self, ids, keys) -> int:
        return sum(self.update(i, k) for i, k in zip(ids, keys))

    def pop(self) -> int:
        while True:
            e = heapq.heappop(self._heap)
            if self._entry[e[2]] is e:
                self._entry[e[2]] = None
                self._n -= 1
                return e[2]

    def pop_n(self, n: int) -> list[int]:
        if n > self._n:
            raise IndexError("pop_n beyond heap size")
        return [self.pop() for _ in range(n)]

    def ids(self) -> list[int]:
        return [i for i, e in enumerate(self._entry) if e is not None]

    def clear(self) -> None:
        self._entry = [None] * len(self._entry)
        self._heap = []
        self._n = 0

    def _maybe_compact(self) -> None:
        if len(self._heap) > 2 * self._n + 64:
            self._heap = [e for e in self._heap if self._entry[e[2]] is e]
            heapq.heapify(self._heap)


def heap_class():
    """The IndexedHeap class to use, building the C++ one on first call."""
    global _CLS
    if _CLS is not None:
        return _CLS
    try:
        from torch.utils.cpp_extension import load

        mod = load(
            name="kvpilot_heap",
            sources=[str(_SRC)],
            extra_cflags=["-O3"],
            verbose=False,
        )
        _CLS = mod.IndexedHeap
        logger.info("KVPilot heap: C++ (%s)", mod.__file__)
    except Exception:
        logger.warning(
            "KVPilot heap: C++ build failed, using python", exc_info=True
        )
        _CLS = PyIndexedHeap
    return _CLS
