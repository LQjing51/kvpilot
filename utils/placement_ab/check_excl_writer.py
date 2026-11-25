"""Byte check of the exclusive-placement writer (lmcache/v1/kvpilot_exclusive.py)
on one GPU, no service: paged KV in vLLM's flash-attn layout, a fake Mooncake
backend that hands out pinned buffers as DRAM replicas and copies SSD writes
out of the writer's staging buffer. Every written value must equal the
token-major [512, 2, hidden] slice of its block and layer; SSD-bound chunks
must not get DRAM replicas and go one bucket per chunk.

    PYTHONPATH=... python3 check_excl_writer.py
"""
import ctypes
import time
from types import SimpleNamespace

import torch

from lmcache.utils import CacheEngineKey
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemLayerwiseGPUConnector
from lmcache.v1.kvpilot_exclusive import ExclusiveWriter

L, NB, BS, NH, HS = 6, 40, 512, 2, 128
dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
torch.manual_seed(0)
kv = [torch.randn(2, NB, BS, NH, HS, dtype=torch.bfloat16, device=dev) for _ in range(L)]


class Backend:
    def __init__(self):
        self.bufs, self.ended, self.removed, self.groups = {}, [], [], {}
        self.disk, self.buckets, self.stored = {}, [], set()

    def kvpilot_set_group_placement(self, groups, tiers, pens):
        self.groups.update(zip(groups, zip(tiers, pens)))
        return len(groups)

    def kvpilot_put_start_local(self, names, sizes):
        out = []
        for n, size in zip(names, sizes):
            if n in self.bufs or "skip" in n:
                out.append(0)
                continue
            self.bufs[n] = torch.zeros(size, dtype=torch.uint8, pin_memory=True)
            out.append(self.bufs[n].data_ptr())
        return out

    def kvpilot_put_end(self, names):
        self.ended += names
        return [0] * len(names)

    def kvpilot_put_revoke(self, names):
        return [0] * len(names)

    def kvpilot_remove_memory(self, names):
        self.removed += names
        return len(names)

    def kvpilot_exist(self, names):
        return [1 if n in self.bufs or n in self.disk or n in self.stored else 0
                for n in names]

    def kvpilot_offload_local(self, names, ptrs, sizes):
        for n, ptr, size in zip(names, ptrs, sizes):
            buf = torch.empty(size, dtype=torch.uint8)
            ctypes.memmove(buf.data_ptr(), ptr, size)
            self.disk[n] = buf
        self.buckets.append(list(names))
        return len(names)


class TokenDB:
    def process_tokens(self, hashes=None, offsets=None, **_):
        start = 0
        for h, o in zip(hashes, offsets):
            yield start, start + o, CacheEngineKey("m", 2, 0, h, torch.bfloat16)
            start += o


backend = Backend()
gc = VLLMPagedMemLayerwiseGPUConnector(
    hidden_dim_size=NH * HS, num_layers=L, use_gpu=True, chunk_size=BS,
    dtype=torch.bfloat16, device=dev,
)
engine = SimpleNamespace(
    gpu_connector=gc, num_layers=L, config=SimpleNamespace(chunk_size=BS),
    token_database=TokenDB(),
    storage_manager=SimpleNamespace(get_active_storage_backends=lambda: [("RemoteBackend", backend)]),
)
w = ExclusiveWriter(engine, kv)
block_ids = [5, 9, 2, 33, 17, 0, 39]
hashes = [0x1000 + i for i in range(len(block_ids))]
names = [[k.to_string() for k in CacheEngineKey("m", 2, 0, h, torch.bfloat16).split_layers(L)]
         for h in hashes]
# Chunk 6's layer 0 is already stored somewhere: only its other layers go.
backend.stored.add(names[6][0])
w.submit({"id": "kvx-0", "req": "r", "tier": "DRAM", "hashes": hashes, "block_ids": block_ids,
          "drop": [1], "dram": [0, 2, 4, 5], "ssd": [1, 6], "dram_pen": [3.0, 3.0, 1.0, 1.0],
          "zero": []})
deadline = time.time() + 30
done = set()
while not done and time.time() < deadline:
    done = w.take_done()
    time.sleep(0.01)
assert done == {"kvx-0"}, done

bad = 0
for c, where in ((0, backend.bufs), (2, backend.bufs), (4, backend.bufs), (5, backend.bufs),
                 (1, backend.disk), (6, backend.disk)):
    for layer in range(L):
        name = names[c][layer]
        if name in backend.stored:
            assert name not in backend.disk
            continue
        ref = kv[layer][:, block_ids[c]].permute(1, 0, 2, 3).reshape(BS, 2, NH * HS)
        got = where[name].view(torch.bfloat16).view(BS, 2, NH * HS)
        if not torch.equal(ref.cpu(), got):
            bad += 1
assert bad == 0, f"{bad} values differ"
assert len(backend.ended) == 4 * L, len(backend.ended)
# SSD-bound chunks never got a DRAM replica; one bucket per chunk.
assert not any(n in backend.bufs for c in (1, 6) for n in names[c])
assert backend.buckets == [names[1], names[6][1:]], backend.buckets
assert len(backend.removed) == L
# Tier codes as the master's KvpTier: DRAM 2, SSD 1.
assert sorted(set(backend.groups.values())) == [(1, 0.0), (2, 1.0), (2, 3.0)], backend.groups
print(f"OK: {4 * L} DRAM + {2 * L - 1} SSD values match the paged KV, "
      f"{len(backend.buckets)} buckets, {L} removed")
