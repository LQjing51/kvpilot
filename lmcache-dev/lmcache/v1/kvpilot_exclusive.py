# SPDX-License-Identifier: Apache-2.0
"""KVPilot exclusive placement, worker side: deferred writes and drops.

Under exclusive-penalty-reqwise a prefill writes nothing below HBM. When the
next turn's tier is known, the scheduler (vllm/v1/core/kvpilot_exclusive.py)
sends each TP rank the same job: which prompt chunks lose their DRAM copy
(they leave DRAM; an SSD copy always stays) and which to write to DRAM or SSD
from HBM blocks it keeps pinned. A
rank runs its jobs in order on one background thread and reports a job id
back through get_finished once done; the scheduler unpins the blocks and drops
their HBM copies when every rank has.

Either write first tells the master the chunk groups' tier and penalty
(SetGroupPlacement); when DRAM fills up the master moves the DRAM groups with
the smallest DRAM -> SSD penalty to SSD.

To DRAM:
    1. PutStart allocates a memory replica per (chunk, layer) value in this
       rank's segment, beside an SSD copy if it has one; 0 = already in
       memory or no room (skipped).
    2. Per layer: gather the chunks from the paged KV into a GPU staging buffer
       (token-major, the value layout), then one batched D2H copies each value
       straight into its replica (the segment is CUDA-registered); no host
       memmove.
    3. PutEnd once the copies are done.

To SSD, without touching the segment (SSD-bound values in DRAM would push the
DRAM tier out):
    1. Values already stored are skipped.
    2. All layers of up to SSD_GROUP_CHUNKS chunks are gathered on the GPU and
       copied into a pinned staging buffer of the writer's own.
    3. Each chunk's values are written to this rank's SSD as one bucket, which
       registers their disk replicas with the master (no memory replica).

None of it runs on the forward thread or stream.

exclusive-penalty-reqwise-unblocking adds the Evictor: an admission short of
HBM borrows held blocks for the prefix its load fills, instead of waiting for
them to be written to DRAM. Their KV goes to DRAM within that load: PutStart on
the forward thread before the load starts, a per-layer copy on the GPU
connector's evict stream that each layer's scatter waits for, then the groups'
tier / penalty and PutEnd on a thread of its own.
"""

# Standard
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional
import queue
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import normalize_kv_and_discover_format
from lmcache.v1 import kvpilot_placement
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)

# A write waiting for DRAM room (the master spilling to SSD) says so this often.
RETRY_LOG_S = 30.0
# Chunks gathered per pass; bounds the staging buffers (512 KiB per chunk
# per layer per rank for Qwen3-30B-A3B at TP 2).
GROUP_CHUNKS = 64
# Chunks staged per SSD pass, all layers each: 192 MiB per rank for
# Qwen3-30B-A3B at TP 2.
SSD_GROUP_CHUNKS = 8


def _remote_backend(engine):
    backends = dict(engine.storage_manager.get_active_storage_backends())
    backend = backends.get("RemoteBackend")
    if backend is None:
        raise RuntimeError("no RemoteBackend")
    return backend


def _chunk_names(engine, hashes: list, chunk: int) -> tuple[list, list[list[str]]]:
    """Cache keys of whole chunks by their LMCache hashes, and each one's
    per-layer Mooncake key strings."""
    keys = [
        key
        for _, _, key in engine.token_database.process_tokens(
            hashes=hashes, offsets=[chunk] * len(hashes)
        )
    ]
    names = [[k.to_string() for k in key.split_layers(engine.num_layers)] for key in keys]
    return keys, names


def _slots(block_ids: list[int], chunk: int) -> torch.Tensor:
    """Slots of whole blocks (block = chunk), block after block, on the CPU."""
    bids = torch.tensor(block_ids, dtype=torch.long)
    return (bids.unsqueeze(1) * chunk + torch.arange(chunk)).flatten()


@dataclass
class Eviction:
    """HBM blocks lent to one request's load, on their way to DRAM.

    slots: int64 CPU, the blocks' slots, block after block.
    ptrs: int64 CPU [layers, blocks], each block's replica per layer; 0 =
        already stored, not copied.
    names: the Mooncake keys of the replicas allocated.
    done: recorded on the evict stream after the last layer's copy (set by
        the GPU connector's start_evict).
    """

    req_id: str
    slots: torch.Tensor
    ptrs: torch.Tensor
    names: list[str]
    skipped: int
    groups: list
    penalties: list[float]
    put_ms: float
    done: Optional[torch.cuda.Event] = None


class Evictor:
    """Writes the KV of HBM blocks the scheduler lent to a load, for one TP
    rank (exclusive-penalty-reqwise-unblocking)."""

    def __init__(self, engine) -> None:
        self._engine = engine
        self._layers = engine.num_layers
        self._chunk = engine.config.chunk_size
        self._nbytes = engine.gpu_connector.value_bytes(self._chunk)
        self._finish = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kvpilot-evict")

    def prepare(self, req_id: str, evict: dict) -> Eviction:
        """A DRAM replica for every layer of every lent block, in this rank's
        Mooncake segment (forward thread, before the load starts).

        Args:
            req_id: The request whose load the blocks were lent to.
            evict: The scheduler's record (vllm/v1/core/kvpilot_exclusive.py):
                "hashes" (LMCache chunk hashes), "block_ids", "dram_pen".

        Returns:
            The eviction, for the GPU connector's start_evict and `finish`.

        Raises:
            RuntimeError: No DRAM room for a value (DRAM keeps room for
                this). A value still stored is skipped: a copy the master
                would not remove while it was being offloaded.
        """
        t0 = time.perf_counter()
        backend = _remote_backend(self._engine)
        keys, names = _chunk_names(self._engine, evict["hashes"], self._chunk)
        n = len(names)
        flat = [names[c][l] for l in range(self._layers) for c in range(n)]
        addrs = backend.kvpilot_put_start_local(flat, [self._nbytes] * len(flat))
        if len(addrs) != len(flat):
            raise RuntimeError(f"KVPilot evict req={req_id}: PutStart failed")
        zero = [j for j, a in enumerate(addrs) if not a]
        if zero:
            exist = backend.kvpilot_exist([flat[j] for j in zero]) or [1] * len(zero)
            lost = sum(1 for e in exist if e != 1)
            if lost:
                backend.kvpilot_put_revoke([k for k, a in zip(flat, addrs) if a])
                raise RuntimeError(
                    f"KVPilot evict req={req_id}: {lost}/{len(flat)} values got "
                    "no DRAM replica"
                )
        return Eviction(
            req_id=req_id,
            slots=_slots(evict["block_ids"], self._chunk),
            ptrs=torch.tensor(addrs, dtype=torch.int64).view(self._layers, n),
            names=[k for k, a in zip(flat, addrs) if a],
            skipped=len(zero),
            groups=[k.chunk_hash for k in keys],
            penalties=list(evict["dram_pen"]),
            put_ms=(time.perf_counter() - t0) * 1e3,
        )

    def finish(self, ev: Eviction) -> None:
        """Once the copies are done: the groups become DRAM with their
        DRAM -> SSD penalty (before PutEnd, which would offload an SSD group),
        then PutEnd. Off the forward thread."""
        if ev.done is None:
            raise RuntimeError(f"KVPilot evict req={ev.req_id}: the load never started it")
        t0 = time.perf_counter()

        def run() -> None:
            ev.done.synchronize()
            copied = (time.perf_counter() - t0) * 1e3
            backend = _remote_backend(self._engine)
            n = len(ev.groups)
            kvpilot_placement.set_groups(backend, ev.groups, ["DRAM"] * n, ev.penalties)
            codes = backend.kvpilot_put_end(ev.names) or []
            logger.info(
                "KVPilot exclusive evict req=%s blocks=%d values=%d skipped=%d "
                "put_start_ms=%.1f copied_ms=%.1f failed=%d",
                ev.req_id, n, len(ev.names), ev.skipped, ev.put_ms, copied,
                sum(1 for c in codes if c != 0),
            )

        self._finish.submit(run)


class ExclusiveWriter:
    """Runs exclusive-placement jobs for one TP rank."""

    def __init__(self, engine, kvcaches: list[torch.Tensor]) -> None:
        """
        Args:
            engine: This rank's LMCacheEngine (token database, Mooncake backend,
                GPU connector for the value layout).
            kvcaches: vLLM's paged KV, one tensor per layer.
        """
        self._engine = engine
        gc = engine.gpu_connector
        self._fmt, self._kvcaches = normalize_kv_and_discover_format(
            kvcaches, EngineType.VLLM, layout_hints=getattr(gc, "layout_hints", None)
        )
        self._device = kvcaches[0].device
        self._dtype = kvcaches[0].dtype
        self._layers = engine.num_layers
        self._chunk = engine.config.chunk_size
        self._shape = gc.get_shape
        self._gc = gc
        # A value: the chunk's token-major KV, then (Qwen3.8) its side page.
        self._kv_bytes = self._chunk * gc.token_bytes()
        self._nbytes = gc.value_bytes(self._chunk)
        self._jobs: "queue.Queue[dict]" = queue.Queue()
        self._done: set[str] = set()
        self._lock = threading.Lock()
        self._gpu: torch.Tensor | None = None
        self._ssd_host: torch.Tensor | None = None
        self._thread = threading.Thread(
            target=self._loop, name="kvpilot-exclusive", daemon=True
        )
        self._thread.start()
        logger.info(
            "KVPilot exclusive writer on %s: %d layers, %d B per value",
            self._device, self._layers, self._nbytes,
        )

    def submit(self, job: dict) -> None:
        """Queue a job (see vllm/v1/core/kvpilot_exclusive.py for its fields)."""
        self._jobs.put(job)

    def take_done(self) -> set[str]:
        """Ids of the jobs finished since the last call."""
        with self._lock:
            done, self._done = self._done, set()
        return done

    def _loop(self) -> None:
        torch.cuda.set_device(self._device)
        self._stream = torch.cuda.Stream(device=self._device)
        while True:
            job = self._jobs.get()
            try:
                self._run(job)
            except Exception:
                logger.exception("KVPilot exclusive job %s failed", job.get("id"))
            with self._lock:
                self._done.add(job["id"])

    def _run(self, job: dict) -> None:
        t0 = time.perf_counter()
        backend = _remote_backend(self._engine)
        keys, names = _chunk_names(self._engine, job["hashes"], self._chunk)
        out: Counter = Counter()
        if job["zero"]:
            # DRAM copies nobody claims any more: first to SSD when DRAM fills.
            kvpilot_placement.set_groups(
                backend,
                [keys[i].chunk_hash for i in job["zero"]],
                ["DRAM"] * len(job["zero"]),
                [0.0] * len(job["zero"]),
            )
        if job.get("repen"):
            # Blockwise: the session moved more of its prompt down, so its DRAM
            # chunks' DRAM -> SSD penalties changed.
            kvpilot_placement.set_groups(
                backend,
                [keys[i].chunk_hash for i, _ in job["repen"]],
                ["DRAM"] * len(job["repen"]),
                [pen for _, pen in job["repen"]],
            )
        if job["drop"]:
            drop = [s for i in job["drop"] for s in names[i]]
            # A memory replica a reader or an offload still holds stays (one
            # extra copy until DRAM demotes it).
            out["drop_ok"] = backend.kvpilot_remove_memory(drop)
            out["drop_values"] = len(drop)
        if job["dram"]:
            # A DRAM group carries the exposed load its move to SSD adds (the
            # master demotes the smallest first).
            self._write(backend, "DRAM", job["dram"], keys, names, job["block_ids"],
                        out, job["dram_pen"])
        if job["ssd"]:
            self._write_ssd(backend, job["ssd"], keys, names, job["block_ids"], out)
        logger.info(
            "KVPilot exclusive job=%s req=%s tier=%s drop=%d/%d dram=%d "
            "ssd=%d chunks written=%d values=%d skipped=%d failed=%d retries=%d "
            "no_room=%d ms=%.1f",
            job["id"], job["req"], job["tier"], out["drop_ok"], out["drop_values"],
            len(job["dram"]), len(job["ssd"]), out["chunks"],
            out["values"],
            out["skipped"], out["failed"], out["retries"], out["no_room"],
            (time.perf_counter() - t0) * 1e3,
        )

    def _write(self, backend, tier, idx, keys, names, block_ids, out, pens) -> None:
        L = self._layers
        kvpilot_placement.set_groups(
            backend, [keys[i].chunk_hash for i in idx], [tier] * len(idx), pens
        )
        flat = [names[i][l] for i in idx for l in range(L)]
        addrs = self._allocate(backend, flat, out)
        rows = [addrs[c * L : (c + 1) * L] for c in range(len(idx))]
        keep = [c for c in range(len(idx)) if any(rows[c])]
        out["skipped"] += len(idx) - len(keep)
        started = [flat[j] for j, a in enumerate(addrs) if a]
        try:
            for g in range(0, len(keep), GROUP_CHUNKS):
                part = keep[g : g + GROUP_CHUNKS]
                self._copy([rows[c] for c in part], [block_ids[idx[c]] for c in part])
        except Exception:
            backend.kvpilot_put_revoke(started)
            raise
        codes = backend.kvpilot_put_end(started) or []
        out["chunks"] += len(keep)
        out["values"] += len(started)
        out["failed"] += sum(1 for c in codes if c != 0)

    def _write_ssd(self, backend, idx, keys, names, block_ids, out) -> None:
        L = self._layers
        kvpilot_placement.set_groups(
            backend, [keys[i].chunk_hash for i in idx], ["SSD"] * len(idx),
            [0.0] * len(idx),
        )
        flat = [names[i][l] for i in idx for l in range(L)]
        exist = backend.kvpilot_exist(flat)
        todo = [c for c in range(len(idx)) if not all(exist[c * L : (c + 1) * L])]
        out["skipped"] += len(idx) - len(todo)
        for g in range(0, len(todo), SSD_GROUP_CHUNKS):
            part = todo[g : g + SSD_GROUP_CHUNKS]
            base = self._stage_ssd([block_ids[idx[c]] for c in part])
            k = len(part)
            for j, c in enumerate(part):
                miss = [l for l in range(L) if not exist[c * L + l]]
                ks = [names[idx[c]][l] for l in miss]
                ptrs = [base + (l * k + j) * self._nbytes for l in miss]
                n = backend.kvpilot_offload_local(ks, ptrs, [self._nbytes] * len(ks))
                if n != len(ks):
                    raise RuntimeError(f"SSD write of {len(ks)} values returned {n}")
                out["chunks"] += 1
                out["values"] += n

    def _stage_ssd(self, block_ids: list[int]) -> int:
        """Copy all layers of these chunks (one vLLM block each) into the SSD
        staging buffer, value (layer l, chunk j) at index l * len + j. Returns
        the buffer's host address."""
        if self._gc.side_caches is not None:
            return self._stage_ssd_side(block_ids)
        n = len(block_ids) * self._chunk
        if self._ssd_host is None:
            self._ssd_host = torch.empty(
                self._shape(SSD_GROUP_CHUNKS * self._chunk * self._layers),
                dtype=self._dtype, pin_memory=True,
            )
        with torch.cuda.stream(self._stream):
            gpu = self._buffer(n)
            slots = self._slots(block_ids)
            for layer in range(self._layers):
                lmc_ops.single_layer_kv_transfer(
                    gpu, self._kvcaches[layer], slots,
                    lmc_ops.TransferDirection.D2H, self._fmt, True,
                )
                self._ssd_host[layer * n : (layer + 1) * n].copy_(gpu, non_blocking=True)
        self._stream.synchronize()
        return self._ssd_host.data_ptr()

    def _stage_ssd_side(self, block_ids: list[int]) -> int:
        """_stage_ssd for values that end with a side page."""
        k = len(block_ids)
        if self._ssd_host is None:
            self._ssd_host = torch.empty(
                SSD_GROUP_CHUNKS * self._layers * self._nbytes,
                dtype=torch.uint8, pin_memory=True,
            )
        base = self._ssd_host.data_ptr()
        with torch.cuda.stream(self._stream):
            gpu = self._buffer(k * self._chunk)
            slots = self._slots(block_ids)
            for layer in range(self._layers):
                lmc_ops.single_layer_kv_transfer(
                    gpu, self._kvcaches[layer], slots,
                    lmc_ops.TransferDirection.D2H, self._fmt, True,
                )
                dst = [base + (layer * k + j) * self._nbytes for j in range(k)]
                lmc_ops.batched_memcpy_d2h(gpu, dst, [self._kv_bytes] * k)
                self._copy_side(layer, block_ids, dst)
        self._stream.synchronize()
        return base

    def _copy_side(self, layer: int, block_ids: list[int], dst: list[int]) -> None:
        """Each block's side page to the end of its value at dst (0 = skip),
        on the current stream."""
        if self._gc.side_caches is None:
            return
        side = self._gc.side_bytes
        src = self._gc.side_caches[layer].data_ptr()
        lmc_ops.batched_memcpy_async(
            [d + self._kv_bytes if d else 0 for d in dst],
            [src + b * side for b in block_ids],
            [side] * len(block_ids),
        )

    def _allocate(self, backend, names: list[str], out) -> list[int]:
        """Replica addresses for `names` (0 = already stored). A value that is
        neither allocated nor stored found DRAM full: wait for the master to
        spill to SSD and try again, however long it takes, since the scheduler
        drops the HBM copy once the job is done (later jobs wait behind it, so
        the HBM blocks stay pinned and the scheduler holds new requests)."""
        addrs = backend.kvpilot_put_start_local(names, [self._nbytes] * len(names))
        t0 = last = time.monotonic()
        while True:
            zero = [j for j, a in enumerate(addrs) if not a]
            if not zero:
                return addrs
            exist = backend.kvpilot_exist([names[j] for j in zero]) or [1] * len(zero)
            todo = [j for j, e in zip(zero, exist) if e != 1]
            if not todo:
                return addrs
            now = time.monotonic()
            if now - last > RETRY_LOG_S:
                logger.warning("KVPilot exclusive: %d values waiting %.0f s for DRAM room",
                               len(todo), now - t0)
                out["no_room"] += 1
                last = now
            out["retries"] += 1
            time.sleep(0.02)
            again = backend.kvpilot_put_start_local(
                [names[j] for j in todo], [self._nbytes] * len(todo)
            )
            for j, a in zip(todo, again or []):
                addrs[j] = a

    def _copy(self, rows: list[list[int]], block_ids: list[int]) -> None:
        """Copy every layer of these chunks (one vLLM block each) into their
        replicas: rows[c][l] is chunk c's layer-l replica address, 0 = skip."""
        n = len(rows) * self._chunk
        with torch.cuda.stream(self._stream):
            gpu = self._buffer(n)
            slots = self._slots(block_ids)
            for layer in range(self._layers):
                lmc_ops.single_layer_kv_transfer(
                    gpu, self._kvcaches[layer], slots,
                    lmc_ops.TransferDirection.D2H, self._fmt, True,
                )
                dst = [row[layer] for row in rows]
                lmc_ops.batched_memcpy_d2h(gpu, dst, [self._kv_bytes] * len(rows))
                self._copy_side(layer, block_ids, dst)
        self._stream.synchronize()

    def _slots(self, block_ids: list[int]) -> torch.Tensor:
        return _slots(block_ids, self._chunk).to(self._device)

    def _buffer(self, n: int) -> torch.Tensor:
        """GPU staging for n tokens of one layer (one stream: a layer's gather
        waits for the previous layer's copies out of it)."""
        if self._gpu is None or self._gpu.shape[0] < n:
            self._gpu = torch.empty(
                self._shape(max(n, GROUP_CHUNKS * self._chunk)),
                dtype=self._dtype, device=self._device,
            )
        return self._gpu[:n]
