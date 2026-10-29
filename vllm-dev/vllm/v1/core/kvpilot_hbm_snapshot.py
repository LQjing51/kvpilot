# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Periodic census of the HBM block pool, for the placement distribution plot.

Every interval this walks `BlockPool.blocks` and writes one JSONL line holding,
per block, how long it has been idle and which request last held it. Two axes
come out of that: idle time is what LRU orders by, and the request's lookahead
load/compute ratio -- joined offline through the request id -- is what KVPilot
orders by. Plotting one against the other shows where the two disagree.

Deliberately independent of the dram-safe feature: nothing here imports the
predictor or the tiered queues, and no verdict is read or written. The plot's
whole purpose is to compare a KVPilot run against a plain-LRU baseline, so this
has to record the same fields when every KVPilot switch is off.

The ratio is not sampled. It is a property of the request, not of the block, and
it is only known after the turn ends -- which for the last turn of a session is
after the blocks are long gone. Sampling records the request id and the offline
table supplies the ratio, so the census stays a census.

Enabled by $GET_HBM_POOL_DISTRIBUTION; writes hbm-pool-<worker>.jsonl into
$LOG_DIR (the directory the launcher already points at this run's logs).
"""

import json
import os
import queue
import threading
import time
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool

logger = init_logger(__name__)

ENABLE_ENV = "GET_HBM_POOL_DISTRIBUTION"

# A full pass is O(num_gpu_blocks) -- a few thousand blocks, around a
# millisecond. At a 10-50ms step that would be a percent or more of scheduling
# time if taken every step, so it is taken about once a second instead: pool
# composition moves on the scale of request lifetimes, not steps. Hardcoded
# because a knob here only invites sampling a run too finely to compare with
# another.
INTERVAL_NS = 1_000_000_000

# Block states. Only CACHED_FREE blocks are eviction candidates and thus the
# subject of the plot; the other two are counted so each line accounts for the
# whole pool.
_IN_USE = 0  # ref_cnt > 0: held by a live request, not up for eviction
_CACHED_FREE = 1  # ref_cnt == 0 with a hash: the eviction candidates
_BLANK = 2  # no hash: nothing to lose by evicting, always goes first


def enabled() -> bool:
    """Whether census sampling is switched on for this process."""
    return os.getenv(ENABLE_ENV, "").lower() in ("1", "true", "yes", "on")


class HbmPoolSampler:
    """Writes a periodic per-block census of one worker's HBM pool."""

    def __init__(self, block_pool: "BlockPool", worker_tag: str | None = None):
        self._pool = block_pool
        log_dir = os.getenv("LOG_DIR") or "."
        tag = worker_tag or _default_worker_tag()
        self._path = os.path.join(log_dir, f"hbm-pool-{tag}.jsonl")
        self._next_ns = 0
        # Serialization and disk I/O happen off the engine thread; the sampler
        # itself only walks the pool and hands over a list. Bounded so a slow
        # disk drops samples instead of growing without limit -- a census is a
        # sample of a continuous process, and a gap in it costs a plot point.
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._dropped = 0
        os.makedirs(log_dir, exist_ok=True)
        self._file = open(self._path, "a", buffering=1024 * 1024)
        self._writer = threading.Thread(
            target=self._drain, name="hbm-pool-sampler", daemon=True
        )
        self._writer.start()
        logger.info(
            "HBM pool distribution sampling enabled (%s): writing %s every %.1fs",
            ENABLE_ENV,
            self._path,
            INTERVAL_NS / 1e9,
        )

    def maybe_sample(self) -> None:
        """Take a census if the interval has elapsed. Safe to call every step."""
        now_ns = time.monotonic_ns()
        if now_ns < self._next_ns:
            return
        self._next_ns = now_ns + INTERVAL_NS
        try:
            self._queue.put_nowait(self._census(now_ns))
        except queue.Full:
            self._dropped += 1

    def _census(self, now_ns: int) -> dict:
        """One pass over the pool, as the record that will be written."""
        blocks = []
        num_blank = 0
        num_in_use = 0
        for block in self._pool.blocks:
            if block.is_null:
                continue
            if block.ref_cnt > 0:
                state = _IN_USE
                num_in_use += 1
            elif block.block_hash is not None:
                state = _CACHED_FREE
            else:
                # Blank blocks carry no KV and no request, so a per-block row
                # would hold nothing but the state. At startup they are nearly
                # the whole pool; counting them keeps early lines small.
                num_blank += 1
                continue
            blocks.append(
                [
                    block.block_id,
                    state,
                    # Blocks untouched since startup have no stamp; -1 marks
                    # that rather than claiming an age measured from the epoch.
                    now_ns - block.kvpilot_last_access_ns
                    if block.kvpilot_last_access_ns
                    else -1,
                    block.kvpilot_last_req_line,
                    block.kvpilot_queue_id,
                ]
            )
        return {
            "ts_ns": now_ns,
            "wall_s": time.time(),
            "pool_blocks": self._pool.num_gpu_blocks,
            "num_blank": num_blank,
            "num_in_use": num_in_use,
            "dropped": self._dropped,
            # [block_id, state, idle_ns, req_line, tier]
            "blocks": blocks,
        }

    def _drain(self) -> None:
        while True:
            record = self._queue.get()
            try:
                self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
                # Flush every record. The writer is a daemon thread with no
                # atexit hook, so anything still buffered when the worker exits
                # is lost: the load-stall JSONL died exactly this way, its ~1 MB
                # of records sitting under a 1 MiB buffer that never filled. A
                # census line is written once a second, so the syscall cost is
                # irrelevant next to losing the run.
                self._file.flush()
            except Exception:
                logger.warning("HBM pool census write failed", exc_info=True)


def _default_worker_tag() -> str:
    """Name the file after the worker, so P and D land side by side."""
    role = os.getenv("DYN_DISAGGREGATION_MODE") or "worker"
    return f"{role}-{os.getpid()}"


def req_line(request_id: str) -> int:
    """The trace line the replay scripts encode in a request id's first field.

    Both replay scripts put the source line of the prompt in the leading UUID
    group so server-side records can be traced back to the conversation and turn
    that produced them; this is the join key to the offline ratio table. Ids
    from any other source (warmup, health checks) have no line and return -1.
    """
    head = request_id.split("-", 1)[0]
    return int(head) if head.isdigit() else -1
