# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How long prefill computation sat idle waiting for KV to arrive.

This is the measurement behind KVPilot's central claim: KV loaded from DRAM is
free only insofar as prefill computation hides it. What is *not* hidden shows up
as the compute stream sitting idle, and that idle time is what this module
measures.

Under layerwise loading a request's KV arrives one layer at a time, and each
attention layer waits until its own layer is resident. There are two such
waiting points per forward, and both are stalls:

  - the prologue, `start_load_kv`, which sets the load up before the forward
    has begun;
  - each `wait_for_layer_load`, once per attention layer, which makes the
    compute stream wait for that layer's copy (on the SSD path it first blocks
    the CPU until the copy is enqueued).

Each is bracketed by a pair of CUDA events on the compute stream; the GPU time
between them is how long that stream sat idle across the wait. When the CPU
has run ahead, earlier layers are still queued and the gap is ~0. The sum over
one forward is the extent to which loading failed to hide behind computation.
Zero means perfectly overlapped -- the goal state.

The worker reports each wait's idle time in order. TP ranks run every layer in
lockstep through an all-reduce, so a layer costs the step its slowest rank's
wait; while one rank waits for its KV the others spin in the collective, which
is not idle. Ranks are therefore combined wait by wait with a max, then summed
(`gpu_stall`); a max over ranks of each rank's own sum undercounts when ranks
stall at different layers.

The scheduler charges the step's total to the prefill in that step, assuming
one prefill per forward (bs=1); a prefill that loaded nothing is charged none.

Enabled by $KVPILOT_LOAD_STALL; the scheduler emits one `KVPilot prefill` log
line per finished prefill. Log-only by design -- see `StallLogger`.
"""

import os
import time

import torch

from vllm.logger import init_logger
from vllm.v1.core.kvpilot_hbm_snapshot import req_line

logger = init_logger(__name__)

ENABLE_ENV = "KVPILOT_LOAD_STALL"


def enabled() -> bool:
    """Whether stall accounting is switched on for this process."""
    return os.getenv(ENABLE_ENV, "").lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Worker side: accumulate one forward's idle time.
# ---------------------------------------------------------------------------

# Resolved once. The timer sits inside the per-layer attention hook, so it runs
# tens of times per forward and must not re-read the environment there.
_ENABLED = enabled()

# The compute-stream event pair around each wait of the current forward, in
# order, or None for a wait timed during graph capture. Order matters: TP ranks
# make the same waits in the same order, and are combined wait by wait. A plain
# module global rather than a class: the attention hook has no handle on any
# object the mixin also holds, and both run on the one engine thread within a
# single forward.
_pairs: list = []


class _Timer:
    """Brackets one wait with compute-stream events for the current forward."""

    __slots__ = ("_ev0",)

    def __enter__(self) -> "_Timer":
        self._ev0 = _record()
        return self

    def __exit__(self, *exc_info) -> None:
        _pairs.append(None if self._ev0 is None else (self._ev0, _record()))


def _record():
    if torch.cuda.is_current_stream_capturing():
        return None
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev


def time_wait() -> "_Timer | None":
    """Wrap a blocking KV wait, or return None when accounting is off.

    Returns None rather than a null context so the disabled path allocates
    nothing: this is called once per attention layer per forward.
    """
    return _Timer() if _ENABLED else None


def begin_step() -> None:
    """Zero the accumulator at the start of a forward."""
    _pairs.clear()


def take_step() -> list[int]:
    """Consume this forward's waits: compute-stream idle ns of each, in order."""
    last = next((p for p in reversed(_pairs) if p is not None), None)
    if last is not None:
        last[1].synchronize()
    idle = [0 if p is None else int(p[0].elapsed_time(p[1]) * 1e6) for p in _pairs]
    _pairs.clear()
    return idle


# ---------------------------------------------------------------------------
# Scheduler side: one line per finished prefill.
# ---------------------------------------------------------------------------


class StallLogger:
    """Emits one log line per finished prefill.

    Deliberately log-only. An earlier version also wrote JSONL through a
    background thread; that file arrived empty because it was opened with a
    1 MiB buffer and drained on a daemon thread, so an exiting worker dropped
    the buffer with no flush or atexit hook. The log line was the copy that
    survived and the one the plotting script parsed, so the file was pure risk:
    a second encoding of the same data whose failure was silent. Log lines are
    already durable, already collected per worker by the launcher, and already
    timestamped.
    """

    # Prefix and field order are the parsers' contract: scripts under utils/
    # and paper-figure/ grep for this line, so renaming a field here means
    # editing their regexes too.
    PREFIX = "KVPilot prefill"

    def __init__(self, worker_tag: str | None = None):
        self._tag = worker_tag or _default_tag()
        logger.info(
            "KVPilot load-stall accounting enabled (%s) on %s: one '%s' line "
            "per finished prefill",
            ENABLE_ENV,
            self._tag,
            self.PREFIX,
        )

    def log(
        self,
        request_id: str,
        num_prompt_tokens: int,
        num_new_tokens: int,
        num_hit_tokens: int,
        num_dram_tokens: int,
        e2e_ns: int,
        num_waits: int,
        num_steps: int,
        gpu_stall_ns: int,
        blocked_ns: int = 0,
    ) -> None:
        """Record one request's finished prefill.

        Fields, in the order they are printed:
          req      the request id
          line     trace line the replay scripts encode in it, or -1; the join
                   key back to the conversation and turn
          prompt   prompt tokens, = new + hit + dram
          new      recomputed from scratch this prefill
          hit      served from the local HBM prefix cache, never stalls
          dram     supplied by the connector; the only tokens that can stall
          e2e      scheduled until prefill computation finished
          waits    timed waits, start_load_kv plus one per attention layer per
                   forward
          steps    forwards this prefill spanned; >1 means chunked prefill, and
                   e2e/waits/gpu_stall are summed over all of them
          gpu_stall  of e2e, compute-stream idle time across the waits: max
                   over TP ranks wait by wait, then summed
          done     wall clock (time.time()) when the prefill finished; minus
                   the client's send time it is TTFT at the end of prefill
          blocked  waiting for HBM: first failed allocation until admitted
                   (not in e2e)
        """
        logger.info(
            "%s req=%s line=%d prompt=%d new=%d hit=%d dram=%d "
            "e2e=%.3fms waits=%d steps=%d gpu_stall=%.3fms done=%.6f blocked=%.3fms",
            self.PREFIX,
            request_id,
            req_line(request_id),
            num_prompt_tokens,
            num_new_tokens,
            num_hit_tokens,
            num_dram_tokens,
            e2e_ns / 1e6,
            num_waits,
            num_steps,
            gpu_stall_ns / 1e6,
            time.time(),
            blocked_ns / 1e6,
        )


def _default_tag() -> str:
    """Identify the emitting worker, so P and D records can be told apart."""
    role = os.getenv("DYN_DISAGGREGATION_MODE") or "worker"
    return f"{role}-{os.getpid()}"
