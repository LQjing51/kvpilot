# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How long prefill computation sat idle waiting for KV to arrive.

This is the measurement behind KVPilot's central claim: KV loaded from DRAM is
free only insofar as prefill computation hides it. What is *not* hidden shows up
as the forward thread blocking, and that blocking is what this module times.

Under layerwise loading a request's KV arrives one layer at a time, and each
attention layer blocks until its own layer is resident. There are two such
blocking points per forward, and both are stalls:

  - the prologue, inside `start_load_kv`, which submits the fetches and then
    blocks on layer 0 before the forward has begun;
  - each `wait_for_layer_load`, once per attention layer, blocking until that
    layer's copy has landed.

Their sum over one forward is the extent to which loading failed to hide behind
computation. Zero means perfectly overlapped -- the goal state.

Attribution is deliberately split across the two processes. A forward is shared
by every request in the batch, so the worker can only honestly report a
*per-step* total; it does that and nothing more. The scheduler, which knows how
many DRAM tokens each request asked for, divides that total among them. Neither
side guesses at what it cannot see.

Enabled by $KVPILOT_LOAD_STALL; the scheduler emits one `KVPilot prefill` log
line per finished prefill. Log-only by design -- see `StallLogger`. Plot with
`plot_load_stall.py`, which parses those lines out of the worker logs.
"""

import os
import time

from vllm.logger import init_logger
from vllm.v1.core.kvpilot_hbm_snapshot import req_line

logger = init_logger(__name__)

ENABLE_ENV = "KVPILOT_LOAD_STALL"


def enabled() -> bool:
    """Whether stall accounting is switched on for this process."""
    return os.getenv(ENABLE_ENV, "").lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Worker side: accumulate one forward's blocking time.
# ---------------------------------------------------------------------------

# Resolved once. The timer sits inside the per-layer attention hook, so it runs
# tens of times per forward and must not re-read the environment there.
_ENABLED = enabled()

# Nanoseconds the current forward has spent blocked, and how many of those
# blocking calls there were. Plain module globals rather than a class: the
# attention hook has no handle on any object the mixin also holds, and both run
# on the one engine thread within a single forward.
_stall_ns = 0
_num_waits = 0


class _Timer:
    """Times one blocking call and folds it into the current forward's total."""

    __slots__ = ("_t0",)

    def __enter__(self) -> "_Timer":
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *exc_info) -> None:
        global _stall_ns, _num_waits
        _stall_ns += time.perf_counter_ns() - self._t0
        _num_waits += 1


def time_wait() -> "_Timer | None":
    """Wrap a blocking KV wait, or return None when accounting is off.

    Returns None rather than a null context so the disabled path allocates
    nothing: this is called once per attention layer per forward.
    """
    return _Timer() if _ENABLED else None


def begin_step() -> None:
    """Zero the accumulator at the start of a forward."""
    global _stall_ns, _num_waits
    _stall_ns = 0
    _num_waits = 0


def take_step() -> tuple[int, int]:
    """Consume this forward's accumulated stall as (nanoseconds, num_waits)."""
    global _stall_ns, _num_waits
    total, waits = _stall_ns, _num_waits
    _stall_ns = 0
    _num_waits = 0
    return total, waits


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

    # Prefix and field order are the parser's contract. `plot_load_stall.py`
    # greps for this, so renaming a field here means editing that regex too.
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
        stall_ns: int,
        num_waits: int,
        num_steps: int,
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
          stall    of that, blocked in start_load_kv + wait_for_layer_load
          waits    blocking calls, roughly one per attention layer per forward
          steps    forwards this prefill spanned; >1 means chunked prefill, and
                   e2e/stall/waits are summed over all of them
        """
        stall_pct = round(100.0 * stall_ns / e2e_ns, 3) if e2e_ns > 0 else 0.0
        logger.info(
            "%s req=%s line=%d prompt=%d new=%d hit=%d dram=%d "
            "e2e=%.3fms stall=%.3fms (%.1f%%) waits=%d steps=%d",
            self.PREFIX,
            request_id,
            req_line(request_id),
            num_prompt_tokens,
            num_new_tokens,
            num_hit_tokens,
            num_dram_tokens,
            e2e_ns / 1e6,
            stall_ns / 1e6,
            stall_pct,
            num_waits,
            num_steps,
        )


def _default_tag() -> str:
    """Identify the emitting worker, so P and D records can be told apart."""
    role = os.getenv("DYN_DISAGGREGATION_MODE") or "worker"
    return f"{role}-{os.getpid()}"


def attribute(step_stall_ns: int, weights: dict[str, int]) -> dict[str, int]:
    """Split one forward's stall among the requests that were loading in it.

    A forward is shared, so the stall it accrued belongs to the batch, not to
    any one request. `weights` maps request id to the number of DRAM tokens that
    request had loading during this forward; the stall is divided in proportion
    to those, which is the only signal available that distinguishes a request
    that caused waiting from one that merely sat in the same batch.

    A request loading nothing gets none of it: it would have run just as fast
    alone. When nothing was loading the stall is unattributable and dropped
    rather than smeared over requests that did not cause it.
    """
    if step_stall_ns <= 0 or not weights:
        return {}
    total = sum(weights.values())
    if total <= 0:
        return {}
    shares = {
        req_id: step_stall_ns * tokens // total
        for req_id, tokens in weights.items()
        if tokens > 0
    }
    # Integer division loses up to one nanosecond per request; hand the
    # remainder to the largest consumer so the shares sum to the measured total
    # exactly. Without this a long run of small batches would quietly
    # under-report the aggregate.
    assigned = sum(shares.values())
    if shares and assigned < step_stall_ns:
        top = max(shares, key=lambda r: weights[r])
        shares[top] += step_stall_ns - assigned
    return shares
