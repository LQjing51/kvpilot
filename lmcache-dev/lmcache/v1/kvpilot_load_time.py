"""Per-request KV load time under layerwise retrieval.

`load` is the GPU-timeline span from just before layer 0's fetch is submitted
to layer N-1's H2D copy completing on the load stream, whether or not compute
hid it. Both ends are CUDA events on the (otherwise idle) load stream, so it
does not depend on when the forward thread gets around to the last layer. `fetch` and `h2d`
are per-layer sums; under heavy compute the H2D kernels share the GPU and run
longer, so take the small-suffix case as the load an idle GPU sees.

Same switch as the exposed part ($KVPILOT_LOAD_STALL). One `KVPilot load` line
per retrieval, per TP rank.
"""

import os
import threading
import time

import torch

from lmcache.logging import init_logger

logger = init_logger(__name__)

ENABLED = os.getenv("KVPILOT_LOAD_STALL", "").lower() in ("1", "true", "yes", "on")


class LoadTimer:
    def __init__(self, req_id: str, num_tokens: int, num_layers: int):
        self.req_id = req_id
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.device = torch.cuda.current_device()
        self.ev0: torch.cuda.Event | None = None
        self.fetch_ns: list[int] = []
        self.copies: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def start(self, stream: torch.cuda.Stream) -> None:
        self.ev0 = torch.cuda.Event(enable_timing=True)
        self.ev0.record(stream)

    def copy_begin(self, stream: torch.cuda.Stream) -> torch.cuda.Event:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(stream)
        return ev

    def copy_end(self, begin: torch.cuda.Event, stream: torch.cuda.Stream) -> None:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(stream)
        self.copies.append((begin, ev))

    def finish(self) -> None:
        # Off the forward thread, so waiting on the last copy is not counted as stall.
        threading.Thread(target=self._emit, daemon=True).start()

    def _emit(self) -> None:
        torch.cuda.set_device(self.device)
        last = self.copies[-1][1]
        last.synchronize()
        load_ms = self.ev0.elapsed_time(last)
        while len(self.fetch_ns) < self.num_layers:
            time.sleep(0.001)
        h2d_ms = sum(a.elapsed_time(b) for a, b in self.copies)
        fetch_ms = sum(self.fetch_ns) / 1e6
        logger.info(
            "KVPilot load req=%s tokens=%d layers=%d fetch=%.3fms h2d=%.3fms load=%.3fms",
            self.req_id,
            self.num_tokens,
            self.num_layers,
            fetch_ms,
            h2d_ms,
            load_ms,
        )


def report(
    req_id: str,
    num_tokens: int,
    num_layers: int,
    timing,
    ssd_values: int = 0,
    start_ms: float = 0.0,
) -> None:
    """Log a C++ loader's timing, off the forward thread (result() blocks).

    With SSD values, `load` runs from before the first SSD read to the last
    layer's scatter, and `h2d` is still copy + scatter only; `ssd` counts the
    values (layer x chunk) read from SSD. `first` is when layer 0 is on the
    GPU, on the same clock as `load`; `start` is the CPU time of the call that
    set the load up (start_layers / start_layers_ssd), before that clock.
    """

    def emit() -> None:
        h2d_ms, load_ms, first_ms = timing.result()
        logger.info(
            "KVPilot load req=%s tokens=%d layers=%d fetch=%.3fms h2d=%.3fms "
            "load=%.3fms ssd=%d first=%.3fms start=%.3fms",
            req_id,
            num_tokens,
            num_layers,
            0.0,
            h2d_ms,
            load_ms,
            ssd_values,
            first_ms,
            start_ms,
        )

    threading.Thread(target=emit, daemon=True).start()
