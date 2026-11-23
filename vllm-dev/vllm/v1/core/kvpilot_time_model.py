# SPDX-License-Identifier: Apache-2.0
"""KVPilot prefill time model: how long a turn's prefill takes with its
prefix split across HBM / DRAM / SSD (paper-figure/Figure3-request-safe/
fit_times2.py fits it; utils/ssd_load_check/mixed_model.py checks it).

For prefix p, suffix s, and D / X prefix tokens loaded from DRAM / SSD (the
rest of the prefix hits HBM):

  compute  the suffix runs in chunks of `chunk` tokens (chunked prefill); a
           chunk of n tokens after context q costs
             d_graph + b_graph * pad(n)          n <= the largest cudagraph
             d_eager + b_eager * n               otherwise
             + a * n * (q + n / 2) + e * q       attention
           pad(n) is the cudagraph size n runs at.
  load     L = c_dram * D + f_dram                          DRAM only
           L = max(c_dram * (D + X) + f_dram, c_ssd * X + f_ssd)   with SSD:
             every value is copied, the SSD part read first
  first    layer 0 on the GPU: g_dram * D + h_dram, or
           g_dram * D + g_ssd * X + h_ssd with SSD
  e2e      the load overlaps the first chunk's compute C0:
             C0 + first + k * L + w * softplus((L + F - C0 - first - k * L) / w)
           with w = omega * C0 + w0, plus the later chunks. A smooth max of
           C0 + first and L + F: near L ~ C0 some layers' loads arrive late.
"""

from __future__ import annotations

import json
import math

VERSION = 2
# Compute coefficients, in compute_features order.
COMPUTE = ("d_graph", "d_eager", "b_graph", "b_eager", "a", "e")


def compute_features(p: float, s: float, chunk: int, capture: list) -> list:
    """Per-chunk compute features, summed; the first chunk's separately.

    Returns:
        (first chunk's features, the later chunks' features), each in
        COMPUTE order.
    """
    first, rest = [0.0] * 6, [0.0] * 6
    q, left, j = p, s, 0
    while left > 0:
        n = min(left, chunk)
        out = first if j == 0 else rest
        pad = next((c for c in capture if n <= c), None)
        if pad is not None:
            out[0] += 1.0
            out[2] += pad
        else:
            out[1] += 1.0
            out[3] += n
        out[4] += n * (q + n / 2.0)
        out[5] += q
        q, left, j = q + n, left - n, j + 1
    return first, rest


def _softplus(z: float) -> float:
    return z if z > 30.0 else math.log1p(math.exp(z))


class TimeModel:
    """Prefill time of a turn whose prefix is split across tiers."""

    @classmethod
    def load(cls, path: str, block: int) -> "TimeModel":
        with open(path) as f:
            k = json.load(f)
        if k.get("version") != VERSION:
            raise ValueError(f"{path}: not a version {VERSION} time model")
        return cls(k, block)

    def __init__(self, k: dict, block: int) -> None:
        self._k = k
        self._comp = [k[name] for name in COMPUTE]
        self.thresh = k["thresh"]
        # Scheduler settings the coefficients were fitted under.
        self.chunk = k["chunk"]
        self.capture = sorted(k["capture"])
        # Tokens per block.
        self.block = block

    def _chunks(self, p: float, s: float) -> tuple[float, float]:
        first, rest = compute_features(p, s, self.chunk, self.capture)
        dot = lambda f: sum(a * b for a, b in zip(f, self._comp))  # noqa: E731
        return dot(first), dot(rest)

    def compute(self, p: float, s: float) -> float:
        """Prefill ms with the whole prefix in HBM."""
        c0, rest = self._chunks(p, s)
        return c0 + rest

    def e2e(self, p: float, s: float, dram: float, ssd: float) -> float:
        """Prefill ms with prefix p, suffix s, `dram` / `ssd` prefix tokens
        read from DRAM / SSD (the rest hit HBM)."""
        c0, rest = self._chunks(p, s)
        if not dram and not ssd:
            return c0 + rest
        k = self._k
        if ssd:
            load = max(k["c_dram"] * (dram + ssd) + k["f_dram"],
                       k["c_ssd"] * ssd + k["f_ssd"])
            first = k["g_dram"] * dram + k["g_ssd"] * ssd + k["h_ssd"]
        else:
            load = k["c_dram"] * dram + k["f_dram"]
            first = k["g_dram"] * dram + k["h_dram"]
        w = k["omega"] * c0 + k["w0"]
        z = (load + k["F"] - c0 - first - k["k"] * load) / w
        return c0 + first + k["k"] * load + w * _softplus(z) + rest

    def times(self, p: float, s: float) -> tuple[float, float, float]:
        """(HBM, DRAM, SSD) prefill ms with the whole prefix in that tier."""
        return self.compute(p, s), self.e2e(p, s, p, 0), self.e2e(p, s, 0, p)
