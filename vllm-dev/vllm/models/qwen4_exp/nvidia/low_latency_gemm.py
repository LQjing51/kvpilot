# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp decode GEMM selection on Blackwell.

vLLM 0.30 swaps in a CuTe-DSL skinny GEMM on SM103 only. The 0.23 backport
runs on Hopper and has no skinny GEMM kernel, so every linear keeps the
standard implementation.
"""

import torch
from torch import nn


def enable_qwen4_exp_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    del module, dtype
