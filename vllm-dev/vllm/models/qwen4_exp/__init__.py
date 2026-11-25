# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp model package."""

from typing import TYPE_CHECKING, Any

from .common.hyperconnection import (
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionBase,
    HyperConnectionConfig,
)

if TYPE_CHECKING:
    from .nvidia.model import (
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
    )


def __getattr__(name: str) -> Any:
    # Backported to vLLM 0.23 (KVPilot): the NVIDIA path only, without MTP.
    if name in {"Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"}:
        from vllm.platforms import current_platform

        if not current_platform.is_cuda():
            raise NotImplementedError("Qwen4Exp is backported for CUDA only")
        from .nvidia.model import (
            Qwen4ExpForCausalLM,
            Qwen4ExpForConditionalGeneration,
        )

        return {
            "Qwen4ExpForCausalLM": Qwen4ExpForCausalLM,
            "Qwen4ExpForConditionalGeneration": (Qwen4ExpForConditionalGeneration),
        }[name]
    raise AttributeError(name)


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
]
