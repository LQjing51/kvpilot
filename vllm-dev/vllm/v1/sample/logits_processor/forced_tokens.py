# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step forced-output LogitsProcessor.

For each request that opts in, force the model's sampled token at step ``i``
to equal a pre-computed ``forced_tokens[i]``. The forced sequence is passed
in via ``SamplingParams.extra_args["forced_output_tokens"]``.

Wire-format
-----------
``forced_output_tokens`` may be either a ``list[int]`` or a JSON-encoded
``list[int]`` string. The JSON form is required when the value travels
through ``CompletionRequest.vllm_xargs`` / ``ChatCompletionRequest.vllm_xargs``
(those fields restrict values to ``str | int | float``).
"""
import json
from collections.abc import Sequence

import torch

from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

EXTRA_ARG_KEY = "forced_output_tokens"


class _ForcedReqLP:
    """Per-request callable. Returns logits whose argmax is forced[step].

    The vLLM adapter framework keeps a reference to the running output
    token list and passes it as ``past_token_ids`` on every forward pass,
    so ``len(past_token_ids)`` is the current decode step.
    """

    def __init__(self, forced_tokens: list[int]):
        self.forced = forced_tokens

    def __call__(
        self, past_token_ids: Sequence[int], logits: torch.Tensor
    ) -> torch.Tensor:
        step = len(past_token_ids)
        if step >= len(self.forced):
            # Defensive: should not trigger when the client sets
            # min_tokens == max_tokens == len(forced). Pass through so
            # any tail token sampled by the engine is left untouched.
            return logits
        target = self.forced[step]
        # One-hot in logit space. -inf elsewhere, 0.0 at target -> any
        # temperature/top_p/top_k still yields ``target`` deterministically.
        new_logits = torch.full_like(logits, float("-inf"))
        new_logits[target] = 0.0
        return new_logits


def _parse_forced_tokens(extra_args: dict | None) -> list[int] | None:
    """Extract and validate the forced-tokens payload from extra_args."""
    if not extra_args:
        return None
    raw = extra_args.get(EXTRA_ARG_KEY)
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        forced = json.loads(raw)
    else:
        forced = list(raw)
    if not forced:
        return None
    if not all(isinstance(t, int) and t >= 0 for t in forced):
        raise ValueError(
            f"{EXTRA_ARG_KEY} must be a list of non-negative ints, "
            f"got {forced[:8]}..."
        )
    return forced


class ForcedOutputTokensLogitsProcessor(AdapterLogitsProcessor):
    """Force per-step output tokens for opt-in requests.

    Requests without ``extra_args['forced_output_tokens']`` are untouched
    (``new_req_logits_processor`` returns None and no hook is registered).
    """

    def is_argmax_invariant(self) -> bool:
        # We overwrite logits with a one-hot mask, so the argmax changes.
        return False

    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> RequestLogitsProcessor | None:
        forced = _parse_forced_tokens(params.extra_args)
        if forced is None:
            return None
        return _ForcedReqLP(forced)
