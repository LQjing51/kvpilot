# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KVPilot online dram-safe feedback (decode -> prefill).

The verdict is only computable once decode has emitted its tool call, which is
after the prefill worker already queued the blocks. The decode worker sends it
back keyed by the prefill request id, which prefill uses to find the blocks that
request left behind; see `FreeKVCacheOnlineDramSafeQueue.retier`.
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class KvpilotReporter:
    """Decides dram-safe on the decode side and reports it to the prefill worker."""

    def __init__(self, tokenizer, prefill_client):
        # Ships with vLLM next to the queue it feeds, so the offline validator
        # and this online path can never disagree on constants or labels.
        from vllm.v1.core import kvpilot_predict

        self._predict = kvpilot_predict
        self._constants = kvpilot_predict.derive_constants()
        self._tokenizer = tokenizer
        self._client = prefill_client
        self._measured = kvpilot_predict.measured_enabled()
        if self._measured:
            logger.info(
                "KVPilot using measured dram-safe table: %s",
                os.environ.get(kvpilot_predict.MEASURED_ENV),
            )

    async def report(
        self,
        kv_params: Dict[str, Any],
        request: Dict[str, Any],
        accumulated_token_ids: Dict[int, list],
    ) -> None:
        """Judge this turn and send the verdict. Never raises into the caller."""
        try:
            remote_request_id = kv_params.get("remote_request_id")
            worker_id = self._prefill_worker_id(request)
            if remote_request_id is None or worker_id is None:
                return

            token_ids = accumulated_token_ids.get(0) or []
            # skip_special_tokens=False: the tool-call regex matches the raw
            # `<function=NAME>` text and stripping tokens can move its bounds.
            text = self._tokenizer.decode(token_ids, skip_special_tokens=False)
            prefix = len(request.get("token_ids") or [])
            dram_safe, label, append, t_load_us, t_comp_us = (
                self._predict.explain_dram_safe(
                    prefix, len(token_ids), text, self._constants
                )
            )

            # The measured arm replaces only the verdict: the label, append and
            # times above still describe what the cost model would have said, so
            # the log line shows both and the two arms stay comparable.
            source = "predicted"
            if self._measured:
                measured = self._predict.measured_dram_safe(remote_request_id)
                if measured is None:
                    source = "predicted(measured-miss)"
                else:
                    dram_safe = measured
                    source = "measured"

            # No label means no tool call, i.e. the session ended this turn.
            logger.info(
                "KVPilot verdict remote_request_id=%s tool_call=%s "
                "prefix=%d decode_len=%d append=%d dram_safe=%s source=%s "
                "t_load_us=%.1f t_comp_us=%.1f margin_us=%.1f "
                "prefill_worker_id=%d",
                remote_request_id,
                label if label is not None else "<none>",
                prefix,
                len(token_ids),
                append,
                dram_safe,
                source,
                t_load_us,
                t_comp_us,
                t_comp_us - t_load_us,
                worker_id,
            )
            # Uncomment to measure the feedback link alone: the lookup, the round
            # trip and the re-tiering still run, but every block lands in the
            # conservative tier, so eviction matches a run with the feature off.
            # Note this lands after the log line, so the logged verdict is the
            # predicted one while `dram_safe=False` is what actually goes out.
            # dram_safe = False
            # The response stream is already in flight whether or not we read
            # it, so drain it: dropping it unread closes the channel early and
            # makes the prefill worker log a publish failure per verdict.
            async for _ in await self._client.direct(
                {"remote_request_id": remote_request_id, "dram_safe": dram_safe},
                worker_id,
            ):
                pass
        except Exception:
            # A lost verdict only costs the blocks their re-tiering; it must
            # never disturb the decode stream.
            logger.info("KVPilot feedback failed", exc_info=True)

    @staticmethod
    def _prefill_worker_id(request: Dict[str, Any]) -> Optional[int]:
        routing = request.get("routing") or {}
        worker_id = routing.get("prefill_worker_id")
        return int(worker_id) if worker_id is not None else None
