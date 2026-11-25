# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KVPilot signal (decode -> prefill).

When a turn finishes decoding, the decode worker sends the prefill worker that
served it the signal for the session's next turn (vllm.v1.core.kvpilot_signal),
keyed by the prefill request id; the prefill side places that request's freed
blocks by it (vllm.v1.core.kvpilot_placement).
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class KvpilotReporter:
    """Computes each finished turn's signal and sends it to the prefill worker."""

    def __init__(self, tokenizer, prefill_client, block_size: int):
        # Ships with vLLM next to the placement it feeds, so both sides share
        # one definition of the signal.
        from vllm.v1.core import kvpilot_signal

        # The prefill side's block size: the launch scripts give both sides
        # the same one.
        self._block = block_size
        self._source = kvpilot_signal.from_env(block_size)
        self._tokenizer = tokenizer
        self._client = prefill_client
        logger.info("KVPilot signal source: %s", type(self._source).__name__)

    async def report(
        self,
        kv_params: Dict[str, Any],
        request: Dict[str, Any],
        accumulated_token_ids: Dict[int, list],
    ) -> None:
        remote_request_id = kv_params["remote_request_id"]
        worker_id = int(request["routing"]["prefill_worker_id"])
        prompt_len = len(request["token_ids"])
        if prompt_len < self._block:
            # No full prompt block: the prefill side tracked nothing.
            return
        token_ids = accumulated_token_ids.get(0) or []

        def text():
            # skip_special_tokens=False: the tool-call regex matches the raw
            # `<function=NAME>` text and stripping tokens can move its bounds.
            return self._tokenizer.decode(token_ids, skip_special_tokens=False)

        signal = self._source.signal(remote_request_id, prompt_len, len(token_ids), text)
        misses = getattr(self._source, "label_misses", None)
        logger.info(
            "KVPilot signal remote_request_id=%s prompt=%d decode_len=%d tier=%s "
            "time=%s prefill_worker_id=%d%s",
            remote_request_id,
            prompt_len,
            len(token_ids),
            signal["tier"],
            signal["time"],
            worker_id,
            "" if misses is None
            else f" label_misses={misses}/{self._source.labels_seen}",
        )
        ok = 0
        async for resp in await self._client.direct(
            {"remote_request_id": remote_request_id, "signal": signal}, worker_id
        ):
            data = resp.data()
            if data.get("status") != "success":
                raise RuntimeError(f"KVPilot signal for {remote_request_id}: {data}")
            ok += 1
        if not ok:
            raise RuntimeError(f"KVPilot signal for {remote_request_id}: no reply")
