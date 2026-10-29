# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import contextlib
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime
from enum import Enum
from http import HTTPStatus
from statistics import mean
from typing import NamedTuple

import aiohttp  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from bench_dataset import (
    ConversationsMap,
    ConvId,
    GenConvArgs,
    MessagesList,
    ShareGptConversations,
    conversations_dict_to_list,
    conversations_list_to_dict,
    generate_conversations,
    parse_input_json_file,
)
from bench_utils import TEXT_SEPARATOR, Color, logger
from transformers import AutoTokenizer  # type: ignore

NUM_TOKENS_FROM_DATASET = 0
# How long a client with no work waits for the feeder before concluding the
# task queue is drained. Only pays out once per client, at the very end.
# How often the progress line prints. The old one fired per finished
# conversation off a blocking queue read; with the queues gone it needs a clock.
PROGRESS_INTERVAL_SEC = 10.0

# Where the arrival schedule is written. Next to the trace it was derived from,
# so a run's timetable can be matched to its input.
DEFAULT_SCHEDULE_NAME = "multiturn_schedule.jsonl"

# Maps each ShareGPT conv_id to the physical line number of every one of its
# user messages in the input JSON file (1-based, matching editor line numbers),
# in conversation order: `_conv_id_to_user_lines[conv_id][turn_idx]`. Populated
# in `main()` by scanning the source file; consumed by `send_turn` to embed the
# line of the exact prompt being sent into the request_id's leading 8 chars, so
# server-side logs map straight back to one request rather than to a whole
# conversation. Worker subprocesses inherit this dict via fork (Linux default).
_conv_id_to_user_lines: dict[str, list[int]] = {}


class ClientArgs(NamedTuple):
    seed: int
    max_num_requests: int | None
    skip_first_turn: bool
    max_turns: int | None
    verbose: bool
    print_content: bool
    verify_output: bool
    max_retries: int


class RequestArgs(NamedTuple):
    chat_url: str
    model: str
    stream: bool
    limit_min_tokens: int  # Use negative value for no limit
    limit_max_tokens: int  # Use negative value for no limit
    timeout_sec: int
    forced_output: bool = False
    # Replay start, as perf_counter_ns(). Every coroutine measures its arrival
    # and its `[req] ts=` stamp from this one origin.
    t0_ns: int = 0


class BenchmarkArgs(NamedTuple):
    url: str
    # Cap on requests in flight at once; 0 = unlimited. Not a client count any
    # more: one coroutine per conversation, this bounds how many may be sending.
    max_inflight_req: int
    # Cap on sessions live at once; 0 = unlimited. A session holds its slot from
    # first turn to last, tool time included, so this bounds how many distinct
    # prefixes the server is asked to keep warm rather than how many requests
    # are in flight.
    max_inflight_session: int = 0
    # Sessions per second, as a Poisson process over conversation start times.
    # 0 = every conversation is ready at t=0 and the in-flight cap sets the pace.
    session_arrival_rate: float = 0.0
    # --tool-time spec: "0", a constant like "1s", or "lognormal[:mu,sigma]".
    tool_time: str = "0"


class ServerResponse(NamedTuple):
    valid: bool
    ttft_ms: float  # time to first chunk
    tpot_ms: float  # time per output chunk (one or more tokens)
    latency_ms: float
    start_time_ms: float
    first_chunk: str  # first chunk of the content
    content: str  # includes the first_chunk
    num_chunks: int

    def __str__(self) -> str:
        return f"ttft_ms {self.ttft_ms:.2f}, tpot_ms {self.tpot_ms:.2f}, latency_ms {self.latency_ms:.2f}"  # noqa: E501


class RequestStats(NamedTuple):
    ttft_ms: float
    tpot_ms: float
    latency_ms: float
    start_time_ms: float
    input_num_turns: int
    input_num_tokens: int
    output_num_tokens: int
    output_num_chunks: int
    output_num_first_chunk_tokens: int
    approx_cached_percent: float
    conversation_id: str
    client_id: int

    def __str__(self) -> str:
        return (
            f"ttft_ms {self.ttft_ms:.2f}, tpot_ms {self.tpot_ms:.2f}, latency_ms {self.latency_ms:.2f}, input_num_tokens {self.input_num_tokens}, "  # noqa: E501
            f"output_num_tokens {self.output_num_tokens} ({self.output_num_chunks} chunks, {self.output_num_first_chunk_tokens} tokens in first chunk), "  # noqa: E501
            f"approx_cached_percent {self.approx_cached_percent:.2f}%"
        )


class MetricStats:
    def __init__(self) -> None:
        self.min: float | None = None
        self.max: float | None = None
        self.avg: float | None = None
        self.sum = 0.0
        self.count = 0

    def update(self, value: float) -> None:
        if self.min is None:
            self.min = value
        else:
            self.min = min(self.min, value)

        if self.max is None:
            self.max = value
        else:
            self.max = max(self.max, value)

        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        if self.count == 0:
            return "no data"
        return f"avg: {self.avg:>10.3f}, min: {self.min:>10.3f}, max: {self.max:>10.3f}"


class MovingAverage:
    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.window = np.zeros(window_size)
        self.index = 0
        self.sum = 0.0
        self.count = 0
        self.avg: float | None = None

    def update(self, new_value: float) -> None:
        if self.count < self.window_size:
            # Filling up the window
            self.sum += new_value
            self.window[self.count] = new_value
            self.count += 1
        else:
            # Window is full, start replacing old values
            old_value = self.window[self.index]
            self.sum = self.sum - old_value + new_value
            self.window[self.index] = new_value
            self.index = (self.index + 1) % self.window_size

        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        if self.count == 0:
            return "no data"
        return f"avg: {self.avg:>10.3f} ({self.count} samples)"


class DebugStats:
    def __init__(self, logger: logging.Logger, window_size: int) -> None:
        self.logger = logger
        self.metrics: dict[str, MovingAverage | MetricStats] = {
            "moving_avg_ttft_ms": MovingAverage(window_size),
            "moving_avg_tpot_ms": MovingAverage(window_size),
            "ttft_ms": MetricStats(),
            "tpot_ms": MetricStats(),
            "latency_ms": MetricStats(),
            "input_num_turns": MetricStats(),
            "input_num_tokens": MetricStats(),
            "output_num_tokens": MetricStats(),
        }

    def update(self, data: RequestStats) -> None:
        self.metrics["ttft_ms"].update(data.ttft_ms)
        self.metrics["moving_avg_ttft_ms"].update(data.ttft_ms)
        self.metrics["tpot_ms"].update(data.tpot_ms)
        self.metrics["moving_avg_tpot_ms"].update(data.tpot_ms)
        self.metrics["latency_ms"].update(data.latency_ms)
        self.metrics["input_num_turns"].update(data.input_num_turns)
        self.metrics["input_num_tokens"].update(data.input_num_tokens)
        self.metrics["output_num_tokens"].update(data.output_num_tokens)

    def print(self) -> None:
        self.logger.info("-" * 50)
        for k, v in self.metrics.items():
            kv_info = f"[{k:25}] {v}"
            self.logger.info(kv_info)
        self.logger.info("-" * 50)


def nanosec_to_millisec(value: float) -> float:
    return value / 1000000.0


def nanosec_to_sec(value: float) -> float:
    return value / 1000000000.0


async def send_request(
    session: aiohttp.ClientSession,
    messages: list[dict[str, str]],
    chat_url: str,
    model: str,
    stream: bool = True,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    timeout_sec: int = 120,
    conversation_id: str | None = None,
    request_id: str | None = None,
    forced_output_tokens: list[int] | None = None,
) -> ServerResponse:
    payload = {
        "model": model,
        "messages": messages,
        "seed": 0,
        "temperature": 0.0,
    }

    if conversation_id is not None:
        payload["conversation_id"] = conversation_id

    if forced_output_tokens:
        # Pins the generated tokens to the trace so repeated runs are
        # comparable. Read by vLLM's ForcedOutputTokensLogitsProcessor; dynamo
        # forwards it through sampling_options (PASSTHROUGH_EXTRA_FIELDS).
        payload["forced_output_tokens"] = forced_output_tokens

    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": False}

    if min_tokens is not None:
        payload["min_tokens"] = min_tokens

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {"Content-Type": "application/json"}
    if request_id is not None:
        # Dynamo's frontend only reads `request-id` or `x-dynamo-request-id`
        # (and only if the value parses as a UUID) — the more conventional
        # `X-Request-Id` is captured for tracing but never used as the
        # internal request_id, so we must use one of dynamo's expected names.
        # See dynamo/lib/runtime/src/logging.rs:TraceParent::from_headers.
        headers["x-dynamo-request-id"] = request_id
        headers["X-Request-Id"] = request_id  # also echoed in response for clients

    # Calculate the timeout for the request
    if max_tokens is not None:
        # Assume TPOT of 200ms and use max_tokens to determine timeout
        token_based_timeout = int(max_tokens * 0.2)
        if token_based_timeout > timeout_sec:
            timeout_sec = token_based_timeout
            logger.info(
                "Using timeout of %ds based on max_tokens %d",
                timeout_sec,
                max_tokens,
            )
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    valid_response = True
    ttft: float | None = None
    chunk_delay: list[int] = []
    latency: float | None = None
    first_chunk = ""
    generated_text = ""

    start_time: int = time.perf_counter_ns()
    most_recent_timestamp: int = start_time

    async with session.post(
        url=chat_url, json=payload, headers=headers, timeout=timeout
    ) as response:
        http_status = HTTPStatus(response.status)
        if http_status == HTTPStatus.OK:
            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue

                chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                if chunk == "[DONE]":
                    # End of stream
                    latency = time.perf_counter_ns() - start_time
                elif stream is False:
                    data = json.loads(chunk)
                    message = data["choices"][0]["message"]
                    assert message["role"] == "assistant"
                    generated_text += message["content"]
                else:
                    timestamp: int = time.perf_counter_ns()
                    data = json.loads(chunk)

                    # Delta is the new content/text/data
                    delta = data["choices"][0]["delta"]
                    if delta.get("content", None):
                        if ttft is None:
                            # First token
                            first_token_time = time.perf_counter_ns()
                            ttft = first_token_time - start_time
                            first_chunk = delta["content"]
                        else:
                            # Decoding phase
                            chunk_delay.append(timestamp - most_recent_timestamp)

                        generated_text += delta["content"]

                    most_recent_timestamp = timestamp
        else:
            valid_response = False
            content = await response.text()
            logger.warning(
                f"{Color.YELLOW}Received HTTP status {http_status.value} "
                f"({http_status.phrase}): {content}{Color.RESET}"
            )

    if latency is None:
        latency = -1.0
        if valid_response:
            # Streaming is disabled, latency was not set
            latency = time.perf_counter_ns() - start_time

    if ttft is None:
        # The response was a single chunk
        ttft = latency

    # Each chunk may include more than one token
    tpot: float = mean(chunk_delay) if len(chunk_delay) > 0 else 0.0
    num_chunks: int = len(chunk_delay)

    sr = ServerResponse(
        valid=valid_response,
        ttft_ms=nanosec_to_millisec(ttft) if ttft > 0.0 else -1.0,
        tpot_ms=nanosec_to_millisec(tpot),
        latency_ms=nanosec_to_millisec(latency),
        start_time_ms=nanosec_to_millisec(start_time),
        first_chunk=first_chunk,
        content=generated_text,
        num_chunks=num_chunks,
    )
    return sr


def get_short_string(input: str) -> str:
    n = 20
    if len(input) < 400:
        return input

    return f"{input[:n]}...{input[-n:]}"


def get_token_count(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def get_messages_token_count(
    tokenizer: AutoTokenizer, messages: list[dict[str, str]]
) -> int:
    token_count = 0
    for m in messages:
        token_count += get_token_count(tokenizer, m["content"])

    return token_count


async def send_turn(
    session: aiohttp.ClientSession,
    client_id: int,
    conv_id: str,
    conversation_messages: MessagesList,
    messages_to_use: int,
    tokenizer: AutoTokenizer,
    req_args: RequestArgs,
    verbose: bool,
    verify_output: bool,
) -> RequestStats | None:
    assert messages_to_use > 0
    assert messages_to_use <= len(conversation_messages)

    messages = conversation_messages[:messages_to_use]

    # Index of the next message (the role should be "user")
    index = messages_to_use - 1

    # Verify that the message has only two keys, "role" and "content"
    assert len(messages[index].keys()) == 2
    assert "role" in messages[index] and "content" in messages[index]
    assert messages[index]["role"] == "user", (
        f"Failed on conversation ID {conv_id}, message role should be user"
    )

    if verbose:
        print(
            f"{Color.CYAN}Messages (conversation ID {conv_id},"
            f" {len(messages)} turns):{Color.RESET}",
            messages,
        )

    # None means that there is no upper/lower limit for the output token count
    min_tokens = None if req_args.limit_min_tokens < 0 else req_args.limit_min_tokens
    max_tokens = None if req_args.limit_max_tokens < 0 else req_args.limit_max_tokens

    forced_output_tokens = None
    if len(conversation_messages) > messages_to_use:
        # The conversation contains an assistant answer for the next user prompt
        if (
            min_tokens == NUM_TOKENS_FROM_DATASET
            or max_tokens == NUM_TOKENS_FROM_DATASET
            or req_args.forced_output
        ):
            # Compute number of tokens in the answer (from the input conversation)
            assistant_answer = conversation_messages[messages_to_use]
            answer_num_tokens = get_token_count(tokenizer, assistant_answer["content"])
            assert assistant_answer["role"] == "assistant"

        if req_args.forced_output:
            forced_output_tokens = tokenizer(
                assistant_answer["content"], add_special_tokens=False
            ).input_ids

        if min_tokens == NUM_TOKENS_FROM_DATASET:
            min_tokens = max(1, answer_num_tokens)

        if max_tokens == NUM_TOKENS_FROM_DATASET:
            max_tokens = max(1, answer_num_tokens)

    # Build a deterministic request id so that server-side block-pool logs
    # (which print the request_id passed via X-Request-Id) can be mapped
    # back to the exact prompt that produced them.
    #
    # Dynamo's frontend only accepts request_id headers that parse as a UUID
    # (lib/runtime/src/logging.rs filter), so we emit a valid UUIDv4 with the
    # line number of *this request's* user message in the input JSON encoded
    # into the leading 8 hex chars (zero-padded). The remaining bits are random
    # so retries of the same turn still get distinct IDs.
    #
    # `index` is the position of the user message inside the conversation, so
    # counting only user messages gives its turn ordinal.
    user_turn_idx = sum(
        1 for _m in conversation_messages[:index] if _m["role"] == "user"
    )
    user_lines = _conv_id_to_user_lines.get(conv_id) or []
    line_no = user_lines[user_turn_idx] if user_turn_idx < len(user_lines) else 0
    # Physical line number from the input JSON, zero-padded to 8 decimal
    # chars (line 16 -> "00000016"). Capped at 99999999. Digits 0-9 are
    # valid hex so the segment is still a valid first UUID group.
    line_prefix = f"{min(line_no, 99999999):08d}"
    # Random remainder; uuid4()'s hex already has the v4 version nibble and
    # RFC4122 variant nibble in the right positions (hex[12]='4', hex[16] in
    # {8,9,a,b}), so slicing preserves a valid UUIDv4.
    rand = uuid.uuid4()
    request_id = (
        f"{line_prefix}-"
        f"{rand.hex[8:12]}-"
        f"{rand.hex[12:16]}-"
        f"{rand.hex[16:20]}-"
        f"{rand.hex[20:32]}"
    )

    # Dispatch time on the shared replay timeline, taken before the request
    # goes out. Mirrors mooncake_trace_replay's start_time_ms: that trace also
    # carries its own arrival times, which this dataset does not, so "when the
    # replay actually sent it" is the only arrival notion available here.
    ts_ms = (time.perf_counter_ns() - req_args.t0_ns) // 1_000_000

    # Send the current conversation to LLM and get a response
    response: ServerResponse = await send_request(
        session,
        messages,
        req_args.chat_url,
        req_args.model,
        req_args.stream,
        min_tokens,
        max_tokens,
        req_args.timeout_sec,
        # Deliberately not sent: the server does nothing with it, and Dynamo's
        # frontend validates the request body strictly, rejecting the whole
        # request with `Unsupported parameter(s): conversation_id`. vLLM's own
        # OpenAI layer tolerates it (extra="allow"), so this only ever cost us
        # nothing. Conversation identity is still tracked client-side via
        # conv_id and carried to the server inside request_id.
        conversation_id=None,
        request_id=request_id,
        forced_output_tokens=forced_output_tokens,
    )

    if response.valid is False:
        # Request failed
        return None

    # Compute number of tokens in input / output
    input_num_tokens = get_messages_token_count(tokenizer, messages)

    # Num tokens in the user's last question
    question_num_tokens = get_token_count(tokenizer, messages[index]["content"])

    # Num tokens in the history/context of the question
    assert input_num_tokens >= question_num_tokens
    history_num_tokens = input_num_tokens - question_num_tokens

    # Num tokens in the LLM's answer (first chunk and full answer)
    first_chunk_tokens = get_token_count(tokenizer, response.first_chunk)

    output_content = response.content
    output_num_tokens = get_token_count(tokenizer, output_content)

    # Prefix caching approximated cached percent
    approx_cached_percent = (
        100.0 * (history_num_tokens / input_num_tokens) if input_num_tokens > 0 else 0.0
    )

    # Compute the correct TTFT and TPOT (based on tokens and not chunks).
    # Required because multiple output tokens may be bundled in a single chunk.
    if output_num_tokens > 1 and output_num_tokens > first_chunk_tokens:
        # More than one token and more than one chunk in the output
        decode_ms = response.latency_ms - response.ttft_ms
        decode_num_tokens = output_num_tokens - first_chunk_tokens
        tpot_ms = decode_ms / decode_num_tokens
    else:
        # In this case: output_num_tokens == first_chunk_tokens
        # Output was a single chunk (output_num_tokens > 1)
        # or even a single token (output_num_tokens == 1)
        tpot_ms = 0.0

    if first_chunk_tokens > 1:
        # First chunk had multiple tokens, adjust TTFT for a single token
        delta_ms = (first_chunk_tokens - 1) * tpot_ms
        ttft_ms = max(0.1, response.ttft_ms - delta_ms)
    else:
        # First chunk had only one token
        ttft_ms = response.ttft_ms

    rs = RequestStats(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        latency_ms=response.latency_ms,
        start_time_ms=response.start_time_ms,
        input_num_turns=len(messages),
        input_num_tokens=input_num_tokens,
        output_num_tokens=output_num_tokens,
        output_num_chunks=response.num_chunks,
        output_num_first_chunk_tokens=first_chunk_tokens,
        approx_cached_percent=approx_cached_percent,
        conversation_id=conv_id,
        client_id=client_id,
    )

    # The two lines analyze_lp_ttft_breakdown.py parses, in the format
    # mooncake_trace_replay.py established. `idx` must equal the decimal in the
    # request_id's leading 8 chars — that is how the analyzer joins these rows
    # to the frontend/prefill logs and the nsys timeline. `hashes` has no
    # counterpart here (it counts mooncake hash_ids), so it stays 0. Printed
    # unconditionally: this is data the analyzer needs, not progress chatter.
    print(
        f"[req] idx={int(line_prefix)} ts={ts_ms}ms"
        f" in={input_num_tokens} out={output_num_tokens} hashes=0"
    )
    print(
        f"[ok ] idx={int(line_prefix)} out={output_num_tokens}/{output_num_tokens}"
        f" ttft={ttft_ms:.1f}ms tpot={tpot_ms:.2f}ms"
        f" e2e={response.latency_ms:.1f}ms chunks={response.num_chunks}"
    )

    if verbose:
        print(
            f"\n{Color.YELLOW}Response ({output_num_tokens} tokens):{Color.RESET}",
            output_content,
        )
        print(f"{Color.YELLOW}Response metrics: {rs}{Color.RESET}")
        print("-" * 70)

    # Save the LLM's answer (will be used as part of the context for the next user turn)
    answer_index = messages_to_use
    if len(conversation_messages) > answer_index:
        assert conversation_messages[answer_index]["role"] == "assistant", (
            f"Failed on conversation ID {conv_id}, message role should be assistant"
        )

        orig_content = conversation_messages[answer_index]["content"]
        if verify_output:
            # Compare the new answer to the answer from the input file
            debug_info = (
                f"LLM/dataset answers do not match ({conv_id}):"
                f"\n'{get_short_string(output_content)}' (len: {len(output_content)}),"
                f"\n'{get_short_string(orig_content)}' (len: {len(orig_content)})"
            )
            if orig_content != output_content:
                raise ValueError(debug_info)

        # Update the answer
        conversation_messages[answer_index]["content"] = output_content
    else:
        # A user prompt that has no answer, add the answer as a new message
        new_answer = {"role": "assistant", "content": output_content}
        conversation_messages.append(new_answer)

    return rs


def parse_tool_time(spec: str):
    """Turn a --tool-time spec into `(label, sampler)`.

    The sampler takes a numpy Generator and returns seconds. Three forms:

      0                 no think time at all — maximum pressure
      <number>[s|ms]    a constant, e.g. `1s`, `250ms`, `5`
      lognormal[:mu,sigma]   heavy tail, defaults mu=0 sigma=1 (median 1s,
                             p99 ~10s), the shape agent tool latencies take

    A constant is the honest default when a trace carries no tool timings;
    lognormal exists because real tool time spans milliseconds (a `cd`) to
    tens of seconds (a test suite), and that spread is what decides whether a
    session's blocks are still resident when its next turn arrives.
    """
    spec = (spec or "0").strip().lower()
    if spec in ("0", "", "none", "off"):
        return "0", (lambda rng: 0.0)
    if spec.startswith("lognormal"):
        mu, sigma = 0.0, 1.0
        if ":" in spec:
            parts = spec.split(":", 1)[1].split(",")
            if len(parts) != 2:
                raise ValueError(
                    f"--tool-time {spec!r}: expected lognormal:mu,sigma"
                )
            mu, sigma = float(parts[0]), float(parts[1])
        return (
            f"lognormal(mu={mu},sigma={sigma})",
            lambda rng: float(rng.lognormal(mu, sigma)),
        )
    scale = 1.0
    value = spec
    if spec.endswith("ms"):
        scale, value = 1e-3, spec[:-2]
    elif spec.endswith("s"):
        value = spec[:-1]
    try:
        seconds = float(value) * scale
    except ValueError:
        raise ValueError(
            f"--tool-time {spec!r}: expected 0, a constant like 1s / 250ms, "
            "or lognormal[:mu,sigma]"
        ) from None
    if seconds < 0:
        raise ValueError(f"--tool-time {spec!r}: must not be negative")
    return f"{seconds}s", (lambda rng: seconds)


def build_schedule(
    conv_ids: list[str],
    conversations: ConversationsMap,
    arrival_rate: float,
    tool_time_spec: str,
    seed: int,
) -> dict:
    """Pre-draw every random quantity the replay needs, before it starts.

    Two things are fixed here and nothing else: when each session first
    arrives, and how long each of its turns spends in its tool. What cannot be
    pre-computed is when a later turn arrives — that is
    `A[t+1] = C[t] + Z[t]`, and `C[t]` is the server's actual completion time,
    known only at run time. Fixing it here instead would turn the replay into a
    rigid timetable that stops reflecting the server once it slows down.

    Drawing up front is what makes a run reproducible: one seed and one
    parameter set give the same arrivals and the same tool times regardless of
    how the coroutines interleave.
    """
    label, sampler = parse_tool_time(tool_time_spec)
    rng = np.random.default_rng(seed)
    arrivals: dict[str, float] = {}
    tool_times: dict[str, list[float]] = {}
    clock = 0.0
    for conv_id in conv_ids:
        if arrival_rate > 0:
            # Exponential gaps: the inter-arrival law of a Poisson process.
            clock += float(rng.exponential(1.0 / arrival_rate))
            arrivals[conv_id] = clock
        else:
            # No arrival process: every session is ready at t=0 and the
            # in-flight cap alone decides the pace.
            arrivals[conv_id] = 0.0
        # One draw per message, which over-provisions (only user turns are
        # sent) and costs nothing, so a turn index can be used directly.
        n = len(conversations.get(conv_id) or [])
        tool_times[conv_id] = [sampler(rng) for _ in range(n + 1)]
    return {
        "params": {
            "seed": seed,
            "session_arrival_rate": arrival_rate,
            "tool_time": label,
            "num_conversations": len(conv_ids),
        },
        "arrivals": arrivals,
        "tool_times": tool_times,
    }


async def exponential_backoff_sleep(
    attempt_cnt: int,
    base_rate: float = 1.0,
    backoff_factor: float = 2.0,
    jitter_fraction: float = 0.10,
    verbose: bool = False,
    rng=None,
) -> None:
    # Sleep with exponential backoff and jitter after a failed request.
    backoff_delay = base_rate * (backoff_factor**attempt_cnt)
    draw = (rng or np.random).uniform
    jittered_delay = backoff_delay * (
        1 + draw(-jitter_fraction, jitter_fraction)
    )

    if verbose:
        logger.info(f"Backoff for {jittered_delay:.3f} seconds...")

    await asyncio.sleep(jittered_delay)


async def conversation_coro(
    conv_id: str,
    ordinal: int,
    messages: MessagesList,
    args: ClientArgs,
    req_args: RequestArgs,
    tokenizer: AutoTokenizer,
    session: aiohttp.ClientSession,
    sem: "asyncio.Semaphore | None",
    session_sem: "asyncio.Semaphore | None",
    budget: "RequestBudget",
    schedule: dict,
    metrics: list,
    debug_stats: "DebugStats",
) -> tuple[str, MessagesList] | None:
    """Replay one conversation, in order, until it ends or the budget runs out.

    The arrival model, per turn:

      * turn 1 fires at its pre-drawn Poisson arrival
      * turn t+1 fires at `C[t] + Z[t]` — the moment turn t completed plus that
        turn's tool time. `C` is measured, not scheduled, so a slow server
        pushes the whole session back instead of letting turns pile up.

    Returns the conversation with the server's answers spliced in, or None if
    it never produced one.
    """
    # Its own stream, derived from the run seed and this conversation's
    # position, so retry jitter stays reproducible however the coroutines
    # happen to interleave.
    rng = np.random.default_rng(args.seed + ordinal + 1)

    arrival = schedule["arrivals"].get(conv_id, 0.0)
    tool_times = schedule["tool_times"].get(conv_id) or [0.0]

    due = arrival - (time.perf_counter_ns() - req_args.t0_ns) / 1e9
    if due > 0:
        # Interruptible: a late-arriving session must not hold the run open
        # after -n is spent, and it has nothing left to contribute anyway.
        if await budget.sleep_unless_spent(due):
            return None
    elif budget.exhausted:
        return None

    turns_count = 0
    if args.skip_first_turn:
        # Warmup already sent this conversation's first exchange.
        turns_count += 2
    if turns_count >= len(messages):
        return None

    # Held for the whole session, not per turn, so a slot means "this
    # conversation is live" — including while it sits out its tool time. That
    # is the point: the cap exists to bound how many distinct prefixes the
    # server's KV cache is being asked to keep warm at once. Acquired after the
    # arrival wait so a session queues for a slot only once it is due.
    if session_sem is None:
        return await _replay_turns(
            conv_id, ordinal, messages, args, req_args, tokenizer, session, sem,
            budget, schedule, metrics, debug_stats, rng, tool_times,
            turns_count,
        )
    async with session_sem:
        return await _replay_turns(
            conv_id, ordinal, messages, args, req_args, tokenizer, session, sem,
            budget, schedule, metrics, debug_stats, rng, tool_times,
            turns_count,
        )


async def _replay_turns(
    conv_id: str,
    ordinal: int,
    messages: MessagesList,
    args: ClientArgs,
    req_args: RequestArgs,
    tokenizer: AutoTokenizer,
    session: aiohttp.ClientSession,
    sem: "asyncio.Semaphore | None",
    budget: "RequestBudget",
    schedule: dict,
    metrics: list,
    debug_stats: "DebugStats",
    rng: "np.random.Generator",
    tool_times: list,
    turns_count: int,
) -> tuple[str, MessagesList] | None:
    """Send one conversation's turns in order. Split out of conversation_coro
    only so the session slot can wrap the whole loop with `async with`."""
    produced = False
    while True:
        max_turns = len(messages)
        if args.max_turns is not None:
            max_turns = min(args.max_turns, max_turns)
        if turns_count >= max_turns:
            break
        if not budget.take():
            break

        current_turn = turns_count + 1
        if current_turn >= len(messages):
            break

        success = False
        exception = False
        for attempt_cnt in range(args.max_retries + 1):
            try:
                exception = False
                # None = unlimited: the arrival schedule alone paces the run.
                async with (sem or contextlib.nullcontext()):
                    result = await send_turn(
                        session,
                        ordinal,
                        conv_id,
                        messages,
                        current_turn,
                        tokenizer,
                        req_args,
                        args.print_content,
                        args.verify_output,
                    )
                if result is not None:
                    metrics.append(result)
                    debug_stats.update(result)
                    success = True
                    break
                logger.warning(
                    f"{Color.YELLOW}Conversation {conv_id} - request rejected "
                    f"(turn: {current_turn}){Color.RESET}"
                )
            except asyncio.exceptions.TimeoutError:
                exception = True
                logger.error(
                    "%sTimeout on conversation %s (turn: %d). Base timeout is "
                    "%ss (set with --request-timeout-sec), but the effective "
                    "timeout may be longer based on max_tokens.%s",
                    Color.RED,
                    conv_id,
                    current_turn,
                    req_args.timeout_sec,
                    Color.RESET,
                )
            except Exception:
                exception = True
                logger.exception(
                    f"{Color.RED}Exception on conversation {conv_id} "
                    f"(turn: {current_turn}){Color.RESET}"
                )

            if not success and attempt_cnt < args.max_retries:
                await exponential_backoff_sleep(
                    attempt_cnt, verbose=args.verbose, rng=rng
                )

        if not success:
            budget.give_back()
            # One failed turn ends this conversation: its next prompt would
            # have to contain an answer the server never gave. Other
            # conversations are unaffected — that isolation is the point of a
            # coroutine per conversation.
            if exception:
                logger.warning(
                    f"{Color.YELLOW}Conversation {conv_id} abandoned after an "
                    f"exception on turn {current_turn}{Color.RESET}"
                )
            break

        produced = True
        # Count the assistant answer too: it is context for the next prompt.
        turns_count = current_turn + 1

        if turns_count >= max_turns:
            break

        # Z: the tool call this turn implies has to run before the next
        # prompt exists. Pre-drawn, so the wait is reproducible. The wait is
        # interruptible: sitting out a tool time only to find the budget spent
        # would add it to the run's tail for nothing.
        think = tool_times[min(current_turn, len(tool_times) - 1)]
        if think > 0 and await budget.sleep_unless_spent(think):
            break

    return (conv_id, messages) if produced else None


class RequestBudget:
    """The shared -n cap. A turn takes a slot before sending and returns it if
    the send failed, so -n counts requests that actually landed.

    It also carries the run's stop signal. Every conversation in the input file
    gets a coroutine up front, and with a session arrival rate the last of them
    is scheduled hours out, so once -n is spent the sessions still waiting for
    their arrival instant have to be woken and told to give up — otherwise the
    run sits idle until the whole arrival schedule has played out.
    """

    def __init__(self, limit: int | None):
        self._limit = limit
        self._used = 0
        self._spent = asyncio.Event()
        # Sessions that gave up while waiting, reported at the end so a short
        # `Finished x/y conversations` is not mistaken for a crash.
        self.dropped_waiting = 0

    def take(self) -> bool:
        if self._limit is not None and self._used >= self._limit:
            return False
        self._used += 1
        if self._limit is not None and self._used >= self._limit:
            self._spent.set()
        return True

    def give_back(self) -> None:
        self._used = max(0, self._used - 1)
        if self._limit is not None and self._used < self._limit:
            # A failed send freed the slot, so sessions still waiting are back
            # in play; -n keeps counting requests that landed.
            self._spent.clear()

    async def sleep_unless_spent(self, seconds: float) -> bool:
        """Sleep, waking early if the budget runs out.

        Returns True if the caller should stop (nothing left to send), False if
        the sleep simply elapsed.
        """
        if self.exhausted:
            self.dropped_waiting += 1
            return True
        try:
            await asyncio.wait_for(self._spent.wait(), timeout=seconds)
        except (TimeoutError, asyncio.TimeoutError):
            return False
        self.dropped_waiting += 1
        return True

    @property
    def used(self) -> int:
        return self._used

    @property
    def exhausted(self) -> bool:
        return self._limit is not None and self._used >= self._limit


def get_client_config(
    args: argparse.Namespace, input_conv: ConversationsMap
) -> tuple[ClientArgs, RequestArgs]:
    if args.max_inflight_req < 0:
        raise ValueError("--max-inflight-req must be >= 0 (0 = unlimited)")

    if args.max_inflight_req and len(input_conv) < args.max_inflight_req:
        raise ValueError(
            "Number of conversations must be equal or larger than "
            "--max-inflight-req"
        )

    if args.max_inflight_session < 0:
        raise ValueError("--max-inflight-session must be >= 0 (0 = unlimited)")

    # -n is a fleet-wide cap now, shared through RequestBudget: there are no
    # per-client shares to divide and no remainder to lose.
    max_num_requests = args.max_num_requests
    if max_num_requests is not None and max_num_requests < 1:
        raise ValueError("Number of requests must be a positive number")

    # Skip the first user turn (as part of the warmup)
    skip_first_turn = args.warmup_step

    client_args = ClientArgs(
        seed=args.seed,
        max_num_requests=max_num_requests,
        skip_first_turn=skip_first_turn,
        max_turns=args.max_turns,
        verbose=args.verbose,
        print_content=args.print_content,
        verify_output=args.verify_output,
        max_retries=args.max_retries,
    )

    if args.limit_min_tokens > 0 or args.limit_max_tokens > 0:
        if args.limit_min_tokens < 1 or args.limit_max_tokens < 1:
            raise ValueError(
                "Invalid min/max tokens limits (both limits should be provided)"
            )
        if args.limit_min_tokens > args.limit_max_tokens:
            raise ValueError(
                "Invalid min/max tokens limits (min should not be larger than max)"
            )

    if args.request_timeout_sec <= 0:
        raise ValueError("Request timeout must be a positive number")

    # Arguments for API requests
    chat_url = f"{args.url}/v1/chat/completions"
    model_name = args.served_model_name if args.served_model_name else args.model

    req_args = RequestArgs(
        chat_url=chat_url,
        model=model_name,
        stream=not args.no_stream,
        limit_min_tokens=args.limit_min_tokens,
        limit_max_tokens=args.limit_max_tokens,
        timeout_sec=args.request_timeout_sec,
        forced_output=args.forced_output,
        # Placeholder: main_mp re-stamps this when the replay actually starts,
        # so arrivals are measured from the first request and not from argument
        # parsing (which happens before the tokenizer loads).
        t0_ns=time.perf_counter_ns(),
    )

    return client_args, req_args


async def main_mp(
    client_args: ClientArgs,
    req_args: RequestArgs,
    bench_args: BenchmarkArgs,
    tokenizer: AutoTokenizer,
    input_conv: ConversationsMap,
) -> tuple[ConversationsMap, list[RequestStats]]:
    """Replay every conversation concurrently, one coroutine each.

    One process, not one per client: a conversation is a sequence of dependent
    turns, so what limits throughput is how many requests may be in flight
    (`--max-inflight-req`), not how many OS processes exist. Coroutines also let
    a conversation wait out its tool time without blocking any other — the old
    model slept the whole client, so a think time froze every conversation that
    client held.

    `--max-inflight-session` is the other axis: it bounds how many
    conversations are live at once, so requests concentrate on a few warm
    prefixes instead of spreading over many sessions that only ever sent turn 1.
    """
    conv_ids = list(input_conv)
    schedule = build_schedule(
        conv_ids,
        input_conv,
        bench_args.session_arrival_rate,
        bench_args.tool_time,
        client_args.seed,
    )
    logger.info(
        f"{Color.GREEN}Replaying {len(conv_ids)} conversations: "
        f"in-flight req cap={bench_args.max_inflight_req or 'unlimited'}, "
        f"in-flight session cap="
        f"{bench_args.max_inflight_session or 'unlimited'}, "
        f"arrival={schedule['params']['session_arrival_rate']} sessions/s, "
        f"tool_time={schedule['params']['tool_time']}{Color.RESET}"
    )

    output_conv: ConversationsMap = {}
    client_metrics: list[RequestStats] = []
    budget = RequestBudget(client_args.max_num_requests)
    # 0 = unlimited for both caps. Without a request cap there is no meaningful
    # multiplier for these two, so fall back to fixed values rather than
    # letting `0 *` collapse them: a 0-sized debug window would print nothing
    # and a 0 RPS threshold would report RPS from the very first sample.
    inflight_cap = bench_args.max_inflight_req
    debug_stats = DebugStats(
        logger, min(15 * inflight_cap, 500) if inflight_cap else 500
    )
    sem = asyncio.Semaphore(inflight_cap) if inflight_cap > 0 else None
    rps_min_samples = 5 * inflight_cap if inflight_cap else 50
    # 0 = unlimited: every conversation's coroutine starts as soon as its
    # arrival instant passes, which is the historical behaviour.
    session_sem = (
        asyncio.Semaphore(bench_args.max_inflight_session)
        if bench_args.max_inflight_session > 0
        else None
    )

    start_time = time.perf_counter_ns()
    req_args = req_args._replace(t0_ns=start_time)
    total_convs = len(conv_ids)
    done = 0

    async def progress() -> None:
        """The old progress print hung off a blocking queue read; with the
        queues gone it needs its own clock, or the run prints nothing at all."""
        while True:
            await asyncio.sleep(PROGRESS_INTERVAL_SEC)
            runtime_sec = nanosec_to_sec(time.perf_counter_ns() - start_time)
            rps: str | float = (
                round(len(client_metrics) / runtime_sec, 3) if runtime_sec else "N/A"
            )
            if len(client_metrics) < rps_min_samples:
                rps = "N/A"
            # Once -n is spent the remaining sessions are being dropped, not
            # run, so `Finished x/y` stops climbing toward y. Say so, or the
            # print looks like a stall.
            spent = " (-n spent, draining)" if budget.exhausted else ""
            logger.info(
                f"{Color.CYAN}Finished {done}/{total_convs} conversations, "
                f"{budget.used} requests, {len(client_metrics)} measurements, "
                f"runtime {runtime_sec:.1f}s, RPS {rps}{spent}{Color.RESET}"
            )
            debug_stats.print()

    # aiohttp's default connector allows 100 sockets; a larger in-flight cap
    # would silently queue inside the client and show up as inflated TTFT.
    # With the cap unlimited the ceiling is one request per conversation, so
    # size the pool to that instead of leaving it at 100 -- otherwise "no cap"
    # would quietly become a 100-request cap enforced by the socket pool.
    conn_limit = (
        max(100, inflight_cap * 2) if inflight_cap else max(100, 2 * len(conv_ids))
    )
    connector = aiohttp.TCPConnector(limit=conn_limit)
    async with aiohttp.ClientSession(connector=connector) as session:
        progress_task = asyncio.create_task(progress())
        tasks = [
            asyncio.create_task(
                conversation_coro(
                    conv_id,
                    ordinal,
                    input_conv[conv_id],
                    client_args,
                    req_args,
                    tokenizer,
                    session,
                    sem,
                    session_sem,
                    budget,
                    schedule,
                    client_metrics,
                    debug_stats,
                )
            )
            for ordinal, conv_id in enumerate(conv_ids)
        ]
        try:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is not None:
                    conv_id, messages = result
                    output_conv[conv_id] = messages
                done += 1
        finally:
            progress_task.cancel()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    runtime_sec = nanosec_to_sec(time.perf_counter_ns() - start_time)
    dropped = (
        f", {budget.dropped_waiting} sessions dropped unsent (-n reached)"
        if budget.dropped_waiting
        else ""
    )
    logger.info(
        f"{Color.GREEN}All conversations finished: {budget.used} requests, "
        f"{len(client_metrics)} measurements, runtime {runtime_sec:.3f} sec, "
        f"requests per second: {len(client_metrics) / runtime_sec:.3f}"
        f"{dropped}{Color.RESET}"
    )
    return output_conv, client_metrics


def get_filename_with_timestamp(label: str, extension: str) -> str:
    time_now = datetime.now()
    timestamp = time_now.strftime("%d-%m-%Y_%H-%M-%S")
    filename = f"{label}__{timestamp}.{extension}"
    return filename


def process_statistics(
    client_metrics: list[RequestStats],
    warmup_percentages: list[float],
    test_params: dict,
    verbose: bool,
    gen_conv_args: GenConvArgs | None = None,
    excel_output: bool = False,
    warmup_runtime_sec: float | None = None,
) -> None:
    if len(client_metrics) == 0:
        logger.info("No samples to process")
        return

    logger.info(f"Processing {len(client_metrics)} samples...")

    raw_data = pd.DataFrame(client_metrics)

    if verbose:
        # Calculate the time between user turns in each conversation (in a new column)
        raw_data = raw_data.sort_values(by=["conversation_id", "start_time_ms"])
        raw_data["time_between_user_turns_sec"] = raw_data.groupby("conversation_id")[
            "start_time_ms"
        ].diff()

        # Convert milliseconds to seconds
        raw_data["time_between_user_turns_sec"] = (
            raw_data["time_between_user_turns_sec"] / 1000.0
        )

    # Final raw data should be sorted by time
    raw_data = raw_data.sort_values(by=["start_time_ms"])
    raw_data["end_time_ms"] = raw_data["start_time_ms"] + raw_data["latency_ms"]

    percentiles = [0.25, 0.5, 0.75, 0.9]

    # Add more percentiles if there are enough samples
    if len(raw_data) >= 100:
        percentiles.append(0.99)

    if len(raw_data) >= 1000:
        percentiles.append(0.999)

    if len(raw_data) >= 10000:
        percentiles.append(0.9999)

    # Set precision for numbers in the output text (the dataframes)
    pd.set_option("display.precision", 2)

    # Exclude parameters from RequestStats
    exclude = [
        "start_time_ms",
        "end_time_ms",
        "output_num_first_chunk_tokens",
        "approx_cached_percent",
        "conversation_id",
        "client_id",
    ]

    print(TEXT_SEPARATOR)
    print(f"{Color.YELLOW}Parameters:{Color.RESET}")
    for k, v in test_params.items():
        print(f"{k}={v}")

    # conversations generation parameters
    if gen_conv_args is not None:
        gen_params = {
            "text_files": ", ".join(gen_conv_args.text_files),
            "input_num_turns": str(gen_conv_args.input_num_turns),
            "input_common_prefix_num_tokens": str(
                gen_conv_args.input_common_prefix_num_tokens
            ),
            "input_prefix_num_tokens": str(gen_conv_args.input_prefix_num_tokens),
            "input_num_tokens": str(gen_conv_args.input_num_tokens),
            "output_num_tokens": str(gen_conv_args.output_num_tokens),
        }

        print(f"{Color.YELLOW}Conversations Generation Parameters:{Color.RESET}")
        for k, v in gen_params.items():
            print(f"{k}={v}")

    print(TEXT_SEPARATOR)

    params_list = []
    df_list = []
    for percent in warmup_percentages:
        # Select samples from the end (tail) of the dataframe
        warmup_count = int(percent * len(raw_data))
        tail_count = len(raw_data) - warmup_count
        if tail_count == 0:
            # No reason to process if the count of samples is zero
            break

        df = raw_data.tail(tail_count)

        # Runtime is the diff between the end of the last request
        # and the start of the first request
        runtime_sec = df["end_time_ms"].iloc[-1] - df["start_time_ms"].iloc[0]

        # Convert milliseconds to seconds
        runtime_sec = runtime_sec / 1000.0
        requests_per_sec = float(len(df)) / runtime_sec
        params = {
            "runtime_sec": runtime_sec,
            "requests_per_sec": requests_per_sec,
        }
        if warmup_runtime_sec is not None:
            params["warmup_runtime_sec"] = warmup_runtime_sec
            params["total_runtime_incl_warmup_sec"] = runtime_sec + warmup_runtime_sec

        # Generate a summary of relevant metrics (and drop irrelevant data)
        df = df.drop(columns=exclude).describe(percentiles=percentiles).transpose()

        # List for Excel file
        params_list.append(params)
        df_list.append(df)

        # Print the statistics summary
        if percent > 0 or len(warmup_percentages) > 1:
            print(
                f"{Color.YELLOW}Statistics summary "
                f"(assuming {percent:.0%} warmup samples):{Color.RESET}"
            )
        else:
            print(f"{Color.YELLOW}Statistics summary:{Color.RESET}")

        for k, v in params.items():
            if isinstance(v, float):
                print(f"{k} = {v:.3f}")
            else:
                print(f"{k} = {v}")
        print(TEXT_SEPARATOR)
        print(df)
        print(TEXT_SEPARATOR)

    if excel_output:
        prefix = f"statistics_{test_params['max_inflight_req']}_inflight_req"
        filename = get_filename_with_timestamp(prefix, "xlsx")

        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            startrow = 0
            test_params_df = pd.DataFrame([test_params])
            test_params_df.to_excel(
                writer, sheet_name="Summary", index=False, startrow=startrow
            )
            startrow += len(test_params_df) + 3

            if gen_conv_args is not None:
                gen_params_df = pd.DataFrame([gen_params])
                gen_params_df.to_excel(
                    writer, sheet_name="Summary", index=False, startrow=(startrow - 1)
                )
                startrow += len(gen_params_df) + 3

            for params, df_stats in zip(params_list, df_list):
                df_params = pd.DataFrame([params])
                df_params.to_excel(
                    writer, sheet_name="Summary", index=False, startrow=startrow
                )
                startrow += len(df_params) + 2
                df_stats.to_excel(
                    writer, sheet_name="Summary", index=True, startrow=startrow
                )
                startrow += len(df_stats) + 3

            raw_data.to_excel(writer, sheet_name="Raw data", index=False, startrow=0)

        logger.info(
            f"{Color.GREEN}Client metrics exported to file: {filename}{Color.RESET}"
        )


async def get_server_info(url: str) -> None:
    logger.info(f"{Color.BLUE}Collecting information from server: {url}{Color.RESET}")
    async with aiohttp.ClientSession() as session:
        # Get server version (not mandatory, "version" endpoint may not exist)
        url_version = f"{url}/version"
        async with session.get(url_version) as response:
            if HTTPStatus(response.status) == HTTPStatus.OK:
                text = await response.text()
                logger.info(f"{Color.BLUE}Server version: {text}{Color.RESET}")

        # Get available models
        url_models = f"{url}/v1/models"
        async with session.get(url_models) as response:
            if HTTPStatus(response.status) == HTTPStatus.OK:
                text = await response.text()
                logger.info(f"{Color.BLUE}Models:{Color.RESET}")
                models_data = json.loads(text)
                models_list = models_data["data"]
                for model in models_list:
                    model_id = model["id"]
                    max_model_len = model.get("max_model_len", "N/A")
                    logger.info(
                        f"{Color.BLUE}\t{model_id=}, {max_model_len=}{Color.RESET}"
                    )
            else:
                logger.info(f"{Color.RED}Failed to get models{Color.RESET}")


def dump_schedule(
    conv_ids: list[str],
    conversations: ConversationsMap,
    arrival_rate: float,
    tool_time_spec: str,
    seed: int,
    path: str,
) -> None:
    """Record what the arrival model drew, before the replay uses it.

    Only the first turn of each conversation has a knowable time: every later
    turn fires at `completion(previous) + Z`, and completion is the server's to
    decide. So this holds the arrival instants and the drawn tool times — the
    inputs to the run — not a full timetable, which cannot exist here.
    """
    schedule = build_schedule(
        conv_ids, conversations, arrival_rate, tool_time_spec, seed
    )
    rows = []
    for ordinal, conv_id in enumerate(conv_ids):
        messages = conversations.get(conv_id) or []
        tool_times = schedule["tool_times"].get(conv_id) or []
        # Only user turns are sent; the assistant answers come back from the
        # server, so they are not requests.
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        rows.append(
            {
                "ordinal": ordinal,
                "conversation_id": conv_id,
                "arrival_sec": round(schedule["arrivals"][conv_id], 6),
                "num_messages": len(messages),
                "num_user_turns": user_turns,
                "tool_seconds": [round(z, 6) for z in tool_times[:user_turns]],
                "tool_seconds_total": round(sum(tool_times[:user_turns]), 6),
            }
        )
    rows.sort(key=lambda r: r["arrival_sec"])
    with open(path, "w") as fh:
        fh.write(json.dumps({"params": schedule["params"]}) + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    arrivals = [r["arrival_sec"] for r in rows]
    total_turns = sum(r["num_user_turns"] for r in rows)
    all_z = [z for r in rows for z in r["tool_seconds"]]
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    logger.info(f"{Color.GREEN}-> {path}{Color.RESET}")
    logger.info(
        f"  {len(rows):,} conversations, {total_turns:,} user turns "
        f"(= requests), params {json.dumps(schedule['params'])}"
    )
    logger.info(
        f"  arrivals span {arrivals[-1] - arrivals[0]:,.1f}s"
        f"  ({(arrivals[-1] - arrivals[0]) / 60:,.1f}min)"
        if len(arrivals) > 1
        else "  single conversation"
    )
    if gaps:
        ordered = sorted(gaps)
        logger.info(
            f"  arrival gap p50={ordered[len(ordered) // 2]:.4f}s "
            f"p90={ordered[int(0.9 * (len(ordered) - 1))]:.4f}s"
        )
    if all_z:
        ordered = sorted(all_z)
        logger.info(
            f"  tool time p50={ordered[len(ordered) // 2]:.4f}s "
            f"p90={ordered[int(0.9 * (len(ordered) - 1))]:.4f}s "
            f"p99={ordered[int(0.99 * (len(ordered) - 1))]:.4f}s "
            f"mean={sum(all_z) / len(all_z):.4f}s"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Benchmark serving with multi-turn conversations",
        description="Benchmark online inference using REST API",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0")

    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        help="Input JSON file with ShareGPT conversations or "
        "configuration file for generation of synthetic conversations",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=None,
        help="Output JSON file containing conversations with updated assistant answers",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for random number generators (default: 0)",
    )

    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path of the LLM model"
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. "
        "If not specified, the model name will be the "
        "same as the `--model` argument. ",
    )

    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL for the LLM API server",
    )

    parser.add_argument(
        "--max-inflight-req",
        type=int,
        default=1,
        help="Cap on requests in flight at once (0 = unlimited). Not a client "
        "count: every conversation gets its own coroutine and this bounds how "
        "many may be sending. Use it to keep a high --session-arrival-rate from "
        "swamping the server; set it to 0 (or leave it high) to let the arrival "
        "rate set the pace.",
    )
    parser.add_argument(
        "--max-inflight-session",
        type=int,
        default=0,
        help="Cap on sessions active at once (0 = unlimited). A session holds a "
        "slot from its first turn until its last, so this bounds how many "
        "distinct conversations the server's KV cache must hold. Without it a "
        "high --session-arrival-rate admits far more sessions than "
        "--max-inflight-req can serve, and most sit having sent only their "
        "first turn: requests spread thin over many cold prefixes instead of "
        "revisiting a few warm ones. Setting it close to --max-inflight-req "
        "makes served requests and live sessions match, so later turns (which "
        "have KV to load) dominate.",
    )
    parser.add_argument(
        "-n",
        "--max-num-requests",
        type=int,
        default=None,
        help="Max number of requests to send across the whole run",
    )

    parser.add_argument(
        "--warmup-step",
        default=False,
        action="store_true",
        help="Run a warmup step (using only the first turn of every conversation), "
        "measurements will not be included in the final benchmark results",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum number of turns/messages per conversation, "
        "includes both user and assistant messages "
        "(a positive number, e.g: 2, 4, 6, etc.), disabled by default",
    )
    parser.add_argument(
        "--limit-max-tokens",
        type=int,
        default=NUM_TOKENS_FROM_DATASET,
        help="Set max_tokens for the output token count of each request "
        "(must also set --limit-min-tokens). "
        "Overrides output token count from the input dataset. "
        "Use a negative value to disable this limit.",
    )
    parser.add_argument(
        "--limit-min-tokens",
        type=int,
        default=NUM_TOKENS_FROM_DATASET,
        help="Set min_tokens for the output token count of each request "
        "(must also set --limit-max-tokens). "
        "Overrides output token count from the input dataset. "
        "Use a negative value to disable this limit.",
    )
    parser.add_argument(
        "--forced-output",
        action="store_true",
        help="Force each response to be the trace's own assistant answer, "
        "token for token, so repeated runs are comparable. Requires the server "
        "to expose ForcedOutputTokensLogitsProcessor.",
    )

    parser.add_argument(
        "--session-arrival-rate",
        type=float,
        default=0,
        help="New conversations per second, drawn as a Poisson process over "
        "their start times. This is an open-loop arrival rate: a conversation "
        "fires at its scheduled instant regardless of how the server is doing, "
        "so raising it past what the server can absorb is what builds a queue "
        "and oversubscribes the KV cache. 0 = every conversation is ready at "
        "once and --max-inflight-req alone paces the run.",
    )
    parser.add_argument(
        "--tool-time",
        type=str,
        default="0",
        help="How long the tool call between two turns takes, i.e. Z in "
        "arrival(turn t+1) = completion(turn t) + Z. Accepts `0` (no wait, "
        "maximum pressure), a constant such as `1s` / `250ms`, or "
        "`lognormal[:mu,sigma]` (default mu=0 sigma=1: median 1s, heavy tail "
        "to ~10s) which is the shape real agent tool latencies take. Drawn "
        "once per turn before the replay starts, so a seed reproduces it.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("MULTITURN_BENCH_MAX_RETRIES", "0")),
        help="Maximum number of retry attempts for timed-out requests. "
        "Default is 0 (no retries). "
        "Set to higher values to retry failed requests and maintain "
        "fair workload distribution. "
        "Can also be set via MULTITURN_BENCH_MAX_RETRIES environment variable.",
    )
    parser.add_argument(
        "--verify-output",
        default=False,
        action="store_true",
        help="Verify the LLM output (compare to the answers in the input JSON file)",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=int,
        default=120,
        help="Timeout in seconds for each API request (default: 120). "
        "Automatically increased if max tokens imply longer decoding.",
    )

    parser.add_argument(
        "--no-stream",
        default=False,
        action="store_true",
        help="Disable stream/streaming mode (set 'stream' to False in the API request)",
    )

    parser.add_argument(
        "-e",
        "--excel-output",
        default=False,
        action="store_true",
        help="Export summary to Excel file (optional)",
    )
    parser.add_argument(
        "--stats-json-output",
        type=str,
        default=None,
        help="Export per-request stats (ttft_ms, tpot_ms, etc.) to a JSON file",
    )
    parser.add_argument(
        "--schedule-out",
        type=str,
        default=None,
        metavar="PATH",
        help="Where to write the arrival schedule (default: "
        f"{DEFAULT_SCHEDULE_NAME} next to the input trace). Records each "
        "conversation's arrival instant and drawn tool times, with the "
        "parameters that produced them on the first line.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--print-content",
        default=False,
        action="store_true",
        help="Print the user prompts and the server's answers",
    )

    parser.add_argument(
        "--warmup-percentages",
        type=str,
        default="0%",
        help="Ignore the first X samples as warmup (X is a percentage)."
        " A comma separated list of percentages can be used "
        "(for example: --warmup-percentages=0%%,50%%)",
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the tokenizer.",
    )

    args = parser.parse_args()

    logger.info(args)

    logger.info(f"{Color.GREEN}Input parameters:{Color.RESET}")
    logger.info(f"url={args.url}")
    logger.info(f"model={args.model}")
    logger.info(f"max_inflight_req={args.max_inflight_req}")
    logger.info(
        f"max_inflight_session={args.max_inflight_session or 'unlimited'}"
    )

    if args.verify_output:
        logger.info(f"{Color.PURPLE}Verify is enabled{Color.RESET}")

    # Calculate the amount of samples to filter (as warmup samples/measurements).
    try:
        warmup_percentages: list[float] = [0.0]
        if not args.warmup_step:
            # Warmup percentage can be used only if the warmup step was used
            warmup_strings: list[str] = args.warmup_percentages.split(",")
            warmup_strings = [x.replace("%", "") for x in warmup_strings]
            warmup_percentages = [float(x) / 100 for x in warmup_strings]

            # Check for valid range (0 to 1)
            for p in warmup_percentages:
                assert p >= 0.0 and p < 1.0

            # Sort from high to low warmup percentage
            warmup_percentages.sort()

            logger.info(
                f"Warmup percentages (percentage of samples): {warmup_percentages}"
            )

    except Exception:
        raise ValueError(
            f"Invalid --warmup-percentage={args.warmup_percentage}"
        ) from None

    # Set global seeds for main process
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )

    await get_server_info(args.url)

    # Load the input file (either conversations of configuration file)
    logger.info(f"Reading input file: {args.input_file}")
    with open(args.input_file) as f:
        input_data = json.load(f)

    gen_conv_args = None
    if isinstance(input_data, list):
        # The conversations are stored as a list of dicts
        logger.info(f"Found {len(input_data)} items in the input file")

        # Map each conv_id to the line number of each of its user messages
        # (1-based, matching editor line numbers). The file is pretty-printed
        # with one `"id": "..."` and one `"role": "..."` per line, so a linear
        # scan is enough: every `"role": "user"` seen after a given `"id"`
        # belongs to that conversation, in order.
        # Used by send_turn to embed the prompt's line into the request_id.
        import re

        id_line_re = re.compile(r'"id"\s*:\s*"([^"\\]*)"')
        user_role_re = re.compile(r'"role"\s*:\s*"user"')
        with open(args.input_file) as _f:
            _current: list[int] | None = None
            for _lineno, _line in enumerate(_f, start=1):
                _m = id_line_re.search(_line)
                if _m:
                    _current = []
                    _conv_id_to_user_lines[_m.group(1)] = _current
                    continue
                if _current is not None and user_role_re.search(_line):
                    _current.append(_lineno)

        # Convert the list to a ConversationsMap
        conversations = conversations_list_to_dict(input_data)

    elif isinstance(input_data, dict):
        # The input file is a configuration file
        # (type is determined by the field 'filetype')
        if "filetype" not in input_data:
            raise Exception(
                f"Input file {args.input_file} is invalid (missing 'filetype')"
            )

        logger.info(f"Using input file with filetype: {input_data['filetype']}")

        gen_conv_args = parse_input_json_file(input_data)

        # Disable warning from "huggingface/tokenizers"
        # (when using python multiprocessing and tokenizers)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        # Generate synthetic conversations
        conversations = generate_conversations(gen_conv_args, tokenizer)

    else:
        raise Exception(f"Input file {args.input_file} is invalid")

    if args.max_turns is not None:
        if args.max_turns < 1:
            raise ValueError("Max turns must be a positive number")
        logger.info(
            f"{Color.PURPLE}Max turns per conversation "
            f"is limited to {args.max_turns}{Color.RESET}"
        )

    # Create benchmark configurations
    client_args, req_args = get_client_config(args, conversations)

    bench_args = BenchmarkArgs(
        url=args.url,
        max_inflight_req=args.max_inflight_req,
        max_inflight_session=args.max_inflight_session,
        session_arrival_rate=args.session_arrival_rate,
        tool_time=args.tool_time,
    )

    dump_schedule(
        list(conversations),
        conversations,
        args.session_arrival_rate,
        args.tool_time,
        args.seed,
        args.schedule_out
        or os.path.join(
            os.path.dirname(os.path.abspath(args.input_file)),
            DEFAULT_SCHEDULE_NAME,
        ),
    )

    warmup_runtime_sec: float | None = None

    # Warm-up step
    if args.warmup_step:
        # Only the first user prompt of every conversation.
        warmup_client_args = client_args._replace(
            skip_first_turn=False, max_turns=1
        )

        # Warmup exists to populate the cache, not to model a workload: fire
        # every conversation at once (no arrival process) and skip the tool
        # time, since there is no second turn to wait for.
        warmup_bench_args = bench_args._replace(
            session_arrival_rate=0.0, tool_time="0"
        )

        logger.info("%sWarmup start%s", Color.PURPLE, Color.RESET)
        warmup_start_ns = time.perf_counter_ns()
        conversations, _ = await main_mp(
            warmup_client_args, req_args, warmup_bench_args, tokenizer, conversations
        )
        warmup_runtime_sec = nanosec_to_sec(time.perf_counter_ns() - warmup_start_ns)
        logger.info(
            "%sWarmup runtime: %.3f sec (%.3f ms)%s",
            Color.PURPLE,
            warmup_runtime_sec,
            warmup_runtime_sec * 1000,
            Color.RESET,
        )
        logger.info("%sWarmup done%s", Color.PURPLE, Color.RESET)

    # Run the benchmark
    benchmark_start_ns = time.perf_counter_ns()
    client_convs, client_metrics = await main_mp(
        client_args, req_args, bench_args, tokenizer, conversations
    )
    benchmark_runtime_sec = nanosec_to_sec(time.perf_counter_ns() - benchmark_start_ns)

    # Calculate requests per second
    requests_per_sec = len(client_metrics) / benchmark_runtime_sec
    benchmark_runtime_ms = benchmark_runtime_sec * 1000.0
    logger.info(
        "%sAll clients finished, benchmark runtime: %.3f sec (%.3f ms), "
        "requests per second: %.3f%s",
        Color.GREEN,
        benchmark_runtime_sec,
        benchmark_runtime_ms,
        requests_per_sec,
        Color.RESET,
    )
    if warmup_runtime_sec is not None:
        total_runtime_sec = benchmark_runtime_sec + warmup_runtime_sec
        logger.info(
            "%sWarmup runtime: %.3f sec (%.3f ms)%s",
            Color.GREEN,
            warmup_runtime_sec,
            warmup_runtime_sec * 1000,
            Color.RESET,
        )
        logger.info(
            "%sTotal runtime (including warmup): %.3f sec (%.3f ms)%s",
            Color.GREEN,
            total_runtime_sec,
            total_runtime_sec * 1000,
            Color.RESET,
        )

    # Benchmark parameters
    params = {
        "model": args.model,
        "max_inflight_req": args.max_inflight_req,
        "max_inflight_session": args.max_inflight_session,
        "num_conversations": len(conversations),
        "session_arrival_rate": args.session_arrival_rate,
        "tool_time": args.tool_time,
        "seed": args.seed,
    }

    if args.limit_min_tokens > 0:
        params["min_tokens"] = args.limit_min_tokens

    if args.limit_max_tokens > 0:
        params["max_tokens"] = args.limit_max_tokens

    # Process and print statistics (and save excel file with the statistics)
    process_statistics(
        client_metrics,
        test_params=params,
        warmup_percentages=warmup_percentages,
        verbose=args.verbose,
        gen_conv_args=gen_conv_args,
        excel_output=args.excel_output,
        warmup_runtime_sec=warmup_runtime_sec,
    )

    if args.stats_json_output is not None:
        # Export per-request metrics as a JSON array for downstream analysis.
        stats_data = [s._asdict() for s in client_metrics]
        logger.info(
            f"{Color.GREEN}Writing per-request stats JSON: "
            f"{args.stats_json_output}{Color.RESET}"
        )
        os.makedirs(
            os.path.dirname(os.path.abspath(args.stats_json_output)), exist_ok=True
        )
        with open(args.stats_json_output, "w") as f:
            json.dump(stats_data, f, indent=2)

    if args.output_file is not None:
        # Write a JSON file with the updated conversations
        # The "assistant" content will contain the answers from the tested LLM
        output_data: ShareGptConversations = conversations_dict_to_list(client_convs)
        logger.info(
            f"{Color.GREEN}Writing conversations file: {args.output_file}{Color.RESET}"
        )
        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
