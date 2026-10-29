# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay a Mooncake trace against a vLLM ``/v1/completions`` endpoint.

Each trace record is sent as an independent request:
    * prompt    -> token ids derived from ``hash_ids`` (block_size tokens
                   per hash, then truncated to ``input_length``).
    * response  -> NOT mapped from the trace. ``hash_ids`` only describe the
                   prompt, so the response is left to the model to generate.
                   We only pin its *length* via ``max_tokens`` / ``min_tokens``
                   = ``output_length`` (with ``ignore_eos``), and do not force
                   any specific output tokens.
    * timing    -> requests are dispatched at ``timestamp_ms / time_scale``
                   relative to start.

Trace line schema::

    {"timestamp": <ms>, "input_length": <int>, "output_length": <int>,
     "hash_ids": [<int>, ...]}

Each ``hash_id`` corresponds to a fixed-size block of tokens (512). Token-id mapping: token_id(h) = (h % vocab_range) + token_offset
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from http import HTTPStatus

import aiohttp  # type: ignore


# ---- trace parsing + token-stream construction -----------------------------


@dataclass(frozen=True)
class TraceRecord:
    timestamp_ms: int
    input_length: int
    output_length: int
    hash_ids: list[int]


def iter_trace(path: str) -> Iterator[TraceRecord]:
    """Yield TraceRecord per line, skipping malformed ones."""
    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield TraceRecord(
                    timestamp_ms=int(obj["timestamp"]),
                    input_length=int(obj["input_length"]),
                    output_length=int(obj["output_length"]),
                    hash_ids=list(obj["hash_ids"]),
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"warn: skip line {line_num}: {e}")


def load_trace(path: str, max_requests: int | None = None) -> list[TraceRecord]:
    records = []
    for rec in iter_trace(path):
        records.append(rec)
        if max_requests is not None and len(records) >= max_requests:
            break
    records.sort(key=lambda r: r.timestamp_ms)
    return records


def hash_to_token(hash_id: int, token_offset: int, vocab_range: int) -> int:
    """Map a hash_id to a token id inside [token_offset, token_offset+vocab_range)."""
    return (hash_id % vocab_range) + token_offset


def build_safe_token_pool(
    model: str,
    token_offset: int,
    vocab_size: int,
) -> list[int]:
    """Scan the model's vocab and return token ids that are safe to use as
    forced output tokens in a streaming OpenAI completion.

    Why: dynamo's frontend has an incremental detokenize bug
    (lib/llm/src/tokenizers.rs:292: ``new_text[prefix_text.len()..]`` slices
    on a byte offset without checking ``is_char_boundary``). When a token's
    decoded text ends mid-UTF-8-character, the next chunk's slice lands
    inside that character and panics, which kills the SSE stream and
    surfaces client-side as
    ``ClientPayloadError: Not enough data to satisfy transfer length header``.

    The simplest robust filter: keep only tokens whose decoded text is
    *pure ASCII*. ASCII bytes are always self-delimiting in UTF-8, so any
    byte slice across ASCII-only output is automatically char-aligned.
    """
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    # Skip every special token id (BOS/EOS/PAD/chat-template markers like
    # <|im_start|>, <think>, <tool_response>, ...). They decode to ASCII
    # but feeding them in the middle of a stream confuses templating /
    # stop-condition logic in the engine.
    special_ids = set(tokenizer.all_special_ids or [])
    added_vocab = tokenizer.get_added_vocab() or {}
    special_ids.update(added_vocab.values())

    safe: list[int] = []
    for tid in range(token_offset, vocab_size):
        if tid in special_ids:
            continue
        try:
            text = tokenizer.decode([tid], skip_special_tokens=False)
        except Exception:
            continue
        if not text:
            continue
        # Printable ASCII only: 0x20 (space) .. 0x7e (~). Single-byte UTF-8
        # so no boundary risk, and excludes control chars (NUL, BEL, etc.)
        # that some downstream layers strip or treat specially.
        if all(0x20 <= ord(c) <= 0x7E for c in text):
            safe.append(tid)

    if not safe:
        raise RuntimeError(
            f"No ASCII-decoded tokens found in vocab [{token_offset}, "
            f"{vocab_size}) for model {model}. Cannot build safe token pool."
        )
    return safe


def _resolve_token(
    hash_id: int,
    token_offset: int,
    vocab_range: int,
    safe_pool: list[int] | None,
) -> int:
    """Map a hash to a token id, either via the raw modulo mapping
    (``safe_pool`` is None) or by indexing into a pre-filtered list of
    ASCII-decoded ids (``safe_pool`` is set).

    Using ``safe_pool`` preserves the *deterministic* hash->token mapping —
    same hash always picks the same token — while restricting the codomain
    to tokens that won't trip dynamo's incremental detokenize bug.
    """
    if safe_pool is None:
        return hash_to_token(hash_id, token_offset, vocab_range)
    return safe_pool[hash_id % len(safe_pool)]


def build_request(
    rec: TraceRecord,
    vocab_range: int,
    block_size: int = 512,
    token_offset: int = 100,
    safe_pool: list[int] | None = None,
) -> list[int]:
    """Materialize ``prompt_token_ids`` for one record.

    ``hash_ids`` describe only the prompt — there is no response hash in the
    trace — so we materialize just the prompt and leave the response to the
    model (its length is pinned by the caller via ``output_length``).

    Strategy:
      1. Expand hash_ids into ``len(hash_ids) * block_size`` tokens.
      2. If that's shorter than ``input_length``, extend with the last block's token.
      3. Truncate to the first ``input_length`` tokens.

    When ``safe_pool`` is supplied, hash->token routing goes through
    ``_resolve_token``, restricting outputs to multi-byte-safe tokens (see
    ``build_safe_token_pool``).
    """
    needed = rec.input_length
    full: list[int] = []
    for h in rec.hash_ids:
        full.extend([_resolve_token(h, token_offset, vocab_range, safe_pool)] * block_size)
        if len(full) >= needed:
            break
    if len(full) < needed:
        pad_token = _resolve_token(
            rec.hash_ids[-1], token_offset, vocab_range, safe_pool
        )
        full.extend([pad_token] * (needed - len(full)))
    return full[:needed]


# ---- replay client ---------------------------------------------------------


@dataclass
class RequestResult:
    record_index: int
    timestamp_ms: int
    input_length: int
    output_length: int
    actual_output_length: int
    start_time_ms: float
    ttft_ms: float
    tpot_ms: float
    latency_ms: float  # = e2e
    output_num_chunks: int
    error: str | None


def _make_request_id(idx: int) -> str:
    """Construct a valid UUIDv4 with `idx` encoded as decimal, zero-padded,
    into the leading 8 chars (e.g. row 21 -> "00000021-...").

    Dynamo's frontend requires the request_id to parse as a UUID
    (see dynamo/lib/runtime/src/logging.rs). Digits 0-9 are valid hex, so a
    zero-padded decimal segment is also a valid first UUID group. The
    remaining 24 hex chars come from `uuid.uuid4().hex`, preserving the
    v4 version nibble (hex[12]='4') and RFC4122 variant nibble (hex[16]).
    Capped at 8 decimal digits (99999999).
    """
    rand = uuid.uuid4().hex
    return (
        f"{min(idx, 99999999):08d}-"
        f"{rand[8:12]}-{rand[12:16]}-{rand[16:20]}-{rand[20:32]}"
    )


async def send_one(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt_ids: list[int],
    output_length: int,
    request_id: str,
    timeout_sec: float,
    guided_regex: str | None,
) -> tuple[float, float, int, int, str | None]:
    """Returns (ttft_ms, e2e_ms, actual_out, num_chunks, error).

    Prefix-cache / LMCache hit metrics are NOT collected here: vLLM's
    ``usage.prompt_tokens_details.cached_tokens`` only reports the local
    GPU prefix cache, which is misleading for disagg + LMCache setups.
    Inspect the server-side logs (vLLM's "External prefix cache hit rate"
    and LMCache's per-request "Retrieved X out of Y" lines) for the real
    reuse picture.
    """
    payload: dict = {
        "model": model,
        "prompt": prompt_ids,
        # Pin the response length only — the response is model-generated, not
        # forced from the trace (``hash_ids`` carries no response hash).
        "max_tokens": output_length,
        "min_tokens": output_length,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # ``return_token_ids`` is intentionally NOT set: the first SSE chunk
        # would echo ``prompt_token_ids`` (~tens of KB for long prompts),
        # blowing past aiohttp's default per-line limit of 128 KB.
    }
    if guided_regex is not None:
        # Dynamo's frontend currently has a UTF-8 boundary bug in incremental
        # detokenization. Constraining generated text to printable ASCII avoids
        # multi-byte output chunks that can panic the frontend mid-stream.
        payload["guided_regex"] = guided_regex

    # Dynamo's frontend only honors `request-id` / `x-dynamo-request-id`
    # (and only if the value parses as a UUID); `X-Request-Id` is captured
    # for tracing but never used as the internal request_id. Send both so
    # the value shows up in server-side logs *and* is echoed back to us.
    # See dynamo/lib/runtime/src/logging.rs:TraceParent::from_headers.
    headers = {
        "Content-Type": "application/json",
        "x-dynamo-request-id": request_id,
        "X-Request-Id": request_id,
    }
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    start_ns = time.perf_counter_ns()
    ttft_ns: int | None = None
    actual_out = 0
    num_chunks = 0

    try:
        async with session.post(
            url, json=payload, headers=headers, timeout=timeout
        ) as resp:
            if HTTPStatus(resp.status) != HTTPStatus.OK:
                body = await resp.text()
                return -1.0, -1.0, 0, 0, f"HTTP {resp.status}: {body[:200]}"
            async for raw in resp.content:
                raw = raw.strip()
                if not raw:
                    continue
                chunk = raw.decode("utf-8").removeprefix("data: ")
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices") or []
                if choices and choices[0].get("text"):
                    num_chunks += 1
                    # First chunk with text => prefill done.
                    if ttft_ns is None:
                        ttft_ns = time.perf_counter_ns() - start_ns
                usage = data.get("usage")
                if usage:
                    # Final chunk (include_usage=True). usage.completion_tokens
                    # is the authoritative output token count.
                    actual_out = int(usage.get("completion_tokens") or 0)
    except (TimeoutError, aiohttp.ClientError) as e:
        return -1.0, -1.0, actual_out, num_chunks, f"{type(e).__name__}: {e}"

    e2e_ns = time.perf_counter_ns() - start_ns
    return (
        (ttft_ns or e2e_ns) / 1e6,
        e2e_ns / 1e6,
        actual_out,
        num_chunks,
        None,
    )


async def replay(args: argparse.Namespace) -> None:
    records = load_trace(args.trace, max_requests=args.max_requests)
    print(f"loaded {len(records)} records from {args.trace}")
    if not records:
        return

    completions_url = args.url.rstrip("/") + "/v1/completions"
    vocab_range = args.vocab_size - args.token_offset
    print(
        f"info: url={completions_url} vocab_size={args.vocab_size}"
        f" token_offset={args.token_offset},"
        f" token id range=[{args.token_offset}, {args.vocab_size})"
    )
    output_regex = None if args.disable_output_regex else args.output_regex
    if output_regex is not None:
        print(f"info: constraining generated text with guided_regex={output_regex!r}")

    # The safe-token pool restricts hash->token mapping to tokens whose decoded
    # text is pure ASCII (see build_safe_token_pool docstring). It only matters
    # for tokens that get streamed back through dynamo's buggy incremental
    # detokenizer — i.e. *output* tokens. Since we no longer force outputs from
    # the trace (the response is model-generated; only prompt tokens come from
    # hash_ids, and prompts are never detokenized into the stream), the pool is
    # off by default. Enable it with --safe-token-pool if a prompt-side path
    # ever surfaces the same UTF-8 boundary issue.
    safe_pool: list[int] | None = None
    if args.safe_token_pool:
        safe_pool = build_safe_token_pool(
            args.model, args.token_offset, args.vocab_size
        )
        print(
            f"info: safe-token pool built from {args.model}: "
            f"{len(safe_pool)} ASCII-decoded tokens in "
            f"[{args.token_offset}, {args.vocab_size})"
        )

    sem = asyncio.Semaphore(args.num_clients)
    results: list[RequestResult] = []
    t0_ms = time.perf_counter_ns() // 1_000_000

    async def schedule(idx: int, rec: TraceRecord) -> None:
        # Honor the trace's arrival time, scaled by time_scale (>1 = faster).
        target_ms = rec.timestamp_ms / max(args.time_scale, 1e-9)
        elapsed_ms = time.perf_counter_ns() // 1_000_000 - t0_ms
        if target_ms > elapsed_ms:
            await asyncio.sleep((target_ms - elapsed_ms) / 1000.0)
        prompt_ids = build_request(
            rec,
            vocab_range=vocab_range,
            token_offset=args.token_offset,
            safe_pool=safe_pool,
        )
        # Parsed by analyze_lp_ttft_breakdown.py.
        print(
            f"[req] idx={idx} ts={rec.timestamp_ms}ms"
            f" in={rec.input_length} out={rec.output_length}"
            f" hashes={len(rec.hash_ids)}"
        )
        async with sem:
            start_time_ms = float(time.perf_counter_ns() // 1_000_000 - t0_ms)
            async with aiohttp.ClientSession() as session:
                ttft, e2e, actual, num_chunks, err = await send_one(
                    session,
                    completions_url,
                    args.model,
                    prompt_ids,
                    rec.output_length,
                    request_id=_make_request_id(idx),
                    timeout_sec=args.timeout,
                    guided_regex=output_regex,
                )
        tpot = (
            (e2e - ttft) / max(actual - 1, 1) if actual > 1 and ttft > 0 else 0.0
        )
        results.append(
            RequestResult(
                record_index=idx,
                timestamp_ms=rec.timestamp_ms,
                input_length=rec.input_length,
                output_length=rec.output_length,
                actual_output_length=actual,
                start_time_ms=start_time_ms,
                ttft_ms=ttft,
                tpot_ms=tpot,
                latency_ms=e2e,
                output_num_chunks=num_chunks,
                error=err,
            )
        )
        if err:
            print(
                f"[ERR] idx={idx} ts={rec.timestamp_ms}ms in={rec.input_length}"
                f" out={rec.output_length}/{actual} {err}"
            )
        else:
            print(
                f"[ok ] idx={idx} out={actual}/{rec.output_length}"
                f" ttft={ttft:.1f}ms tpot={tpot:.2f}ms e2e={e2e:.1f}ms"
                f" chunks={num_chunks}"
            )

    await asyncio.gather(*(schedule(i, r) for i, r in enumerate(records)))

    _print_summary(results, args)


SEP = "-" * 100


def _print_summary(results: list[RequestResult], args: argparse.Namespace) -> None:
    import pandas as pd  # local import: only needed at end-of-run

    ok = [r for r in results if r.error is None]
    err_count = sum(1 for r in results if r.error is not None)

    print(SEP)
    print("\033[93mParameters:\033[0m")
    print(f"model={args.model}")
    print(f"num_clients={args.num_clients}")
    print(f"num_requests={len(results)}")
    print(f"trace={args.trace}")
    print(f"time_scale={args.time_scale}")
    print(f"vocab_size={args.vocab_size}")
    print(f"errors={err_count}")
    print(SEP)

    if not ok:
        print("no successful requests")
        return

    rows = [
        {
            "ttft_ms": r.ttft_ms,
            "tpot_ms": r.tpot_ms,
            "latency_ms": r.latency_ms,
            "input_num_tokens": r.input_length,
            "output_num_tokens": r.actual_output_length,
            "output_num_chunks": r.output_num_chunks,
        }
        for r in ok
    ]
    df = pd.DataFrame(rows)

    # Runtime = last request end - first request start.
    starts = [r.start_time_ms for r in ok]
    ends = [r.start_time_ms + r.latency_ms for r in ok]
    runtime_sec = (max(ends) - min(starts)) / 1000.0
    requests_per_sec = len(ok) / runtime_sec if runtime_sec > 0 else 0.0

    print("\033[93mStatistics summary:\033[0m")
    print(f"runtime_sec = {runtime_sec:.3f}")
    print(f"requests_per_sec = {requests_per_sec:.3f}")
    print(SEP)

    pd.set_option("display.precision", 2)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    summary = df.describe(percentiles=[0.5, 0.99]).transpose()
    # Keep only the requested columns in the requested order.
    summary = summary[["count", "mean", "min", "50%", "99%", "max"]]
    print(summary)
    print(SEP)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i", "--trace", required=True, help="path to *.jsonl trace"
    )
    p.add_argument(
        "-m", "--model", required=True, help="model name served by vLLM"
    )
    p.add_argument(
        "-u","--url",default="http://localhost:8000",
        help="Base URL for the LLM API server (no path).",
    )
    p.add_argument(
        "-p", "--num-clients", type=int, default=32,
        help="Number of parallel client coroutines",
    )
    p.add_argument(
        "-n", "--max-requests", type=int, default=None,
        help="Max number of requests to send",
    )
    p.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="speed factor: 1.0=real-time, 10.0=10x faster, 0.1=10x slower",
    )
    p.add_argument("--token-offset", type=int, default=100)
    p.add_argument(
        "--vocab-size",
        type=int,
        default=150000,
        help=(
            "model vocab size (read from HF model config.json's"
            " 'vocab_size'). Hash->token mapping uses vocab_range ="
            " vocab_size - token_offset, so token ids land in"
            " [token_offset, vocab_size)."
        ),
    )
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument(
        "--output-regex",
        default="[ -~]*",
        help=(
            "Regex constraint for generated text. The default permits only "
            "printable ASCII to avoid Dynamo frontend UTF-8 detokenization "
            "panics during streaming."
        ),
    )
    p.add_argument(
        "--disable-output-regex",
        action="store_true",
        help="Disable the default guided_regex output constraint.",
    )
    p.add_argument(
        "--safe-token-pool",
        action="store_true",
        help=(
            "Restrict prompt hash->token mapping to ASCII-decoded tokens "
            "(loads the HF tokenizer). Off by default; only relevant if a "
            "prompt-side path hits dynamo's incremental-detokenize UTF-8 bug."
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(replay(args))


if __name__ == "__main__":
    main()
