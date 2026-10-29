# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover Dynamo workers, optionally warming every prefill/decode NIXL pair."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Sequence
from pathlib import Path

from dynamo.runtime import DistributedRuntime


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for Dynamo workers; optionally warm pinned NIXL pairs."
    )
    parser.add_argument("--namespace", default="dynamo")
    parser.add_argument("--store-backend", default="etcd")
    parser.add_argument("--request-plane", default="tcp")
    parser.add_argument("--expected-prefill", type=int, default=2)
    parser.add_argument("--expected-decode", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--stable-polls", type=int, default=1)
    parser.add_argument("--extra-worker-grace", type=float, default=15.0)
    parser.add_argument("--warm-nixl-pairs", action="store_true")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--log-check-timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    if args.expected_prefill <= 0 or args.expected_decode <= 0:
        parser.error("expected worker counts must be positive")
    if args.stable_polls <= 0:
        parser.error("stable polls must be positive")
    if min(
        args.timeout,
        args.poll_interval,
        args.extra_worker_grace,
        args.request_timeout,
        args.log_check_timeout,
    ) <= 0:
        parser.error("timeouts and intervals must be positive")
    if args.warm_nixl_pairs and (not args.model or not args.log_dir):
        parser.error("--warm-nixl-pairs requires --model and --log-dir")
    return args


async def discover(args: argparse.Namespace) -> dict[str, list[int]]:
    loop = asyncio.get_running_loop()
    runtime: DistributedRuntime | None = None
    last_prefill_ids: list[int] = []
    last_decode_ids: list[int] = []
    try:
        try:
            async with asyncio.timeout(args.timeout):
                runtime = DistributedRuntime(loop, args.store_backend, args.request_plane)
                prefill_client, decode_client = await asyncio.gather(
                    runtime.endpoint(f"{args.namespace}.prefill.generate").client(),
                    runtime.endpoint(f"{args.namespace}.backend.generate").client(),
                )

                stable_candidate: tuple[tuple[int, ...], tuple[int, ...]] | None = None
                stable_count = 0
                extra_workers_since: float | None = None
                while True:
                    last_prefill_ids = sorted(set(prefill_client.instance_ids()))
                    last_decode_ids = sorted(set(decode_client.instance_ids()))
                    now = loop.time()

                    has_extra = (
                        len(last_prefill_ids) > args.expected_prefill
                        or len(last_decode_ids) > args.expected_decode
                    )
                    if has_extra:
                        extra_workers_since = extra_workers_since or now
                        if now - extra_workers_since >= args.extra_worker_grace:
                            raise RuntimeError(
                                "Extra Dynamo workers remained registered: "
                                f"prefill={last_prefill_ids} decode={last_decode_ids}"
                            )
                    else:
                        extra_workers_since = None

                    exact = (
                        len(last_prefill_ids) == args.expected_prefill
                        and len(last_decode_ids) == args.expected_decode
                    )
                    candidate = (tuple(last_prefill_ids), tuple(last_decode_ids))
                    if exact and candidate == stable_candidate:
                        stable_count += 1
                    else:
                        stable_candidate = candidate if exact else None
                        stable_count = 1 if exact else 0
                    if stable_count >= args.stable_polls:
                        return {
                            "prefill_ids": last_prefill_ids,
                            "decode_ids": last_decode_ids,
                        }
                    await asyncio.sleep(args.poll_interval)
        except TimeoutError as exc:
            raise TimeoutError(
                "Timed out discovering a stable exact Dynamo worker set: "
                f"prefill={last_prefill_ids} (expected {args.expected_prefill}), "
                f"decode={last_decode_ids} (expected {args.expected_decode})"
            ) from exc
    finally:
        if runtime is not None:
            runtime.shutdown()


def post_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout: float,
) -> tuple[int, dict]:
    data = json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode())
        except Exception:
            return exc.code, {}
    except Exception as exc:
        return 0, {"error": str(exc)}


def request_logged(log_dir: Path, request_id: str, prefill_id: int, decode_id: int) -> bool:
    prefill_ok = False
    decode_ok = False
    nixl_ok = False
    prefill_token = str(prefill_id)
    decode_token = str(decode_id)

    for path in log_dir.glob("prefill-*.log"):
        with path.open(errors="replace") as file:
            for line in file:
                if request_id in line and "request completed" in line and prefill_token in line:
                    prefill_ok = True
                    break
        if prefill_ok:
            break

    for path in log_dir.glob("decode-*.log"):
        with path.open(errors="replace") as file:
            for line in file:
                if request_id not in line:
                    continue
                decode_ok |= "request completed" in line and decode_token in line
                nixl_ok |= "NIXL recv metadata queued" in line
        if decode_ok and nixl_ok:
            break
    return prefill_ok and decode_ok and nixl_ok


def warm_nixl_pairs(
    args: argparse.Namespace,
    prefill_ids: list[int],
    decode_ids: list[int],
) -> None:
    log_dir = Path(args.log_dir)
    url = args.base_url.rstrip("/") + "/v1/completions"
    total = len(prefill_ids) * len(decode_ids)
    warmed = 0

    print(f"  Prefill IDs: {' '.join(map(str, prefill_ids))}", flush=True)
    print(f"  Decode IDs:  {' '.join(map(str, decode_ids))}", flush=True)
    print(f"Warming all {total} prefill x decode NIXL pairs...", flush=True)

    for prefill_id in prefill_ids:
        for decode_id in decode_ids:
            request_id = str(uuid.uuid4())
            payload = {
                "model": args.model,
                "prompt": [100 + ((prefill_id + decode_id + i) % 1000) for i in range(4096)],
                "max_tokens": 1,
                "min_tokens": 1,
                "ignore_eos": True,
                "stream": False,
                "temperature": 0,
                "guided_regex": "[ -~]*",
                "nvext": {
                    "extra_fields": ["worker_id"],
                    "prefill_worker_id": prefill_id,
                    "decode_worker_id": decode_id,
                    "backend_instance_id": decode_id,
                },
            }
            headers = {
                "x-dynamo-request-id": request_id,
                "x-prefill-instance-id": str(prefill_id),
                "x-worker-instance-id": str(decode_id),
                "x-prefill-dp-rank": "0",
                "x-dp-rank": "0",
            }
            status, response = post_json(url, payload, headers, args.request_timeout)
            valid = (
                status == 200
                and isinstance(response.get("choices"), list)
                and response["choices"]
            )

            nixl_log = False
            if valid:
                log_deadline = time.monotonic() + args.log_check_timeout
                while time.monotonic() < log_deadline:
                    nixl_log = request_logged(log_dir, request_id, prefill_id, decode_id)
                    if nixl_log:
                        break
                    time.sleep(1)
                valid = nixl_log

            if not valid:
                nvext = response.get("nvext") or {} if isinstance(response, dict) else {}
                worker = nvext.get("worker_id") or {}
                raise RuntimeError(
                    "NIXL pair warm failed: "
                    f"prefill={prefill_id} decode={decode_id} http={status} "
                    f"response_prefill={worker.get('prefill_worker_id', 'missing')} "
                    f"response_decode={worker.get('decode_worker_id', 'missing')} "
                    f"nixl_log={nixl_log}"
                )

            warmed += 1
            print(
                f"  [{warmed}/{total}] warmed prefill={prefill_id} "
                f"decode={decode_id} request_id={request_id}",
                flush=True,
            )
    print(f"All {total} prefill x decode NIXL pairs are ready.", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = asyncio.run(discover(args))
        if args.warm_nixl_pairs:
            warm_nixl_pairs(args, result["prefill_ids"], result["decode_ids"])
        else:
            print("prefill\t" + " ".join(str(i) for i in result["prefill_ids"]))
            print("decode\t" + " ".join(str(i) for i in result["decode_ids"]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
