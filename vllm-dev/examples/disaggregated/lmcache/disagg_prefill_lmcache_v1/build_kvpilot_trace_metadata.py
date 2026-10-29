#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--breakdown-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-exposed-pct", type=float, default=5.0)
    parser.add_argument(
        "--force-unsafe-block-indices", type=int, nargs="*", default=[0]
    )
    args = parser.parse_args()
    force_unsafe_blocks = set(args.force_unsafe_block_indices)

    records = [
        json.loads(line)
        for line in args.trace.read_text().splitlines()
        if line.strip()
    ]
    metrics = {
        int(row["idx"]): row
        for row in csv.DictReader(args.breakdown_csv.open())
    }
    if set(metrics) != set(range(len(records))):
        raise ValueError("breakdown CSV does not cover every trace request")

    dram_safe = {
        idx: float(row["kv_load_exposed_pct"]) < args.safe_exposed_pct
        for idx, row in metrics.items()
    }

    # Parent is the latest prior request with the longest matching stable
    # prefix. A request's final trace hash is excluded because it may represent
    # a partial block and is not stable across turns.
    prior_by_prefix: dict[tuple[int, ...], list[int]] = defaultdict(list)
    parents: dict[int, int] = {}
    for idx, record in enumerate(records):
        hashes = tuple(record["hash_ids"])
        for prefix_len in range(len(hashes), 1, -1):
            candidates = prior_by_prefix.get(hashes[:prefix_len])
            if candidates:
                parents[idx] = candidates[-1]
                break
        stable_prefix = hashes[:-1]
        if len(stable_prefix) >= 2:
            prior_by_prefix[stable_prefix].append(idx)

    # Compute the next consumer independently for every stable trace block.
    # This handles the few branched sessions in the trace: inherited blocks
    # may have a different next consumer from newly appended blocks.
    next_by_prefix: dict[tuple[int, ...], int] = {}
    block_next_indices: dict[int, list[int]] = {}
    block_next_safe: dict[int, list[bool]] = {}
    for idx in range(len(records) - 1, -1, -1):
        stable_hashes = records[idx]["hash_ids"][:-1]
        next_indices = []
        next_safe = []
        for block_idx in range(len(stable_hashes)):
            prefix = tuple(stable_hashes[: block_idx + 1])
            next_idx = next_by_prefix.get(prefix, -1)
            next_indices.append(next_idx)
            next_safe.append(
                block_idx not in force_unsafe_blocks
                and (next_idx < 0 or dram_safe[next_idx])
            )
            next_by_prefix[prefix] = idx
        block_next_indices[idx] = next_indices
        block_next_safe[idx] = next_safe

    for idx, record in enumerate(records):
        next_indices = block_next_indices[idx]
        next_safe = block_next_safe[idx]
        record["kvpilot"] = {
            "kvpilot_parent_request_idx": parents.get(idx, -1),
            "kvpilot_dram_safe": dram_safe[idx],
            "kvpilot_next_request_idx_by_block": next_indices,
            "kvpilot_next_dram_safe_by_block": next_safe,
        }

    with args.output.open("w") as out:
        for record in records:
            out.write(json.dumps(record, separators=(",", ":")) + "\n")

    safe_count = sum(dram_safe.values())
    edge_count = sum(parent >= 0 for parent in parents.values())
    print(
        f"wrote {args.output}: requests={len(records)} "
        f"dram_safe={safe_count} dram_unsafe={len(records) - safe_count} "
        f"parent_edges={edge_count}"
    )


if __name__ == "__main__":
    main()
