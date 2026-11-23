# SPDX-License-Identifier: Apache-2.0
"""KVPilot placement (KVPILOT_PLACEMENT, see vllm/v1/core/kvpilot_placement.py)
as LMCache sees it, and chunk-group placement on the Mooncake master.

A group is the chunk-hash field of the Mooncake key, so every layer and TP rank
of a chunk shares it. Its tier is where the chunk should live (exclusive), or
the class it evicts in (inclusive-safe); its penalty is the exposed load its
move to the next tier down adds (the penalty strategies). The scheduler process sets groups over
the lookup channel, which rank 0's lookup server forwards to the master; the
exclusive writer sets them itself before writing.
"""

# Standard
from typing import TYPE_CHECKING, Optional
import os

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine

LRU = ""
INCLUSIVE = "inclusive-safe-reqwise"
# Inclusive, but HBM and DRAM evict by penalty instead of tier class.
INCLUSIVE_PENALTY = "inclusive-penalty-reqwise"
# Copies as LRU; HBM and DRAM evict the farthest next use (penalty order).
BELADY = "belady"
EXCLUSIVE = "exclusive-penalty-reqwise"
# Exclusive, but the prefix an admission loads borrows held HBM blocks instead
# of waiting for their write to DRAM (kvpilot_exclusive.Evictor).
EXCLUSIVE_UNBLOCKING = "exclusive-penalty-reqwise-unblocking"
# Unblocking, with each block's tier and penalty from the time model.
EXCLUSIVE_BLOCKWISE = "exclusive-penalty-blockwise-unblocking"

# Group tier codes, as the Mooncake master's KvpTier.
TIER_CODE = {None: 0, "SSD": 1, "DRAM": 2, "HBM": 3}
# The master's kKvpOrderCode: the "penalty" carries the releasing request's
# sequence number times ORDER_POS_SPAN plus the chunk's index in its prompt.
ORDER_CODE = 4
ORDER_POS_SPAN = 1 << 16

# Sent in the request_configs slot of a lookup message; never valid JSON, so it
# cannot be mistaken for a real lookup's configs.
HINT_MARKER = "__kvpilot_group_placement__"


def strategy() -> str:
    """The configured strategy; an unknown value is a configuration error."""
    name = os.environ.get("KVPILOT_PLACEMENT", "")
    if name not in (LRU, BELADY, INCLUSIVE, INCLUSIVE_PENALTY, EXCLUSIVE,
                    EXCLUSIVE_UNBLOCKING, EXCLUSIVE_BLOCKWISE):
        raise ValueError(f"KVPILOT_PLACEMENT={name!r} is not a KVPilot strategy")
    return name


def exclusive() -> bool:
    """Whether the strategy keeps one copy per block (any exclusive one)."""
    return strategy() in (EXCLUSIVE, EXCLUSIVE_UNBLOCKING, EXCLUSIVE_BLOCKWISE)


def chunk_hash_hex(chunk_hash: int | bytes) -> str:
    """The chunk-hash field exactly as CacheEngineKey.to_string renders it.

    Args:
        chunk_hash: A chunk hash from the token database.

    Returns:
        Its rendering in Mooncake key strings.
    """
    if isinstance(chunk_hash, bytes):
        return chunk_hash.hex()
    return f"{chunk_hash:x}"


def set_groups(
    backend,
    chunk_hashes: list,
    tiers: list[Optional[str]],
    penalties: list[float],
) -> int:
    """Set chunk groups' tier and penalty on the Mooncake master.

    Args:
        backend: The RemoteBackend.
        chunk_hashes: One per chunk.
        tiers: Tier per chunk (None = no session needs it).
        penalties: Penalty (ms) per chunk.

    Returns:
        Groups the master updated.
    """
    return int(
        backend.kvpilot_set_group_placement(
            [chunk_hash_hex(h) for h in chunk_hashes],
            [TIER_CODE[t] for t in tiers],
            [float(p) for p in penalties],
        )
    )


def set_order(backend, chunk_hashes: list, seq: int) -> int:
    """Tell the Mooncake master which request last released these chunks and
    where each sits in its prompt (index), so DRAM evicts it from its tail.

    Args:
        backend: The RemoteBackend.
        chunk_hashes: The request's full prompt chunks, head first.
        seq: The release's sequence number (later releases are larger).

    Returns:
        Groups the master updated.
    """
    if len(chunk_hashes) >= ORDER_POS_SPAN:
        raise ValueError(f"KVPilot: {len(chunk_hashes)} chunks past the order span")
    return int(
        backend.kvpilot_set_group_placement(
            [chunk_hash_hex(h) for h in chunk_hashes],
            [ORDER_CODE] * len(chunk_hashes),
            [float(seq * ORDER_POS_SPAN + i) for i in range(len(chunk_hashes))],
        )
    )


def apply_hint(
    engine: "LMCacheEngine",
    chunk_hashes: list,
    tiers: list[Optional[str]],
    penalties: list[float],
) -> int:
    """The lookup server's side of a group placement (rank 0 sets it).

    Args:
        engine: The lookup server's engine.
        chunk_hashes: One per chunk.
        tiers: Tier per chunk.
        penalties: Penalty (ms) per chunk.

    Returns:
        Groups the master updated; 0 on other ranks.
    """
    if engine.metadata.worker_id != 0:
        return 0
    backends = dict(engine.storage_manager.get_active_storage_backends())
    return set_groups(backends["RemoteBackend"], chunk_hashes, tiers, penalties)
