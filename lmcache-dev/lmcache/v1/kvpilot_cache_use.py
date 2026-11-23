"""Per-request control over which tiers a prefill may reuse KV from.

`cache_use_control` on the request is a list drawn from
  1  this worker's HBM prefix cache (vLLM, handled in Request)
  2  Mooncake DRAM on this host
  3  Mooncake DRAM on another host
  4  Mooncake SSD offload on this host, whichever prefill worker wrote it
It only restricts lookup; what gets stored is unchanged. The scheduler forwards
it through request_configs together with how many tokens the HBM prefix cache
already covers. Chunks past that point are the ones loaded, so each must sit in
an allowed tier, and the hit stops at the first one that does not.

The prelookup (a layerwise lookup) itself picks each value's source from
the allowed local tiers (DRAM first, then SSD), so `limit_hit` only applies the
token cap there. `[4]` loads from SSD even when DRAM holds the same value.

`cache_use_max_tokens` caps the reused prefix across all tiers, so a request can
be held to a fixed prefix even after an identical prompt has stored more. As a
list it has one entry per `cache_use_control` tier: consecutive spans from the
start of the prompt, each loaded only from its tier (a chunk takes the span its
first token falls in), capped at their sum. `[2, 4]` with `[8192, 4096]` loads
the first 8K from DRAM and the next 4K from SSD.
"""

from enum import Enum
from itertools import accumulate
from typing import TYPE_CHECKING, Callable, Optional

from lmcache.logging import init_logger

if TYPE_CHECKING:
    from lmcache.v1.cache_engine import LMCacheEngine

logger = init_logger(__name__)

EXTRA_ARGS_KEY = "cache_use_control"
MAX_TOKENS_EXTRA_ARGS_KEY = "cache_use_max_tokens"
CONTROL_KEY = "lmcache.kvpilot.cache_use_control"
MAX_TOKENS_KEY = "lmcache.kvpilot.cache_use_max_tokens"
HBM_TOKENS_KEY = "lmcache.kvpilot.hbm_tokens"
LOCAL_DRAM, REMOTE_DRAM, LOCAL_SSD = 2, 3, 4


class ReplicaLocation(Enum):
    MISSING = 0
    LOCAL = LOCAL_DRAM
    REMOTE = REMOTE_DRAM


def endpoint_host(endpoint: str) -> str:
    return endpoint.rsplit(":", 1)[0]


def max_tokens(request_configs: Optional[dict]) -> Optional[int]:
    """Cap on the reused prefix: the integer, or the sum of per-tier spans."""
    cap = (request_configs or {}).get(MAX_TOKENS_KEY)
    return sum(cap) if isinstance(cap, list) else cap


def local_tiers(
    request_configs: Optional[dict],
) -> tuple[Callable[[int], tuple[bool, bool]], int]:
    """Which local tiers a prelookup may source loaded chunks from.

    Returns (a function from a chunk's start token to (DRAM allowed, SSD
    allowed), tokens the HBM prefix cache already covers). Chunks within the
    HBM part are not loaded, so any tier will do for them. No control on the
    request allows every tier.
    """
    request_configs = request_configs or {}
    allowed = request_configs.get(CONTROL_KEY)
    hbm = request_configs.get(HBM_TOKENS_KEY, 0)
    if allowed is None:
        return (lambda start: (True, True)), hbm
    spans = request_configs.get(MAX_TOKENS_KEY)
    if not isinstance(spans, list):
        both = (LOCAL_DRAM in allowed, LOCAL_SSD in allowed)
        return (lambda start: both), hbm
    ends = list(accumulate(spans))

    def tier_at(start: int) -> tuple[bool, bool]:
        for tier, end in zip(allowed, ends, strict=True):
            if start < end:
                return tier == LOCAL_DRAM, tier == LOCAL_SSD
        return False, False

    return tier_at, hbm


def limit_hit(
    engine: "LMCacheEngine",
    hashes: list[int],
    offsets: list[int],
    hit_tokens: int,
    request_configs: Optional[dict],
    lookup_id: str,
) -> int:
    """Cut a lookup result back to the prefix the request may reuse.

    Args:
        engine: the engine that served the lookup.
        hashes: chunk hashes of the lookup.
        offsets: chunk lengths of the lookup.
        hit_tokens: prefix length found in LMCache, HBM part included.
        request_configs: carries CONTROL_KEY / MAX_TOKENS_KEY and HBM_TOKENS_KEY
            when the request set them; otherwise the hit is returned as is.
        lookup_id: request id, for the log line.

    Returns:
        The reusable prefix length, never more than hit_tokens.
    """
    request_configs = request_configs or {}
    cap = max_tokens(request_configs)
    if cap is not None and hit_tokens > cap:
        # Chunk-aligned, matching the HBM side which caps at whole blocks.
        ends = []
        for _, end, _ in engine.token_database.process_tokens(
            hashes=hashes, offsets=offsets, make_key=False
        ):
            ends.append(end)
        hit_tokens = max([e for e in ends if e <= cap], default=0)
    allowed = request_configs.get(CONTROL_KEY)
    plan = engine.lookup_views.get(lookup_id)
    if plan is not None:
        # The prelookup already sourced every chunk from an allowed tier.
        if allowed is not None or cap is not None:
            _log(
                lookup_id, allowed, cap, request_configs.get(HBM_TOKENS_KEY, -1),
                hit_tokens, hit_tokens, plan.num_dram_chunks, 0, plan.num_ssd_chunks,
            )
        return hit_tokens
    if allowed is None or (LOCAL_DRAM in allowed and REMOTE_DRAM in allowed):
        if cap is not None:
            _log(lookup_id, allowed, cap, -1, hit_tokens, hit_tokens, 0, 0)
        return hit_tokens
    hbm = request_configs[HBM_TOKENS_KEY]
    res = min(hit_tokens, hbm)
    if hit_tokens <= hbm or (LOCAL_DRAM not in allowed and REMOTE_DRAM not in allowed):
        _log(lookup_id, allowed, cap, hbm, hit_tokens, res, 0, 0)
        return res

    chunks = [
        (end, key.get_first_layer() if engine.use_layerwise else key)
        for _, end, key in engine.token_database.process_tokens(
            hashes=hashes, offsets=offsets, request_configs=request_configs
        )
        if hbm < end <= hit_tokens
    ]
    remote_backend = dict(engine.storage_manager.get_active_storage_backends())[
        "RemoteBackend"
    ]
    locations = remote_backend.batched_replica_location([key for _, key in chunks])
    num_local = sum(loc is ReplicaLocation.LOCAL for loc in locations)
    num_remote = sum(loc is ReplicaLocation.REMOTE for loc in locations)
    for (end, _), loc in zip(chunks, locations, strict=True):
        if loc.value not in allowed:
            break
        res = end
    _log(lookup_id, allowed, cap, hbm, hit_tokens, res, num_local, num_remote)
    return res


def _log(
    lookup_id: str,
    allowed: Optional[list[int]],
    cap: Optional[int],
    hbm: int,
    hit: int,
    res: int,
    num_local: int,
    num_remote: int,
    num_ssd: int = 0,
) -> None:
    logger.info(
        "KVPilot cache_use req=%s allow=%s max=%s hbm=%d hit=%d reusable=%d "
        "local_chunks=%d remote_chunks=%d ssd_chunks=%d",
        lookup_id,
        allowed,
        cap,
        hbm,
        hit,
        res,
        num_local,
        num_remote,
        num_ssd,
    )
