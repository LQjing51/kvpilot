# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, no_type_check
import asyncio
import ctypes
import hashlib
import json
import os
import socket
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.kvpilot_cache_use import ReplicaLocation, endpoint_host
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.system_detection import NUMADetector

logger = init_logger(__name__)

# TODO(baoloongmao): Remove these in the future
# Legacy positional-arg key order for old mooncake setup API
_LEGACY_SETUP_KEYS = [
    "local_hostname",
    "metadata_server",
    "global_segment_size",
    "local_buffer_size",
    "protocol",
    "device_name",
    "master_server_address",
]


# Legacy keys that must be passed as ``int`` to the old
# positional-arg ``store.setup()`` API (C++/pybind11).
_LEGACY_INT_KEYS = {"global_segment_size", "local_buffer_size"}

# Keys whose values may contain credentials and must be
# redacted when the setup dict is logged.
_SENSITIVE_SETUP_KEYS = {"metadata_server", "master_server_address"}


def _sanitize_setup_config(
    setup_config: Dict[str, str],
) -> Dict[str, str]:
    """Return a copy of *setup_config* with sensitive values
    redacted, safe for logging."""
    sanitized: Dict[str, str] = {}
    for k, v in setup_config.items():
        lower_k = k.lower()
        if (
            k in _SENSITIVE_SETUP_KEYS
            or "password" in lower_k
            or "token" in lower_k
            or "secret" in lower_k
            or "key" in lower_k
        ):
            sanitized[k] = "***REDACTED***"
        else:
            sanitized[k] = v
    return sanitized


# Mooncake ``ErrorCode`` names, from mooncake-store/include/types.h.
# Mooncake does not export the enum to Python, so the codes that
# ``setup_internal()`` can return are mirrored here to make the failure
# readable. INVALID_PARAMS and INTERNAL_ERROR are returned directly;
# the others reach the caller via the propagated errors of
# RegisterLocalMemory() and MountSegment().
_MOONCAKE_ERROR_NAMES: Dict[int, str] = {
    -1: "INTERNAL_ERROR",
    -10: "BUFFER_OVERFLOW",
    -101: "SEGMENT_NOT_FOUND",
    -102: "SEGMENT_ALREADY_EXISTS",
    -103: "CLIENT_NOT_FOUND",
    -200: "NO_AVAILABLE_HANDLE",
    -300: "INVALID_VERSION",
    -600: "INVALID_PARAMS",
}


def setup_mooncake_store(
    store: Any,
    config: "MooncakeStoreConfig",
) -> None:
    """Initialize a MooncakeDistributedStore instance.

    Calls ``store.setup()`` using the dict-based API introduced
    in Mooncake PR #1445 when available, and transparently falls
    back to the legacy positional-arg API otherwise.  This lets
    LMCache work with both new and old mooncake builds without
    requiring callers to know which API is present.

    Args:
        store: A MooncakeDistributedStore instance (already
            constructed but not yet setup).  Must expose a
            ``setup()`` method.
        config: A :class:`MooncakeStoreConfig` carrying the
            setup parameters in ``config.setup_config``.

    Returns:
        None.  ``store`` is mutated in place.

    Raises:
        RuntimeError: If ``store.setup()`` returns a non-zero mooncake
            ``ErrorCode``, meaning the store was not initialized.
        Exception: Any exception raised by ``store.setup()``
            other than :class:`TypeError` is propagated to the
            caller (``TypeError`` is caught and used as the
            signal to fall back to the legacy API).
    """
    setup_dict = config.setup_config
    try:
        # New API (Mooncake PR #1445): setup(config: dict)
        rc = store.setup(setup_dict)
        logger.info("Using dict-based setup API (new)")
    except TypeError:
        # Legacy API: setup with positional arguments.
        # Some keys (e.g. ``global_segment_size``) must be int.
        logger.info(
            "Dict-based setup not available, "
            "falling back to positional-arg API (legacy)"
        )
        args: List[Any] = []
        for k in _LEGACY_SETUP_KEYS:
            v: Any = setup_dict.get(k, "")
            if k in _LEGACY_INT_KEYS:
                try:
                    v = int(v) if v != "" else 0
                except (TypeError, ValueError):
                    v = 0
            args.append(v)
        rc = store.setup(*args)

    # ``setup()`` reports failure by return code, not by raising: it
    # returns 0 on success and a negative mooncake ``ErrorCode``
    # otherwise. Leaving it unchecked keeps ``RealClient::client_``
    # null while LMCache proceeds as if the store were connected,
    # which only surfaces later as "Client is not initialized".
    if isinstance(rc, int) and rc != 0:
        name = _MOONCAKE_ERROR_NAMES.get(rc)
        raise RuntimeError(
            "Mooncake store.setup() failed: "
            f"code={rc}" + (f" ({name})" if name else "")
        )


# Prefix for keys that should be forwarded to mooncake setup.
# e.g. ``mooncake_local_hostname`` -> ``local_hostname``
_MOONCAKE_PREFIX = "mooncake_"

# Keys that are consumed by LMCache only (never sent to mooncake).
_LMCACHE_ONLY_KEYS = {
    "transfer_timeout",
    "storage_root_dir",
    "prefer_local_alloc",
    "mooncake_transfer_timeout",
    "mooncake_storage_root_dir",
    "mooncake_prefer_local_alloc",
}

# Legacy keys that are forwarded without prefix (for compat).
_LEGACY_PASSTHROUGH_KEYS = {
    "local_hostname",
    "metadata_server",
    "global_segment_size",
    "local_buffer_size",
    "protocol",
    "device_name",
    "master_server_address",
}

# Default values for mooncake setup keys
_SETUP_DEFAULTS: Dict[str, str] = {
    "global_segment_size": "3355443200",
    "local_buffer_size": "1073741824",
    "protocol": "tcp",
    "device_name": "",
}

MOONCAKE_PRELOAD_EXPECTED_ENGINES_ENV = "MOONCAKE_PRELOAD_EXPECTED_ENGINES"
MOONCAKE_PRELOAD_DIR = "/tmp/mooncake-shm-preload"
MOONCAKE_PRELOAD_TIMEOUT_SEC = 600


def _int_value(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class MooncakeStoreConfig:
    """Configuration for MooncakeDistributedStore.

    Mooncake setup keys are stored in ``setup_config`` dict
    and passed through to ``store.setup()`` as-is.  This
    means LMCache does **not** need to change when mooncake
    adds or removes setup keys.

    LMCache-only knobs (``transfer_timeout``,
    ``storage_root_dir``, ``prefer_local_alloc``) are kept
    as explicit attributes.
    """

    def __init__(
        self,
        setup_config: Dict[str, str],
        transfer_timeout: int = 1,
        storage_root_dir: str = "",
        prefer_local_alloc: bool = False,
    ):
        self.setup_config = dict(setup_config)
        self.transfer_timeout = transfer_timeout
        self.storage_root_dir = storage_root_dir
        self.prefer_local_alloc = prefer_local_alloc

    def __repr__(self) -> str:
        # Redact sensitive values to keep logs safe.
        return (
            "MooncakeStoreConfig("
            "setup_config=%r, "
            "transfer_timeout=%r, "
            "storage_root_dir=%r, "
            "prefer_local_alloc=%r)"
            % (
                _sanitize_setup_config(self.setup_config),
                self.transfer_timeout,
                self.storage_root_dir,
                self.prefer_local_alloc,
            )
        )

    # ----------------------------------------------------------
    # Deprecated attribute-style accessors for backward compat
    # ----------------------------------------------------------
    def _get_setup_key(self, key: str) -> str:
        return self.setup_config.get(key, "")

    def _set_setup_key(self, key: str, value: str) -> None:
        self.setup_config[key] = value

    @property
    def local_hostname(self) -> str:  # deprecated
        return self._get_setup_key("local_hostname")

    @property
    def metadata_server(self) -> str:  # deprecated
        return self._get_setup_key("metadata_server")

    @property
    def global_segment_size(self) -> str:  # deprecated
        return self._get_setup_key("global_segment_size")

    @property
    def local_buffer_size(self) -> str:  # deprecated
        return self._get_setup_key("local_buffer_size")

    @property
    def protocol(self) -> str:  # deprecated
        return self._get_setup_key("protocol")

    @property
    def device_name(self) -> str:  # deprecated
        return self._get_setup_key("device_name")

    @device_name.setter
    def device_name(self, value: str) -> None:  # deprecated
        self._set_setup_key("device_name", value)

    @property
    def master_server_address(self) -> str:  # deprecated
        return self._get_setup_key("master_server_address")

    @master_server_address.setter
    def master_server_address(  # deprecated
        self, value: str
    ) -> None:
        self._set_setup_key("master_server_address", value)

    # ----------------------------------------------------------
    # Factory methods
    # ----------------------------------------------------------
    @staticmethod
    def _build_setup_dict(
        raw: Dict[str, Any],
    ) -> Dict[str, str]:
        """Extract mooncake setup keys from *raw* config dict.

        Rules (evaluated in order):
        1. Keys in ``_LMCACHE_ONLY_KEYS`` are skipped.
        2. Keys starting with ``mooncake_`` have the prefix
           stripped and are forwarded (highest priority).
        3. Keys in ``_LEGACY_PASSTHROUGH_KEYS`` are forwarded
           as-is (backward compat, lower priority).
        4. All other keys are ignored.

        Missing keys with known defaults are filled in.
        """
        setup: Dict[str, str] = {}
        # Prefixed keys have higher priority than legacy keys;
        # collect them separately and apply last.
        prefixed_setup: Dict[str, str] = {}
        prefix_len = len(_MOONCAKE_PREFIX)

        for k, v in raw.items():
            if k in _LMCACHE_ONLY_KEYS or v is None:
                continue
            if k.startswith(_MOONCAKE_PREFIX):
                stripped = k[prefix_len:]
                if stripped:
                    prefixed_setup[stripped] = str(v)
            elif k in _LEGACY_PASSTHROUGH_KEYS:
                setup[k] = str(v)

        # Prefixed keys override legacy keys on conflict.
        setup.update(prefixed_setup)

        # Apply defaults for keys not provided
        for k, default_v in _SETUP_DEFAULTS.items():
            setup.setdefault(k, default_v)
        return setup

    @staticmethod
    def from_file(
        file_path: str,
    ) -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            raw = json.load(fin)
        return MooncakeStoreConfig(
            setup_config=MooncakeStoreConfig._build_setup_dict(raw),
            transfer_timeout=raw.get("transfer_timeout", 1),
            storage_root_dir=raw.get("storage_root_dir", ""),
            prefer_local_alloc=raw.get("mooncake_prefer_local_alloc", False),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """Load config from MOONCAKE_CONFIG_PATH env var."""
        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeStoreConfig.from_file(config_file_path)

    @staticmethod
    def load_from_lmcache_config(
        config: "LMCacheEngineConfig",
    ) -> "MooncakeStoreConfig":
        """Load config from LMCacheEngineConfig.extra_config."""
        extra_config = config.extra_config
        if extra_config is None:
            raise ValueError("The extra config is not set.")
        return MooncakeStoreConfig(
            setup_config=MooncakeStoreConfig._build_setup_dict(extra_config),
            transfer_timeout=extra_config.get("transfer_timeout", 1),
            storage_root_dir=extra_config.get("storage_root_dir", ""),
            prefer_local_alloc=extra_config.get("mooncake_prefer_local_alloc", False),
        )


class MooncakestoreConnector(RemoteConnector):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        lmcache_config: Optional[LMCacheEngineConfig],
        plugin_name: Optional[str] = None,
    ):
        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        try:
            # Third Party
            from mooncake.store import (
                MooncakeDistributedStore,
                ReplicaStatus,
                ReplicateConfig,
            )
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e

        self.lmcache_config = lmcache_config

        try:
            self.store = MooncakeDistributedStore()
            config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
            if config_file_path is not None:
                self.config = MooncakeStoreConfig.from_file(config_file_path)
            elif lmcache_config is not None:
                self.config = MooncakeStoreConfig.load_from_lmcache_config(
                    lmcache_config
                )
            else:
                raise ValueError("MOONCAKE_CONFIG_PATH/lmcache_config must be provided")

            logger.info("Mooncake Configuration loaded. config: %s", self.config)
            self._configure_mooncake_feature_flags()
            self._configure_ssd_offload()

            # Check if storage_root_dir exists and set env var
            if self.config.storage_root_dir:
                os.environ["MOONCAKE_STORAGE_ROOT_DIR"] = self.config.storage_root_dir
                logger.info(
                    "Set MOONCAKE_STORAGE_ROOT_DIR to: %s",
                    self.config.storage_root_dir,
                )

            logger.info(
                "Setting up Mooncake store with setup_config: %s",
                _sanitize_setup_config(self.config.setup_config),
            )

            try:
                numa_mapping = getattr(
                    local_cpu_backend.memory_allocator, "numa_mapping", None
                )
                if numa_mapping is None and lmcache_config is not None:
                    numa_mapping = NUMADetector.get_numa_mapping(lmcache_config)

                if numa_mapping:
                    current_device_id = torch_dev.current_device()
                    gpu_to_numa = getattr(numa_mapping, "gpu_to_numa_mapping", {})
                    numa_id = gpu_to_numa.get(current_device_id)
                    logger.info(
                        f"NUMA mapping detected (pre-Mooncake setup): {gpu_to_numa}"
                    )
                    try:
                        # Third Party
                        from mooncake.store import bind_to_numa_node

                        if numa_id is not None:
                            bind_to_numa_node(numa_id)
                            logger.info(
                                f"GPU {current_device_id}, "
                                f"NUMA node {numa_id} binding done"
                            )
                        else:
                            logger.info(
                                f"NUMA mapping not found for GPU {current_device_id}"
                            )
                    except ImportError:
                        logger.warning(
                            "unable to import bind_to_numa_node from mooncake.store"
                        )
                else:
                    logger.info("NUMA mapping unavailable or disabled")
            except Exception as e:
                logger.warning(
                    f"Failed to determine NUMA mapping before Mooncake setup: {e}"
                )

            setup_mooncake_store(self.store, self.config)
            logger.info("Mooncake store setup completed successfully")

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.metadata = local_cpu_backend.metadata
        self.registered_buffer_ptr = None

        # Initialize ReplicateConfig
        self.replica_config = ReplicateConfig()
        self.replica_config.replica_num = 1
        self.complete_replica_status = ReplicaStatus.COMPLETE

        # Set preferred_segment based on configuration
        if self.config.prefer_local_alloc:
            self.replica_config.preferred_segment = self.store.get_hostname()

        # Register CPU buffer for zero-copy operations
        self._register_cpu_buffer()
        self._preload_shared_memory_peers()

        logger.info("MooncakeConnector initialized successfully.")

    def _register_cpu_buffer(self):
        """Register CPU buffer for zero-copy operations."""
        try:
            allocator = self.local_cpu_backend.memory_allocator
            if hasattr(allocator, "pin_allocator") and hasattr(
                allocator.pin_allocator, "buffer"
            ):
                buffer = allocator.pin_allocator.buffer
                self.registered_buffer_ptr = buffer.data_ptr()
                result = self.store.register_buffer(buffer.data_ptr(), buffer.numel())
                if result == 0:
                    logger.info(
                        f"Registered: {hex(buffer.data_ptr())}, {buffer.numel()} bytes"
                    )
                else:
                    logger.warning(f"Buffer registration failed: error={result}")
                    self.registered_buffer_ptr = None
            else:
                self.registered_buffer_ptr = None
        except Exception as e:
            logger.error(f"Buffer registration error: {e}")
            self.registered_buffer_ptr = None


    def _configure_mooncake_feature_flags(self) -> None:
        # KVPilot always runs the zero-copy path: values stored raw (no chunk
        # meta), read in place from this process's segment, and segments in
        # shared memory so same-host peers' replicas are readable too.
        self.enable_mooncake_cpu_backend = True
        self.enable_local_mooncake_hbm = True
        self.enable_shm_mooncake_hbm = True

    def _configure_ssd_offload(self) -> None:
        # KVPilot: KVPILOT_SSD_OFFLOAD_PATH turns on Mooncake's SSD offload in
        # a directory of its own per process. Bucket files are named by
        # millisecond timestamp + sequence, so two clients in one directory can
        # overwrite each other, and a directory left by an earlier run would be
        # scanned back in at startup.
        self.ssd_offload_path = ""
        root = os.getenv("KVPILOT_SSD_OFFLOAD_PATH", "")
        if not root:
            return
        path = os.path.join(root, f"pid{os.getpid()}")
        os.makedirs(path, exist_ok=True)
        self.config.setup_config["enable_ssd_offload"] = "true"
        self.config.setup_config["ssd_offload_path"] = path
        self.ssd_offload_path = path
        logger.info("KVPilot: Mooncake SSD offload enabled at %s", path)

    def _preload_extra_config(self) -> dict:
        if self.lmcache_config is None or self.lmcache_config.extra_config is None:
            return {}
        return self.lmcache_config.extra_config

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _preload_record(self) -> dict:
        extra = self._preload_extra_config()
        metadata = self.metadata
        rank = _int_value(extra.get("mooncake_preload_rank"), metadata.worker_id)
        tp_size = max(1, _int_value(extra.get("mooncake_preload_tp_size"), 1))
        pp_size = max(1, _int_value(extra.get("mooncake_preload_pp_size"), 1))
        return {
            "endpoint": self.store.get_hostname(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "model": str(metadata.model_name),
            "cache_signature": "|".join(
                [
                    str(metadata.model_name),
                    str(metadata.kv_dtype),
                    str(metadata.kv_shape),
                    str(metadata.chunk_size),
                    str(self.meta_fmt),
                    str(tp_size),
                    str(pp_size),
                ]
            ),
            "rank": rank,
            "tp_rank": _int_value(extra.get("mooncake_preload_tp_rank"), rank % tp_size),
            "pp_rank": _int_value(
                extra.get("mooncake_preload_pp_rank"), (rank // tp_size) % pp_size
            ),
            "tp_size": tp_size,
            "pp_size": pp_size,
            "engine_id": str(extra.get("mooncake_preload_engine_id", "")),
            "time": time.time(),
        }

    def _preload_rendezvous_dir(self, record: dict) -> Path:
        key = "|".join(
            [
                str(self.config.master_server_address),
                record["host"],
                record["cache_signature"],
            ]
        )
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return Path(MOONCAKE_PRELOAD_DIR) / digest

    def _publish_preload_record(self, directory: Path, record: dict) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        endpoint_hash = hashlib.sha256(record["endpoint"].encode()).hexdigest()[:8]
        path = directory / f"{record['pid']}-{endpoint_hash}.json"
        tmp_path = directory / f".{path.name}.tmp"
        tmp_path.write_text(json.dumps(record), encoding="utf-8")
        os.replace(tmp_path, path)
        return path

    def _load_preload_records(self, directory: Path, local_record: dict) -> list[dict]:
        records: list[dict] = []
        for path in directory.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                pid = int(record.get("pid", -1))
                if not self._pid_alive(pid):
                    path.unlink(missing_ok=True)
                    continue
                if record.get("host") != local_record["host"]:
                    continue
                if record.get("cache_signature") != local_record["cache_signature"]:
                    continue
                records.append(record)
            except Exception:
                logger.warning("Ignoring invalid Mooncake preload record: %s", path)
        return records

    def _preload_shared_memory_peers(self) -> None:
        extra = self._preload_extra_config()
        if not self.enable_shm_mooncake_hbm:
            return
        if not hasattr(self.store, "preload_shared_memory_segments"):
            raise RuntimeError("Mooncake preload_shared_memory_segments is unavailable")

        record = self._preload_record()
        expected_engines = max(
            1,
            _int_value(
                extra.get(
                    "mooncake_preload_expected_engines",
                    os.getenv(MOONCAKE_PRELOAD_EXPECTED_ENGINES_ENV),
                ),
                1,
            ),
        )
        expected_workers = expected_engines * record["tp_size"] * record["pp_size"]
        timeout_s = max(
            1,
            _int_value(
                extra.get("mooncake_preload_timeout_sec"),
                MOONCAKE_PRELOAD_TIMEOUT_SEC,
            ),
        )
        directory = self._preload_rendezvous_dir(record)
        path = self._publish_preload_record(directory, record)
        logger.info(
            "shared_memory_preload_wait endpoint=%s expected_workers=%d dir=%s",
            record["endpoint"],
            expected_workers,
            directory,
        )

        deadline = time.monotonic() + timeout_s
        records: list[dict] = []
        while True:
            records = self._load_preload_records(directory, record)
            endpoints = {item.get("endpoint") for item in records}
            if len(endpoints) >= expected_workers:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Mooncake shared memory preload timed out: "
                    f"endpoint={record['endpoint']} ready={len(endpoints)} "
                    f"expected={expected_workers} dir={directory}"
                )
            time.sleep(0.2)

        # Every other worker on this host, whatever its TP / PP rank: once a
        # worker's own segment is full, Mooncake places its keys in any segment
        # with room, another rank's included, and an unmapped one fails the
        # whole lookup.
        peer_endpoints = sorted(
            item["endpoint"]
            for item in records
            if item.get("endpoint") != record["endpoint"]
        )
        mapped = self.store.preload_shared_memory_segments(peer_endpoints)
        if mapped < 0:
            raise RuntimeError(
                "Mooncake shared memory preload failed: "
                f"endpoint={record['endpoint']} peers={peer_endpoints}"
            )
        logger.info(
            "shared_memory_preload_ready endpoint=%s peers=%d mapped_segments=%d record=%s",
            record["endpoint"],
            len(peer_endpoints),
            mapped,
            path,
        )

    def _unregister_cpu_buffer(self):
        """Unregister CPU buffer."""
        if self.registered_buffer_ptr is not None:
            result = self.store.unregister_buffer(self.registered_buffer_ptr)
            if result == 0:
                logger.info(f"Unregistered buffer: {hex(self.registered_buffer_ptr)}")
            else:
                logger.warning(f"Buffer unregistration failed: error={result}")
            self.registered_buffer_ptr = None

    def support_batched_get(self) -> bool:
        """
        Check if the connector supports batched get

        Returns:
            True if batched get is supported, False otherwise
        """
        return True

    async def exists(self, key: CacheEngineKey) -> bool:
        return self.store.is_exist(key.to_string())

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return self.store.is_exist(key.to_string())

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """
        Batch get operation - the only supported get method.
        Uses batch_get_into (with metadata) or batch_get_buffer (without metadata).
        """
        if not keys:
            return []

        # Check if we have metadata for zero-copy operations
        if self.save_chunk_meta:
            # Use legacy mode with metadata stored in remote
            return await self._batch_get_buffer(keys)
        else:
            # Use optimized mode with local metadata
            return await self._batch_get_into(keys)

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        """Mooncake only supports batched get (single get raises
        NotImplementedError). The base class default implementation fetches
        keys one-by-one via self.get(), which is unusable here and breaks the
        layerwise retrieve path. Reuse batched_get and return the consecutive
        valid prefix, releasing any objects after the first miss to match the
        base-class semantics."""
        if not keys:
            return []
        results = await self.batched_get(keys)

        memory_objs: List[MemoryObj] = []
        found_failure = False
        for result in results:
            if found_failure:
                # Release subsequent memory objects to avoid memory leak
                if isinstance(result, MemoryObj):
                    result.ref_count_down()
            elif isinstance(result, MemoryObj):
                memory_objs.append(result)
            else:
                # First failure encountered (None)
                found_failure = True
        return memory_objs

    def support_batched_async_contains(self) -> bool:
        return True

    def support_batched_contains(self) -> bool:
        return True

    def batched_contains(self, keys: List[CacheEngineKey]) -> int:
        """Prefix of keys present, in one master RPC (1 = found)."""
        if not keys:
            return 0
        rets = self.store.batch_is_exist([key.to_string() for key in keys])
        hit = 0
        for ret in rets:
            if ret != 1:
                break
            hit += 1
        return hit

    def batched_local_raw_views(
        self, keys: List[CacheEngineKey]
    ) -> Optional[List[MemoryObj]]:
        """Zero-copy views of the keys' host replicas, in one master RPC.

        The same query a layerwise retrieve would issue per layer, done once for
        every layer up front (the prelookup). The query pins each key's memory
        replica (refcnt); call
        batch_unpin_memory_replicas when the views are done.

        Returns:
            One view per key, or None if any key has no usable local replica.
        """
        return self._try_batch_get_local_raw_views(keys)

    def batch_unpin_memory_replicas(self, keys: List[str]) -> int:
        """Drop GetReplicaList refcnt pins for keys (priority eviction)."""
        fn = getattr(self.store, "batch_unpin_memory_replicas", None)
        if fn is None or not keys:
            return 0
        return int(fn(keys))

    def batched_local_sources(
        self, keys: List[CacheEngineKey]
    ) -> Tuple[List[int], List[Tuple[str, int, int]]]:
        """Where each key can be loaded from on this host, in one master RPC.

        Same query as `batched_local_raw_views`; under priority eviction it
        also pins memory replicas (refcnt) until batch_unpin_memory_replicas.

        Returns:
            (ptrs, ssd, key_strs): ptrs[i] is the host address of key i's DRAM
            replica in this process, 0 if none; ssd[i] is (bucket file, value
            offset, value size) of a complete SSD replica written by any client
            on this host -- the files sit on the same disks, so another prefill
            worker's are read as fast as our own -- or ("", 0, 0) if none;
            key_strs[i] is the key's store name.
        """
        if not keys:
            return [], [], []
        empty = ("", 0, 0)
        key_strs = [key.to_string() for key in keys]
        if self.save_chunk_meta:
            return [0] * len(keys), [empty] * len(keys), key_strs
        replica_map = self.store.batch_get_replica_desc(key_strs)
        local_endpoint = self.store.get_hostname()
        local_host = endpoint_host(local_endpoint)
        min_size: dict = {}
        ptrs, ssd = [], []
        for key, key_str in zip(keys, key_strs, strict=True):
            ptr, loc = 0, empty
            same_host = False
            for replica in replica_map.get(key_str, []):
                if replica.status != self.complete_replica_status:
                    continue
                if replica.is_local_disk_replica():
                    disk = replica.get_local_disk_descriptor()
                    if (
                        loc is empty
                        and disk.file_path
                        and endpoint_host(disk.transport_endpoint) == local_host
                    ):
                        loc = (disk.file_path, disk.value_offset, disk.object_size)
                    continue
                if ptr or not self.enable_local_mooncake_hbm:
                    continue
                if not replica.is_memory_replica():
                    continue
                buffer = replica.get_memory_descriptor().buffer_descriptor
                if buffer.transport_endpoint == local_endpoint:
                    layer = getattr(key, "layer_id", None)
                    if layer not in min_size:
                        shapes, dtypes, _ = self._metadata_for_batch_get_into_key(key)
                        min_size[layer] = self._logical_size_bytes(shapes, dtypes)
                    if int(buffer.size) >= min_size[layer]:
                        ptr = int(buffer.buffer_address)
                    continue
                same_host |= endpoint_host(buffer.transport_endpoint) == local_host
            if not ptr and same_host and self.enable_shm_mooncake_hbm:
                raise RuntimeError(
                    "Mooncake shared-memory replica was not preloaded or rewritten "
                    f"for key {key_str}"
                )
            ptrs.append(ptr)
            ssd.append(loc)
        return ptrs, ssd, key_strs

    def batched_locate_local_ssd(
        self, keys: List[CacheEngineKey]
    ) -> List[Tuple[str, int, int]]:
        """Where each key's value sits on this client's SSD, if offload is on.

        Returns per key (bucket file path, byte offset of the value, value
        size); the path is empty for keys not on this client's SSD. Local map
        lookup in the Mooncake client, no RPC.
        """
        if not self.ssd_offload_path or not keys:
            return [("", 0, 0)] * len(keys)
        return self.store.batch_locate_local_ssd([key.to_string() for key in keys])

    def kvpilot_put_start_local(self, keys: List[str], sizes: List[int]) -> List[int]:
        """Allocate a memory replica per key (sizes in bytes) in this client's
        own segment, to be written in place and completed with
        kvpilot_put_end. Per key, the host address to write the value to, or 0
        (already stored, or not in this segment; revoked)."""
        return self.store.batch_put_start_local(keys, sizes, self.replica_config)

    def kvpilot_put_end(self, keys: List[str]) -> List[int]:
        """Complete write-back puts once their buffers hold the values."""
        return self.store.batch_put_end(keys)

    def kvpilot_put_revoke(self, keys: List[str]) -> List[int]:
        """Drop write-back puts that will not be written."""
        return self.store.batch_put_revoke(keys)

    def kvpilot_exist(self, keys: List[str]) -> List[int]:
        """Per key 1 if a completed replica exists, else 0 (one master RPC)."""
        return list(self.store.batch_is_exist(keys))

    def kvpilot_remove_memory(self, keys: List[str]) -> int:
        """Drop keys' memory replicas, keeping disk ones (exclusive placement:
        a chunk leaving DRAM). A replica a reader or an offload holds stays.
        Returns how many keys the master processed."""
        n = int(self.store.batch_remove_memory_replicas(keys))
        if n < 0:
            raise RuntimeError("Mooncake batch_remove_memory_replicas failed")
        return n

    def kvpilot_offload_local(
        self, keys: List[str], ptrs: List[int], sizes: List[int]
    ) -> int:
        """Write values from host memory (the caller's staging buffer, not the
        segment) to this client's SSD as one bucket and register them with
        the master (exclusive placement). Returns how many were written."""
        return int(self.store.batch_offload_local(keys, ptrs, sizes))

    def kvpilot_set_group_placement(
        self, groups: List[str], tiers: List[int], penalties_ms: List[float]
    ) -> int:
        """Set chunk groups' tier and penalty on the master.

        Args:
            groups: Chunk-hash fields of the keys (one per chunk).
            tiers: Tier code per group (kvpilot_placement.TIER_CODE).
            penalties_ms: Penalty per group, same order.

        Returns:
            Groups updated.
        """
        return int(self.store.batch_set_group_placement(groups, tiers, penalties_ms))

    def batched_replica_location(self, keys: List[CacheEngineKey]) -> list:
        # Same-host peers' shm replicas come back rewritten to our own
        # endpoint, so comparing hosts covers both them and our own segment.
        key_strs = [key.to_string() for key in keys]
        replica_map = self.store.batch_get_replica_desc(key_strs)
        host = endpoint_host(self.store.get_hostname())
        locations = []
        for key_str in key_strs:
            loc = ReplicaLocation.MISSING
            for replica in replica_map.get(key_str, []):
                if (
                    replica.status != self.complete_replica_status
                    or not replica.is_memory_replica()
                ):
                    continue
                buffer = replica.get_memory_descriptor().buffer_descriptor
                if endpoint_host(buffer.transport_endpoint) == host:
                    loc = ReplicaLocation.LOCAL
                    break
                loc = ReplicaLocation.REMOTE
            locations.append(loc)
        return locations

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        for key in keys:
            if not self.store.is_exist(key.to_string()):
                break
            num_hit_counts += 1
        return num_hit_counts

    def _metadata_for_batch_get_into_key(
        self, key: CacheEngineKey
    ) -> tuple[list[torch.Size], list[torch.dtype], MemoryFormat]:
        layer_id = getattr(key, "layer_id", None)
        if layer_id is None:
            return self.meta_shapes, self.meta_dtypes, self.meta_fmt

        remaining_layer_id = int(layer_id)
        for shape, dtype in zip(self.meta_shapes, self.meta_dtypes, strict=False):
            if remaining_layer_id >= shape[1]:
                remaining_layer_id -= shape[1]
                continue

            if self.meta_fmt == MemoryFormat.KV_MLA_FMT:
                return [torch.Size([shape[2], shape[3]])], [dtype], self.meta_fmt

            if self.meta_fmt == MemoryFormat.KV_2LTD:
                if self.local_cpu_backend.config.enable_blending:
                    return (
                        [torch.Size([shape[0], shape[2], shape[3]])],
                        [dtype],
                        MemoryFormat.KV_2TD,
                    )
                return (
                    [torch.Size([shape[2], shape[0], shape[3]])],
                    [dtype],
                    MemoryFormat.KV_T2D,
                )

            raise ValueError(f"Unsupported layerwise metadata format: {self.meta_fmt}")

        raise ValueError(
            f"layer_id {layer_id} is out of range for meta_shapes={self.meta_shapes}"
        )

    @staticmethod
    def _logical_size_bytes(
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
    ) -> int:
        return sum(
            shape.numel() * dtype.itemsize
            for shape, dtype in zip(shapes, dtypes, strict=True)
        )

    @staticmethod
    def _raw_tensor_from_host_ptr(ptr: int, size: int) -> torch.Tensor:
        buffer_type = ctypes.c_uint8 * size
        host_buffer = buffer_type.from_address(ptr)
        tensor = torch.frombuffer(host_buffer, dtype=torch.uint8, count=size)
        # Keep the ctypes view alive for the lifetime of the tensor wrapper.
        tensor._lmcache_mooncake_host_buffer = host_buffer  # type: ignore[attr-defined]
        return tensor

    def _make_local_raw_memory_obj(
        self,
        ptr: int,
        physical_size: int,
        shapes: list[torch.Size],
        dtypes: list[torch.dtype],
        fmt: MemoryFormat,
    ) -> TensorMemoryObj:
        logical_size = self._logical_size_bytes(shapes, dtypes)
        if physical_size < logical_size:
            raise ValueError(
                f"Mooncake local range is too small: physical={physical_size}, "
                f"logical={logical_size}"
            )
        raw_tensor = self._raw_tensor_from_host_ptr(ptr, logical_size)
        return TensorMemoryObj(
            raw_data=raw_tensor,
            metadata=MemoryObjMetadata(
                shapes[0],
                dtypes[0],
                ptr,
                physical_size,
                1,
                0,
                fmt,
                shapes=shapes,
                dtypes=dtypes,
            ),
            parent_allocator=None,
        )

    def _try_batch_get_local_raw_views(
        self, keys: List[CacheEngineKey]
    ) -> Optional[List[MemoryObj]]:
        if not self.enable_local_mooncake_hbm or self.save_chunk_meta:
            return None

        batch_get_replica_desc = getattr(self.store, "batch_get_replica_desc", None)
        if batch_get_replica_desc is None:
            if self.enable_local_mooncake_hbm:
                raise RuntimeError("Mooncake batch_get_replica_desc is unavailable")
            return None

        key_strs = [key.to_string() for key in keys]
        try:
            replica_map = batch_get_replica_desc(key_strs)
            local_endpoint = self.store.get_hostname()
        except Exception as exc:
            logger.debug("Unable to query Mooncake local replicas: %s", exc)
            if self.enable_shm_mooncake_hbm:
                raise RuntimeError("Mooncake shared-memory replica query failed") from exc
            return None

        memory_objs: list[MemoryObj] = []
        local_segment_addrs = 0
        cross_segment_addrs = 0
        for key, key_str in zip(keys, key_strs, strict=True):
            try:
                shapes, dtypes, fmt = self._metadata_for_batch_get_into_key(key)
            except Exception as exc:
                logger.debug(
                    "Unable to infer metadata for local Mooncake key %s: %s",
                    key,
                    exc,
                )
                if self.enable_shm_mooncake_hbm:
                    raise RuntimeError(
                        f"Unable to infer Mooncake shared-memory metadata for key {key}"
                    ) from exc
                return None

            selected_buffer = None
            any_same_host = False
            local_host = endpoint_host(local_endpoint)
            for replica in replica_map.get(key_str, []):
                if replica.status != self.complete_replica_status:
                    continue
                if not replica.is_memory_replica():
                    continue
                buffer = replica.get_memory_descriptor().buffer_descriptor
                if buffer.transport_endpoint == local_endpoint:
                    selected_buffer = buffer
                    break
                any_same_host |= endpoint_host(buffer.transport_endpoint) == local_host

            if selected_buffer is None:
                # Only another host has it: fetch the whole batch over the
                # network (cache_use_control can admit remote DRAM).
                if self.enable_shm_mooncake_hbm and any_same_host:
                    raise RuntimeError(
                        "Mooncake shared-memory replica was not preloaded or rewritten "
                        f"for key {key_str}"
                    )
                return None

            ptr = int(selected_buffer.buffer_address)
            physical_size = int(selected_buffer.size)
            segment_base = int(getattr(selected_buffer, "segment_base", 0) or 0)
            segment_size = int(getattr(selected_buffer, "segment_size", 0) or 0)
            shared_memory = bool(getattr(selected_buffer, "shared_memory", False))
            # A rewritten peer address keeps the owner's segment_base/size but
            # points into our own mapping, so it falls outside the owner range.
            is_cross_segment_addr = (
                shared_memory
                and segment_base > 0
                and segment_size > 0
                and not (segment_base <= ptr < segment_base + segment_size)
            )
            if is_cross_segment_addr:
                cross_segment_addrs += 1
            else:
                local_segment_addrs += 1
            if ptr <= 0 or physical_size <= 0:
                logger.debug(
                    "Invalid local Mooncake range for key %s: ptr=%s size=%s",
                    key,
                    ptr,
                    physical_size,
                )
                if self.enable_shm_mooncake_hbm:
                    raise RuntimeError(f"Invalid Mooncake shared-memory range for key {key}")
                return None

            try:
                memory_objs.append(
                    self._make_local_raw_memory_obj(
                        ptr,
                        physical_size,
                        shapes,
                        dtypes,
                        fmt,
                    )
                )
            except Exception as exc:
                logger.debug(
                    "Unable to build local Mooncake raw view for key %s: %s",
                    key,
                    exc,
                )
                if self.enable_shm_mooncake_hbm:
                    raise RuntimeError(
                        f"Unable to build Mooncake shared-memory raw view for key {key}"
                    ) from exc
                return None

        if cross_segment_addrs == 0:
            address_source = "local_segment"
        elif local_segment_addrs == 0:
            address_source = "cross_segment"
        else:
            address_source = "mixed"
        logger.info(
            "Using local Mooncake raw-view get: keys=%d address_source=%s ",
            len(keys),
            address_source,
        )
        return memory_objs

    async def _batch_get_into(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """
        Zero-copy batch get using batch_get_into when metadata is available locally.
        This is used when save_chunk_meta=False (metadata not stored remotely).
        """
        if not self.meta_shapes or not self.meta_dtypes or not self.meta_fmt:
            logger.error(
                f"Metadata required for batch_get_into but not available: "
                f"meta_shapes={self.meta_shapes}, "
                f"meta_dtypes={self.meta_dtypes}, "
                f"meta_fmt={self.meta_fmt}"
            )
            return [None] * len(keys)

        local_raw_views = self._try_batch_get_local_raw_views(keys)
        if local_raw_views is not None:
            return local_raw_views

        logger.debug(f"Using batch_get_into for {len(keys)} keys (zero-copy mode)")

        # Reserve a buffer for every requested chunk
        memory_objs: list[Optional[MemoryObj]] = []
        valid_idx: list[int] = []

        key_strs: list[str] = []
        buffer_ptrs: list[int] = []
        buffer_sizes: list[int] = []

        torch.cuda.nvtx.range_push(
            f"LMCache.mooncake.batch_get_into.alloc.keys_{len(keys)}"
        )
        try:
            for i, key in enumerate(keys):
                try:
                    shapes, dtypes, fmt = self._metadata_for_batch_get_into_key(key)
                except Exception as exc:
                    logger.error(
                        "Failed to infer local metadata for key %s: %s", key, exc
                    )
                    memory_objs.append(None)
                    continue

                obj = self.local_cpu_backend.allocate(shapes, dtypes, fmt)
                memory_objs.append(obj)
                if obj is not None and obj.raw_tensor is not None:
                    valid_idx.append(i)

                    # Prepare the argument lists for the C++ call
                    key_strs.append(keys[i].to_string())
                    buffer_ptrs.append(obj.data_ptr)
                    buffer_sizes.append(obj.get_size())
        finally:
            torch.cuda.nvtx.range_pop()

        if not valid_idx:
            logger.warning("Batch-get aborted: unable to allocate any buffers.")
            return [None] * len(keys)

        try:
            # Single RPC call for multiple chunks
            logger.debug(f"Calling batch_get_into with {len(key_strs)} keys")

            def _batch_get_into_with_nvtx():
                torch.cuda.nvtx.range_push(
                    f"LMCache.mooncake.batch_get_into.store.keys_{len(key_strs)}"
                )
                try:
                    return self.store.batch_get_into(
                        key_strs, buffer_ptrs, buffer_sizes
                    )
                finally:
                    torch.cuda.nvtx.range_pop()

            bytes_read_list = await asyncio.to_thread(_batch_get_into_with_nvtx)
            logger.debug(f"batch_get_into returned: {bytes_read_list}")

            # Assemble the final result list
            results: list[Optional[MemoryObj]] = [None] * len(keys)

            for i, n_read in zip(valid_idx, bytes_read_list, strict=False):
                if n_read <= 0:
                    logger.warning(
                        f"batch_get_into failed for key {keys[i]} (code={n_read})"
                    )
                    memory_objs[i].ref_count_down()  # type: ignore
                    continue

                try:
                    results[i] = self.reshape_partial_chunk(
                        memory_objs[i],  # type: ignore
                        n_read,
                    )
                except Exception as exc:
                    logger.error(f"Reshape failed for key {keys[i]}: {exc}")
                    memory_objs[i].ref_count_down()  # type: ignore

            return results

        except Exception as exc:
            logger.error(f"batch_get_into threw exception: {str(exc)}")
            # Release any buffers we successfully allocated
            for i in valid_idx:
                memory_objs[i].ref_count_down()  # type: ignore
            return [None] * len(keys)

    async def _batch_get_buffer(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """
        Batch get using batch_get_buffer when metadata is stored remotely.
        This is used when save_chunk_meta=True (metadata stored with data).
        """
        key_strs = [key.to_string() for key in keys]

        try:
            buffers = await asyncio.to_thread(self.store.batch_get_buffer, key_strs)
        except Exception as e:
            logger.error(f"batch_get_buffer failed: {str(e)}")
            return [None] * len(keys)

        results: list[Optional[MemoryObj]] = []
        for i, buffer in enumerate(buffers):
            if buffer is None:
                logger.warning(f"Buffer {i} is None for key {key_strs[i]}")
                results.append(None)
                continue
            try:
                memory_obj = self._process_buffer_with_metadata(buffer)
                results.append(memory_obj)
            except Exception as e:
                logger.error(
                    f"Failed to process buffer {i} for key {key_strs[i]}: {str(e)}"
                )
                results.append(None)
        return results

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Single get method - NOT SUPPORTED.
        Use batched_get instead for all operations.
        """
        logger.error("Single get operation is not supported. Use batched_get instead.")
        raise NotImplementedError(
            "Single get is not supported. Use batched_get([key]) instead."
        )

    def _process_buffer_with_metadata(self, buffer: bytes) -> Optional[MemoryObj]:
        """
        Process buffer that contains metadata + data.
        Used when save_chunk_meta=True (metadata stored remotely).
        """
        retrieved_view = memoryview(buffer)
        metadata_bytes = retrieved_view[: self.remote_metadata_bytes]
        if metadata_bytes is None or len(metadata_bytes) != self.remote_metadata_bytes:
            return None

        metadata = RemoteMetadata.deserialize(metadata_bytes)

        memory_obj = self.local_cpu_backend.allocate(
            metadata.shapes,
            metadata.dtypes,
            metadata.fmt,
        )
        assert len(retrieved_view) == metadata.length + self.remote_metadata_bytes

        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        if memory_obj.raw_tensor is not None:
            temp_tensor = torch.frombuffer(
                buffer,
                dtype=torch.uint8,
                offset=self.remote_metadata_bytes,
                count=metadata.length,
            )

            memory_obj.raw_tensor.copy_(temp_tensor)
            return memory_obj
        else:
            return None

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """
        Put operation with metadata-consistent handling.
        Uses put_from (without metadata) or
        put_parts (with metadata) to match get behavior.
        """
        key_str = key.to_string()

        # Check metadata handling mode to match get behavior
        if self.save_chunk_meta:
            # Use put_parts with metadata stored remotely
            await self._put_with_metadata(key_str, memory_obj)
        else:
            # Use put_from without metadata (zero-copy)
            await self._put_without_metadata(key_str, memory_obj)

    def support_batched_put(self) -> bool:
        return True

    async def batched_put(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ):
        """
        Batched put with clear split by metadata mode.
        - save_chunk_meta False: use Mooncake's batch_put_from (zero-copy).
        - save_chunk_meta True: no batch API; fall back to sequential put_parts.
        """
        if not keys:
            return

        if self.save_chunk_meta:
            await self._batched_put_with_metadata(keys, memory_objs)
        else:
            await self._batched_put_zero_copy(keys, memory_objs)

    async def _batched_put_zero_copy(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        key_strs = [k.to_string() for k in keys]
        buffer_ptrs: list[int] = []
        buffer_sizes: list[int] = []
        for obj in memory_objs:
            assert obj.raw_tensor is not None
            buffer_ptrs.append(obj.data_ptr)
            buffer_sizes.append(obj.get_size())

        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    self.store.batch_put_from,
                    key_strs,
                    buffer_ptrs,
                    buffer_sizes,
                    self.replica_config,
                ),
                timeout=self.config.transfer_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Timeout during batch_put_from; some decoders may redo prefill."
            )

    async def _batched_put_with_metadata(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> None:
        for key, obj in zip(keys, memory_objs, strict=False):
            await self._put_with_metadata(key.to_string(), obj)

    async def _put_without_metadata(self, key_str: str, memory_obj: MemoryObj):
        """
        Zero-copy put using put_from when metadata is not stored remotely.
        This is used when save_chunk_meta=False (matches _batch_get_into).
        """
        try:
            assert memory_obj.raw_tensor is not None
            buffer_ptr = memory_obj.data_ptr
            buffer_size = memory_obj.get_size()

            await asyncio.wait_for(
                asyncio.to_thread(
                    self.store.put_from,
                    key_str,
                    buffer_ptr,
                    buffer_size,
                    self.replica_config,
                ),
                timeout=self.config.transfer_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout when putting key {key_str} using put_from. "
                "Decode instance may redo prefill."
            )
        except Exception as e:
            logger.error(
                f"Failed to put key {key_str} using put_from: "
                f"{type(e).__name__}: {str(e)}"
            )
            raise

    async def _put_with_metadata(self, key_str: str, memory_obj: MemoryObj):
        """
        Put using put_parts when metadata is stored remotely.
        This is used when save_chunk_meta=True (matches _batch_get_buffer).
        """
        try:
            # Serialize data and metadata
            kv_bytes = memory_obj.byte_array
            kv_shapes = memory_obj.get_shapes()
            kv_dtypes = memory_obj.get_dtypes()
            memory_format = memory_obj.get_memory_format()

            metadata_bytes = RemoteMetadata(
                len(kv_bytes), kv_shapes, kv_dtypes, memory_format
            ).serialize()
            assert len(metadata_bytes) == self.remote_metadata_bytes

            await asyncio.wait_for(
                asyncio.to_thread(
                    self.store.put_parts, key_str, metadata_bytes, kv_bytes
                ),
                timeout=self.config.transfer_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout when putting key {key_str} using put_parts. "
                "Decode instance may redo prefill."
            )
        except Exception as e:
            logger.error(
                f"Failed to put key {key_str} using put_parts: "
                f"{type(e).__name__}: {str(e)}"
            )
            raise

    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        # Unregister buffer before closing the store
        self._unregister_cpu_buffer()

        self.store.close()
        logger.info("Closed the mooncake store connection")
