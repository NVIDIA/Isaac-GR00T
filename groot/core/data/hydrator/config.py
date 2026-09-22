# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Configuration for Data Hydrator."""

import dataclasses
from dataclasses import dataclass
from typing import Any

from groot.infra.job.data_credentials import DataCredentials


@dataclass
class HydratorConfig:
    """Configuration for the data hydrator service.

    Attributes:
        max_workers: Number of concurrent download workers
        max_cache_size_gb: Maximum total cache size in GB (triggers LRU eviction)
        tmp_root: Root directory for local cache storage (if None, uses cwd/hydrate_cache)
        download_timeout: Timeout in seconds for downloading a single file
        retry_attempts: Number of retry attempts for failed downloads
        retry_delay: Initial retry delay in seconds (exponential backoff)
        eviction_buffer: Evict until size < max_cache_size_gb * buffer (0.9 = 90%)
        cleanup_on_shutdown: Whether to delete all cached files on shutdown
        backend: Storage backend type ("boto3", "s5cmd", "aioboto3", "lustre")
        backend_config: Backend-specific configuration dict
        credentials_profile: Profile name in ~/.gear/data_credentials.
            ``None`` (default) means auto-resolve: the first profile with working
            credentials from :class:`DataSourceResolver` is used.
        credentials_file: Path to data credentials file (default: ~/.gear/data_credentials).
        runtime_env: Optional Ray runtime_env for worker pool (e.g. tests use py_executable).
        worker_node_affinity: Keep I/O workers on the process's Ray node. Enable this when the
            cache directory is node-local rather than shared storage.
    """

    # None = auto-detect: num_nodes * 8 (requires Ray to be initialized).
    # Falls back to 8 if Ray is unavailable.
    max_workers: int | None = None
    max_cache_size_gb: float = 50.0
    # In distributed Ray training, tmp_root must point to shared storage unless
    # worker_node_affinity is enabled and actors share the cache mount. Node affinity keeps
    # I/O on the owning machine; container deployments must also provide a shared mount.
    tmp_root: str | None = None
    download_timeout: float = 300.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    eviction_buffer: float = 0.9
    cleanup_on_shutdown: bool = True
    backend: str = "boto3"
    backend_config: dict = dataclasses.field(default_factory=dict)
    credentials_profile: str | None = None
    credentials_file: str = DataCredentials.DEFAULT_CREDENTIALS_FILE
    runtime_env: dict[str, Any] | None = None
    worker_node_affinity: bool = False
    # Enable shared DuckDB download index on Lustre for cross-rank file discovery. The
    # in-memory files dict is sufficient for single-rank or non-distributed use cases.
    #
    # ``None`` means unspecified: ``with_distributed_defaults`` resolves it from the ambient
    # process group. Pass a value to state the intent explicitly and it is never overwritten —
    # an actor-owned cache asking for ``False`` keeps it even inside a DDP worker.
    enable_download_index: bool | None = None
    # Distributed identity used by download_index when ``enable_download_index`` resolves true.
    # Under the default ``IdentityStrategy`` every rank downloads what it registers and
    # dedup is best-effort via cross-rank adoption (a rank registering after another
    # rank has published the row reads from the index instead of re-fetching).
    #
    # ``None`` means unspecified, resolved the same way; an explicit value wins.
    rank: int | None = None
    world_size: int | None = None
    # Pluggable strategy that decides ``is_owner`` and DuckDB ``db_key``
    # for each path. ``None`` (default) selects ``IdentityStrategy()`` when
    # ``enable_download_index=True``; ignored otherwise. ``HashPartitionStrategy``
    # is selectable but requires the registrant set for every URI to include the
    # hash-picked owner (e.g. eager full-set registration up-front at every rank);
    # see ``cache.partition_strategy``.
    #
    # Typed ``Any`` (not ``PartitionStrategy | None``) so OmegaConf can wrap
    # HydratorConfig as a structured config — it can't validate against an
    # ABC. CacheManager treats the value as duck-typed against PartitionStrategy.
    download_partition_strategy: Any = None

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_workers is not None and self.max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {self.max_workers}")
        if self.max_cache_size_gb <= 0:
            raise ValueError(f"max_cache_size_gb must be positive, got {self.max_cache_size_gb}")
        if not 0 < self.eviction_buffer <= 1.0:
            raise ValueError(f"eviction_buffer must be in (0, 1], got {self.eviction_buffer}")
        if type(self.worker_node_affinity) is not bool:
            raise TypeError("worker_node_affinity must be a boolean")
        # rank/world_size validate once resolved; ``None`` is "unspecified", not invalid.
        if self.world_size is not None and self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}")
        if self.rank is not None:
            limit = self.world_size if self.world_size is not None else 1
            if not 0 <= self.rank < limit:
                raise ValueError(
                    f"rank must satisfy 0 <= rank < world_size, "
                    f"got rank={self.rank} world_size={limit}"
                )

    def with_distributed_defaults(self) -> "HydratorConfig":
        """Return a copy with *unspecified* distributed fields resolved from torch.

        ``rank``, ``world_size`` and ``enable_download_index`` default to ``None`` meaning
        "unspecified". Only those are filled: when ``torch.distributed`` is initialized with
        ``world_size > 1`` they resolve to the ambient process group with the shared index on,
        otherwise to the single-rank values with it off. The partition strategy is left as
        ``None`` so ``CacheManager`` selects the safe default (``IdentityStrategy``) — see
        ``cache.manager.CacheManager`` for the rationale.

        A field the caller passed explicitly is never overwritten. That matters for caches
        that are actor-owned rather than rank-shared: a trainer rank publishing a checkpoint
        runs inside DDP, but its private upload cache must not adopt cross-rank download
        state just because a process group happens to exist around it.

        torch is imported lazily so importing this module stays light.
        """
        ambient_rank, ambient_world_size, distributed = 0, 1, False
        try:
            import torch.distributed as dist
        except ImportError:
            pass
        else:
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                ambient_rank = dist.get_rank()
                ambient_world_size = dist.get_world_size()
                distributed = True
        return dataclasses.replace(
            self,
            rank=self.rank if self.rank is not None else ambient_rank,
            world_size=(self.world_size if self.world_size is not None else ambient_world_size),
            enable_download_index=(
                self.enable_download_index
                if self.enable_download_index is not None
                else distributed
            ),
        )
