# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    """

    # None = auto-detect: num_nodes * 8 (requires Ray to be initialized).
    # Falls back to 8 if Ray is unavailable.
    max_workers: int | None = None
    max_cache_size_gb: float = 50.0
    # NOTE: In distributed Ray training, tmp_root must point to a LustreFS mount
    # (e.g. /lustre/...). Because downloads are executed by remote Ray actors that may
    # run on different machines than the DataLoader workers, a shared filesystem is
    # required so that files downloaded on one node are accessible by readers on another.
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
    # Enable shared DuckDB download index on Lustre for cross-rank file
    # discovery. Off by default; the in-memory files dict is sufficient for
    # single-rank or non-distributed use cases.
    enable_download_index: bool = False
    # Distributed identity used by download_index when ``enable_download_index=True``.
    # Under the default ``IdentityStrategy`` every rank downloads what it registers and
    # dedup is best-effort via cross-rank adoption (a rank registering after another
    # rank has published the row reads from the index instead of re-fetching).
    rank: int = 0
    world_size: int = 1
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
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank must satisfy 0 <= rank < world_size, "
                f"got rank={self.rank} world_size={self.world_size}"
            )

    def with_distributed_defaults(self) -> "HydratorConfig":
        """Return a copy with rank/world_size/enable_download_index resolved from torch.

        When ``torch.distributed`` is initialized with ``world_size > 1``, populate
        ``rank`` and ``world_size`` from the current process group and flip
        ``enable_download_index=True``. The partition strategy is left as ``None``
        so ``CacheManager`` selects the safe default (``IdentityStrategy``) — see
        ``cache.manager.CacheManager`` for the rationale.

        Single-rank and non-distributed runs return ``self`` unchanged.

        torch is imported lazily so importing this module stays light.
        """
        try:
            import torch.distributed as dist
        except ImportError:
            return self
        if not (dist.is_available() and dist.is_initialized()):
            return self
        world_size = dist.get_world_size()
        if world_size <= 1:
            return self
        return dataclasses.replace(
            self,
            rank=dist.get_rank(),
            world_size=world_size,
            enable_download_index=True,
        )
