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

"""Partition strategies for distributed file ownership and DuckDB shard routing.

A ``PartitionStrategy`` decides two things, side-by-side:

1. ``is_owner(path, rank, world_size)`` — which rank should download a path.
   Drives ownership under ``download_index`` so the same file isn't fetched by
   multiple ranks.
2. ``db_key(path)`` — which DuckDB shard stores the (uri, local_path) record.
   Lookup is decoupled from ``is_owner`` so a ``world_size`` change between
   runs reshuffles owners but doesn't strand existing entries.

Two built-ins:

- ``HashPartitionStrategy``: sha1-based; uniform owner load AND uniform DB
  shard load. Default for new ``download_index`` users.
- ``AlphanumericPrefixStrategy``: filename-first-character partitioning;
  preserves the existing s5cmd directory-sync layout.
"""

from abc import ABC, abstractmethod
import hashlib


class PartitionStrategy(ABC):
    """Pluggable owner + DB-routing strategy for download_index entries."""

    @abstractmethod
    def is_owner(self, path: str, rank: int, world_size: int) -> bool:
        """Return True iff this rank should download ``path``."""

    @abstractmethod
    def db_key(self, path: str) -> str:
        """Return the DuckDB shard key for ``path`` (used as ``sync_{key}.duckdb``)."""


class HashPartitionStrategy(PartitionStrategy):
    """Uniform sha1-based ownership + bucketed DB routing.

    Ownership: ``sha1(path) % world_size == rank``.
    DB key: ``sha1(path) % n_db_buckets`` — same hash, different modulus.
    sha1 is used (not Python's ``hash()``) so results are deterministic
    across processes regardless of ``PYTHONHASHSEED``.

    **Caller contract**: the registrant set for any given URI must include
    the hash-picked owner. Equivalently — every rank that knows about a
    URI must call ``register_file`` on it (or at least the one that
    happens to be the owner must). When the contract holds, ownership is
    deterministic and exactly one rank downloads per URI. When it
    doesn't (e.g. VLA-style partial-shard registration where the
    hash-picked rank never registers the URI), pollers deadlock waiting
    for an owner that never appears — use ``IdentityStrategy`` instead.
    """

    def __init__(self, n_db_buckets: int = 64) -> None:
        self.n_db_buckets = n_db_buckets

    def _h(self, path: str) -> int:
        return int.from_bytes(hashlib.sha1(path.encode("utf-8")).digest()[:8], "big")

    def is_owner(self, path: str, rank: int, world_size: int) -> bool:
        return self._h(path) % world_size == rank

    def db_key(self, path: str) -> str:
        return str(self._h(path) % self.n_db_buckets)


class IdentityStrategy(HashPartitionStrategy):
    """Every registrant downloads — no ownership coordination.

    ``is_owner`` always returns True. Use when the registrant set is
    unpredictable (e.g. VLA training where each rank registers only its
    own shard, so the union has overlap but no rank knows which others
    will register a given URI).

    Dedup becomes **best-effort** via cross-rank adoption: a rank
    registering after another rank already published hits the index and
    adopts (no enqueue). Ranks registering before any publish all
    enqueue and duplicate-download — the worst case matches main's
    pre-dedup behavior, never worse.

    Inherits ``db_key`` from ``HashPartitionStrategy`` so DuckDB shard
    routing stays uniform regardless of strategy choice.

    Future stronger dedup (deferred): an atomic
    ``DownloadIndex.try_claim`` (``INSERT … ON CONFLICT DO NOTHING
    RETURNING``) would let the first registrant win the row and the
    others see in-flight without enqueueing — guaranteeing 1 download
    per URI even under fully concurrent registration. Requires UPSERT
    on completion + an in-flight row state (NULL local_path AND NULL
    error_message), plus crash-recovery for stranded claims. Out of
    scope for the current iteration.
    """

    def is_owner(self, path: str, rank: int, world_size: int) -> bool:
        return True


class AlphanumericPrefixStrategy(PartitionStrategy):
    """Filename-first-character partitioning (preserves s5cmd-style layout).

    65 prefixes (0-9, A-Z, a-z, ``_``, ``-``, ``.``). Ownership is round-robin
    over prefixes; DB routing matches (one DuckDB file per prefix). Files
    whose filename starts with a character outside this set fall back to the
    ``_`` bucket.
    """

    _ALPHANUMERIC: list[str] = (
        [chr(ord("0") + i) for i in range(10)]
        + [chr(ord("A") + i) for i in range(26)]
        + [chr(ord("a") + i) for i in range(26)]
    )
    _OTHERS: list[str] = ["_", "-", "."]
    PREFIXES: list[str] = _ALPHANUMERIC + _OTHERS

    def all_prefixes(self) -> list[str]:
        return self.PREFIXES

    @staticmethod
    def _prefix_of(path: str) -> str:
        filename = path.rsplit("/", 1)[-1]
        return filename[0] if filename else "_"

    def is_owner(self, path: str, rank: int, world_size: int) -> bool:
        prefix = self._prefix_of(path)
        if prefix not in self.PREFIXES:
            prefix = "_"
        return self.PREFIXES.index(prefix) % world_size == rank

    def db_key(self, path: str) -> str:
        prefix = self._prefix_of(path)
        return prefix if prefix in self.PREFIXES else "_"
