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
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import datetime as dt
from datetime import datetime
import enum
import json
import os
from typing import ClassVar

from botocore import exceptions as botocore_exceptions

from groot.infra.ci import db, log
from groot.infra.ci import pipeline as ci_pipeline

_S3_KEY_PREFIX = "rws/ci/tests"
_ZOMBIE_CACHE_KEY_PREFIX = "rws/ci/zombie_cache/v1"
_ZOMBIE_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours


def _is_first_shard() -> bool:
    """Return whether this job is the first test shard.

    PYTEST_SHARD is set by the unit-test matrix as "1/6", "2/6", etc. Fall
    back to GitLab's CI_NODE_INDEX if the caller is using GitLab parallel
    indexing instead.
    """
    pytest_shard = os.getenv("PYTEST_SHARD", "").strip()
    if pytest_shard:
        shard_index = pytest_shard.split("/", 1)[0]
        try:
            return int(shard_index) == 1
        except ValueError:
            return True

    ci_node_index = os.getenv("CI_NODE_INDEX", "").strip()
    if ci_node_index:
        try:
            return int(ci_node_index) == 1
        except ValueError:
            return True

    return True


class TestType(enum.Enum):
    E2E = "e2e"
    INTEGRATION = "integration"
    UNIT = "unit"


@dataclass
class Test:
    """An abstraction to represent a test in the CI/CD pipeline."""

    name: str
    owner: str
    type: TestType
    state: str | None = None
    last_modified_by: str | None = None
    issue_number: int | None = None

    _db: ClassVar[db.Database] = db.Database()
    _logger: ClassVar = log.init(__name__)

    def get_canonical_name(self) -> str:
        """Get the database key representation (aka. canonical name) of the test.

        The canonical name is a string that is used to uniquely identify a test in the database.
        It is constructed by concatenating the sanitized test name and the test type.

        Note: Common pytest test name characters are sanitized for S3 keys:
        - :: (pytest separators) → __ (double underscore)
        - / (path separators) → _
        - [ ] (parametrized test markers) → _
        - spaces → _
        """
        # Sanitize common pytest test name characters for S3 key compatibility
        sanitized_name = self.name
        # Replace :: first (before individual colons)
        sanitized_name = sanitized_name.replace("::", "__")
        for char in ["/", "[", "]", " "]:
            sanitized_name = sanitized_name.replace(char, "_")

        return f"{sanitized_name}_{self.type.value}"

    async def async_persist(self) -> None:
        """Persist the test to the database."""
        key = f"{_S3_KEY_PREFIX}/{self.type.value}/{self.get_canonical_name()}.json"
        if not self.state or self.issue_number is None:
            try:
                existing_json = await Test._db.async_read(key)
                existing_data = json.loads(existing_json)
                if not self.state:
                    self.state = existing_data.get("state")
                if self.issue_number is None:
                    self.issue_number = existing_data.get("issue_number")
            except (
                botocore_exceptions.BotoCoreError,
                botocore_exceptions.ClientError,
                json.JSONDecodeError,
                KeyError,
            ) as e:
                Test._logger.debug(f"No existing record for {key}: {e}")
        json_object = {
            "name": self.name,
            "owner": self.owner,
            "type": self.type.value,
            "state": self.state,
            "last_modified_by": self.last_modified_by,
            "issue_number": self.issue_number,
        }
        await Test._db.async_write(key, json.dumps(json_object))

    @classmethod
    async def get_all_paginated(
        cls, test_type: TestType, page: int = 1, page_size: int = 50
    ) -> tuple[list[Test], int]:
        """Fetch a paginated subset of tests of a specific type from S3.

        Args:
            test_type: Type of tests to fetch (UNIT, INTEGRATION, or E2E)
            page: Page number (1-indexed)
            page_size: Number of tests per page

        Returns:
            Tuple of (list of Test objects for the page, total count of tests)
        """
        db_ = cls._db
        # Use test type in prefix for efficient S3 filtering
        prefix = f"{_S3_KEY_PREFIX}/{test_type.value}/"
        keys = await db_.async_list(prefix)
        total_count = len(keys)

        cls._logger.info(
            f"Found {total_count} {test_type.value} test files in S3, "
            f"fetching page {page} (size {page_size})..."
        )

        # Paginate the keys first (before reading S3)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_keys = keys[start_idx:end_idx]

        cls._logger.info(f"Reading {len(paginated_keys)} test files for page {page}...")

        # Read only the paginated test files
        test_jsons = await asyncio.gather(
            *[db_.async_read(key) for key in paginated_keys], return_exceptions=True
        )

        # Parse into Test objects
        tests = []
        errors = 0
        for test_json in test_jsons:
            if isinstance(test_json, Exception):
                errors += 1
                continue
            try:
                data = json.loads(test_json)
                tests.append(
                    cls(
                        name=data["name"],
                        owner=data["owner"],
                        type=TestType[data["type"].upper()],
                        state=data.get("state"),
                        last_modified_by=data.get("last_modified_by"),
                        issue_number=data.get("issue_number"),
                    )
                )
            except (json.JSONDecodeError, KeyError):
                errors += 1
                continue

        if errors > 0:
            cls._logger.warning(f"Skipped {errors} files due to read/parse errors")

        cls._logger.info(
            f"Successfully parsed {len(tests)} {test_type.value} tests "
            f"for page {page}/{(total_count + page_size - 1) // page_size}"
        )
        return tests, total_count

    @classmethod
    async def _async_get_all(cls, test_type: TestType) -> list[Test]:
        """Async implementation: fetch all tests of the given type, paginating as needed."""
        all_tests: list[Test] = []
        page = 1
        page_size = 50
        while True:
            tests, total = await cls.get_all_paginated(test_type, page=page, page_size=page_size)
            all_tests.extend(tests)
            if page * page_size >= total:
                break
            page += 1
        return all_tests

    @classmethod
    def get_all(cls, test_type: TestType) -> list[Test]:
        """Fetch all tests of the given type from the DB, paginating as needed."""
        return asyncio.run(cls._async_get_all(test_type))

    @classmethod
    async def _read_zombie_cache(cls, test_type: TestType) -> list[str] | None:
        """Read zombie names from S3 cache. Returns None if missing or older than TTL."""
        key = f"{_ZOMBIE_CACHE_KEY_PREFIX}/{test_type.value}.json"
        try:
            raw = await cls._db.async_read(key)
            data = json.loads(raw)
            cached_at = datetime.fromisoformat(data["cached_at"])
            age = (datetime.now(dt.timezone.utc) - cached_at).total_seconds()
            if age <= _ZOMBIE_CACHE_TTL_SECONDS:
                cls._logger.info(
                    f"[zombie-cache] hit for {test_type.value} (age {age:.0f}s / "
                    f"{_ZOMBIE_CACHE_TTL_SECONDS}s TTL)"
                )
                return data["zombie_names"]
            cls._logger.info(
                f"[zombie-cache] stale for {test_type.value} (age {age:.0f}s > "
                f"{_ZOMBIE_CACHE_TTL_SECONDS}s TTL) — will refresh"
            )
        except (
            botocore_exceptions.BotoCoreError,
            botocore_exceptions.ClientError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
        ):
            cls._logger.info(f"[zombie-cache] miss for {test_type.value} — will query DB")
        return None

    @classmethod
    async def _write_zombie_cache(cls, test_type: TestType, zombie_names: list[str]) -> None:
        """Write zombie names to S3 cache with current timestamp."""
        key = f"{_ZOMBIE_CACHE_KEY_PREFIX}/{test_type.value}.json"
        payload = {
            "cached_at": datetime.now(dt.timezone.utc).isoformat(),
            "zombie_names": zombie_names,
        }
        try:
            await cls._db.async_write(key, json.dumps(payload))
            cls._logger.info(
                f"[zombie-cache] wrote {len(zombie_names)} name(s) for {test_type.value}"
            )
        except (botocore_exceptions.BotoCoreError, botocore_exceptions.ClientError) as e:
            cls._logger.warning(f"[zombie-cache] write failed ({e}) — continuing without cache")

    @classmethod
    def get_in_zombie_states(cls, test_type: TestType) -> list[str]:
        """Return names of tests in zombie/sloth state from the DB.

        Cache strategy (based on Pipeline.is_main_branch()):
        - Main branch first shard: run the full DB query and write the result to cache.
        - Main branch later shards: read from cache if fresh (< TTL), fall back to DB.
        - Non-main (MR branches, local dev): read from cache if fresh (< TTL),
          fall back to full query on cache miss/stale. Never write cache.

        Uses a deferred import of TestState to avoid circular imports
        (test_state.py imports from test.py).
        """
        from groot.infra.ci import test_state

        zombie_states = frozenset(
            {test_state.TestState.ZOMBIE.value, test_state.TestState.SLOTH.value}
        )
        on_main = ci_pipeline.Pipeline().is_main_branch()
        refresh_cache = on_main and _is_first_shard()

        async def _run() -> list[str]:
            if not refresh_cache:
                cached = await cls._read_zombie_cache(test_type)
                if cached is not None:
                    return cached

            try:
                tests = await cls._async_get_all(test_type)
            except (
                botocore_exceptions.BotoCoreError,
                botocore_exceptions.ClientError,
                json.JSONDecodeError,
                KeyError,
            ) as e:
                cls._logger.warning(f"DB query failed ({e}) — skipping zombie state filter")
                return []

            excluded = [t.name for t in tests if t.state in zombie_states]
            if refresh_cache:
                await cls._write_zombie_cache(test_type, excluded)
            return excluded

        excluded = asyncio.run(_run())
        if excluded:
            cls._logger.info(
                f"Found {len(excluded)} {test_type.value} test(s) in zombie/sloth state: "
                f"{', '.join(excluded)}"
            )
        else:
            cls._logger.info(f"No {test_type.value} tests in zombie/sloth state")
        return excluded

    @classmethod
    async def get_by_name(cls, name: str, test_type: TestType) -> Test | None:
        """Load a single test by name and type directly from S3."""
        dummy = cls(name=name, owner="ci", type=test_type)
        key = f"{_S3_KEY_PREFIX}/{test_type.value}/{dummy.get_canonical_name()}.json"
        try:
            test_json = await cls._db.async_read(key)
            data = json.loads(test_json)
            return cls(
                name=data["name"],
                owner=data["owner"],
                type=TestType[data["type"].upper()],
                state=data.get("state"),
                last_modified_by=data.get("last_modified_by"),
                issue_number=data.get("issue_number"),
            )
        except (
            botocore_exceptions.BotoCoreError,
            botocore_exceptions.ClientError,
            json.JSONDecodeError,
            KeyError,
        ):
            return None
