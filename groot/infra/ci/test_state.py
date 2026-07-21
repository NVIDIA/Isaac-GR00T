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

"""Test state query functions for the CI/CD pipeline."""
from __future__ import annotations

import asyncio
import datetime as dt
from datetime import datetime
import enum
import json
from typing import TYPE_CHECKING

from groot.infra.ci import db, log, test_run
from groot.infra.ci import test as ci_test

if TYPE_CHECKING:
    from groot.infra.ci import issue as ci_issue

logger = log.init(__name__)

_SLOW_RATIO_THRESHOLD = 0.60  # 60% of runs must exceed the threshold
_TEST_RUN_SUFFIX = ".json"
_FALLBACK_READ_BATCH_SIZE = 2
_FALLBACK_READ_MAX_ATTEMPTS = 5


class TestState(enum.Enum):
    PASSING = "passing"
    FAILING = "failing"
    CONSISTENTLY_FAILING = "consistently_failing"
    FLAKY = "flaky"
    ZOMBIE = "zombie"
    SLOTH = "sloth"  # slow but passing; degrades to bad states as needed


# Per-type slow thresholds (seconds). 80% of runs must exceed this to be SLOTH.
SLOW_THRESHOLD_BY_TYPE: dict[ci_test.TestType, int] = {
    ci_test.TestType.UNIT: 300,  # 5 minutes
    ci_test.TestType.INTEGRATION: 900,  # 15 minutes
    ci_test.TestType.E2E: 86400,  # 24 hours
}


def is_slow(active_runs: list, test_type: ci_test.TestType | None) -> bool:
    """Return True if 80%+ of non-SKIPPED runs exceed the per-type slow threshold."""
    if not active_runs or test_type is None:
        return False
    threshold = SLOW_THRESHOLD_BY_TYPE.get(test_type)
    if threshold is None:
        return False
    slow_count = sum(1 for r in active_runs if r.duration > threshold)
    return slow_count / len(active_runs) >= _SLOW_RATIO_THRESHOLD


def _is_zombie(issue: ci_issue.Issue | None, threshold_seconds: int) -> bool:
    """Check if an open issue is older than threshold_seconds."""
    if issue is None:
        return False
    now = datetime.now(dt.timezone.utc)
    created_at_str = getattr(issue, "created_at", "") or ""
    if not created_at_str:
        return False
    try:
        created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        age = (now - created_at).total_seconds()
        return age > threshold_seconds
    except (ValueError, AttributeError):
        return False


class TestStateDB:
    def __init__(self):
        self._db = db.Database()

    @staticmethod
    def _test_run_key_sort_value(key: str) -> tuple[int, int | str]:
        """Return a descending-sortable value for a test-run S3 key.

        Test run keys end with ``_{run_id}.json``. In CI, ``run_id`` is the
        GitLab job id, which is monotonically increasing. Local runs use
        ``local-{timestamp_ms}``. Unknown legacy shapes fall back to the key
        string so callers still get deterministic behavior.
        """
        stem = key.removesuffix(_TEST_RUN_SUFFIX)
        run_id = stem.rsplit("_", 1)[-1]
        if run_id.startswith("local-"):
            try:
                return (2, int(run_id.removeprefix("local-")))
            except ValueError:
                return (0, key)
        try:
            return (2, int(run_id))
        except ValueError:
            return (0, key)

    @staticmethod
    def _filter_runs(
        runs: list[test_run.TestRun], n: int, max_age_days: float
    ) -> list[test_run.TestRun]:
        runs.sort(key=lambda r: r.timestamp, reverse=True)
        if max_age_days:
            cutoff = datetime.now(dt.timezone.utc) - dt.timedelta(days=max_age_days)
            runs = [r for r in runs if r.timestamp >= cutoff]
        return runs[:n]

    async def _read_run_files_until(
        self, keys: list[str], n: int, test: ci_test.Test
    ) -> list[test_run.TestRun]:
        runs = []
        pending_keys = list(keys)
        attempts_by_key: dict[str, int] = {}
        while pending_keys and len(runs) < n:
            remaining = n - len(runs)
            batch_size = min(_FALLBACK_READ_BATCH_SIZE, remaining)
            batch_keys = pending_keys[:batch_size]
            pending_keys = pending_keys[batch_size:]
            run_jsons = await asyncio.gather(
                *[self._db.async_read(k) for k in batch_keys], return_exceptions=True
            )

            retry_keys = []
            for key, result in zip(batch_keys, run_jsons, strict=True):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to read a test run: {result}")
                    attempts = attempts_by_key.get(key, 1)
                    if attempts < _FALLBACK_READ_MAX_ATTEMPTS:
                        attempts_by_key[key] = attempts + 1
                        retry_keys.append(key)
                    continue
                try:
                    data = json.loads(result)
                    runs.append(test_run.TestRun.init_from_dict(data))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse a test run: {e}")
                    continue

            if retry_keys:
                pending_keys.extend(retry_keys)
                await asyncio.sleep(min(2.0, 0.25 * max(attempts_by_key[k] for k in retry_keys)))

        if len(runs) < n:
            logger.info(
                "Read %d/%d requested run(s) for %s after exhausting %d key(s)",
                len(runs),
                n,
                test.get_canonical_name(),
                len(keys),
            )
        return runs

    async def query_tests(
        self, test_type: ci_test.TestType, page: int = 1, page_size: int = 50
    ) -> tuple[list[ci_test.Test], int]:
        """Query tests of a given type from S3, paginated."""
        return await ci_test.Test.get_all_paginated(test_type, page=page, page_size=page_size)

    async def query_last_n_results(
        self, test: ci_test.Test, n: int = 30, max_age_days: float = 7
    ) -> list[test_run.TestRun]:
        """Query the last N test run results for a given test, sorted by timestamp desc.

        Args:
            test: The test to query runs for.
            n: Maximum number of runs to return.
            max_age_days: Discard runs older than this many days. Default 7.

        Returns:
            List of TestRun objects sorted by timestamp descending.
        """
        prefix = f"rws/ci/test_runs/{test.type.value}/{test.get_canonical_name()}_"
        keys = await self._db.async_list(prefix)

        if not keys:
            return []

        sorted_keys = sorted(keys, key=self._test_run_key_sort_value, reverse=True)
        logger.info(
            "Found %d run files for %s, reading until %d successful run(s)...",
            len(keys),
            test.get_canonical_name(),
            n,
        )

        runs = await self._read_run_files_until(sorted_keys, n=n, test=test)

        return self._filter_runs(runs, n=n, max_age_days=max_age_days)

    def determine_state(
        self,
        test_runs: list,
        issue: ci_issue.Issue | None,
        current_state: TestState | None = None,
        test_type: ci_test.TestType | None = None,
    ) -> TestState:
        """Determine the state of a test based on its recent runs.

        Args:
            test_runs: List of TestRun objects (may include SKIPPED)
            issue: Open GitLab issue for this test, or None
            current_state: The current persisted TestState, if known. Used for ZOMBIE exit logic.
            test_type: The type of test, used for slow threshold determination.

        Returns:
            TestState enum value
        """
        # Filter out SKIPPED runs
        active_runs = [r for r in test_runs if r.status != test_run.TestStatus.SKIPPED]

        if not active_runs:
            return TestState.PASSING

        # If currently ZOMBIE, require 10 consecutive non-SKIPPED PASSed runs to exit.
        # A healed-but-slow test exits to SLOTH rather than PASSING.
        if current_state == TestState.ZOMBIE:
            consecutive_passes = 0
            for run in active_runs:
                if run.status == test_run.TestStatus.PASSED:
                    consecutive_passes += 1
                else:
                    break
            if consecutive_passes >= 10:
                return TestState.SLOTH if is_slow(active_runs, test_type) else TestState.PASSING
            return TestState.ZOMBIE

        # Last 3 non-SKIPPED all FAILED → CONSISTENTLY_FAILING, or ZOMBIE as soon as
        # the consistently-failing issue exists.
        last_3 = active_runs[:3]
        if len(last_3) >= 3 and all(r.status == test_run.TestStatus.FAILED for r in last_3):
            if issue is not None:
                return TestState.ZOMBIE
            return TestState.CONSISTENTLY_FAILING

        # Count transitions for FLAKY detection
        transitions = 0
        for i in range(len(active_runs) - 1):
            curr = active_runs[i].status
            nxt = active_runs[i + 1].status
            if (curr == test_run.TestStatus.PASSED and nxt == test_run.TestStatus.FAILED) or (
                curr == test_run.TestStatus.FAILED and nxt == test_run.TestStatus.PASSED
            ):
                transitions += 1

        if transitions / len(active_runs) > 0.10:
            if _is_zombie(issue, threshold_seconds=172800):
                return TestState.ZOMBIE
            return TestState.FLAKY

        # Latest non-SKIPPED FAILED → FAILING
        if active_runs[0].status == test_run.TestStatus.FAILED:
            return TestState.FAILING

        # Slow but passing → SLOTH (other states take precedence; E2E excluded)
        if is_slow(active_runs, test_type):
            return TestState.SLOTH

        return TestState.PASSING
