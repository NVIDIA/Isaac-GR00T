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

"""Regression tests for the corrupt-stats-cache trap.

Background: a 0-byte ``meta/relative_stats.json`` (left behind by a previous
writer that was killed mid-flush — ENOSPC, SIGKILL, runner reboot) used to
poison every subsequent caller of ``generate_rel_stats`` with a
``json.JSONDecodeError``. Observed taking down 6 of 8 retried test.unit.gpu
jobs after a /shared NFS ENOSPC event (jobs 312671660-312671673).

These CPU-only tests pin the contract that ``_load_stats_cache`` and
``_dump_stats_cache_atomic`` together survive any unreadable-cache scenario
without raising and without writing a partially-truncated file the next
caller will trip over.
"""

import json
from pathlib import Path

from gr00t.data.stats import _dump_stats_cache_atomic, _load_stats_cache
import pytest


class TestLoadStatsCacheTreatsUnreadableAsMissing:
    """``_load_stats_cache`` must return ``{}`` for every unreadable state."""

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert _load_stats_cache(tmp_path / "absent.json") == {}

    def test_zero_byte_file_returns_empty(self, tmp_path: Path) -> None:
        """The exact failure shape from job 312671660: ENOSPC truncated the file."""
        p = tmp_path / "stats.json"
        p.touch()
        assert p.stat().st_size == 0
        assert _load_stats_cache(p) == {}

    def test_truncated_json_returns_empty(self, tmp_path: Path) -> None:
        """Writer killed after the opening ``{`` made it to disk."""
        p = tmp_path / "stats.json"
        p.write_text('{"action.foo": {"mea')
        assert _load_stats_cache(p) == {}

    def test_garbage_returns_empty(self, tmp_path: Path) -> None:
        """Any non-JSON content (e.g. a stray binary blob) falls back to regenerate."""
        p = tmp_path / "stats.json"
        p.write_bytes(b"\x00\x01\x02not json")
        assert _load_stats_cache(p) == {}

    def test_valid_json_returns_parsed_dict(self, tmp_path: Path) -> None:
        """The fast path must still load real caches verbatim."""
        p = tmp_path / "stats.json"
        payload = {"action.right_arm": {"mean": [1.0, 2.0], "std": [0.5, 0.5]}}
        p.write_text(json.dumps(payload))
        assert _load_stats_cache(p) == payload


class TestDumpStatsCacheAtomic:
    """``_dump_stats_cache_atomic`` must leave the destination either fully
    written or untouched — never half-written."""

    def test_writes_payload_and_removes_tmp(self, tmp_path: Path) -> None:
        """Happy path: tmp sibling must not survive a successful write."""
        p = tmp_path / "stats.json"
        payload = {"action.x": {"mean": [1.0]}}
        _dump_stats_cache_atomic(p, payload)

        assert json.loads(p.read_text()) == payload
        assert not list(tmp_path.glob("*.tmp")), "tmp sibling must not be left behind"

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """Subsequent calls replace the previous content atomically."""
        p = tmp_path / "stats.json"
        p.write_text('{"old": true}')
        _dump_stats_cache_atomic(p, {"new": True})
        assert json.loads(p.read_text()) == {"new": True}

    def test_serializer_failure_preserves_original_and_cleans_tmp(self, tmp_path: Path) -> None:
        """If json.dump raises (e.g. non-serializable value), the destination
        must still hold the old content and the tmp sibling must be unlinked.

        This is the contract that prevents a SIGKILL-during-write from
        creating the very 0-byte file this MR exists to defang.
        """
        p = tmp_path / "stats.json"
        p.write_text('{"sentinel": "preserved"}')

        with pytest.raises(TypeError):
            # ``set`` is not JSON-serializable → json.dump raises mid-write.
            _dump_stats_cache_atomic(p, {"bad": {1, 2, 3}})

        assert json.loads(p.read_text()) == {"sentinel": "preserved"}, (
            "atomic-write contract violated: destination was modified despite "
            "the writer raising before completion"
        )
        assert not list(tmp_path.glob("*.tmp")), (
            "tmp sibling must be cleaned up after a write failure to avoid "
            "littering meta/ with abandoned shards"
        )


class TestRoundtripSurvivesCorruption:
    """End-to-end: a corrupt cache from the previous run must not block the
    next writer from producing a valid one."""

    def test_load_then_dump_replaces_corrupt_file(self, tmp_path: Path) -> None:
        """Mirrors the production sequence in ``generate_rel_stats``:
        read existing cache (possibly corrupt) → fill in missing keys →
        write atomically. The new file must be a valid JSON dict, not an
        accidental concatenation of the old garbage and the new payload.
        """
        p = tmp_path / "stats.json"
        p.write_bytes(b"")  # 0-byte corruption, the smoking-gun failure mode.

        stats = _load_stats_cache(p)
        assert stats == {}
        stats["action.regenerated"] = {"mean": [42.0]}
        _dump_stats_cache_atomic(p, stats)

        assert json.loads(p.read_text()) == {"action.regenerated": {"mean": [42.0]}}
