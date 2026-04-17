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

"""
Test generate_stats() and generate_rel_stats() using the bundled demo dataset.

These top-down tests exercise deep call chains through:
- stats.py (calculate_dataset_statistics, check_stats_validity, RelativeActionLoader)
- lerobot_episode_loader.py (parquet + video data loading)
- pose.py (EndEffectorPose, JointPose)
- action_chunking.py (relative_chunking)
- utils.py (to_json_serializable)

Dataset path: ``resolve_libero_demo_dataset_path`` (shared drive, env override, or Git LFS).
"""

import functools
import json
from pathlib import Path
import shutil

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import (
    calculate_dataset_statistics,
    check_stats_validity,
    generate_rel_stats,
    generate_stats,
)
import numpy as np
import pyarrow.parquet as pq
import pytest
from test_support.runtime import get_root, resolve_libero_demo_dataset_path


ROOT = get_root()
EMBODIMENT = EmbodimentTag("libero_sim")


def _parquet_readable(path: Path) -> bool:
    """Return True if *path* is a real parquet file (not a Git LFS pointer stub)."""
    try:
        pq.read_schema(path)
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _libero_demo_dataset() -> Path:
    return resolve_libero_demo_dataset_path(ROOT)


def _dataset_usable() -> bool:
    """Check that the demo dataset exists and its parquet files are readable."""
    try:
        ds = _libero_demo_dataset()
    except (AssertionError, FileNotFoundError):
        return False
    parquets = sorted(ds.glob("data/*/*.parquet"))
    return len(parquets) > 0 and _parquet_readable(parquets[0])


# Skip when the libero_demo dataset is not available (e.g. shallow clone without
# Git LFS, or CI runner without the shared drive mount).  The tests run in full
# CI (where LFS data is present) and on any dev machine after `git lfs pull`.
requires_libero_demo = pytest.mark.skipif(
    not _dataset_usable(),
    reason="libero_demo dataset not available or parquet files are Git LFS stubs",
)


@pytest.fixture
def tmp_dataset(tmp_path):
    """Copy the demo dataset to a temp directory so we can modify stats files."""
    dest = tmp_path / "libero_demo"
    shutil.copytree(_libero_demo_dataset(), dest)
    (dest / "meta" / "stats.json").unlink(missing_ok=True)
    (dest / "meta" / "relative_stats.json").unlink(missing_ok=True)
    return dest


@pytest.fixture
def demo_parquet_paths():
    """Sorted parquet episode files from the bundled libero demo dataset."""
    paths = sorted(_libero_demo_dataset().glob("data/*/*.parquet"))
    assert len(paths) > 0, "No parquet files found in demo dataset"
    return paths


@requires_libero_demo
class TestCalculateDatasetStatistics:
    """Test the low-level statistics computation on parquet data."""

    def test_returns_all_requested_features(self, demo_parquet_paths):
        features = ["observation.state", "action"]
        stats = calculate_dataset_statistics(demo_parquet_paths, features=features)
        assert set(stats.keys()) == set(features)

    def test_stat_keys_present(self, demo_parquet_paths):
        stats = calculate_dataset_statistics(demo_parquet_paths, features=["observation.state"])
        for stat_name in ("mean", "std", "min", "max", "q01", "q99"):
            assert stat_name in stats["observation.state"]
            assert len(stats["observation.state"][stat_name]) > 0

    def test_values_are_finite(self, demo_parquet_paths):
        stats = calculate_dataset_statistics(demo_parquet_paths, features=["observation.state"])
        for stat_name in ("mean", "std", "min", "max", "q01", "q99"):
            values = np.array(stats["observation.state"][stat_name])
            assert np.all(np.isfinite(values)), f"{stat_name} contains non-finite values"

    def test_mathematical_invariants(self, demo_parquet_paths):
        """Verify invariants that must hold for any valid dataset statistics.

        These catch computational bugs (e.g. wrong axis, wrong quantile order)
        that structural or finiteness checks would miss.
        """
        stats = calculate_dataset_statistics(
            demo_parquet_paths, features=["observation.state", "action"]
        )
        for feature in ("observation.state", "action"):
            s = {k: np.array(v) for k, v in stats[feature].items()}
            assert np.all(s["std"] >= 0), f"{feature}: std has negative values"
            assert np.all(s["min"] <= s["mean"]), f"{feature}: min > mean"
            assert np.all(s["mean"] <= s["max"]), f"{feature}: mean > max"
            assert np.all(s["min"] <= s["q01"]), f"{feature}: min > q01"
            assert np.all(s["q01"] <= s["q99"]), f"{feature}: q01 > q99"
            assert np.all(s["q99"] <= s["max"]), f"{feature}: q99 > max"
            dims = {k: len(v) for k, v in s.items()}
            assert len(set(dims.values())) == 1, (
                f"{feature}: inconsistent dimensions across stats: {dims}"
            )

    def test_auto_discovers_all_columns(self, demo_parquet_paths):
        stats = calculate_dataset_statistics(demo_parquet_paths, features=None)
        expected_columns = {"observation.state", "action", "timestamp"}
        assert expected_columns.issubset(set(stats.keys())), (
            f"Expected at least {expected_columns}, got {set(stats.keys())}"
        )


class TestCheckStatsValidity:
    """Test the stats file validity checker."""

    @requires_libero_demo
    def test_valid_existing_stats(self):
        assert check_stats_validity(_libero_demo_dataset(), ["observation.state", "action"])

    @requires_libero_demo
    def test_missing_feature_returns_false(self):
        assert not check_stats_validity(_libero_demo_dataset(), ["nonexistent_feature_xyz"])

    def test_missing_file_returns_false(self, tmp_path):
        assert not check_stats_validity(tmp_path, ["anything"])

    @requires_libero_demo
    def test_partial_features_returns_false(self):
        assert not check_stats_validity(
            _libero_demo_dataset(), ["observation.state", "no_such_key"]
        )


@requires_libero_demo
class TestGenerateStats:
    """Test end-to-end stats generation from demo dataset."""

    def test_creates_stats_file(self, tmp_dataset):
        stats_path = tmp_dataset / "meta" / "stats.json"
        assert not stats_path.exists()
        generate_stats(tmp_dataset)
        assert stats_path.exists()

    def test_generated_stats_are_mathematically_correct(self, tmp_dataset):
        """Verify generated stats match independent computation from raw parquet data."""
        import pandas as pd

        generate_stats(tmp_dataset)
        with open(tmp_dataset / "meta" / "stats.json") as f:
            generated = json.load(f)

        assert "observation.state" in generated
        assert "action" in generated

        parquet_paths = sorted(tmp_dataset.glob("data/*/*.parquet"))
        raw = pd.concat([pd.read_parquet(p) for p in parquet_paths], axis=0)

        for feature in ("observation.state", "action"):
            data = np.vstack([np.asarray(x, dtype=np.float32) for x in raw[feature]])
            expected = {
                "mean": np.mean(data, axis=0),
                "std": np.std(data, axis=0),
                "min": np.min(data, axis=0),
                "max": np.max(data, axis=0),
                "q01": np.quantile(data, 0.01, axis=0),
                "q99": np.quantile(data, 0.99, axis=0),
            }
            for stat_name, expected_vals in expected.items():
                assert stat_name in generated[feature], (
                    f"Missing {stat_name} in generated stats for {feature}"
                )
                gen_vals = np.array(generated[feature][stat_name])
                np.testing.assert_allclose(
                    gen_vals,
                    expected_vals,
                    rtol=1e-5,
                    err_msg=f"{feature}.{stat_name} incorrect",
                )

    def test_skips_when_stats_already_valid(self, tmp_dataset):
        generate_stats(tmp_dataset)
        stats_path = tmp_dataset / "meta" / "stats.json"
        mtime_before = stats_path.stat().st_mtime_ns
        generate_stats(tmp_dataset)
        mtime_after = stats_path.stat().st_mtime_ns
        assert mtime_before == mtime_after, "stats.json should not be rewritten"


@requires_libero_demo
class TestGenerateRelStats:
    """Test relative stats generation.

    libero_sim uses ABSOLUTE actions (action_configs=None), so generate_rel_stats
    should return early at the `if action_config.action_configs is None: return`
    guard without writing any file. We verify this early-return contract explicitly.
    """

    def test_early_return_and_idempotent_for_absolute_actions(self, tmp_dataset):
        """libero_sim has only ABSOLUTE actions — no relative_stats.json should be created.

        Calls generate_rel_stats twice to also verify idempotency.
        """
        rel_stats_path = tmp_dataset / "meta" / "relative_stats.json"
        generate_rel_stats(tmp_dataset, EMBODIMENT)
        assert not rel_stats_path.exists(), (
            "relative_stats.json should NOT be created for embodiments with only ABSOLUTE actions"
        )
        generate_rel_stats(tmp_dataset, EMBODIMENT)
        assert not rel_stats_path.exists(), "Idempotency broken on second call"
