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

"""CPU-only tests for the per-feature schema-fingerprint cache in ``generate_stats``.

Without the fingerprint guard, an existing ``meta/stats.json`` was reused as
long as every float feature name was still present, even after the underlying
``info.json`` schema (``dtype`` / ``shape``) had changed -- silently degrading
normalization at training/eval time. The fingerprint hashes the per-feature
schema so any drift invalidates just that feature's cached entry.
"""

import json

from gr00t.data.stats import (
    LE_ROBOT_STATS_FILENAME,
    STATS_FINGERPRINTS_KEY,
    _compute_stats_fingerprint,
    _stale_features,
    check_stats_validity,
    generate_stats,
)
import pytest


_STATE_META = {"dtype": "float32", "shape": [17]}
_ACTION_META = {"dtype": "float32", "shape": [17]}
_TIMESTAMP_META = {"dtype": "float32", "shape": [1]}
_TASK_META = {"dtype": "int64", "shape": [1]}


def _info_json(features: dict) -> dict:
    return {
        "codebase_version": "v2.1",
        "robot_type": "test",
        "total_episodes": 1,
        "total_frames": 0,
        "fps": 15,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "chunks_size": 1000,
        "splits": {"train": "0:1"},
        "features": features,
    }


def _write_meta(dataset_path, features: dict, stats: dict | None = None) -> None:
    meta = dataset_path / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(_info_json(features)))
    if stats is not None:
        (meta / "stats.json").write_text(json.dumps(stats))


def _stub_stat_dict(dim: int = 17) -> dict[str, list[float]]:
    return {
        "mean": [0.0] * dim,
        "std": [1.0] * dim,
        "min": [-1.0] * dim,
        "max": [1.0] * dim,
        "q01": [-0.99] * dim,
        "q99": [0.99] * dim,
    }


@pytest.fixture
def dataset(tmp_path):
    return tmp_path


@pytest.fixture
def lowdim_features():
    return {
        "observation.state": _STATE_META,
        "action": _ACTION_META,
        "timestamp": _TIMESTAMP_META,
    }


@pytest.fixture
def mock_calculate(monkeypatch):
    """Replace the heavy parquet-driven computation with a counter-stub.

    Returns a list of (features-arg) calls so tests can assert exactly which
    features were recomputed on each ``generate_stats`` invocation.
    """
    calls: list[list[str]] = []

    def fake(parquet_paths, features=None):
        calls.append(list(features) if features is not None else [])
        return {f: _stub_stat_dict() for f in (features or [])}

    monkeypatch.setattr("gr00t.data.stats.calculate_dataset_statistics", fake)
    return calls


# ---------------------------------------------------------------------------
# _compute_stats_fingerprint -- pure helper, no I/O
# ---------------------------------------------------------------------------


class TestStatsFingerprintHelper:
    def test_deterministic(self):
        a = _compute_stats_fingerprint("action", _ACTION_META)
        b = _compute_stats_fingerprint("action", _ACTION_META)
        assert a == b

    def test_format_prefix(self):
        fp = _compute_stats_fingerprint("action", _ACTION_META)
        assert fp.startswith("sha256:") and len(fp) == len("sha256:") + 64

    def test_distinct_per_feature(self):
        a = _compute_stats_fingerprint("action", _ACTION_META)
        b = _compute_stats_fingerprint("observation.state", _STATE_META)
        assert a != b

    def test_changes_with_dtype(self):
        baseline = _compute_stats_fingerprint("action", {"dtype": "float32", "shape": [17]})
        widened = _compute_stats_fingerprint("action", {"dtype": "float64", "shape": [17]})
        assert baseline != widened

    def test_changes_with_shape(self):
        baseline = _compute_stats_fingerprint("action", {"dtype": "float32", "shape": [17]})
        grown = _compute_stats_fingerprint("action", {"dtype": "float32", "shape": [21]})
        assert baseline != grown


# ---------------------------------------------------------------------------
# _stale_features -- targeted behaviour with hand-shaped stats payloads
# ---------------------------------------------------------------------------


class TestStaleFeatures:
    def test_none_stats_returns_full_list(self, lowdim_features):
        stale = _stale_features(None, lowdim_features, list(lowdim_features))
        assert stale == list(lowdim_features)

    def test_empty_stats_returns_full_list(self, lowdim_features):
        assert _stale_features({}, lowdim_features, list(lowdim_features)) == list(lowdim_features)

    def test_full_match_returns_empty(self, lowdim_features):
        stats = {f: _stub_stat_dict(meta["shape"][0]) for f, meta in lowdim_features.items()}
        stats[STATS_FINGERPRINTS_KEY] = {
            f: _compute_stats_fingerprint(f, meta) for f, meta in lowdim_features.items()
        }
        assert _stale_features(stats, lowdim_features, list(lowdim_features)) == []

    def test_mismatched_fingerprint_marks_only_that_feature_stale(self, lowdim_features):
        stats = {f: _stub_stat_dict(meta["shape"][0]) for f, meta in lowdim_features.items()}
        stats[STATS_FINGERPRINTS_KEY] = {
            f: _compute_stats_fingerprint(f, meta) for f, meta in lowdim_features.items()
        }
        stats[STATS_FINGERPRINTS_KEY]["action"] = "sha256:" + "0" * 64
        assert _stale_features(stats, lowdim_features, list(lowdim_features)) == ["action"]

    def test_missing_fingerprint_marks_feature_stale(self, lowdim_features):
        stats = {f: _stub_stat_dict(meta["shape"][0]) for f, meta in lowdim_features.items()}
        stats[STATS_FINGERPRINTS_KEY] = {
            f: _compute_stats_fingerprint(f, meta) for f, meta in lowdim_features.items()
        }
        del stats[STATS_FINGERPRINTS_KEY]["timestamp"]
        assert _stale_features(stats, lowdim_features, list(lowdim_features)) == ["timestamp"]

    def test_missing_stat_field_marks_feature_stale(self, lowdim_features):
        stats = {f: _stub_stat_dict(meta["shape"][0]) for f, meta in lowdim_features.items()}
        stats[STATS_FINGERPRINTS_KEY] = {
            f: _compute_stats_fingerprint(f, meta) for f, meta in lowdim_features.items()
        }
        del stats["action"]["q01"]
        assert _stale_features(stats, lowdim_features, list(lowdim_features)) == ["action"]

    def test_legacy_file_without_fingerprints_marks_all_stale(self, lowdim_features):
        legacy = {f: _stub_stat_dict(meta["shape"][0]) for f, meta in lowdim_features.items()}
        assert _stale_features(legacy, lowdim_features, list(lowdim_features)) == list(
            lowdim_features
        )


# ---------------------------------------------------------------------------
# check_stats_validity -- end-to-end disk read with info.json reconciliation
# ---------------------------------------------------------------------------


class TestCheckStatsValidity:
    def test_returns_false_when_stats_missing(self, dataset, lowdim_features):
        _write_meta(dataset, lowdim_features)
        assert not check_stats_validity(dataset, list(lowdim_features))

    def test_returns_false_when_info_missing(self, dataset, lowdim_features):
        # Synthesize a fingerprint-complete stats.json but omit info.json -- without
        # info.json we cannot recompute the expected fingerprint and must not
        # trust the cache.
        stats = {f: _stub_stat_dict(meta["shape"][0]) for f, meta in lowdim_features.items()}
        stats[STATS_FINGERPRINTS_KEY] = {
            f: _compute_stats_fingerprint(f, meta) for f, meta in lowdim_features.items()
        }
        (dataset / "meta").mkdir(parents=True)
        (dataset / "meta" / "stats.json").write_text(json.dumps(stats))
        assert not check_stats_validity(dataset, list(lowdim_features))

    def test_returns_true_on_full_match(self, dataset, lowdim_features):
        stats = {f: _stub_stat_dict(meta["shape"][0]) for f, meta in lowdim_features.items()}
        stats[STATS_FINGERPRINTS_KEY] = {
            f: _compute_stats_fingerprint(f, meta) for f, meta in lowdim_features.items()
        }
        _write_meta(dataset, lowdim_features, stats)
        assert check_stats_validity(dataset, list(lowdim_features))

    def test_returns_false_when_dtype_drifts(self, dataset, lowdim_features):
        stats = {f: _stub_stat_dict(meta["shape"][0]) for f, meta in lowdim_features.items()}
        stats[STATS_FINGERPRINTS_KEY] = {
            f: _compute_stats_fingerprint(f, meta) for f, meta in lowdim_features.items()
        }
        # info.json now reports float64 for `action`; cached fingerprint was float32.
        drifted_features = {**lowdim_features, "action": {"dtype": "float64", "shape": [17]}}
        _write_meta(dataset, drifted_features, stats)
        assert not check_stats_validity(dataset, list(drifted_features))

    def test_returns_false_when_shape_drifts(self, dataset, lowdim_features):
        stats = {f: _stub_stat_dict(meta["shape"][0]) for f, meta in lowdim_features.items()}
        stats[STATS_FINGERPRINTS_KEY] = {
            f: _compute_stats_fingerprint(f, meta) for f, meta in lowdim_features.items()
        }
        drifted_features = {**lowdim_features, "action": {"dtype": "float32", "shape": [21]}}
        _write_meta(dataset, drifted_features, stats)
        assert not check_stats_validity(dataset, list(drifted_features))

    def test_returns_false_when_feature_absent_from_info(self, dataset, lowdim_features):
        stats = {f: _stub_stat_dict(meta["shape"][0]) for f, meta in lowdim_features.items()}
        stats[STATS_FINGERPRINTS_KEY] = {
            f: _compute_stats_fingerprint(f, meta) for f, meta in lowdim_features.items()
        }
        _write_meta(dataset, lowdim_features, stats)
        # Caller asks about a feature that does not exist in info.json.
        assert not check_stats_validity(dataset, ["never_seen_feature"])


# ---------------------------------------------------------------------------
# generate_stats -- end-to-end caching with mocked compute
# ---------------------------------------------------------------------------


class TestGenerateStatsCache:
    def test_first_run_computes_and_persists_fingerprints(
        self, dataset, lowdim_features, mock_calculate
    ):
        _write_meta(dataset, lowdim_features)
        generate_stats(dataset)

        assert sorted(mock_calculate[-1]) == sorted(lowdim_features)
        payload = json.loads((dataset / LE_ROBOT_STATS_FILENAME).read_text())
        assert STATS_FINGERPRINTS_KEY in payload
        for f, meta in lowdim_features.items():
            assert f in payload
            assert payload[STATS_FINGERPRINTS_KEY][f] == _compute_stats_fingerprint(f, meta)

    def test_second_run_is_full_cache_hit(self, dataset, lowdim_features, mock_calculate):
        _write_meta(dataset, lowdim_features)
        generate_stats(dataset)
        mock_calculate.clear()

        generate_stats(dataset)

        assert mock_calculate == [], "fresh fingerprints must produce zero recompute"

    def test_dtype_change_recomputes_only_that_feature(
        self, dataset, lowdim_features, mock_calculate
    ):
        _write_meta(dataset, lowdim_features)
        generate_stats(dataset)
        mock_calculate.clear()

        # info.json now reports float64 for `action`; cached fingerprint was float32.
        drifted = {**lowdim_features, "action": {"dtype": "float64", "shape": [17]}}
        (dataset / "meta" / "info.json").write_text(json.dumps(_info_json(drifted)))

        generate_stats(dataset)

        assert mock_calculate == [["action"]], (
            f"only `action` should recompute on dtype change; got {mock_calculate}"
        )
        payload = json.loads((dataset / LE_ROBOT_STATS_FILENAME).read_text())
        assert payload[STATS_FINGERPRINTS_KEY]["action"] == _compute_stats_fingerprint(
            "action", drifted["action"]
        )

    def test_new_feature_in_info_recomputes_only_that_feature(
        self, dataset, lowdim_features, mock_calculate
    ):
        _write_meta(dataset, lowdim_features)
        generate_stats(dataset)
        mock_calculate.clear()

        added = {**lowdim_features, "extra": {"dtype": "float32", "shape": [3]}}
        (dataset / "meta" / "info.json").write_text(json.dumps(_info_json(added)))

        generate_stats(dataset)

        assert mock_calculate == [["extra"]]
        payload = json.loads((dataset / LE_ROBOT_STATS_FILENAME).read_text())
        assert "extra" in payload
        assert "extra" in payload[STATS_FINGERPRINTS_KEY]

    def test_legacy_stats_without_fingerprints_recomputes_all(
        self, dataset, lowdim_features, mock_calculate
    ):
        legacy = {f: _stub_stat_dict(meta["shape"][0]) for f, meta in lowdim_features.items()}
        _write_meta(dataset, lowdim_features, legacy)

        generate_stats(dataset)

        assert sorted(mock_calculate[-1]) == sorted(lowdim_features)
        payload = json.loads((dataset / LE_ROBOT_STATS_FILENAME).read_text())
        for f in lowdim_features:
            assert f in payload[STATS_FINGERPRINTS_KEY]

    def test_partial_cache_only_recomputes_stale(self, dataset, lowdim_features, mock_calculate):
        partial = {f: _stub_stat_dict(meta["shape"][0]) for f, meta in lowdim_features.items()}
        partial[STATS_FINGERPRINTS_KEY] = {
            "observation.state": _compute_stats_fingerprint(
                "observation.state", lowdim_features["observation.state"]
            ),
            "timestamp": _compute_stats_fingerprint("timestamp", lowdim_features["timestamp"]),
        }
        _write_meta(dataset, lowdim_features, partial)

        generate_stats(dataset)

        assert mock_calculate == [["action"]], (
            f"only `action` (no fingerprint) should be recomputed; got {mock_calculate}"
        )

    def test_feature_removed_from_info_drops_phantom_entries(
        self, dataset, lowdim_features, mock_calculate
    ):
        """A feature dropped from ``info.json`` (e.g. sensor / DOF removed
        in an upstream dataset rev) must not survive on disk as a
        phantom stat dict or fingerprint — under feature churn on shared
        NFS that grows ``stats.json`` indefinitely and leaves it
        inconsistent with ``info.json``."""
        _write_meta(dataset, lowdim_features)
        generate_stats(dataset)
        mock_calculate.clear()

        shrunk = {k: v for k, v in lowdim_features.items() if k != "action"}
        (dataset / "meta" / "info.json").write_text(json.dumps(_info_json(shrunk)))

        generate_stats(dataset)

        assert mock_calculate == [], "no recompute needed when only removals happened"
        payload = json.loads((dataset / LE_ROBOT_STATS_FILENAME).read_text())
        assert "action" not in payload, "stat dict for removed feature must be dropped"
        assert "action" not in payload[STATS_FINGERPRINTS_KEY], (
            "fingerprint for removed feature must be dropped"
        )
        for f in shrunk:
            assert f in payload
            assert f in payload[STATS_FINGERPRINTS_KEY]

    def test_corrupt_fingerprint_dict_is_self_healed(
        self, dataset, lowdim_features, mock_calculate
    ):
        # ``__fingerprints__`` is a top-level reserved key; if a bug or external
        # writer ever sets it to something non-dict (e.g. a list), generate_stats
        # must recover rather than crash.
        broken = {f: _stub_stat_dict(meta["shape"][0]) for f, meta in lowdim_features.items()}
        broken[STATS_FINGERPRINTS_KEY] = ["not", "a", "dict"]
        _write_meta(dataset, lowdim_features, broken)

        generate_stats(dataset)

        payload = json.loads((dataset / LE_ROBOT_STATS_FILENAME).read_text())
        assert isinstance(payload[STATS_FINGERPRINTS_KEY], dict)
        for f in lowdim_features:
            assert f in payload[STATS_FINGERPRINTS_KEY]
