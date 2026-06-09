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

"""CPU-only tests for the schema-fingerprint cache in ``generate_rel_stats``.

Without the fingerprint guard, an existing ``meta/relative_stats.json`` was
reused whenever a per-embodiment ``action_key`` name matched, regardless of
whether the inputs that drive the computation (``delta_indices``, ``format``,
``state_key``, ...) had since changed — silently corrupting normalization at
training time.
"""

from dataclasses import replace
import json
from unittest.mock import patch

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import (
    LE_ROBOT_REL_STATS_FILENAME,
    STATS_FINGERPRINTS_KEY,
    _compute_relative_action_fingerprint,
    generate_rel_stats,
)
from gr00t.data.types import ActionFormat
import numpy as np
import pytest


# Real registered embodiment with two RELATIVE action keys (eef_9d, joint_position)
# and one ABSOLUTE (gripper_position) — exercises the cache path without
# needing parquet data once calculate_stats_for_key is mocked.
EMBODIMENT = EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
RELATIVE_KEYS = ("eef_9d", "joint_position")


def _stub_stats():
    """Synthetic stats payload shaped like calculate_stats_for_key's return."""
    return {
        "max": np.ones(9, dtype=np.float32),
        "min": -np.ones(9, dtype=np.float32),
        "q01": -np.ones(9, dtype=np.float32) * 0.99,
        "q99": np.ones(9, dtype=np.float32) * 0.99,
        "mean": np.zeros(9, dtype=np.float32),
        "std": np.ones(9, dtype=np.float32),
    }


@pytest.fixture
def dataset_dir(tmp_path):
    (tmp_path / "meta").mkdir()
    return tmp_path


@pytest.fixture
def mock_calculate(monkeypatch):
    """Replace the heavy parquet-driven computation with a counter-stub."""
    calls = []

    def fake(dataset_path, embodiment_tag, group_key, max_episodes=-1):
        calls.append((str(dataset_path), embodiment_tag.value, group_key))
        return _stub_stats()

    monkeypatch.setattr("gr00t.data.stats.calculate_stats_for_key", fake)
    return calls


# ---------------------------------------------------------------------------
# _compute_relative_action_fingerprint — pure helper, no I/O
# ---------------------------------------------------------------------------


class TestFingerprintHelper:
    def test_deterministic(self):
        a = _compute_relative_action_fingerprint(EMBODIMENT, "eef_9d")
        b = _compute_relative_action_fingerprint(EMBODIMENT, "eef_9d")
        assert a == b

    def test_format_prefix(self):
        fp = _compute_relative_action_fingerprint(EMBODIMENT, "eef_9d")
        assert fp.startswith("sha256:") and len(fp) == len("sha256:") + 64

    def test_distinct_per_action_key(self):
        a = _compute_relative_action_fingerprint(EMBODIMENT, "eef_9d")
        b = _compute_relative_action_fingerprint(EMBODIMENT, "joint_position")
        assert a != b

    @pytest.mark.parametrize(
        "mutator",
        [
            pytest.param(
                lambda cfg: replace(cfg, format=ActionFormat.XYZ_ROTVEC),
                id="format-change",
            ),
            pytest.param(
                lambda cfg: replace(cfg, state_key="other_state_key"),
                id="state-key-change",
            ),
        ],
    )
    def test_changes_when_action_config_changes(self, mutator):
        """Mutating fields the loader actually reads must invalidate the cache."""
        cfgs = MODALITY_CONFIGS[EMBODIMENT.value]
        action_modality = cfgs["action"]
        idx = action_modality.modality_keys.index("eef_9d")
        original_action_config = action_modality.action_configs[idx]

        baseline = _compute_relative_action_fingerprint(EMBODIMENT, "eef_9d")

        new_configs = list(action_modality.action_configs)
        new_configs[idx] = mutator(original_action_config)
        with patch.dict(
            MODALITY_CONFIGS[EMBODIMENT.value]["action"].__dict__,
            {"action_configs": new_configs},
        ):
            mutated = _compute_relative_action_fingerprint(EMBODIMENT, "eef_9d")

        assert mutated != baseline

    def test_changes_when_delta_indices_change(self):
        baseline = _compute_relative_action_fingerprint(EMBODIMENT, "eef_9d")
        action_modality = MODALITY_CONFIGS[EMBODIMENT.value]["action"]
        with patch.dict(
            action_modality.__dict__,
            {"delta_indices": [0, 1, 2]},
        ):
            mutated = _compute_relative_action_fingerprint(EMBODIMENT, "eef_9d")
        assert mutated != baseline


# ---------------------------------------------------------------------------
# generate_rel_stats — cache hit / miss / mixed via mocked calculate_stats_for_key
# ---------------------------------------------------------------------------


class TestGenerateRelStatsCache:
    def test_first_run_computes_and_persists_fingerprints(self, dataset_dir, mock_calculate):
        generate_rel_stats(dataset_dir, EMBODIMENT)

        assert sorted(c[2] for c in mock_calculate) == sorted(RELATIVE_KEYS)

        with open(dataset_dir / LE_ROBOT_REL_STATS_FILENAME) as f:
            payload = json.load(f)
        assert STATS_FINGERPRINTS_KEY in payload
        for key in RELATIVE_KEYS:
            assert key in payload
            assert key in payload[STATS_FINGERPRINTS_KEY]
            assert payload[STATS_FINGERPRINTS_KEY][key].startswith("sha256:")

    def test_second_run_is_full_cache_hit(self, dataset_dir, mock_calculate):
        generate_rel_stats(dataset_dir, EMBODIMENT)
        mock_calculate.clear()

        generate_rel_stats(dataset_dir, EMBODIMENT)

        assert mock_calculate == [], "fresh fingerprints must produce zero recompute"

    def test_legacy_file_without_fingerprints_is_regenerated(self, dataset_dir, mock_calculate):
        """Pre-existing relative_stats.json from before this fix must be recomputed."""
        legacy_payload = {key: {"_legacy": True} for key in RELATIVE_KEYS}
        with open(dataset_dir / LE_ROBOT_REL_STATS_FILENAME, "w") as f:
            json.dump(legacy_payload, f)

        generate_rel_stats(dataset_dir, EMBODIMENT)

        assert sorted(c[2] for c in mock_calculate) == sorted(RELATIVE_KEYS)
        with open(dataset_dir / LE_ROBOT_REL_STATS_FILENAME) as f:
            payload = json.load(f)
        for key in RELATIVE_KEYS:
            assert "_legacy" not in payload[key]
            assert payload[STATS_FINGERPRINTS_KEY][key].startswith("sha256:")

    def test_stale_fingerprint_triggers_regenerate(self, dataset_dir, mock_calculate):
        generate_rel_stats(dataset_dir, EMBODIMENT)

        with open(dataset_dir / LE_ROBOT_REL_STATS_FILENAME) as f:
            payload = json.load(f)
        payload[STATS_FINGERPRINTS_KEY]["eef_9d"] = "sha256:" + "0" * 64
        with open(dataset_dir / LE_ROBOT_REL_STATS_FILENAME, "w") as f:
            json.dump(payload, f)

        mock_calculate.clear()
        generate_rel_stats(dataset_dir, EMBODIMENT)

        recomputed = [c[2] for c in mock_calculate]
        assert recomputed == ["eef_9d"], f"only the stale key should regenerate; got {recomputed}"

    def test_modality_config_change_invalidates_cache(self, dataset_dir, mock_calculate):
        """The whole point of the fingerprint: silently changed config must not get a stale cache hit."""
        generate_rel_stats(dataset_dir, EMBODIMENT)
        mock_calculate.clear()

        action_modality = MODALITY_CONFIGS[EMBODIMENT.value]["action"]
        idx = action_modality.modality_keys.index("eef_9d")
        new_configs = list(action_modality.action_configs)
        new_configs[idx] = replace(new_configs[idx], format=ActionFormat.XYZ_ROTVEC)
        with patch.dict(action_modality.__dict__, {"action_configs": new_configs}):
            generate_rel_stats(dataset_dir, EMBODIMENT)

        recomputed = [c[2] for c in mock_calculate]
        assert "eef_9d" in recomputed, "format change must trigger eef_9d regeneration"
        assert "joint_position" not in recomputed, (
            "joint_position config did not change; must not be regenerated"
        )

    def test_partial_cache_only_recomputes_missing(self, dataset_dir, mock_calculate):
        """Pre-fill cache for one key only; the other should be the only one computed."""
        eef_fp = _compute_relative_action_fingerprint(EMBODIMENT, "eef_9d")
        prefilled = {
            "eef_9d": {k: v.tolist() for k, v in _stub_stats().items()},
            STATS_FINGERPRINTS_KEY: {"eef_9d": eef_fp},
        }
        with open(dataset_dir / LE_ROBOT_REL_STATS_FILENAME, "w") as f:
            json.dump(prefilled, f)

        generate_rel_stats(dataset_dir, EMBODIMENT)

        assert [c[2] for c in mock_calculate] == ["joint_position"]
        with open(dataset_dir / LE_ROBOT_REL_STATS_FILENAME) as f:
            payload = json.load(f)
        for key in RELATIVE_KEYS:
            assert key in payload
            assert key in payload[STATS_FINGERPRINTS_KEY]


# ---------------------------------------------------------------------------
# Backward compat — downstream consumers must not see __fingerprints__ as data
# ---------------------------------------------------------------------------


def test_fingerprints_key_does_not_collide_with_action_keys():
    """Reserved sentinel must not match any registered action key string anywhere."""
    for tag, cfg in MODALITY_CONFIGS.items():
        action = cfg.get("action")
        if action is None or action.action_configs is None:
            continue
        assert STATS_FINGERPRINTS_KEY not in action.modality_keys, (
            f"reserved sentinel {STATS_FINGERPRINTS_KEY!r} collides with an "
            f"action key in embodiment {tag!r}"
        )


def test_per_action_key_payload_unchanged_shape(dataset_dir, mock_calculate):
    """The per-action_key dict written to disk must contain only the stat fields,
    not the fingerprint — downstream loader iterates ``stats[key].keys()``
    expecting ``{mean, std, min, max, q01, q99}``.
    """
    generate_rel_stats(dataset_dir, EMBODIMENT)
    with open(dataset_dir / LE_ROBOT_REL_STATS_FILENAME) as f:
        payload = json.load(f)
    for key in RELATIVE_KEYS:
        assert set(payload[key].keys()) == {"mean", "std", "min", "max", "q01", "q99"}
