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
Test StateActionProcessor: normalization, denormalization, sin/cos encoding.

Uses the same modality configs and statistics from the test fixtures directory
(libero_sim embodiment).
"""

import json
from pathlib import Path

from gr00t.data.state_action.state_action_processor import StateActionProcessor
import numpy as np
import pytest


FIXTURE_DIR = Path(__file__).parent.parent.parent.parent / "fixtures" / "processor_config"
EMBODIMENT = "libero_sim"


def _load_fixture_configs():
    """Load modality configs and statistics from test fixtures."""
    with open(FIXTURE_DIR / "processor_config.json") as f:
        proc_config = json.load(f)
    modality_configs = proc_config["processor_kwargs"]["modality_configs"]

    with open(FIXTURE_DIR / "statistics.json") as f:
        statistics = json.load(f)
    return modality_configs, statistics


@pytest.fixture
def fixture_data():
    modality_configs, statistics = _load_fixture_configs()
    state_keys = modality_configs[EMBODIMENT]["state"]["modality_keys"]
    action_keys = modality_configs[EMBODIMENT]["action"]["modality_keys"]
    proc = StateActionProcessor(
        modality_configs=modality_configs,
        statistics=statistics,
        use_percentiles=False,
        clip_outliers=True,
    )
    return proc, state_keys, action_keys, statistics


@pytest.fixture
def processor(fixture_data):
    return fixture_data[0]


@pytest.fixture
def modality_keys(fixture_data):
    return fixture_data[1], fixture_data[2]


@pytest.fixture
def statistics(fixture_data):
    return fixture_data[3]


def _get_dim_from_stats(statistics, embodiment, modality, key):
    """Get the dimension of a key from its statistics (length of min array)."""
    return len(statistics[embodiment][modality][key]["min"])


def _random_state(state_keys, statistics):
    """Create random state dict with dimensions matching statistics."""
    return {
        k: np.random.randn(1, _get_dim_from_stats(statistics, EMBODIMENT, "state", k)).astype(
            np.float32
        )
        for k in state_keys
    }


def _random_action(action_keys, statistics, horizon=16):
    """Create random action dict with dimensions matching statistics."""
    return {
        k: np.random.randn(
            horizon, _get_dim_from_stats(statistics, EMBODIMENT, "action", k)
        ).astype(np.float32)
        for k in action_keys
    }


class TestStateNormalization:
    """Test state normalization and denormalization roundtrip."""

    def test_apply_state_returns_all_keys(self, processor, modality_keys, statistics):
        state_keys, _ = modality_keys
        raw = _random_state(state_keys, statistics)
        result = processor.apply_state(raw, EMBODIMENT)
        assert set(result.keys()) == set(state_keys)

    def test_normalized_state_within_bounds(self, processor, modality_keys, statistics):
        state_keys, _ = modality_keys
        raw = _random_state(state_keys, statistics)
        result = processor.apply_state(raw, EMBODIMENT)
        for key, val in result.items():
            assert val.min() >= -1.0, f"{key}: value {val.min()} < -1"
            assert val.max() <= 1.0, f"{key}: value {val.max()} > 1"

    def test_state_roundtrip(self, processor, modality_keys, statistics):
        """Normalize then denormalize should recover original values."""
        state_keys, _ = modality_keys
        raw = _random_state(state_keys, statistics)
        # Keep values within normalization range to avoid clipping
        for k in raw:
            stats = statistics[EMBODIMENT]["state"][k]
            low = np.array(stats["min"])
            high = np.array(stats["max"])
            raw[k] = low + (high - low) * np.random.rand(*raw[k].shape).astype(np.float32)
        normalized = processor.apply_state(raw, EMBODIMENT)
        recovered = processor.unapply_state(normalized, EMBODIMENT)
        for key in state_keys:
            np.testing.assert_allclose(
                recovered[key], raw[key], atol=1e-4, err_msg=f"Roundtrip failed for {key}"
            )

    def test_missing_key_raises(self, processor):
        with pytest.raises(KeyError, match="not found in state dict"):
            processor.apply_state({"nonexistent": np.zeros(1)}, EMBODIMENT)


class TestActionNormalization:
    """Test action normalization and denormalization."""

    def test_apply_action_returns_all_keys(self, processor, modality_keys, statistics):
        _, action_keys = modality_keys
        raw = _random_action(action_keys, statistics)
        result = processor.apply_action(raw, EMBODIMENT)
        assert set(result.keys()) == set(action_keys)

    def test_normalized_action_clipped(self, processor, modality_keys, statistics):
        _, action_keys = modality_keys
        raw = _random_action(action_keys, statistics)
        result = processor.apply_action(raw, EMBODIMENT)
        for key, val in result.items():
            assert val.min() >= -1.0, f"{key}: action value {val.min()} < -1"
            assert val.max() <= 1.0, f"{key}: action value {val.max()} > 1"

    def test_action_roundtrip(self, processor, modality_keys, statistics):
        _, action_keys = modality_keys
        # Use values within the normalization range to avoid clipping
        raw = _random_action(action_keys, statistics)
        # Scale down to stay within range for perfect roundtrip
        for k in raw:
            stats = statistics[EMBODIMENT]["action"][k]
            low = np.array(stats["min"])
            high = np.array(stats["max"])
            raw[k] = low + (high - low) * np.random.rand(*raw[k].shape).astype(np.float32)
        normalized = processor.apply_action(raw, EMBODIMENT)
        recovered = processor.unapply_action(normalized, EMBODIMENT)
        for key in action_keys:
            np.testing.assert_allclose(
                recovered[key], raw[key], atol=1e-4, err_msg=f"Action roundtrip failed for {key}"
            )


class TestApplyConvenience:
    """Test the combined apply/unapply convenience methods."""

    def test_apply_returns_tuple(self, processor, modality_keys, statistics):
        state_keys, action_keys = modality_keys
        raw_state = _random_state(state_keys, statistics)
        raw_action = _random_action(action_keys, statistics)
        result = processor.apply(raw_state, raw_action, EMBODIMENT)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_full_roundtrip(self, processor, modality_keys, statistics):
        state_keys, action_keys = modality_keys
        raw_state = _random_state(state_keys, statistics)
        raw_action = _random_action(action_keys, statistics)
        # Keep values within normalization range for exact roundtrip
        for k in raw_state:
            stats = statistics[EMBODIMENT]["state"][k]
            low = np.array(stats["min"])
            high = np.array(stats["max"])
            raw_state[k] = low + (high - low) * np.random.rand(*raw_state[k].shape).astype(
                np.float32
            )
        for k in raw_action:
            stats = statistics[EMBODIMENT]["action"][k]
            low = np.array(stats["min"])
            high = np.array(stats["max"])
            raw_action[k] = low + (high - low) * np.random.rand(*raw_action[k].shape).astype(
                np.float32
            )
        norm_state, norm_action = processor.apply(raw_state, raw_action, EMBODIMENT)
        rec_state, rec_action = processor.unapply(norm_state, norm_action, EMBODIMENT)
        for key in state_keys:
            np.testing.assert_allclose(rec_state[key], raw_state[key], atol=1e-4)
        for key in action_keys:
            np.testing.assert_allclose(rec_action[key], raw_action[key], atol=1e-4)


class TestSinCosEncoding:
    """Test sin/cos state encoding mode."""

    def test_sincos_doubles_dimension(self, modality_keys, statistics):
        state_keys, _ = modality_keys
        modality_configs, statistics_local = _load_fixture_configs()
        modality_configs[EMBODIMENT]["state"]["sin_cos_embedding_keys"] = state_keys
        proc = StateActionProcessor(
            modality_configs=modality_configs,
            statistics=statistics_local,
            apply_sincos_state_encoding=True,
        )
        raw = _random_state(state_keys, statistics)
        result = proc.apply_state(raw, EMBODIMENT)
        for key in state_keys:
            assert result[key].shape[-1] == raw[key].shape[-1] * 2, (
                f"{key}: sin/cos should double dimension"
            )


class TestStatistics:
    """Test statistics management."""

    def test_set_statistics(self):
        modality_configs, statistics = _load_fixture_configs()
        proc = StateActionProcessor(modality_configs=modality_configs)
        # No statistics yet — normalization params empty
        assert len(proc.norm_params) == 0
        proc.set_statistics(statistics)
        assert EMBODIMENT in proc.norm_params

    def test_set_statistics_no_override(self):
        modality_configs, statistics = _load_fixture_configs()
        proc = StateActionProcessor(modality_configs=modality_configs, statistics=statistics)
        original_min = proc.norm_params[EMBODIMENT]["state"]["x"]["min"].copy()
        # Modify statistics
        modified = json.loads(json.dumps(statistics))
        modified[EMBODIMENT]["state"]["x"]["min"] = [999.0]
        proc.set_statistics(modified, override=False)
        # Should NOT have changed
        np.testing.assert_array_equal(
            proc.norm_params[EMBODIMENT]["state"]["x"]["min"], original_min
        )

    def test_set_statistics_override(self):
        modality_configs, statistics = _load_fixture_configs()
        proc = StateActionProcessor(modality_configs=modality_configs, statistics=statistics)
        modified = json.loads(json.dumps(statistics))
        modified[EMBODIMENT]["state"]["x"]["min"] = [999.0]
        proc.set_statistics(modified, override=True)
        assert proc.norm_params[EMBODIMENT]["state"]["x"]["min"][0] == pytest.approx(999.0)


class TestTrainEvalMode:
    """Test train/eval mode switching."""

    def test_default_is_train(self, processor):
        assert processor.training is True

    def test_eval(self, processor):
        processor.eval()
        assert processor.training is False

    def test_train(self, processor):
        processor.eval()
        processor.train()
        assert processor.training is True
