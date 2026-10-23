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

"""Tests for the state/action normalization primitives in gr00t.data.utils.

Covers:
- Min/max and mean/std normalize -> unnormalize roundtrips
- The documented degenerate cases (min == max, std == 0)
- Per-step (2D) bounds alongside per-feature (1D) bounds
- Result dtype: integer inputs must not truncate the normalized values
"""

from gr00t.data.utils import (
    normalize_values_meanstd,
    normalize_values_minmax,
    unnormalize_values_meanstd,
    unnormalize_values_minmax,
)
import numpy as np
import pytest


SEED = 20260823
TIMESTEPS = 6
FEATURES = 4


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


class TestMinMaxNormalization:
    def test_roundtrip_recovers_values_in_range(self, rng):
        low = rng.uniform(-5.0, 0.0, size=FEATURES)
        high = low + rng.uniform(0.1, 5.0, size=FEATURES)
        values = rng.uniform(low, high, size=(TIMESTEPS, FEATURES))
        params = {"min": low, "max": high}

        roundtripped = unnormalize_values_minmax(normalize_values_minmax(values, params), params)

        np.testing.assert_allclose(roundtripped, values, rtol=1e-6, atol=1e-8)

    def test_normalized_values_span_minus_one_to_one(self, rng):
        low = rng.uniform(-5.0, 0.0, size=FEATURES)
        high = low + rng.uniform(0.1, 5.0, size=FEATURES)
        values = rng.uniform(low, high, size=(TIMESTEPS, FEATURES))

        normalized = normalize_values_minmax(values, {"min": low, "max": high})

        assert np.all(normalized >= -1.0)
        assert np.all(normalized <= 1.0)

    def test_bounds_map_to_endpoints(self):
        params = {"min": np.array([-2.0, 0.0]), "max": np.array([2.0, 10.0])}

        normalized = normalize_values_minmax(np.array([[-2.0, 0.0], [2.0, 10.0]]), params)

        np.testing.assert_allclose(normalized, np.array([[-1.0, -1.0], [1.0, 1.0]]))

    def test_degenerate_range_normalizes_to_zero(self):
        """A feature whose min equals its max carries no information and maps to 0."""
        params = {"min": np.array([1.0, 3.0]), "max": np.array([1.0, 7.0])}

        normalized = normalize_values_minmax(np.array([[1.0, 7.0]]), params)

        assert normalized[0, 0] == 0.0
        np.testing.assert_allclose(normalized[0, 1], 1.0)

    def test_per_step_bounds(self, rng):
        """2D bounds apply a different range at every timestep."""
        low = rng.uniform(-5.0, 0.0, size=(TIMESTEPS, FEATURES))
        high = low + rng.uniform(0.1, 5.0, size=(TIMESTEPS, FEATURES))
        values = rng.uniform(low, high)
        params = {"min": low, "max": high}

        roundtripped = unnormalize_values_minmax(normalize_values_minmax(values, params), params)

        np.testing.assert_allclose(roundtripped, values, rtol=1e-6, atol=1e-8)

    def test_batched_values(self, rng):
        low = np.zeros(FEATURES)
        high = np.full(FEATURES, 10.0)
        values = rng.uniform(low, high, size=(3, TIMESTEPS, FEATURES))
        params = {"min": low, "max": high}

        roundtripped = unnormalize_values_minmax(normalize_values_minmax(values, params), params)

        np.testing.assert_allclose(roundtripped, values, rtol=1e-6, atol=1e-8)


class TestMeanStdNormalization:
    def test_roundtrip_recovers_values(self, rng):
        mean = rng.uniform(-5.0, 5.0, size=FEATURES)
        std = rng.uniform(0.1, 3.0, size=FEATURES)
        values = rng.normal(mean, std, size=(TIMESTEPS, FEATURES))
        params = {"mean": mean, "std": std}

        roundtripped = unnormalize_values_meanstd(normalize_values_meanstd(values, params), params)

        np.testing.assert_allclose(roundtripped, values, rtol=1e-6, atol=1e-8)

    def test_normalization_is_zero_mean_unit_std(self, rng):
        values = rng.normal(2.0, 3.0, size=(4096, 1))
        params = {"mean": values.mean(axis=0), "std": values.std(axis=0)}

        normalized = normalize_values_meanstd(values, params)

        np.testing.assert_allclose(normalized.mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(normalized.std(), 1.0, atol=1e-6)

    def test_zero_std_passes_values_through(self):
        """A constant feature has no scale, so both directions leave it untouched."""
        params = {"mean": np.array([100.0, 0.0]), "std": np.array([0.0, 2.0])}
        values = np.array([[7.0, 4.0]])

        normalized = normalize_values_meanstd(values, params)

        np.testing.assert_allclose(normalized[0, 0], 7.0)
        np.testing.assert_allclose(normalized[0, 1], 2.0)
        np.testing.assert_allclose(unnormalize_values_meanstd(normalized, params), values)


class TestResultDtype:
    """Normalization divides, so the result must be floating point regardless of input dtype."""

    def test_integer_input_is_not_truncated_minmax(self):
        params = {"min": np.array([0.0, 0.0]), "max": np.array([10.0, 10.0])}
        values = np.array([[5, 7]], dtype=np.int64)

        normalized = normalize_values_minmax(values, params)

        assert np.issubdtype(normalized.dtype, np.floating)
        np.testing.assert_allclose(normalized, np.array([[0.0, 0.4]]), atol=1e-8)

    def test_integer_input_is_not_truncated_meanstd(self):
        params = {"mean": np.array([4.0, 4.0]), "std": np.array([3.0, 3.0])}
        values = np.array([[5, 7]], dtype=np.int64)

        normalized = normalize_values_meanstd(values, params)

        assert np.issubdtype(normalized.dtype, np.floating)
        np.testing.assert_allclose(normalized, np.array([[1.0 / 3.0, 1.0]]), rtol=1e-6)

    def test_integer_input_is_not_truncated_unnormalize_meanstd(self):
        params = {"mean": np.array([4.0, 4.0]), "std": np.array([3.0, 3.0])}

        unnormalized = unnormalize_values_meanstd(np.array([[0, 1]], dtype=np.int64), params)

        assert np.issubdtype(unnormalized.dtype, np.floating)
        np.testing.assert_allclose(unnormalized, np.array([[4.0, 7.0]]))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_float_dtype_is_preserved(self, dtype):
        params = {"min": np.array([0.0, 0.0]), "max": np.array([10.0, 10.0])}

        normalized = normalize_values_minmax(np.array([[5, 7]], dtype=dtype), params)

        assert normalized.dtype == dtype
