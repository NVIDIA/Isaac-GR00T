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

from gr00t.data.utils import (
    normalize_values_meanstd,
    normalize_values_minmax,
    unnormalize_values_meanstd,
)
import numpy as np


def test_minmax_integer_midpoint_stays_at_zero():
    values = np.array([[0, 5], [10, 10]], dtype=np.int64)
    params = {"min": np.array([0, 0]), "max": np.array([10, 10])}

    normalized = normalize_values_minmax(values, params)

    # 5 is halfway from 0 to 10, so the normalized value is 0.
    np.testing.assert_allclose(normalized, [[-1.0, 0.0], [1.0, 1.0]])


def test_meanstd_integer_quotient_keeps_the_fraction():
    values = np.array([[1, 2], [3, 4]], dtype=np.int64)
    params = {"mean": np.array([0, 0]), "std": np.array([2, 0])}

    normalized = normalize_values_meanstd(values, params)

    # (1 - 0) / 2 = 0.5 and (3 - 0) / 2 = 1.5. A zero std keeps the original value.
    np.testing.assert_allclose(normalized, [[0.5, 2.0], [1.5, 4.0]])


def test_meanstd_unnormalize_integer_keeps_the_fraction():
    normalized_values = np.array([[1, 0]], dtype=np.int64)
    params = {"mean": np.array([0.0, 0.0]), "std": np.array([0.5, 0.0])}

    restored = unnormalize_values_meanstd(normalized_values, params)

    # 1 * 0.5 + 0 = 0.5. A zero std keeps the normalized value.
    np.testing.assert_allclose(restored, [[0.5, 0.0]])


def test_float32_minmax_keeps_dtype():
    values = np.array([[0, 5], [10, 10]], dtype=np.float32)
    params = {"min": np.array([0, 0]), "max": np.array([10, 10])}

    normalized = normalize_values_minmax(values, params)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, [[-1.0, 0.0], [1.0, 1.0]])
