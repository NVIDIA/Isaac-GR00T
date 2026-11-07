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
Test the DROID euler -> rot6d convention.

DROID stores euler angles in the extrinsic fixed-frame ``xyz`` convention
(``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``), which matches scipy ``"xyz"`` and
``tfg.rotation_matrix_3d.from_euler``. scipy ``"XYZ"`` (uppercase) is the
intrinsic convention ``R = Rx @ Ry @ Rz`` and produces a different rotation
matrix for the same angles. These tests pin the load-bearing lowercase
spelling so a future edit cannot silently reintroduce the intrinsic
convention (see commit c5cadf5, which did exactly that under the false claim
that scipy ``"XYZ"`` is equivalent to tfg).
"""

from __future__ import annotations

from gr00t.data.state_action.droid_frame import (
    DROID_EEF_ROTATION_CORRECT,
    compute_eef_9d,
    euler_to_rot6d,
)
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


NON_DEGENERATE_EULERS = [
    np.array([0.3, -0.2, 0.5]),
    np.array([-1.2, 1.1, 2.6]),
    np.array([0.7, 0.4, -1.8]),
    np.array([2.9, -0.1, 0.4]),  # roll near pi (common in DROID teleop)
]

ALL_EULERS = [
    np.array([0.0, 0.0, 0.0]),
    *NON_DEGENERATE_EULERS,
    np.array([0.0, np.pi / 2, 0.0]),  # pitch gimbal lock
]


def _rot6d_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return matrix[:2, :].reshape(6)


@pytest.mark.parametrize("euler", ALL_EULERS)
def test_euler_to_rot6d_matches_tfg_extrinsic_xyz(euler: np.ndarray) -> None:
    """euler->matrix must equal tfg.rotation_matrix_3d.from_euler (Rz @ Ry @ Rx)."""
    expected = _rot6d_from_matrix(
        Rotation.from_euler("xyz", euler).as_matrix() @ DROID_EEF_ROTATION_CORRECT
    )
    np.testing.assert_allclose(euler_to_rot6d(euler), expected, atol=1e-12)


@pytest.mark.parametrize("euler", NON_DEGENERATE_EULERS)
def test_euler_to_rot6d_differs_from_intrinsic_xyz(euler: np.ndarray) -> None:
    """Guard: scipy "XYZ" (intrinsic, Rx @ Ry @ Rz) is NOT the DROID convention."""
    intrinsic = _rot6d_from_matrix(
        Rotation.from_euler("XYZ", euler).as_matrix() @ DROID_EEF_ROTATION_CORRECT
    )
    assert not np.allclose(euler_to_rot6d(euler), intrinsic, atol=1e-9)


def test_euler_to_rot6d_batched() -> None:
    eulers = np.stack(NON_DEGENERATE_EULERS)  # (4, 3)
    result = euler_to_rot6d(eulers)
    assert result.shape == (4, 6)
    for i, euler in enumerate(eulers):
        np.testing.assert_allclose(result[i], euler_to_rot6d(euler), atol=1e-12)


def test_compute_eef_9d() -> None:
    cart = np.array([[1.0, 2.0, 3.0, 0.3, -0.2, 0.5], [4.0, 5.0, 6.0, 0.7, 0.4, -1.8]])
    eef_9d = compute_eef_9d(cart)
    assert eef_9d.shape == (2, 9)
    np.testing.assert_allclose(eef_9d[:, :3], cart[:, :3], atol=1e-12)
    np.testing.assert_allclose(eef_9d[:, 3:], euler_to_rot6d(cart[:, 3:]), atol=1e-12)
