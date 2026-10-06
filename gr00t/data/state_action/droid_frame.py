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

"""DROID end-effector frame correction — single source of truth.

The pretrained OXE DROID model expects eef rotations in the egocentric TFG
convention: the robot's euler->matrix rotation is post-multiplied by
``DROID_EEF_ROTATION_CORRECT`` before the rot6d is taken. The dataset builder,
the rotation-verify script, and the real-robot client must all agree on this
matrix; a drifted copy silently produces wrong eef rotations with no crash.

``examples/DROID/main_gr00t.py`` runs on a slim robot install without the
``gr00t`` package and cannot import this module, so it vendors a tagged mirror.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


# Egocentric frame correction applied after euler->matrix conversion to match
# the OXE DROID training pipeline (TFG convention).
DROID_EEF_ROTATION_CORRECT = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    dtype=np.float64,
)


def euler_to_rot6d(euler_angles: np.ndarray) -> np.ndarray:
    """Convert euler angles (3D) to rotation 6D representation.

    Uses extrinsic XYZ Euler convention (scipy ``"XYZ"``, equivalent to
    ``tfg.rotation_matrix_3d.from_euler``) and post-multiplies by
    ``DROID_EEF_ROTATION_CORRECT`` to match the pretrained model.

    Args:
        euler_angles: (..., 3) array of euler angles

    Returns:
        (..., 6) array of rot6d representation
    """
    shape = euler_angles.shape[:-1]
    flat = euler_angles.reshape(-1, 3)
    rot_matrices = Rotation.from_euler("XYZ", flat).as_matrix()  # (N, 3, 3)
    rot_matrices = rot_matrices @ DROID_EEF_ROTATION_CORRECT
    rot6d = rot_matrices[:, :2, :].reshape(-1, 6)  # (N, 6)
    return rot6d.reshape(*shape, 6)


def compute_eef_9d(cartesian_position: np.ndarray) -> np.ndarray:
    """Convert cartesian_position (XYZ + euler 3D) to eef_9d (XYZ + rot6d).

    Args:
        cartesian_position: (..., 6) array [x, y, z, euler_x, euler_y, euler_z]

    Returns:
        (..., 9) array [x, y, z, rot6d_0..5]
    """
    xyz = cartesian_position[..., :3]
    euler = cartesian_position[..., 3:]
    rot6d = euler_to_rot6d(euler)
    return np.concatenate([xyz, rot6d], axis=-1)
