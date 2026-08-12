# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Rotation utilities using the first-two-rows 6D convention.

Convention: rot6d = rotation_matrix[:2, :].flatten()  ->  [r00, r01, r02, r10, r11, r12]

Vendored from groot/control/utils/rotation_utils.py for groot.core self-containment.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """Apply Gram-Schmidt process to a set of vectors.

    vectors: batchsize, N, D
    return: batchsize, N, D
    """
    if len(vectors.shape) == 2:
        vectors = vectors[None]

    basis = np.zeros_like(vectors)
    basis[:, 0] = vectors[:, 0] / np.linalg.norm(vectors[:, 0], axis=-1, keepdims=True)
    for i in range(1, vectors.shape[1]):
        v = vectors[:, i]
        for j in range(i):
            v -= np.sum(v * basis[:, j], axis=-1, keepdims=True) * basis[:, j]
        basis[:, i] = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return basis


def rot_mat_to_rot_6d(rot_mat: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to 6d representation. rot_mat: N, 3, 3 -> N, 6"""
    rot_6d = rot_mat[:, :2, :]  # N, 2, 3
    return rot_6d.reshape(-1, 6)  # N, 6


def rot_6d_to_rot_mat(rot_6d: np.ndarray) -> np.ndarray:
    """Convert a 6d representation to rotation matrix. rot_6d: N, 6 -> N, 3, 3"""
    rot_6d = rot_6d.reshape(-1, 2, 3)
    if not np.allclose(np.sum(rot_6d[:, 0] * rot_6d[:, 1], axis=-1), 0):
        rot_6d = gram_schmidt(rot_6d)

    rot_mat = np.zeros((rot_6d.shape[0], 3, 3))
    rot_mat[:, :2, :] = rot_6d
    rot_mat[:, 2, :] = np.cross(rot_6d[:, 0], rot_6d[:, 1])
    return rot_mat


def compute_se3_relative(
    actions_xyz_rot6d: np.ndarray,
    ref_xyz_rot6d: np.ndarray,
) -> np.ndarray:
    """Compute SE(3) relative transform: T_ref^{-1} @ T_action (vectorized).

    Args:
        actions_xyz_rot6d: (H, 9) target poses [xyz(3) + rot6d(6)]
        ref_xyz_rot6d: (9,) reference pose [xyz(3) + rot6d(6)]

    Returns:
        (H, 9) relative poses [delta_xyz(3) + delta_rot6d(6)]
    """
    act_xyz = actions_xyz_rot6d[:, :3]
    act_rot = rot_6d_to_rot_mat(actions_xyz_rot6d[:, 3:9])  # (H, 3, 3)
    ref_xyz = ref_xyz_rot6d[:3]
    ref_rot = rot_6d_to_rot_mat(ref_xyz_rot6d[3:9].reshape(1, 6))[0]  # (3, 3)
    ref_rot_inv = ref_rot.T  # orthogonal -> inverse = transpose
    delta_xyz = (act_xyz - ref_xyz) @ ref_rot_inv.T  # (H, 3)
    delta_rot = np.einsum("ij,hjk->hik", ref_rot_inv, act_rot)  # (H, 3, 3)
    delta_rot6d = rot_mat_to_rot_6d(delta_rot)  # (H, 6)
    return np.concatenate([delta_xyz, delta_rot6d], axis=1).astype(np.float32)


def apply_se3_delta(
    ref_xyz_rot6d: np.ndarray,
    delta_xyz_rot6d: np.ndarray,
) -> np.ndarray:
    """Apply SE(3) delta to a reference pose: T_result = T_ref @ T_delta (vectorized).

    Inverse of compute_se3_relative.

    Args:
        ref_xyz_rot6d: (9,) reference pose [xyz(3) + rot6d(6)]
        delta_xyz_rot6d: (H, 9) relative poses [delta_xyz(3) + delta_rot6d(6)]

    Returns:
        (H, 9) absolute poses [xyz(3) + rot6d(6)]
    """
    delta_xyz = delta_xyz_rot6d[:, :3]
    delta_rot = rot_6d_to_rot_mat(delta_xyz_rot6d[:, 3:9])  # (H, 3, 3)
    ref_xyz = ref_xyz_rot6d[:3]
    ref_rot = rot_6d_to_rot_mat(ref_xyz_rot6d[3:9].reshape(1, 6))[0]  # (3, 3)
    result_xyz = delta_xyz @ ref_rot.T + ref_xyz  # (H, 3)
    result_rot = np.einsum("ij,hjk->hik", ref_rot, delta_rot)  # (H, 3, 3)
    result_rot6d = rot_mat_to_rot_6d(result_rot)  # (H, 6)
    return np.concatenate([result_xyz, result_rot6d], axis=1).astype(np.float32)


def euler_to_rot_6d(euler: np.ndarray, format="XYZ") -> np.ndarray:
    """Convert euler angles to 6d representation. euler: N, 3"""
    rot_mat = Rotation.from_euler(format, euler, degrees=False).as_matrix()
    return rot_mat_to_rot_6d(rot_mat)


def rot_6d_to_euler(rot_6d: np.ndarray, format="XYZ"):
    """Convert 6d representation to euler angles. rot_6d: N, 6"""
    rot_mat = rot_6d_to_rot_mat(rot_6d)
    return Rotation.from_matrix(rot_mat).as_euler(format, degrees=False)
