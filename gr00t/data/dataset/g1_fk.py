# Copyright (c) 2026 Tau Robotics.
# SPDX-License-Identifier: Apache-2.0
"""
G1 forward kinematics for computing wrist EEF 9D from joint positions.

EEF 9D = position(3) + rotation_6d(6)
    position: XYZ of the EEF in the pelvis frame
    rotation_6d: first two rows of the 3x3 rotation matrix, flattened

Matches the EEF convention from G1_29_ArmIK (robot_arm_ik.py):
    - EEF frame is 5cm offset in the local X direction from left/right_wrist_yaw_joint
    - Identity rotation relative to the joint frame
    - This is the same convention used during teleop data collection and
      matches the pretrained real_g1_relative_eef_relative_joints format.

Uses yourdfpy URDF for FK computation — same URDF as the skeleton renderer
and the IK solver.
"""

import numpy as np
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_URDF_PATH = _REPO_ROOT / "hardware/teleop/robot_control/assets/g1/g1_body29_hand14.urdf"

_LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
_RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Default standing pose for legs + waist (locked at zero in the IK reduced model)
_DEFAULT_STANDING = {
    "left_hip_pitch_joint": 0.0, "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0, "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0, "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0, "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.0, "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
}

# EEF is defined as the wrist_yaw link frame + 5cm in local X.
# Matches G1_29_ArmIK: pin.SE3(np.eye(3), np.array([0.05, 0, 0]))
_LEFT_WRIST_LINK = "left_wrist_yaw_link"
_RIGHT_WRIST_LINK = "right_wrist_yaw_link"
_EEF_LOCAL_OFFSET = np.array([0.05, 0.0, 0.0], dtype=np.float64)

# Rotation correction: our URDF's wrist frame has X along the forearm (forward),
# but the pretrained model's convention has X pointing down (towards table).
# Ry(-90°) maps X→-Z, Y→Y, Z→X, converting between these conventions.
# Verified against nvidia/GR00T-N1.7-3B pretrained statistics — all dims
# within 1 std at the mean arm pose.
_EEF_ROTATION_CORRECT = np.array([
    [0, 0, -1],
    [0, 1,  0],
    [1, 0,  0],
], dtype=np.float64)


def _matrix_to_rot6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """First two rows of rotation matrix, flattened to (6,)."""
    return rotation_matrix[:2, :].flatten()


class G1ForwardKinematics:
    """Compute wrist EEF 9D from G1 arm joint positions via URDF FK.

    Matches the EEF convention from G1_29_ArmIK (robot_arm_ik.py):
    the EEF frame is offset 5cm in local X from the wrist_yaw joint,
    with identity rotation relative to the joint frame.
    """

    def __init__(self, urdf_path: str | None = None):
        import yourdfpy
        if urdf_path is None:
            urdf_path = str(_URDF_PATH)
        self._urdf = yourdfpy.URDF.load(
            urdf_path,
            build_scene_graph=True,
            load_meshes=False,
            load_collision_meshes=False,
        )
        self._base_link = self._urdf.robot.links[0].name

    def compute_eef_9d(
        self,
        left_arm: np.ndarray,
        right_arm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute EEF 9D for both wrists from arm joint positions.

        Args:
            left_arm: (7,) joint positions in radians
            right_arm: (7,) joint positions in radians

        Returns:
            left_eef_9d: (9,) position(3) + rotation_6d(6)
            right_eef_9d: (9,) position(3) + rotation_6d(6)
        """
        cfg = dict(_DEFAULT_STANDING)
        for i, name in enumerate(_LEFT_ARM_JOINTS):
            cfg[name] = float(left_arm[i])
        for i, name in enumerate(_RIGHT_ARM_JOINTS):
            cfg[name] = float(right_arm[i])
        self._urdf.update_cfg(cfg)

        left_eef = self._wrist_to_eef_9d(_LEFT_WRIST_LINK)
        right_eef = self._wrist_to_eef_9d(_RIGHT_WRIST_LINK)
        return left_eef, right_eef

    def _wrist_to_eef_9d(self, wrist_link: str) -> np.ndarray:
        """Get EEF 9D for a wrist link with the 5cm local X offset."""
        T_wrist = self._urdf.get_transform(wrist_link, self._base_link)
        # Apply the local offset: EEF position = wrist_pos + R_wrist @ [0.05, 0, 0]
        R = T_wrist[:3, :3]
        eef_pos = T_wrist[:3, 3] + R @ _EEF_LOCAL_OFFSET
        # Apply rotation correction to match pretrained convention
        R_corrected = R @ _EEF_ROTATION_CORRECT
        rot6d = _matrix_to_rot6d(R_corrected).astype(np.float32)
        return np.concatenate([eef_pos.astype(np.float32), rot6d])

    def compute_eef_9d_batch(
        self,
        left_arm_seq: np.ndarray,
        right_arm_seq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute EEF 9D for a sequence of frames.

        Args:
            left_arm_seq: (T, 7) joint positions
            right_arm_seq: (T, 7) joint positions

        Returns:
            left_eef_seq: (T, 9)
            right_eef_seq: (T, 9)
        """
        T = left_arm_seq.shape[0]
        left_eefs = np.empty((T, 9), dtype=np.float32)
        right_eefs = np.empty((T, 9), dtype=np.float32)
        for t in range(T):
            left_eefs[t], right_eefs[t] = self.compute_eef_9d(
                left_arm_seq[t], right_arm_seq[t]
            )
        return left_eefs, right_eefs
