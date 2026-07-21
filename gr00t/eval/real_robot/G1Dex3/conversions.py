# SPDX-License-Identifier: Apache-2.0
"""Conversions between the TRI ``policy_interfaces`` gRPC contract and the
nested observation/action dicts that :class:`gr00t.policy.gr00t_policy.Gr00tPolicy`
consumes and produces, for the Unitree G1 (bimanual, Dex3 hands) checkpoint.

The finetuned checkpoint uses embodiment tag ``new_embodiment`` with a 32-dim
state/action laid out as::

    left_arm_eef[0:9] | left_hand[9:16] | right_arm_eef[16:25] | right_hand[25:32]

- Arm keys are 9D = ``xyz + rot6d``. Actions are RELATIVE end-effector; the
  policy un-relativizes them to ABSOLUTE at decode time using the ``state`` we
  pass in, so the returned pose lands in the same frame as the state EEF pose we
  read out of the observation (this avoids a separate calibration transform).
- Hand keys are 7D absolute Dex3 joint targets. On the wire a Dex3 hand travels
  as 7 named scalar grippers (``{prefix}_0..6``), which anzu regroups by
  splitting on ``_hand_``.

rot6d <-> rotation-matrix conversions reuse :class:`EndEffectorPose` so encode
and decode are guaranteed inverse (same basis-vector convention as training).
"""

from dataclasses import dataclass, field

from gr00t.data.state_action.pose import EndEffectorPose
import numpy as np
from policy_interfaces.robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers
from pydrake.math import RigidTransform, RotationMatrix


# GR00T language modality key for the finetuned G1-Dex3 checkpoint.
LANGUAGE_KEY = "annotation.human.task_description"

# Dex3 hand joint order, from GR00T's convert_mcap_to_lerobot.py
# (``HAND_JOINT_NAMES``). This order IS the checkpoint's ``left_hand[0:7]`` /
# ``right_hand[0:7]`` index order and must not be reordered. The full anzu joint
# names are ``{side}_hand_{name}_joint`` (see g1_poses.py G1_DEX3_TABLETOP_RESET_POSE).
HAND_JOINT_ORDER = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]


def _default_hand_joint_map() -> dict[str, list[str]]:
    return {
        "left_hand": [f"left_hand_{name}_joint" for name in HAND_JOINT_ORDER],
        "right_hand": [f"right_hand_{name}_joint" for name in HAND_JOINT_ORDER],
    }


@dataclass
class G1DexConversionConfig:
    """Name maps between anzu's ``MultiarmObservation`` / ``PosesAndGrippers``
    and GR00T's modality keys.

    Defaults are the actual anzu names traced from
    ``robot_policy_system_params.yaml`` (topics) and GR00T's
    ``convert_mcap_to_lerobot.py`` (camera/EE/hand-joint mapping). The one thing
    to verify on-robot is the hand joint-name strings (see README): the ORDER is
    fixed by training, but the exact ``HandStatus`` joint names must match.
    """

    # anzu ``obs.visuo`` key (the camera ROS topic string)  ->  GR00T video key.
    # ego_view is the ZED head LEFT eye; the head RIGHT eye is unused.
    camera_map: dict[str, str] = field(
        default_factory=lambda: {
            "/head_camera/zed_node/left/image_rect_color/compressed": "ego_view",
            "/left_wrist_camera/color/image_rect_raw/compressed": "left_wrist",
            "/right_wrist_camera/color/image_rect_raw/compressed": "right_wrist",
        }
    )
    # GR00T arm state/action key  ->  anzu EE-pose key in ``obs.robot.actual.poses``
    # (the PoseStatus ``topic_name``, i.e. the actual-EE-pose topic string).
    arm_state_pose_map: dict[str, str] = field(
        default_factory=lambda: {
            "left_arm_eef": "/current_left_hand_ee_link",
            "right_arm_eef": "/current_right_hand_ee_link",
        }
    )
    # GR00T arm action key  ->  anzu output pose key (what RosRobotPolicyRunner
    # publishes as ``/ee_target_{left,right}``).
    arm_action_out_map: dict[str, str] = field(
        default_factory=lambda: {
            "left_arm_eef": "/ee_target_left",
            "right_arm_eef": "/ee_target_right",
        }
    )
    # GR00T hand state/action key  ->  ORDERED list of anzu Dex3 joint names.
    # Read/written in this exact order (== the checkpoint's per-hand index order).
    # On output, anzu's joints_to_joint_state filters grippers by the ``{side}_hand_``
    # prefix and republishes them to ``/joint_cmd_hand_{left,right}``.
    hand_joint_map: dict[str, list[str]] = field(default_factory=_default_hand_joint_map)
    language_key: str = LANGUAGE_KEY


def _rigid_transform_to_eef9d(transform: RigidTransform) -> np.ndarray:
    """Drake ``RigidTransform`` -> 9D ``[xyz, rot6d]`` (float32)."""
    rotation = np.asarray(transform.rotation().matrix(), dtype=np.float64)
    translation = np.asarray(transform.translation(), dtype=np.float64)
    rot6d = EndEffectorPose(rotation=rotation, rotation_type="matrix").to_rotation("rot6d")
    return np.concatenate([translation, rot6d]).astype(np.float32)


def _eef9d_to_rigid_transform(eef9d: np.ndarray) -> RigidTransform:
    """9D ``[xyz, rot6d]`` -> Drake ``RigidTransform``."""
    eef9d = np.asarray(eef9d, dtype=np.float64)
    translation = eef9d[:3]
    matrix = EndEffectorPose(rotation=eef9d[3:9], rotation_type="rot6d").to_rotation("matrix")
    return RigidTransform(R=RotationMatrix(matrix), p=translation)


def multiarm_obs_to_gr00t(obs: MultiarmObservation, cfg: G1DexConversionConfig) -> dict:
    """Build the nested ``{"video", "state", "language"}`` dict Gr00tPolicy
    expects, with each entry shaped ``(B=1, T=1, ...)``.
    """
    video = {}
    for anzu_key, view in cfg.camera_map.items():
        rgb = np.asarray(obs.visuo[anzu_key].rgb.array)  # (H, W, 3) uint8
        video[view] = rgb[np.newaxis, np.newaxis, ...]  # (1, 1, H, W, 3)

    actual = obs.robot.actual
    state = {}
    for gr00t_key, pose_key in cfg.arm_state_pose_map.items():
        eef9d = _rigid_transform_to_eef9d(actual.poses[pose_key])  # (9,)
        state[gr00t_key] = eef9d[np.newaxis, np.newaxis, :]  # (1, 1, 9)
    for gr00t_key, joint_names in cfg.hand_joint_map.items():
        joints = np.array(
            [actual.grippers[name] for name in joint_names], dtype=np.float32
        )
        state[gr00t_key] = joints[np.newaxis, np.newaxis, :]  # (1, 1, len(joint_names))

    instruction = obs.language_instruction or ""
    language = {cfg.language_key: [[instruction]]}  # (B=1, T=1)

    return {"video": video, "state": state, "language": language}


def gr00t_action_to_poses_and_grippers(
    action: dict, step: int, cfg: G1DexConversionConfig
) -> PosesAndGrippers:
    """Convert one timestep of a GR00T action chunk (``{key: (B=1, T, D)}``,
    absolute after decode) into a ``PosesAndGrippers``.
    """
    poses = {}
    for gr00t_key, out_key in cfg.arm_action_out_map.items():
        eef9d = np.asarray(action[gr00t_key][0, step], dtype=np.float64)  # (9,)
        poses[out_key] = _eef9d_to_rigid_transform(eef9d)

    grippers = {}
    for gr00t_key, joint_names in cfg.hand_joint_map.items():
        joints = np.asarray(action[gr00t_key][0, step], dtype=np.float64)  # (len(joint_names),)
        for name, value in zip(joint_names, joints):
            grippers[name] = float(value)

    return PosesAndGrippers(poses=poses, grippers=grippers)
