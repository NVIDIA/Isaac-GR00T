# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from groot.core.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

MODALITY_CONFIGS = {
    "gr1_unified_seq": {
        "video": ModalityConfig(
            delta_indices=[0, 7, 15],
            modality_keys=["ego_view_bg_crop_pad_res256_freq20"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"],
            sin_cos_embedding_keys=[
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "waist",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.coarse_action"],
        ),
        "rl_info": ModalityConfig(
            delta_indices=[0],
            modality_keys=[],
        ),
    },
    "gr1_unified": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view_bg_crop_pad_res256_freq20"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"],
            sin_cos_embedding_keys=[
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "waist",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.coarse_action"],
        ),
        "rl_info": ModalityConfig(
            delta_indices=[0],
            modality_keys=[],
        ),
    },
    "xdof": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_camera-images-rgb_320_240",
                "top_camera-images-rgb_320_240",
                "right_camera-images-rgb_320_240",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "gripper_pos_obs_left",
                "gripper_pos_obs_right",
                "joint_pos_obs_left",
                "joint_pos_obs_right",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 50)),
            modality_keys=[
                "gripper_pos_action_left",
                "gripper_pos_action_right",
                "joint_pos_action_left",
                "joint_pos_action_right",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_right",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_pos_obs_right",
                ),
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.task"]),
        "rl_info": ModalityConfig(
            delta_indices=[0],
            modality_keys=[],
        ),
    },
    "arx_x5": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_joints",
                "left_gripper",
                "right_joints",
                "right_gripper",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=[
                "left_joints",
                "left_gripper",
                "right_joints",
                "right_gripper",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="left_joints",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="left_gripper",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="right_joints",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="right_gripper",
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
        "rl_info": ModalityConfig(
            delta_indices=[0],
            modality_keys=[],
        ),
    },
    "xdof_dream": {
        "video": ModalityConfig(
            # 8 frames per chunk (0..42 step 6). With num_chunks=4, dataset adds +1 frame
            # to make kn+1 => 8*4+1=33 frames total, which keeps VAE temporal blocks aligned
            # (latent frames = 1+(33-1)/4=9, and (9-1) divisible by num_frame_per_block=2).
            delta_indices=list(range(0, 48, 6)),
            modality_keys=[
                "left_camera-images-rgb_320_240",
                "top_camera-images-rgb_320_240",
                "right_camera-images-rgb_320_240",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "gripper_pos_obs_left",
                "gripper_pos_obs_right",
                "joint_pos_obs_left",
                "joint_pos_obs_right",
            ],
        ),
        "action": ModalityConfig(
            # Using 48 to match chunk_length from video deltas (8*6=48)
            delta_indices=list(range(48)),
            modality_keys=[
                "gripper_pos_action_left",
                "gripper_pos_action_right",
                "joint_pos_action_left",
                "joint_pos_action_right",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_right",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_pos_obs_right",
                ),
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.task"]),
    },
    "xdof_oss_data": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "top",
                "left",
                "right",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_joint_pos",
                "left_gripper_pos",
                "right_joint_pos",
                "right_gripper_pos",
            ],
            sin_cos_embedding_keys=[
                "left_joint_pos",
                "right_joint_pos",
                "left_gripper_pos",
                "right_gripper_pos",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 50)),
            modality_keys=[
                "left_joint_pos",
                "left_gripper_pos",
                "right_joint_pos",
                "right_gripper_pos",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    },
    "xdof_h16": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_camera-images-rgb_320_240",
                "top_camera-images-rgb_320_240",
                "right_camera-images-rgb_320_240",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "gripper_pos_obs_left",
                "gripper_pos_obs_right",
                "joint_pos_obs_left",
                "joint_pos_obs_right",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "gripper_pos_action_left",
                "gripper_pos_action_right",
                "joint_pos_action_left",
                "joint_pos_action_right",
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.task"]),
    },
    "robocasa_panda_omron": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["res256_image_side_0", "res256_image_side_1", "res256_image_wrist_0"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "end_effector_position_relative",
                "end_effector_rotation_relative",
                "gripper_qpos",
                "base_position",
                "base_rotation",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "end_effector_position",
                "end_effector_rotation",
                "gripper_close",
                "base_motion",
                "control_mode",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.action.task_description"]
        ),
    },
    "robocasa365_panda_omron": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["robot0_eye_in_hand", "robot0_agentview_left", "robot0_agentview_right"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "end_effector_position_relative",
                "end_effector_rotation_relative",
                "gripper_qpos",
                "base_position",
                "base_rotation",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "end_effector_position",
                "end_effector_rotation",
                "gripper_close",
                "base_motion",
                "control_mode",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    },
    "language_table_sim": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["rgb_pad_res256_freq10"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "effector_translation",
                "effector_target_translation",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "action",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.language.instruction"]
        ),
    },
    "oxe_fractal": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "image_pad_res256_freq03",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "eef_position",
                "eef_rotation",
                "gripper_closedness_commanded",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "world_vector",
                "rotation_delta",
                "gripper_position",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.language.natural_language_instruction"]
        ),
    },
    "oxe_bridge": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "image_0",
                "image_1",
                "image_2",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "eef_position",
                "eef_rotation",
                "gripper_closed",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "eef_position",
                "eef_rotation",
                "gripper_position",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.language.language_instruction"]
        ),
    },
    "oxe_droid_joint_position_relative": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "exterior_image_1_left",
                "wrist_image_left",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "joint_position",
                "gripper_position",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 32)),
            modality_keys=[
                "joint_position",
                "gripper_position",
            ],
            action_configs=[
                # joint_position
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_position
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "annotation.language.language_instruction",
            ],
        ),
    },
    "oxe_droid": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "exterior_image_1_left_pad_res256_freq15",
                "exterior_image_2_left_pad_res256_freq15",
                "wrist_image_left_pad_res256_freq15",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "joint_position",
                "gripper_position",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "joint_position",
                "gripper_position",
            ],
            action_configs=[
                # joint_position
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_position
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.language.language_instruction"]
        ),
    },
    "oxe_droid_dream": {
        "video": ModalityConfig(
            delta_indices=list(range(0, 24, 3)),
            modality_keys=[
                "exterior_image_1_left",
                "exterior_image_2_left",
                "wrist_image_left",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "joint_position",
                "gripper_position",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 24)),
            modality_keys=[
                "joint_position",
                "gripper_position",
            ],
            action_configs=[
                # joint_position
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # gripper_position
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "annotation.language.language_instruction",
                "annotation.language.language_instruction_2",
                "annotation.language.language_instruction_3",
            ],
        ),
    },
    "simpler_env_widowx": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image_0"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "pad",
                "gripper",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 8)),
            modality_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "gripper",
            ],
            mean_std_embedding_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "simpler_env_google": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "x",
                "y",
                "z",
                "rx",
                "ry",
                "rz",
                "rw",
                "gripper",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 8)),
            modality_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "gripper",
            ],
            mean_std_embedding_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "libero_sim": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image", "wrist_image"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "gripper",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "gripper",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "vlabench_sim": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image", "second_image", "wrist_image"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "droid_sim": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "exterior_image_1_left_pad_res256_freq15",
                "wrist_image_left_pad_res256_freq15",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "joint_position",
                "gripper_position",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "joint_position",
                "gripper_position",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.language.language_instruction"]
        ),
    },
    "sim_behavior_r1_pro": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "observation.images.rgb.head_256_256",
                "observation.images.rgb.left_wrist_256_256",
                "observation.images.rgb.right_wrist_256_256",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "robot_pos",  # 3
                "robot_ori_cos",  # 3
                "robot_ori_sin",  # 3
                "robot_2d_ori",  # 1
                "robot_2d_ori_cos",  # 1
                "robot_2d_ori_sin",  # 1
                "robot_lin_vel",  # 3
                "robot_ang_vel",  # 3
                "arm_left_qpos",  # 7
                "arm_left_qpos_sin",  # 7
                "arm_left_qpos_cos",  # 7
                "eef_left_pos",  # 3
                "eef_left_quat",  # 4
                "gripper_left_qpos",  # 2
                "arm_right_qpos",  # 7
                "arm_right_qpos_sin",  # 7
                "arm_right_qpos_cos",  # 7
                "eef_right_pos",  # 3
                "eef_right_quat",  # 4
                "gripper_right_qpos",  # 2
                "trunk_qpos",  # 4
            ],
        ),  # dim = 82
        "action": ModalityConfig(
            delta_indices=list(range(0, 32)),
            modality_keys=[
                "base",
                "torso",
                "left_arm",
                "left_gripper",
                "right_arm",
                "right_gripper",
            ],
            action_configs=[
                # base
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # torso
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="trunk_qpos",
                ),
                # left_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="arm_left_qpos",
                ),
                # left_gripper
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="arm_right_qpos",
                ),
                # right_gripper
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.coarse_action"],
        ),
    },
    "r1_pro": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_centercrop_res256",
                "right_ego_view_centercrop_res256",
                "left_wrist_view_centercrop_res256",
                "right_wrist_view_centercrop_res256",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm_joints",
                "right_arm_joints",
                "left_gripper",
                "right_gripper",
            ],
            sin_cos_embedding_keys=[
                "left_arm_joints",
                "right_arm_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=[
                "left_arm_joints",
                "right_arm_joints",
                "left_gripper",
                "right_gripper",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.coarse_description"]
        ),
    },
    "r1_pro_single-view": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_centercrop_res256",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm_joints",
                "right_arm_joints",
                "left_gripper",
                "right_gripper",
            ],
            sin_cos_embedding_keys=[
                "left_arm_joints",
                "right_arm_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=[
                "left_arm_joints",
                "right_arm_joints",
                "left_gripper",
                "right_gripper",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.coarse_description"]
        ),
    },
    "moz1": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_centercrop_res256",
                "left_wrist_view_centercrop_res256",
                "right_wrist_view_centercrop_res256",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm_joints",
                "right_arm_joints",
                "left_gripper",
                "right_gripper",
            ],
            sin_cos_embedding_keys=[
                "left_arm_joints",
                "right_arm_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=[
                "left_arm_joints",
                "right_arm_joints",
                "left_gripper",
                "right_gripper",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.coarse_description"]
        ),
    },
    "agibot": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "top_head_pad_res256_freq10",
                "hand_left_pad_res256_freq10",
                "hand_right_pad_res256_freq10",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm_joint_position",
                "right_arm_joint_position",
                "left_effector_position",
                "right_effector_position",
                "head_position",
                "waist_pitch",
                "waist_lift",
            ],
            sin_cos_embedding_keys=[
                "left_arm_joint_position",
                "right_arm_joint_position",
                "head_position",
                "waist_pitch",
            ],  # Apply sin/cos only to angular states, not effector positions or waist_lift
        ),
        "action": ModalityConfig(
            delta_indices=list(range(50)),
            modality_keys=[
                "left_arm_joint_position",
                "right_arm_joint_position",
                "left_effector_position",
                "right_effector_position",
                "head_position",
                "waist_pitch",
                "waist_lift",
                "robot_velocity",
            ],
            action_configs=[
                # left_arm_joint_position
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_arm_joint_position
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # left_effector_position
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_effector_position
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # head_position
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # waist_pitch
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # waist_lift
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # robot_velocity
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.language.action_text"]
        ),
    },
    "agibot_dream": {
        "video": ModalityConfig(
            # 8 frames per chunk (0..42 step 6). With num_chunks=4, dataset adds +1 frame
            # to make kn+1 => 8*4+1=33 frames total, which keeps VAE temporal blocks aligned
            # (latent frames = 1+(33-1)/4=9, and (9-1) divisible by num_frame_per_block=2).
            delta_indices=list(range(0, 48, 6)),
            modality_keys=[
                "top_head",
                "hand_left",
                "hand_right",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm_joint_position",
                "right_arm_joint_position",
                "left_effector_position",
                "right_effector_position",
                "head_position",
                "waist_pitch",
                "waist_lift",
            ],
            sin_cos_embedding_keys=[
                "left_arm_joint_position",
                "right_arm_joint_position",
                "head_position",
                "waist_pitch",
            ],  # Apply sin/cos only to angular states, not effector positions or waist_lift
        ),
        "action": ModalityConfig(
            delta_indices=list(range(48)),
            modality_keys=[
                "left_arm_joint_position",
                "right_arm_joint_position",
                "left_effector_position",
                "right_effector_position",
                "head_position",
                "waist_pitch",
                "waist_lift",
                "robot_velocity",
            ],
            action_configs=[
                # left_arm_joint_position
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_arm_joint_position
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # left_effector_position
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_effector_position
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # head_position
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # waist_pitch
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # waist_lift
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # robot_velocity
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["episode_meta.detailed_global_instruction_concise"]
        ),
    },
    "mecka_hands": {
        "video": ModalityConfig(
            delta_indices=list(range(0, 48, 6)),
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["left_wrist", "left_hand", "right_wrist", "right_hand"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(48)),
            modality_keys=[
                "left_wrist",
                "left_hand",
                "right_wrist",
                "right_hand",
                "waist",
                "neck",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "annotation.human.fine_action",
                "annotation.human.coarse_action",
            ],
        ),
    },
    "unitree_g1_full_body_with_waist_height_nav_cmd": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(50)),
            modality_keys=[
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "waist",
                "base_height_command",
                "navigate_command",
            ],
            action_configs=[
                # left_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # left_hand
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,  # G1 hand is controlled by binary signals like a gripper
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_hand
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,  # G1 hand is controlled by binary signals like a gripper
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # waist
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # base_height_command
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # navigate_command
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_full_body_with_waist_height_nav_cmd_with_wrist_cams": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view", "left_wrist", "right_wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(50)),
            modality_keys=[
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "waist",
                "base_height_command",
                "navigate_command",
            ],
            action_configs=[
                # left_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # left_hand
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,  # G1 hand is controlled by binary signals like a gripper
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_hand
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,  # G1 hand is controlled by binary signals like a gripper
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # waist
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # base_height_command
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # navigate_command
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "real_r1_pro_sharpa": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_res320x240_freq20",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm_joints",
                "right_arm_joints",
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
        ),
        # TODO:  # noqa: TD002, TD003
        # 1. Dataset owner's responsibility to compute EEF and their relative forms normalization stats
        # 2. we need standardlize the data storage for EEF action, either homogeneous or 9dof (xyz+rot6d)
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_r1_pro_sharpa_relative_eef": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_res320x240_freq20",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "unitree_g1_whole_body_teleop_smpl": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(30)),
            modality_keys=[
                "smpl_joints",
                "target_body_orientation",
                "left_hand_joints",
                "right_hand_joints",
                "left_wrist_joints",
                "right_wrist_joints",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_whole_body_teleop_latent": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=[
                "motion_token",
                "left_hand_joints",
                "right_hand_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_whole_body_teleop_latent_gripper": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view", "left_wrist", "right_wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=[
                "motion_token",
                "left_gripper_width_mm",
                "right_gripper_width_mm",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_whole_body_teleop_latent_gripper_no_hand_state": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view", "left_wrist", "right_wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=[
                "motion_token",
                "left_gripper_width_mm",
                "right_gripper_width_mm",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_whole_body_teleop_latent_shared_proj": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=[
                "motion_token",
                "left_hand_joints",
                "right_hand_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_whole_body_teleop_latent_gripper_shared_proj": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view", "left_wrist", "right_wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=[
                "motion_token",
                "left_gripper_width_mm",
                "right_gripper_width_mm",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_whole_body_teleop_latent_relative_eef": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand",
                "right_hand",
                "left_arm",
                "right_arm",
                "waist",
                "left_leg",
                "right_leg",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand",
                "right_hand",
                "motion_token",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_whole_body_teleop_latent_relative_eef_new_proj": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand",
                "right_hand",
                "left_arm",
                "right_arm",
                "waist",
                "left_leg",
                "right_leg",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand",
                "right_hand",
                "motion_token",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_whole_body_teleop_latent_relative_eef_gripper": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view", "left_wrist", "right_wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand",
                "right_hand",
                "left_arm",
                "right_arm",
                "waist",
                "left_leg",
                "right_leg",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_gripper_width_mm",
                "right_gripper_width_mm",
                "motion_token",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_whole_body_teleop_latent_relative_eef_gripper_new_proj": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view", "left_wrist", "right_wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand",
                "right_hand",
                "left_arm",
                "right_arm",
                "waist",
                "left_leg",
                "right_leg",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_gripper_width_mm",
                "right_gripper_width_mm",
                "motion_token",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_whole_body_teleop_latent_no_hands": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "projected_gravity",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(50)),
            modality_keys=[
                "motion_token",
                "left_hand_joints",
                "right_hand_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "unitree_g1_full_body_with_waist_height_nav_cmd_sim": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(30)),
            modality_keys=[
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "waist",
                "base_height_command",
                "navigate_command",
            ],
            action_configs=[
                # left_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_arm
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # left_hand
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,  # G1 hand is controlled by binary signals like a gripper
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_hand
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,  # G1 hand is controlled by binary signals like a gripper
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # waist
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # base_height_command
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # navigate_command
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "so100_2rgb": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image", "image2"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "main_shoulder_pan",
                "main_shoulder_lift",
                "main_elbow_flex",
                "main_wrist_flex",
                "main_wrist_roll",
                "main_gripper",
            ],
            sin_cos_embedding_keys=[
                "main_shoulder_pan",
                "main_shoulder_lift",
                "main_elbow_flex",
                "main_wrist_flex",
                "main_wrist_roll",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "main_shoulder_pan",
                "main_shoulder_lift",
                "main_elbow_flex",
                "main_wrist_flex",
                "main_wrist_roll",
                "main_gripper",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.task"],
        ),
    },
    "so100_3rgb": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image", "image2", "image3"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "main_shoulder_pan",
                "main_shoulder_lift",
                "main_elbow_flex",
                "main_wrist_flex",
                "main_wrist_roll",
                "main_gripper",
            ],
            sin_cos_embedding_keys=[
                "main_shoulder_pan",
                "main_shoulder_lift",
                "main_elbow_flex",
                "main_wrist_flex",
                "main_wrist_roll",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "main_shoulder_pan",
                "main_shoulder_lift",
                "main_elbow_flex",
                "main_wrist_flex",
                "main_wrist_roll",
                "main_gripper",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.task"],
        ),
    },
    "robomind_agilex_3rgb": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "camera_front",
                "camera_left_wrist",
                "camera_right_wrist",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm",
                "right_arm",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "left_arm",
                "right_arm",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.coarse_action"],
        ),
    },
    "robomind_franka_3rgb": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "camera_left",
                "camera_right",
                "camera_top",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "arm",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "arm",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.coarse_action"],
        ),
    },
    "robomind_ur_1rgb": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "camera_top",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "arm",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "arm",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.coarse_action"],
        ),
    },
    "robomind_tienkung_gello_1rgb": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "camera_top",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm",
                "left_hand",
                "right_arm",
                "right_hand",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "left_arm",
                "left_hand",
                "right_arm",
                "right_hand",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.coarse_action"],
        ),
    },
    "robomind_franka_dual_4rgb": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "camera_front",
                "camera_left",
                "camera_right",
                "camera_top",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm",
                "right_arm",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "left_arm",
                "right_arm",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.coarse_action"],
        ),
    },
    "robomind_franka_sim_4rgb": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "camera_front_external",
                "camera_handeye",
                "camera_left_external",
                "camera_right_external",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "arm",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "arm",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.coarse_action"],
        ),
    },
    "robomind_tienkung_prod1_gello_1rgb": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "camera_top",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm",
                "left_hand",
                "right_arm",
                "right_hand",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "left_arm",
                "left_hand",
                "right_arm",
                "right_hand",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.coarse_action"],
        ),
    },
    "robomind_sim_tienkung_1rgb": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "camera_chest",
                "camera_head",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm",
                "left_hand",
                "right_arm",
                "right_hand",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "left_arm",
                "left_hand",
                "right_arm",
                "right_hand",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.coarse_action"],
        ),
    },
    "robomind_franka_1rgb": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "camera_top",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "arm",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "arm",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.coarse_action"],
        ),
    },
    "robomind_tienkung_xsens_1rgb": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "camera_top",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm",
                "right_arm",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "left_arm",
                "right_arm",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.coarse_action"],
        ),
    },
    "molmoact_franka_3rgb": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "first_view",
                "second_view",
                "wrist_image",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "arm",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "arm",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.task"],
        ),
    },
    "galaxea_r1_4rgb": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "head_rgb",
                "head_right_rgb",
                "left_wrist_rgb",
                "right_wrist_rgb",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "joint_position_arm_left",
                "joint_position_arm_right",
                "gripper_state_left",
                "gripper_state_right",
                "joint_position_torso",
                "base_velocity",
            ],
            sin_cos_embedding_keys=[
                "joint_position_arm_left",
                "joint_position_arm_right",
                "joint_position_torso",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=[
                "left_arm",
                "left_gripper",
                "right_arm",
                "right_gripper",
                "torso",
                "base",
            ],
            action_configs=[
                # left_arm
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # left_gripper
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_arm
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # right_gripper
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # torso
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                # base
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.coarse_action"],
        ),
    },
    "oxe_droid_relative_eef": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "exterior_image_1_left",
                "wrist_image_left",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "eef_9d",
                "gripper_position",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "eef_9d",
                "gripper_position",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "annotation.language.language_instruction",
                "annotation.language.language_instruction_2",
                "annotation.language.language_instruction_3",
            ],
        ),
    },
    "oxe_droid_relative_eef_relative_joint": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "exterior_image_1_left",
                "wrist_image_left",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "eef_9d",
                "gripper_position",
                "joint_position",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "eef_9d",
                "gripper_position",
                "joint_position",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "annotation.language.language_instruction",
                "annotation.language.language_instruction_2",
                "annotation.language.language_instruction_3",
            ],
        ),
    },
    "bridge_relative_eef": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "image_0",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "eef_9d",
                "gripper",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "eef_9d",
                "gripper",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "annotation.language.language_instruction",
            ],
        ),
    },
    "fractal_relative_eef": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "image",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "eef_9d",
                "gripper",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "eef_9d",
                "gripper",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "annotation.human.action.task_description",
            ],
        ),
    },
    "real_r1_pro_sharpa_relative_eef_human": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_res320x240_freq20",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_r1_pro_sharpa_relative_eef_egodex": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_r1_pro_sharpa_relative_eef_maxinsights": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_cropratio_res320x240_freq30",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "mecka_relative_eef_active_head": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_cropratio_res320x240_freq30",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
                "head_eef",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
                "head_eef",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_r1_pro_sharpa_relative_eef_mecka": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_cropratio_res320x240_freq30",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    # ------------- New SHARPA Embodiment Configs -------------
    "real_r1_pro_sharpa_add_view": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_cropped",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm_joints",
                "right_arm_joints",
                "left_hand_joints",
                "right_hand_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_arm_joints",
                "right_arm_joints",
                "left_hand_joints",
                "right_hand_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_r1_pro_sharpa_relative_arm_joint": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_res320x240_freq20",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
                "left_arm_joints",
                "right_arm_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
                "left_arm_joints",
                "right_arm_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_r1_pro_sharpa_relative_eef_relative_hand_relative_joint": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_res320x240_freq20",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
                "left_arm_joints",
                "right_arm_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
                "left_arm_joints",
                "right_arm_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_r1_pro_sharpa_relative_joint": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_res320x240_freq20",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm_joints",
                "right_arm_joints",
                "left_hand_joints",
                "right_hand_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_arm_joints",
                "right_arm_joints",
                "left_hand_joints",
                "right_hand_joints",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_r1_pro_sharpa_absolute_eef": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_res320x240_freq20",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_r1_pro_sharpa_relative_eef_add_view": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_cropped",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_r1_pro_sharpa_relative_eef_relative_hand": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_res320x240_freq20",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_r1_pro_sharpa_relative_eef_human_add_view": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_cropped",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_r1_pro_sharpa_relative_eef_human_relative_hand": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_res320x240_freq20",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_r1_pro_sharpa_relative_eef_egodex_relative_hand": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view_res320x240_freq20"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_r1_pro_sharpa_relative_eef_egodex_wrist_only": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view_res320x240_freq20"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["left_wrist_eef", "right_wrist_eef"],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=["left_wrist_eef", "right_wrist_eef"],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_r1_pro_sharpa_relative_eef_mecka_relative_hand": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view_cropratio_res320x240_freq30"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_r1_pro_sharpa_relative_eef_maxinsights_relative_hand": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view_cropratio_res320x240_freq30"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    # ------------- Real G1 Embodiment Configs -------------
    "real_g1_relative_eef_absolute_joints": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef_9d",
                "right_wrist_eef_9d",
                "left_hand",
                "right_hand",
                "left_arm",
                "right_arm",
                "waist",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef_9d",
                "right_wrist_eef_9d",
                "left_hand",
                "right_hand",
                "left_arm",
                "right_arm",
                "waist",
                "base_height_command",
                "navigate_command",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    },
    "real_g1_relative_eef_absolute_joints_wrist_cam": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view", "left_wrist", "right_wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef_9d",
                "right_wrist_eef_9d",
                "left_hand",
                "right_hand",
                "left_arm",
                "right_arm",
                "waist",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef_9d",
                "right_wrist_eef_9d",
                "left_hand",
                "right_hand",
                "left_arm",
                "right_arm",
                "waist",
                "base_height_command",
                "navigate_command",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    },
    "real_g1_relative_eef_relative_joints": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef_9d",
                "right_wrist_eef_9d",
                "left_hand",
                "right_hand",
                "left_arm",
                "right_arm",
                "waist",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef_9d",
                "right_wrist_eef_9d",
                "left_hand",
                "right_hand",
                "left_arm",
                "right_arm",
                "waist",
                "base_height_command",
                "navigate_command",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    },
    "xdof_relative_eef_relative_joint": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "top_camera-images-rgb_320_240",
                "left_camera-images-rgb_320_240",
                "right_camera-images-rgb_320_240",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "gripper_pos_obs_left",
                "gripper_pos_obs_right",
                "joint_pos_obs_left",
                "joint_pos_obs_right",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "gripper_pos_action_left",
                "gripper_pos_action_right",
                "joint_pos_action_left",
                "joint_pos_action_right",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_right",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_pos_obs_right",
                ),
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.task"]),
    },
    "xdof_relative_eef_relative_joint_subtask": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "top_camera-images-rgb_320_240",
                "left_camera-images-rgb_320_240",
                "right_camera-images-rgb_320_240",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "gripper_pos_obs_left",
                "gripper_pos_obs_right",
                "joint_pos_obs_left",
                "joint_pos_obs_right",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "gripper_pos_action_left",
                "gripper_pos_action_right",
                "joint_pos_action_left",
                "joint_pos_action_right",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_right",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_pos_obs_right",
                ),
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.sub_task"]),
    },
    "xdof_relative_eef_relative_joint_recap_long": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "top_camera-images-rgb_320_240",
                "left_camera-images-rgb_320_240",
                "right_camera-images-rgb_320_240",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "gripper_pos_obs_left",
                "gripper_pos_obs_right",
                "joint_pos_obs_left",
                "joint_pos_obs_right",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "gripper_pos_action_left",
                "gripper_pos_action_right",
                "joint_pos_action_left",
                "joint_pos_action_right",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_right",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_pos_obs_right",
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.task_recaption_long", "annotation.sub_task_recaption"],
        ),
    },
    "xdof_relative_eef_relative_joint_umi_keys": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "top_camera-images-rgb_320_240",
                "left_camera-images-rgb_320_240",
                "right_camera-images-rgb_320_240",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "gripper_pos_obs_left",
                "gripper_pos_obs_right",
                "joint_pos_obs_left",
                "joint_pos_obs_right",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "gripper_pos_action_left",
                "gripper_pos_action_right",
                "joint_pos_action_left",
                "joint_pos_action_right",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_right",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_pos_obs_right",
                ),
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.task"]),
    },
    "xdof_relative_eef_relative_joint_umi_keys_4views": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "top_camera-images-rgb_320_240",
                "left_camera-images-rgb_320_240",
                "right_camera-images-rgb_320_240",
                "bottom_camera-images-rgb_320_240",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "gripper_pos_obs_left",
                "gripper_pos_obs_right",
                "joint_pos_obs_left",
                "joint_pos_obs_right",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "gripper_pos_action_left",
                "gripper_pos_action_right",
                "joint_pos_action_left",
                "joint_pos_action_right",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_right",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_pos_obs_right",
                ),
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.task"]),
    },
    "xdof_relative_joint": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "top_camera-images-rgb_320_240",
                "left_camera-images-rgb_320_240",
                "right_camera-images-rgb_320_240",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "joint_pos_obs_left",
                "joint_pos_obs_right",
                "gripper_pos_obs_left",
                "gripper_pos_obs_right",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 50)),
            modality_keys=[
                "joint_pos_action_left",
                "joint_pos_action_right",
                "gripper_pos_action_left",
                "gripper_pos_action_right",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_pos_obs_right",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_right",
                ),
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.task"]),
    },
    "xdof_relative_eef": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "top_camera-images-rgb_320_240",
                "left_camera-images-rgb_320_240",
                "right_camera-images-rgb_320_240",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "gripper_pos_obs_left",
                "gripper_pos_obs_right",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 50)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "gripper_pos_action_left",
                "gripper_pos_action_right",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_left",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_pos_obs_right",
                ),
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.task"]),
    },
    "real_dexmate_sharpa_relative_eef": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_res320x240_freq20",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            exclude_state=False,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_dexmate_sharpa_relative_eef_new_controller": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_res320x240_freq20",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            exclude_state=False,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_dexmate_sharpa_relative_eef_cropped": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_cropped",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            exclude_state=False,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_dexmate_sharpa_relative_eef_relative_joint": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_res320x240_freq20",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
                "left_arm_joints",
                "right_arm_joints",
            ],
            exclude_state=False,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
                "left_arm_joints",
                "right_arm_joints",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                    state_key="left_wrist_eef",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                    state_key="right_wrist_eef",
                ),
                # left_hand_joints (22-dim) — absolute (cannot be made relative
                # to the 7-dim arm joints).
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="left_hand_joints",
                ),
                # right_hand_joints (22-dim) — absolute.
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="right_hand_joints",
                ),
                # left_arm_joints (7-dim) — relative to its own reference frame.
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="left_arm_joints",
                ),
                # right_arm_joints (7-dim) — relative to its own reference frame.
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="right_arm_joints",
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "real_dexmate_sharpa_relative_eef_with_tactile": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_res320x240_freq20",
                "left_wrist_view_res320x240_freq20",
                "right_wrist_view_res320x240_freq20",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
                # tactile_f6: 10 fingers × 6-DoF (60 dims total)
                "left_thumb_f6",
                "left_index_f6",
                "left_middle_f6",
                "left_ring_f6",
                "left_pinky_f6",
                "right_thumb_f6",
                "right_index_f6",
                "right_middle_f6",
                "right_ring_f6",
                "right_pinky_f6",
            ],
            exclude_state=False,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
    "gripper_umi_4camera": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_umi_camera.rgb",
                "left_trigger_camera.rgb",
                "right_umi_camera.rgb",
                "right_trigger_camera.rgb",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(1, 41)),
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    },
    "gripper_umi_2camera_recaption": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["left_umi_camera.rgb", "right_umi_camera.rgb"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(1, 41)),
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.action_description"]
        ),
    },
    # ------------------------------------------------------------------
    # UMI variants used by the zarr→LeRobot conversion pipeline
    # (groot/core/data/conversion/umi/). Ported from
    # groot/vla/omni/configs/data/embodiment_configs.py on k2/umi_gr00t.
    # NOTE: per-arm interleaved key ordering and delta_indices=range(0,50)
    # are preserved verbatim so existing checkpoints remain compatible.
    # masked_keys=[<all state keys>] was collapsed to exclude_state=True
    # because ModalityConfig on groot.core.data.types has no masked_keys.
    # ------------------------------------------------------------------
    "single_gripper_umi": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["right_umi_camera.rgb"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "right_arm.eef",
                "right_gripper.distance",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 50)),
            modality_keys=["right_arm.eef", "right_gripper.distance"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "gripper_umi": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_umi_camera.rgb",
                "right_umi_camera.rgb",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm.eef",
                "left_gripper.distance",
                "right_arm.eef",
                "right_gripper.distance",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 50)),
            modality_keys=[
                "left_arm.eef",
                "left_gripper.distance",
                "right_arm.eef",
                "right_gripper.distance",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "gripper_umi_trigger_camera": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_umi_camera.rgb",
                "right_umi_camera.rgb",
                "left_trigger_camera.rgb",
                "right_trigger_camera.rgb",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
                # "right_arm_to_left_arm.eef",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    # Same state/action as gripper_umi_trigger_camera (no head), with two
    # table-mounted ZED RGB streams added. The dataset also stores ZED depth
    # (`*_zed_camera.depth`, byte-split lossless), but depth is NOT a model
    # input here -- it has no decode/recombine path in training yet, so only
    # the RGB streams are listed as video modality keys.
    "gripper_umi_trigger_two_zed_camera": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_umi_camera.rgb",
                "right_umi_camera.rgb",
                "left_trigger_camera.rgb",
                "right_trigger_camera.rgb",
                "left_zed_camera.rgb",
                "right_zed_camera.rgb",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
                # "right_arm_to_left_arm.eef",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    # Same as gripper_umi_trigger_two_zed_camera, with a third table-mounted
    # ZED RGB stream (middle_zed_camera.rgb). As with the two-ZED tag the
    # dataset also stores ZED depth (`*_zed_camera.depth`, byte-split lossless),
    # but depth is NOT a model input here -- it has no decode/recombine path in
    # training yet, so only the RGB streams are listed as video modality keys.
    "gripper_umi_trigger_three_zed_camera": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_umi_camera.rgb",
                "right_umi_camera.rgb",
                "left_trigger_camera.rgb",
                "right_trigger_camera.rgb",
                "left_zed_camera.rgb",
                "middle_zed_camera.rgb",
                "right_zed_camera.rgb",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
                # "right_arm_to_left_arm.eef",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    # Same state/action as gripper_umi_trigger_camera (no head), with four
    # table-mounted RealSense RGB streams (left / middle_left / middle_right /
    # right). Like the ZED tags the dataset also stores RealSense depth
    # (`*_realsense_camera.depth`, byte-split lossless) and per-camera
    # intrinsics, but depth is NOT a model input here -- it has no
    # decode/recombine path in training yet, so only the RGB streams are listed
    # as video modality keys.
    "gripper_umi_trigger_four_realsense_camera": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_umi_camera.rgb",
                "right_umi_camera.rgb",
                "left_trigger_camera.rgb",
                "right_trigger_camera.rgb",
                "left_realsense_camera.rgb",
                "middle_left_realsense_camera.rgb",
                "middle_right_realsense_camera.rgb",
                "right_realsense_camera.rgb",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
                # "right_arm_to_left_arm.eef",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "gripper_umi_trigger_head_camera": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_umi_camera.rgb",
                "right_umi_camera.rgb",
                "left_trigger_camera.rgb",
                "right_trigger_camera.rgb",
                "head_tracking_camera.rgb",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "head.eef",
                "left_gripper.distance",
                "right_gripper.distance",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "head.eef",
                "left_gripper.distance",
                "right_gripper.distance",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    # Same as gripper_umi_trigger_head_camera, but adds right_arm_to_left_arm.eef
    # (the right arm's pose expressed in the left arm's eef frame, derived from
    # the two head-relative poses by the matching processor in
    # groot/core/data/conversion/umi/gripper_umi_processor.py).
    # TODO: adding eef-relative state, and configure mask state keys  # noqa: TD002, TD003
    "gripper_umi_trigger_head_camera_w_rels": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_umi_camera.rgb",
                "right_umi_camera.rgb",
                "left_trigger_camera.rgb",
                "right_trigger_camera.rgb",
                "head_tracking_camera.rgb",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "head.eef",
                "right_arm_to_left_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 50)),
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "head.eef",
                "right_arm_to_left_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    # Same as gripper_umi_trigger_head_camera_w_rels, but the head RGB
    # stream is sourced from `head_umi_camera` (a UMI/fisheye head camera)
    # instead of `head_tracking_camera`. The head_tracking_camera is still
    # used upstream by the matching processor to derive `head.eef` and the
    # head-relative arm poses; only the policy's video input changes.
    # TODO: adding eef-relative state, and configure mask state keys  # noqa: TD002, TD003
    "gripper_umi_trigger_two_head_camera_w_rels": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_umi_camera.rgb",
                "right_umi_camera.rgb",
                "left_trigger_camera.rgb",
                "right_trigger_camera.rgb",
                "head_umi_camera.rgb",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "head.eef",
                "right_arm_to_left_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "head.eef",
                "right_arm_to_left_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    # Single right-hand DexUMI exoskeleton with tactile -- FULL info (eef +
    # 16-DOF hand joints + 30-dim Orange-hand tactile `right_hand.tactile`). The
    # dataset always carries everything; the policy decides usage. Shares an
    # identical schema with single_dexumi_teleop for co-training.
    # See SingleDexUMIProcessor / orange_hand_tactile.py.
    "single_dexumi": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["right_umi_camera.rgb"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "right_arm.eef",
                "right_hand.joint_positions",
                "right_hand.tactile",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=["right_arm.eef", "right_hand.joint_positions"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    # Single right-hand DexUMI without tactile.
    # intentionally dropped (see ``SingleDexUMIWoTacProcessor``).
    "single_dexumi_wo_tac": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["right_umi_camera.rgb"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "right_arm.eef",
                "right_hand.joint_positions",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=["right_arm.eef", "right_hand.joint_positions"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    # Same as gripper_umi_trigger_two_head_camera_w_rels, plus two ZED scene
    # cameras in video -- each contributing an RGB and a metric depth stream
    # (left/right_zed_camera.rgb + left/right_zed_camera.depth) -- and
    # left_arm_to_head.eef / right_arm_to_head.eef added to state. The two
    # arm-to-head keys are inert
    # under exclude_state=True (state is zeroed before the model) and are NOT in
    # action; they are present in the schema for future use.
    "gripper_umi_trigger_two_head_two_zed_camera_w_rels": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_umi_camera.rgb",
                "right_umi_camera.rgb",
                "left_trigger_camera.rgb",
                "right_trigger_camera.rgb",
                "head_umi_camera.rgb",
                "left_zed_camera.rgb",
                "right_zed_camera.rgb",
                "left_zed_camera.depth",
                "right_zed_camera.depth",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "head.eef",
                "left_arm_to_head.eef",
                "right_arm_to_head.eef",
                "right_arm_to_left_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=[
                "left_arm.eef",
                "right_arm.eef",
                "head.eef",
                "right_arm_to_left_arm.eef",
                "left_gripper.distance",
                "right_gripper.distance",
            ],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    # Single right-hand UR5e + Orange dexterous hand teleop with tactile.
    # EEF from robot FK; action from recorded ur5e_action / right_hand_action
    # streams. Tactile is an observation -> state only, never action: one 30-dim
    # `right_hand.tactile` column (triplet-major (sensor, point) -> fx, fy, fz),
    # identical schema to single_dexumi for co-training.
    "single_dexumi_teleop": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["right_umi_camera.rgb"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "right_arm.eef",
                "right_hand.joint_positions",
                "right_hand.tactile",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=["right_arm.eef", "right_hand.joint_positions"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "single_dexumi_teleop_wo_tac": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["right_umi_camera.rgb"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "right_arm.eef",
                "right_hand.joint_positions",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=["right_arm.eef", "right_hand.joint_positions"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "single_dexumi_teleop_wo_tac_abs_hand": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["right_umi_camera.rgb"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "right_arm.eef",
                "right_hand.joint_positions",
            ],
            exclude_state=True,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 40)),
            modality_keys=["right_arm.eef", "right_hand.joint_positions"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    },
    "real_tianji_sharpa_relative_eef": {
        # 3 cameras (ego + L/R wrist) — matches the deploy contract in
        # mapping/tianji_sharpa.yaml. The data's 4th camera (ego_view_right) is
        # recorded but unused; declaring it here would push the view count past
        # --model.max_n_cameras and hard-crash at processor.py:761.
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "ego_view_res320x240_freq30",
                "left_wrist_view_res320x240_freq30",
                "right_wrist_view_res320x240_freq30",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            exclude_state=False,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(0, 48)),
            modality_keys=[
                "left_wrist_eef",
                "right_wrist_eef",
                "left_hand_joints",
                "right_hand_joints",
            ],
            normalize_rotation=True,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.coarse_action"]
        ),
    },
}

# convert_recorder_session_tianji DEFAULTS to the `_with_tactile` tag, so every tianji
# dataset carries it in meta/info.json -- and generate_rel_stats indexes MODALITY_CONFIGS
# with a bare [tag] while merge_lerobot_datasets passes the dataset's own tag. Without an
# entry here every tianji merge died with KeyError at the very last step, after the whole
# dataset had already been written.
#
# The four groups below are DELIBERATELY the base tag's, unchanged: the ten fingertip F6
# vectors are extra STATE COLUMNS on disk (136D parquet vs 76D), not extra selected state.
# Tactile is consumed as its own "tactile" modality group feeding the frozen Tactile-VAE,
# which additionally needs a loader/processor seam this file does not provide. Keep this an
# alias so that the richer entry stays a clean superset, not a conflict; do NOT fold F6 into
# "state" here, that would silently hand the state encoder 122D instead of 62D.
MODALITY_CONFIGS["real_tianji_sharpa_relative_eef_with_tactile"] = dict(
    MODALITY_CONFIGS["real_tianji_sharpa_relative_eef"]
)


def register_modality_config(embodiment_tag: str, config: dict):
    MODALITY_CONFIGS[embodiment_tag] = config
