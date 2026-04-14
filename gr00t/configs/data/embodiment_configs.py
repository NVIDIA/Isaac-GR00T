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

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


MODALITY_CONFIGS = {
    ##### Pre-registered pretrain configurations #####
    "oxe_droid_relative_eef_relative_joint": {
        "video": ModalityConfig(
            delta_indices=[-15, 0],
            modality_keys=["exterior_image_1_left", "wrist_image_left"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["eef_9d", "gripper_position", "joint_position"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=["eef_9d", "gripper_position", "joint_position"],
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.EEF,
                    format=ActionFormat.XYZ_ROT6D,
                    state_key="eef_9d",
                ),
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="gripper_position",
                ),
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="joint_position",
                ),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.language.language_instruction"],
        ),
    },
    ##### Pre-registered posttrain configurations #####
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
    "libero_sim": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image", "wrist_image"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
    "simpler_env_widowx": {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["image_0"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(8)),
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
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
            modality_keys=["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(8)),
            modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    },
}


def register_modality_config(
    config: dict, embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
):
    assert embodiment_tag.value not in MODALITY_CONFIGS, (
        f"Embodiment tag {embodiment_tag} already registered"
    )
    MODALITY_CONFIGS[embodiment_tag.value] = config
