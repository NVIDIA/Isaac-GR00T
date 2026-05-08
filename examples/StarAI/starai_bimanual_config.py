"""GR00T modality config for Star AI Robot Arm (bimanual).

Star AI bi_so_follower:
  - Left:  5 arm joints + 1 gripper = 6 DOF
  - Right: 5 arm joints + 1 gripper = 6 DOF
  - Total: 12 DOF
  - Joint names: left_shoulder_pan .. left_gripper, right_shoulder_pan .. right_gripper
  - Cameras: left_rear, left_side

Usage:
  python gr00t/experiment/launch_finetune.py \
      --base-model-path nvidia/GR00T-N1.6-3B \
      --dataset-path <groot_v21_dataset> \
      --embodiment-tag NEW_EMBODIMENT \
      --modality-config-path examples/StarAI/starai_bimanual_config.py \
      --num-gpus 1 \
      --max-steps 2000
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

starai_bimanual_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_rear", "left_side"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
        ],
        action_configs=[
            # left_arm (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_gripper
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_arm (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
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
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(starai_bimanual_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
