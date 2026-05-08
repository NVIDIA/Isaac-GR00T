"""GR00T modality config for Star AI Robot Arm (single arm).

Star AI single arm:
  - 6 arm motors + 1 gripper = 7 control channels
  - Joint names in the recorded dataset: Motor_0 .. Motor_5, gripper
  - Cameras consumed by this config: side, rear, onhand

Usage:
  python gr00t/experiment/launch_finetune.py \
      --base-model-path nvidia/GR00T-N1.6-3B \
      --dataset-path <groot_v21_dataset> \
      --embodiment-tag NEW_EMBODIMENT \
      --modality-config-path examples/StarAI/starai_single_arm_config.py \
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

starai_single_arm_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["side", "rear", "onhand"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "single_arm",
            "gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "single_arm",
            "gripper",
        ],
        action_configs=[
            # single_arm (Motor_0 .. Motor_5)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper
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

register_modality_config(starai_single_arm_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
