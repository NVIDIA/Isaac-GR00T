from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# Joint Space Configuration - Action Horizon 16
piper_joint_space_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["hand_camera", "third_camera"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "joint_states",
            "gripper_distance",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),  # Action horizon = 16
        modality_keys=[
            "joint_states",
            "gripper_distance",
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
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(piper_joint_space_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

