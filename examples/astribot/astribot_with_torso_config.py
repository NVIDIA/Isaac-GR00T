from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


astribot_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head", "hand_left", "hand_right", "torso"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "arm_left", # 7
            "arm_right", # 7
            "gripper_left", # 1
            "gripper_right", # 1
            "head", # 2
            "torso", # 4
        ], # 22
    ), 
    "action": ModalityConfig(
        delta_indices=list(range(0, 30)),
        modality_keys=[
            "arm_left", # 7
            "arm_right", # 7
            "gripper_left", # 1
            "gripper_right", # 1
            "head", # 2
            "torso", # 4
        ],
        action_configs=[
            # arm_left
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="arm_left",
            ),
            # arm_right
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="arm_right",
            ),
            # gripper_left
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper_right
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # head
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="head",
            ),
            # torso
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="torso",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(astribot_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
