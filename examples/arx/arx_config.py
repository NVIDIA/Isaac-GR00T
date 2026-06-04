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
        modality_keys=["head", "left_wrist", "right_wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "arm_left", # 6
            "gripper_left", # 1
            "arm_right", # 6
            "gripper_right", # 1
        ], # 14
    ), 
    "action": ModalityConfig(
        delta_indices=list(range(0, 30)),
        modality_keys=[
            "arm_left", # 6
            "gripper_left", # 1
            "arm_right", # 6
            "gripper_right", # 1
        ],
        action_configs=[
            # arm_left
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="arm_left",
            ),
            # gripper_left
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # arm_right
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="arm_right",
            ),
            # gripper_right
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
}

register_modality_config(astribot_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
