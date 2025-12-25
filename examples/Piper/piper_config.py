from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# Joint Space Configuration
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
        delta_indices=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        ],
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


# # Task Space Rot6D Configuration
# piper_task_space_rot6d_config = {
#     "video": ModalityConfig(
#         delta_indices=[0],
#         modality_keys=["hand_camera", "third_camera"],
#     ),
#     "state": ModalityConfig(
#         delta_indices=[0],
#         modality_keys=[
#             "eef_pos",
#             "eef_rpy_external",
#             "gripper_distance",
#         ],
#     ),
#     "action": ModalityConfig(
#         delta_indices=[
#             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
#         ],
#         modality_keys=[
#             "eef_pos",
#             "eef_rpy_external",
#             "gripper_distance",
#         ],
#         action_configs=[
#             ActionConfig(
#                 rep=ActionRepresentation.ABSOLUTE,
#                 type=ActionType.EEF,
#                 format=ActionFormat.POSITION_ROTATION6D,
#             ),
#         ],
#     ),
#     "language": ModalityConfig(
#         delta_indices=[0],
#         modality_keys=["annotation.human.action.task_description"],
#     ),
# }


# # Task Space RPY Configuration
# piper_task_space_rpy_config = {
#     "video": ModalityConfig(
#         delta_indices=[0],
#         modality_keys=["hand_camera", "third_camera"],
#     ),
#     "state": ModalityConfig(
#         delta_indices=[0],
#         modality_keys=[
#             "eef_pos",
#             "eef_rpy_external",
#             "gripper_distance",
#         ],
#     ),
#     "action": ModalityConfig(
#         delta_indices=[
#             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
#         ],
#         modality_keys=[
#             "eef_pos",
#             "eef_rpy_external",
#             "gripper_distance",
#         ],
#         action_configs=[
#             ActionConfig(
#                 rep=ActionRepresentation.ABSOLUTE,
#                 type=ActionType.EEF,
#                 format=ActionFormat.POSITION_EULER_RPY,
#             ),
#         ],
#     ),
#     "language": ModalityConfig(
#         delta_indices=[0],
#         modality_keys=["annotation.human.action.task_description"],
#     ),
# }


# # Task Space Quaternion Configuration
# piper_task_space_quat_config = {
#     "video": ModalityConfig(
#         delta_indices=[0],
#         modality_keys=["hand_camera", "third_camera"],
#     ),
#     "state": ModalityConfig(
#         delta_indices=[0],
#         modality_keys=[
#             "eef_pos",
#             "eef_rpy_external",
#             "gripper_distance",
#         ],
#     ),
#     "action": ModalityConfig(
#         delta_indices=[
#             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
#         ],
#         modality_keys=[
#             "eef_pos",
#             "eef_rpy_external",
#             "gripper_distance",
#         ],
#         action_configs=[
#             ActionConfig(
#                 rep=ActionRepresentation.ABSOLUTE,
#                 type=ActionType.EEF,
#                 format=ActionFormat.POSITION_QUATERNION,
#             ),
#         ],
#     ),
#     "language": ModalityConfig(
#         delta_indices=[0],
#         modality_keys=["annotation.human.action.task_description"],
#     ),
# }


# Register configurations - uncomment the one you need
# register_modality_config(piper_joint_space_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
# register_modality_config(piper_task_space_rot6d_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
# register_modality_config(piper_task_space_rpy_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
# register_modality_config(piper_task_space_quat_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

# Default: use joint space configuration
register_modality_config(piper_joint_space_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

