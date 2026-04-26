# N1.7 equivalent of the N1.5 `allex_thetwo_ck40_egostereo_config`.
#
# Original N1.5 definition: /rlwrld2/home/david/gr00t/gr00t/experiment/data_config.py
#   (class `allex_thetwo_ck40_egostereo_config`, registered under the same key in
#    DATA_CONFIG_MAP).
#
# In N1.7 there is no per-embodiment subclass with transforms anymore. Transforms
# (video crop/resize/color jitter, q99 normalization, ConcatTransform, GR00TTransform)
# are handled internally by the N1.7 model processor. You only need to declare the
# modality layout (which keys live under video/state/action, their delta indices,
# and an ActionConfig per action key).
#
# Usage:
#   torchrun --nproc_per_node=N gr00t/experiment/launch_finetune.py \
#       --base_model_path nvidia/GR00T-N1.7-2B \
#       --dataset_path /rlwrld-dataset/.../61-Mouse_Box_Packing-d748af07 \
#       --embodiment_tag new_embodiment \
#       --modality_config_path configs/allex_thetwo_ck40_egostereo.py \
#       --state_dropout_prob 0.0      # 0.1 / 0.3 / 0.8 to reproduce SD10 / SD30 / SD80
#       ...
#
# The dataset's meta/modality.json must expose these bucket keys (verified against
# /rlwrld-dataset/foundry-dvc/data/gold/teleop-real/V3/224/61-Mouse_Box_Packing-d748af07):
#   state:  right_arm_joints, left_arm_joints, right_hand_joints, left_hand_joints,
#           neck_joints, waist_joints
#   action: same 6 keys
#   video:  camera_ego_left, camera_ego_right
#   annotation: human.task_description

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# Six joint groups shared by state and action for the ALLEX bimanual robot.
JOINT_KEYS = [
    "right_arm_joints",
    "left_arm_joints",
    "right_hand_joints",
    "left_hand_joints",
    "neck_joints",
    "waist_joints",
]

ACTION_HORIZON = 40  # "ck40" in the original config name (chunk-40)

allex_thetwo_ck40_egostereo_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["camera_ego_left", "camera_ego_right"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=JOINT_KEYS,
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=JOINT_KEYS,
        # All six groups are joint-space (NON_EEF). Using RELATIVE aligns with the
        # N1.7 relative-EEF/relative-joint pretrain recipe (deltas from current
        # state) and matches how the pretrain humanoid data is represented.
        # If your recorded actions are absolute joint targets and you want to keep
        # them that way, change `rep=ActionRepresentation.ABSOLUTE`.
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in JOINT_KEYS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(
    allex_thetwo_ck40_egostereo_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
