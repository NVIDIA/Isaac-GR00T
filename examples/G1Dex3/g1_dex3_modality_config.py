# SPDX-License-Identifier: Apache-2.0
"""GR00T modality config for the Unitree G1 (bimanual, Dex3 hands) dataset
produced by convert_mcap_to_lerobot.py.

This is what actually makes the action space *relative end-effector* at training
time -- meta/modality.json only carries the index splits, it cannot express
rep/type/format. The dataset on disk stores ABSOLUTE poses; GR00T's processor
converts absolute -> relative because the arm actions below are marked
rep=RELATIVE, type=EEF, format=XYZ_ROT6D (the N1.7 9D [x,y,z + rot6d] EEF space,
mirroring the shipped ``oxe_droid_relative_eef_relative_joint`` config).

Custom robots must register under EmbodimentTag.NEW_EMBODIMENT. Use it when
loading / finetuning:

    --embodiment-tag NEW_EMBODIMENT --modality-config-path g1_dex3_modality_config.py

State/action keys and index splits must match meta/modality.json exactly:
    left_arm_eef(9) | left_hand(7) | right_arm_eef(9) | right_hand(7)
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


ACTION_HORIZON = 16  # number of future steps the model predicts

g1_dex3_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view", "left_wrist", "right_wrist"],
    ),
    # Proprioception: absolute EEF poses (9D each) + Dex3 joint positions (7D each).
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm_eef", "left_hand", "right_arm_eef", "right_hand"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["left_arm_eef", "left_hand", "right_arm_eef", "right_hand"],
        action_configs=[
            # left_arm_eef: relative end-effector (9D xyz+rot6d), delta vs left_arm_eef state
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="left_arm_eef",
            ),
            # left_hand: absolute Dex3 joint targets (following the G1 hand convention)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_hand",
            ),
            # right_arm_eef: relative end-effector (9D xyz+rot6d)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="right_arm_eef",
            ),
            # right_hand: absolute Dex3 joint targets
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_hand",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1_dex3_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
