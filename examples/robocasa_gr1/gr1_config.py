# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modality config for RoboCasa GR1 Tabletop Tasks.
# Embodiment: GR1ArmsAndWaistFourierHands.
# This mirrors DiT4DiT's FourierGr1ArmsWaistDataConfig:
#   state/action keys: left_arm, right_arm, left_hand, right_hand, waist
#   state preprocessing: sin/cos encoding

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# State/action joint groups in DiT4DiT and RoboCasa deployment order:
#   left_arm (7), right_arm (7), left_hand (6), right_hand (6), waist (3)
GR1_STATE_KEYS = [
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "waist",
]

GR1_ACTION_KEYS = GR1_STATE_KEYS


gr1_config = {
    # Video: ego-centric camera (key matches "video" entries in modality.json)
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    # State: current proprioceptive reading
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=GR1_STATE_KEYS,
        sin_cos_embedding_keys=GR1_STATE_KEYS,
    ),
    # Action: 16-step prediction horizon
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=GR1_ACTION_KEYS,
        action_configs=[
            # Joint-position absolute targets for all groups (matches teleop data format)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
        ] * len(GR1_ACTION_KEYS),
    ),
    # Language: task description annotations from dataset
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.coarse_action"],
    ),
}

register_modality_config(gr1_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
