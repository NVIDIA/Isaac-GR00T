# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N1.7ES YAM bowl-stacking golden-run experiment."""

from __future__ import annotations

from groot.infra.experiments.training.configs._n17es_base_compute_config import (
    N17ESBaseComputeConfig,
)
from groot.infra.experiments.training.configs._n17es_base_data_config import (
    N17ESBaseDataConfig,
)
from groot.infra.experiments.training.configs._n17es_base_training_config import (
    N17ESBaseTrainingConfig,
)
from groot.infra.launcher.one_launcher.training_experiment import TrainingExperiment

N17ES_YAM_BOWL_STACKING_DATA = {
    "datasets": [
        {
            "dataset_paths": ["yam_lerobot_v8/xdof.bowl-stacking-yam_2026-04-24_203traj_v4_USA"],
            "mix_ratio": 1.0,
            "embodiment_tag": "xdof_relative_eef_relative_joint",
        }
    ],
    "modality_configs": {
        "xdof_relative_eef_relative_joint": {
            "video": {
                "modality_keys": [
                    "top_camera-images-rgb_320_240",
                    "left_camera-images-rgb_320_240",
                    "right_camera-images-rgb_320_240",
                    "bottom_camera-images-rgb_320_240",
                ]
            }
        }
    },
    "mode": "single_turn",
    "video_backend": "torchcodec",
    "override_pretraining_statistics": True,
}

EXPERIMENT = TrainingExperiment(
    data_config=N17ESBaseDataConfig(data=N17ES_YAM_BOWL_STACKING_DATA).to_data_config(),
    training_config=N17ESBaseTrainingConfig(global_batch_size=1024).to_training_config(),
    compute_config=N17ESBaseComputeConfig(
        h100_batch_size=16,
        gb300_batch_size=32,
    ).to_compute_config(),
)
