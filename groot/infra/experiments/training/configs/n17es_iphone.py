# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N1.7ES iPhone golden-run experiment."""

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

N17ES_IPHONE_DATA = {
    "datasets": [
        {
            "dataset_paths": ["datasets/yam_v7/xdof.iphone_v3-100traj_2026-04-16_19-02-47_v4_USA"],
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
    data_config=N17ESBaseDataConfig(data=N17ES_IPHONE_DATA).to_data_config(),
    training_config=N17ESBaseTrainingConfig(global_batch_size=640).to_training_config(),
    compute_config=N17ESBaseComputeConfig(
        h100_batch_size=16,
        gb300_batch_size=32,
    ).to_compute_config(),
)
