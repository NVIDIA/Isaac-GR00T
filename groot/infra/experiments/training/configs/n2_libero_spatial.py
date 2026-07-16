# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N2 LIBERO Spatial golden-run experiment."""

from __future__ import annotations

from groot.infra.experiments.training.configs._n2_base_compute_config import N2BaseComputeConfig
from groot.infra.experiments.training.configs._n2_base_data_config import N2BaseDataConfig
from groot.infra.experiments.training.configs._n2_base_training_config import (
    N2BaseTrainingConfig,
)
from groot.infra.launcher.one_launcher.training_experiment import TrainingExperiment

N2_LIBERO_SPATIAL_DATA = {
    "download_cache": True,
    "datasets": [
        {
            "dataset_paths": ["libero/libero_spatial_no_noops_1.0.0_lerobot"],
            "mix_ratio": 1.0,
            "embodiment_tag": "libero_sim",
        }
    ],
    "modality_configs": {"libero_sim": {"video": {"delta_indices": [0, 2, 4, 6, 8, 10, 12, 14]}}},
    "seed": 24,
    "mode": "single_turn",
    "allow_padding": True,
    "video_backend": "torchcodec",
    "override_pretraining_statistics": True,
}

EXPERIMENT = TrainingExperiment(
    data_config=N2BaseDataConfig(data=N2_LIBERO_SPATIAL_DATA).to_data_config(),
    training_config=N2BaseTrainingConfig(wandb_project="n2-libero-golden").to_training_config(),
    compute_config=N2BaseComputeConfig(h100_batch_size=4).to_compute_config(),
)
