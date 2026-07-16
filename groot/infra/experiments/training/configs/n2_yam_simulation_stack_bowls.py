# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N2 YAM simulation stack-bowls experiment."""

from __future__ import annotations

from groot.infra.experiments.training.configs._n2_base_compute_config import N2BaseComputeConfig
from groot.infra.experiments.training.configs._n2_base_data_config import N2BaseDataConfig
from groot.infra.experiments.training.configs._n2_base_training_config import (
    N2BaseTrainingConfig,
)
from groot.infra.launcher.one_launcher.training_experiment import TrainingExperiment

_EMBODIMENT = "xdof_relative_eef_relative_joint"

N2_YAM_SIMULATION_STACK_BOWLS_DATA = {
    "datasets": [
        {
            "dataset_paths": ["datasets/yam_workstation_stack_bowls_20260430_170700"],
            "embodiment_tag": _EMBODIMENT,
            "mix_ratio": 1.0,
        }
    ],
    "seed": 24,
    "mode": "single_turn",
    "allow_padding": True,
    "video_backend": "torchcodec",
    "override_pretraining_statistics": True,
    "modality_configs": {
        _EMBODIMENT: {
            "video": {"delta_indices": [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]}
        }
    },
}

EXPERIMENT = TrainingExperiment(
    data_config=N2BaseDataConfig(data=N2_YAM_SIMULATION_STACK_BOWLS_DATA).to_data_config(),
    training_config=N2BaseTrainingConfig(
        wandb_project="groot_n2_posttrain",
        max_n_cameras=3,
        save_steps=1000,
        upload_every=1000,
        start_from_checkpoint=(
            "ruijiez/n2_yam_pretrain-bidrectional-full-200k-source/checkpoint-200000"
        ),
        torch_compile=True,
        torch_compile_mode="reduce-overhead",
    ).to_training_config(),
    compute_config=N2BaseComputeConfig(h100_batch_size=4, gb300_batch_size=12).to_compute_config(),
)
