# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N1.7ES YAM simulation stack-bowls experiment."""

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

_EMBODIMENT = "xdof_relative_eef_relative_joint"

N17ES_YAM_SIMULATION_STACK_BOWLS_DATA = {
    "datasets": [
        {
            "dataset_paths": ["datasets/yam_workstation_stack_bowls_20260430_170700"],
            "mix_ratio": 1.0,
            "embodiment_tag": _EMBODIMENT,
        }
    ],
    "mode": "single_turn",
    "video_backend": "torchcodec",
    "override_pretraining_statistics": True,
}

EXPERIMENT = TrainingExperiment(
    data_config=N17ESBaseDataConfig(data=N17ES_YAM_SIMULATION_STACK_BOWLS_DATA).to_data_config(),
    training_config=N17ESBaseTrainingConfig(global_batch_size=1024).to_training_config(),
    compute_config=N17ESBaseComputeConfig(h100_batch_size=16).to_compute_config(),
)
