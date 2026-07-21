# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N2 Min-WAM iPhone DeltaCAT experiment."""

from __future__ import annotations

from groot.infra.experiments.training.configs._n2_minwam_deltacat_common import (
    data_config,
    h100_compute_config,
    training_config,
)
from groot.infra.launcher.one_launcher.training_experiment import TrainingExperiment

DATA_CONFIG = data_config(
    session_ids=(
        "2026-04-13-18-06-40-YAM-19",
        "2026-04-13-20-16-46-YAM-19",
        "2026-04-16-17-04-56-YAM-19",
    ),
    video_config={
        "modality_keys": [
            "top_camera-images-rgb_320_240",
            "left_camera-images-rgb_320_240",
            "right_camera-images-rgb_320_240",
            "bottom_camera-images-rgb_320_240",
        ],
    },
)

TRAINING_CONFIG = training_config(
    max_n_cameras=4,
    gradient_accumulation_steps=2,
)
COMPUTE_CONFIG = h100_compute_config(per_device_batch_size=2)

EXPERIMENT = TrainingExperiment(
    data_config=DATA_CONFIG,
    training_config=TRAINING_CONFIG,
    compute_config=COMPUTE_CONFIG,
)
