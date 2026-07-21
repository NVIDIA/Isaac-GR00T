# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""N2 Min-WAM backpack DeltaCAT experiment."""

from __future__ import annotations

from groot.infra.experiments.training.configs._n2_minwam_deltacat_common import (
    data_config,
    h100_compute_config,
    training_config,
)
from groot.infra.launcher.one_launcher.training_experiment import TrainingExperiment

DATA_CONFIG = data_config(
    session_ids=(
        "2026-04-20-20-35-13-YAM-22",
        "2026-04-21-16-58-29-YAM-22",
        "2026-04-21-16-31-29-YAM-21",
        "2026-04-21-17-21-37-YAM-21",
        "2026-04-20-20-31-20-YAM-21",
    )
)

TRAINING_CONFIG = training_config(max_n_cameras=3)
COMPUTE_CONFIG = h100_compute_config(per_device_batch_size=4)

EXPERIMENT = TrainingExperiment(
    data_config=DATA_CONFIG,
    training_config=TRAINING_CONFIG,
    compute_config=COMPUTE_CONFIG,
)
