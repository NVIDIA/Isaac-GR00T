# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared construction for N2 Turbo LIBERO posttraining experiments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from groot.infra.experiments.training.configs.n2_turbo_airplane import (
    TRAINING_CONFIG as BASE_TURBO_TRAINING_CONFIG,
)
from groot.infra.launcher.one_launcher.training_experiment import TrainingExperiment

LIBERO_ACTION_HORIZON = 40


def make_n2_turbo_libero_experiment(
    base_experiment: TrainingExperiment,
) -> TrainingExperiment:
    """Apply the N2 Turbo model profile to one canonical LIBERO experiment."""
    base_training_config = base_experiment.training_config
    turbo_model = deepcopy(BASE_TURBO_TRAINING_CONFIG.model)
    # Preserve the checkpoint-compatible 48-token action head while avoiding
    # padding every LIBERO sample beyond its canonical 40-step horizon.
    turbo_model["action-cfg"]["max-action-horizon"] = LIBERO_ACTION_HORIZON

    training_config = replace(
        base_training_config,
        training=replace(
            base_training_config.training,
            wandb_project="n2-turbo-libero-golden",
            # The standard N2 checkpoint uses Wan2.1 latents and cannot be
            # loaded by the Turbo model's Wan2.2 patch embedding/output head.
            start_from_checkpoint=BASE_TURBO_TRAINING_CONFIG.training.start_from_checkpoint,
        ),
        model_name=BASE_TURBO_TRAINING_CONFIG.model_name,
        model=turbo_model,
    )

    return TrainingExperiment(
        data_config=base_experiment.data_config,
        training_config=training_config,
        compute_config=base_experiment.compute_config,
    )
