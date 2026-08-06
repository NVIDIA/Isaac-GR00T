# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a SteamInMicrowave-trained checkpoint on WaffleReheat."""

from dataclasses import replace

from groot.infra.experiments.training.configs.n2_turbo_robocasa_steam_in_microwave_colossus import (
    EXPERIMENT as STEAM_IN_MICROWAVE_EXPERIMENT,
)

assert STEAM_IN_MICROWAVE_EXPERIMENT.evaluation_config is not None

# WaffleReheat is a Composite-Unseen probe. Its target demonstrations must never
# enter training, so retain the complete SteamInMicrowave experiment and change
# only the task evaluated against the resulting checkpoint.
EXPERIMENT = replace(
    STEAM_IN_MICROWAVE_EXPERIMENT,
    evaluation_config=replace(
        STEAM_IN_MICROWAVE_EXPERIMENT.evaluation_config,
        task="WaffleReheat",
    ),
)
