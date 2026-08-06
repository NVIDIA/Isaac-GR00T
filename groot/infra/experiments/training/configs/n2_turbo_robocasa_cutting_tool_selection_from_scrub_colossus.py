# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a ScrubCuttingBoard-trained checkpoint on CuttingToolSelection."""

from dataclasses import replace

from groot.infra.experiments.training.configs.n2_turbo_robocasa_scrub_cutting_board_colossus import (
    EXPERIMENT as SCRUB_CUTTING_BOARD_EXPERIMENT,
)

assert SCRUB_CUTTING_BOARD_EXPERIMENT.evaluation_config is not None

# CuttingToolSelection is a Composite-Unseen probe. Its target demonstrations
# must never enter training, so retain the complete ScrubCuttingBoard experiment
# and change only the task evaluated against the resulting checkpoint.
EXPERIMENT = replace(
    SCRUB_CUTTING_BOARD_EXPERIMENT,
    evaluation_config=replace(
        SCRUB_CUTTING_BOARD_EXPERIMENT.evaluation_config,
        task="CuttingToolSelection",
    ),
)
