# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""N2 LIBERO-Object post-training experiment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

from groot.infra.launcher.one_launcher.training_experiment import TrainingExperiment
from groot.infra.launcher.one_launcher.training_experiment_resolver import (
    TrainingExperimentResolver,
)

_BASE_EXPERIMENT = cast(
    TrainingExperiment,
    TrainingExperimentResolver.init_default(experiment_dir=Path(__file__).parent).load(
        "n2_oss_pretrain"
    ),
)

EXPERIMENT = replace(
    _BASE_EXPERIMENT,
    data_config=replace(
        _BASE_EXPERIMENT.data_config,
        data={
            "download_cache": True,
            "datasets": [
                {
                    "dataset_paths": ["datasets/libero/libero_object_no_noops_1.0.0_lerobot"],
                    "mix_ratio": 1.0,
                    "embodiment_tag": "libero_sim",
                }
            ],
            "modality_configs": {
                "libero_sim": {"video": {"delta_indices": [0, 2, 4, 6, 8, 10, 12, 14]}}
            },
            "seed": 24,
            "mode": "single_turn",
            "allow_padding": True,
            "video_backend": "torchcodec",
            "override_pretraining_statistics": True,
        },
    ),
    training_config=replace(
        _BASE_EXPERIMENT.training_config,
        training=replace(
            _BASE_EXPERIMENT.training_config.training,
            learning_rate=5e-5,
            weight_decay=1e-5,
            logging_steps=10,
            save_steps=1000,
            save_total_limit=2,
            save_final_model=True,
            use_async_checkpoint_save=False,
            max_steps=10000,
            torch_compile=False,
            torch_compile_mode="default",
            lr_scheduler_type="cosine",
            dataloader_num_workers=4,
            gradient_accumulation_steps=12,
        ),
        global_batch_size=384,
    ),
    compute_config=replace(
        _BASE_EXPERIMENT.compute_config,
        families={
            name: replace(profile, per_device_batch_size=4)
            for name, profile in _BASE_EXPERIMENT.compute_config.families.items()
        },
    ),
)
