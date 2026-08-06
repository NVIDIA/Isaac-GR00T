# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared construction for single-task N2 Turbo RoboCasa Colossus runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

from groot.infra.experiments.training.configs.n2_robocasa365 import (
    EXPERIMENT as BASE_ROBOCASA365_EXPERIMENT,
)
from groot.infra.experiments.training.configs.n2_turbo_airplane_colossus import (
    TRAINING_CONFIG as BASE_TURBO_TRAINING_CONFIG,
)
from groot.infra.launcher.one_launcher.experiment_compute_config import ExperimentComputeConfig
from groot.infra.launcher.one_launcher.training_experiment import TrainingExperiment
from groot.infra.launcher.one_launcher.training_experiment_compute_config import (
    TrainingExperimentComputeConfig,
)
from groot.infra.launcher.one_launcher.training_experiment_evaluation_config import (
    TrainingExperimentEvaluationConfig,
)

SINGLE_TASK_MAX_STEPS = 500
_PROXY_RECIPE_SCHEMA = "n2-turbo-robocasa-proxy-recipe-v1"
_PROXY_RECIPE_TRAINING_FIELDS = {
    "learning_rate",
    "lr_scheduler_type",
    "max_grad_norm",
    "muon_impl",
    "muon_lr",
    "optim",
    "use_legacy_wd_application",
    "use_muon",
    "warmup_ratio",
    "warmup_steps",
    "weight_decay",
    "wsd_decay_type",
    "wsd_stable_ratio",
}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"proxy recipe contains a non-finite JSON constant: {value}")


def _load_proxy_recipe() -> dict[str, Any]:
    path = Path(__file__).with_name("n2_turbo_robocasa_proxy_recipe.json")
    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    if not isinstance(payload, dict) or set(payload) != {"schema", "training", "model"}:
        raise ValueError("proxy recipe must contain exactly schema, training, and model")
    if payload["schema"] != _PROXY_RECIPE_SCHEMA:
        raise ValueError("proxy recipe has an unsupported schema")
    training = payload["training"]
    if not isinstance(training, dict) or set(training) != _PROXY_RECIPE_TRAINING_FIELDS:
        raise ValueError("proxy recipe has an unexpected training field set")
    model = payload["model"]
    if (
        not isinstance(model, dict)
        or set(model) != {"action-cfg"}
        or not isinstance(model["action-cfg"], dict)
        or set(model["action-cfg"]) != {"state-dropout-prob"}
    ):
        raise ValueError("proxy recipe has an unexpected model field set")
    return payload


_PROXY_RECIPE = _load_proxy_recipe()

# Reuse the canonical RoboCasa-365 dataset metadata while keeping this module a
# construction helper rather than exposing an unsupported aggregate experiment.
BASE_DATA_CONFIG = replace(
    BASE_ROBOCASA365_EXPERIMENT.data_config,
    data={
        **deepcopy(BASE_ROBOCASA365_EXPERIMENT.data_config.data),
        "multiprocessing_context": "fork",
        # Fail fast when a staged dataset is incomplete instead of generating
        # statistics independently on each Ray worker.
        "auto_generate_stats": False,
    },
)

_ROBOCASA365_TRAINING_CONFIG = BASE_ROBOCASA365_EXPERIMENT.training_config
_TURBO_ROBOCASA_MODEL = deepcopy(BASE_TURBO_TRAINING_CONFIG.model)
# RoboCasa provides 16 action steps. Keep the checkpoint-compatible 48-token
# action head, but do not pad every sample with 32 masked tokens.
_TURBO_ROBOCASA_MODEL["action-cfg"]["max-action-horizon"] = 16
# RoboCasa has exactly three real camera modalities. Avoid appending the fully
# masked zero view inherited from the four-camera Airplane profile.
_TURBO_ROBOCASA_MODEL["max_n_cameras"] = 3
_TURBO_ROBOCASA_MODEL["action-cfg"]["state-dropout-prob"] = _PROXY_RECIPE["model"]["action-cfg"][
    "state-dropout-prob"
]

BASE_TRAINING_CONFIG = replace(
    _ROBOCASA365_TRAINING_CONFIG,
    training=replace(
        _ROBOCASA365_TRAINING_CONFIG.training,
        **_PROXY_RECIPE["training"],
        output_dir="/data/groot_ol/shared/checkpoints/{USER_SLUG}/{JOB_NAME}",
        logging_steps=1000,
        torch_compile=False,
        dataloader_num_workers=4,
        use_wandb=False,
        # The RoboCasa checkpoint-80000 is Wan2.1 and is not Turbo-compatible.
        start_from_checkpoint=BASE_TURBO_TRAINING_CONFIG.training.start_from_checkpoint,
        upload_checkpoints=False,
        msc_checkpoint_url="",
        msc_profile_url="",
        enable_profiling=False,
        save_final_model=True,
        # The async saver forks inside a Ray actor and has triggered
        # task_receiver objects_valid crashes; prefer resumable sync saves.
        use_async_checkpoint_save=False,
        gradient_accumulation_steps=1,
        log_distributed_grad_norm=False,
        straggler=replace(
            _ROBOCASA365_TRAINING_CONFIG.training.straggler,
            enabled=False,
        ),
        use_ddp=True,
        ddp_bucket_cap_mb=256,
    ),
    model_name=BASE_TURBO_TRAINING_CONFIG.model_name,
    model=_TURBO_ROBOCASA_MODEL,
)

# Dataclasses do not enforce runtime annotations. Reject malformed candidate
# JSON locally even though the resolved-config guard independently default-denies
# its values and shape.
for _numeric_name in (
    "learning_rate",
    "max_grad_norm",
    "muon_lr",
    "warmup_ratio",
    "weight_decay",
    "wsd_stable_ratio",
):
    _numeric_value = getattr(BASE_TRAINING_CONFIG.training, _numeric_name)
    if isinstance(_numeric_value, bool) or not isinstance(_numeric_value, int | float):
        raise TypeError(f"proxy recipe {_numeric_name} must be numeric")
    if not math.isfinite(float(_numeric_value)):
        raise ValueError(f"proxy recipe {_numeric_name} must be finite")
if not 0 < BASE_TRAINING_CONFIG.training.learning_rate <= 1:
    raise ValueError("proxy recipe learning_rate must be within (0, 1]")
if not 0 <= BASE_TRAINING_CONFIG.training.max_grad_norm <= 100:
    raise ValueError("proxy recipe max_grad_norm must be within [0, 100]")
if not 0 < BASE_TRAINING_CONFIG.training.muon_lr <= 1:
    raise ValueError("proxy recipe muon_lr must be within (0, 1]")
if not 0 <= BASE_TRAINING_CONFIG.training.warmup_ratio <= 1:
    raise ValueError("proxy recipe warmup_ratio must be within [0, 1]")
if not 0 <= BASE_TRAINING_CONFIG.training.weight_decay <= 1:
    raise ValueError("proxy recipe weight_decay must be within [0, 1]")
if not 0 <= BASE_TRAINING_CONFIG.training.wsd_stable_ratio <= 1:
    raise ValueError("proxy recipe wsd_stable_ratio must be within [0, 1]")
if (
    isinstance(BASE_TRAINING_CONFIG.training.warmup_steps, bool)
    or not isinstance(BASE_TRAINING_CONFIG.training.warmup_steps, int)
    or not 0 <= BASE_TRAINING_CONFIG.training.warmup_steps <= 10_000
):
    raise ValueError("proxy recipe warmup_steps must be an integer within [0, 10000]")
if BASE_TRAINING_CONFIG.training.lr_scheduler_type not in {"cosine", "linear", "wsd"}:
    raise ValueError("proxy recipe lr_scheduler_type is unsupported")
if BASE_TRAINING_CONFIG.training.wsd_decay_type not in {"cosine", "linear", "sqrt"}:
    raise ValueError("proxy recipe wsd_decay_type is unsupported")
if BASE_TRAINING_CONFIG.training.muon_impl not in {"dist", "single_batched"}:
    raise ValueError("proxy recipe muon_impl is unsupported")
if BASE_TRAINING_CONFIG.training.optim not in {
    "adafactor",
    "adamw_torch",
    "adamw_torch_fused",
}:
    raise ValueError("proxy recipe optim is unsupported")
for _boolean_name in ("use_legacy_wd_application", "use_muon"):
    if not isinstance(getattr(BASE_TRAINING_CONFIG.training, _boolean_name), bool):
        raise TypeError(f"proxy recipe {_boolean_name} must be boolean")
_state_dropout_probability = _TURBO_ROBOCASA_MODEL["action-cfg"]["state-dropout-prob"]
if (
    isinstance(_state_dropout_probability, bool)
    or not isinstance(_state_dropout_probability, int | float)
    or not math.isfinite(float(_state_dropout_probability))
    or not 0 <= _state_dropout_probability <= 1
):
    raise ValueError("proxy recipe state-dropout-prob must be numeric within [0, 1]")

# One Launcher maps Colossus to the gb300 compute family while enforcing one
# training GPU per station.
COMPUTE_CONFIG = TrainingExperimentComputeConfig(
    families={"gb300": ExperimentComputeConfig(per_device_batch_size=128)},
    env_vars={
        "GEAR_PRESERVE_CHECKPOINT_ROOT": "1",
        "GROOT_DISTRIBUTED_TIMEOUT_MINUTES": "30",
        "NCCL_SOCKET_NTHREADS": "4",
        "NCCL_NSOCKS_PERTHREAD": "4",
        "TORCH_NCCL_HIGH_PRIORITY": "1",
    },
)


def make_robocasa_single_task_experiment(
    *,
    dataset_path: str,
    evaluation_task: str,
    evaluation_num_trials: int,
    evaluation_seed: int = 0,
    evaluation_local_checkpoint: str | None = None,
    evaluation_remote_checkpoint: str | None = None,
    max_steps: int = SINGLE_TASK_MAX_STEPS,
    wandb_project: str,
) -> TrainingExperiment:
    """Build a one-GB300 experiment for one staged RoboCasa pretrain task."""
    data = deepcopy(BASE_DATA_CONFIG.data)
    data["datasets"][0]["dataset_paths"] = [dataset_path]
    data_config = replace(BASE_DATA_CONFIG, data=data)

    training_config = replace(
        BASE_TRAINING_CONFIG,
        training=replace(
            BASE_TRAINING_CONFIG.training,
            max_steps=max_steps,
            logging_steps=10,
            use_wandb=True,
            wandb_project=wandb_project,
        ),
        # One GB300 x 128 samples/GPU x one gradient-accumulation step.
        global_batch_size=128,
    )

    return TrainingExperiment(
        data_config=data_config,
        training_config=training_config,
        compute_config=COMPUTE_CONFIG,
        evaluation_config=TrainingExperimentEvaluationConfig(
            task=evaluation_task,
            num_trials=evaluation_num_trials,
            seed=evaluation_seed,
            local_checkpoint=evaluation_local_checkpoint,
            remote_checkpoint=evaluation_remote_checkpoint,
        ),
    )
