# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python-authored training experiment definition."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from dataclasses import dataclass
from typing import Any

from groot.infra.launcher.one_launcher.resolved_training_experiment import (
    ResolvedTrainingExperiment,
)
from groot.infra.launcher.one_launcher.training_experiment_compute_config import (
    TrainingExperimentComputeConfig,
)
from groot.infra.launcher.one_launcher.training_experiment_data_config import (
    TrainingExperimentDataConfig,
)
from groot.infra.launcher.one_launcher.training_experiment_evaluation_config import (
    TrainingExperimentEvaluationConfig,
)
from groot.infra.launcher.one_launcher.training_experiment_training_config import (
    TrainingExperimentTrainingConfig,
)
from groot.infra.launcher.one_launcher.training_launch_compute_profile import (
    TrainingLaunchComputeProfile,
)


@dataclass(frozen=True)
class TrainingExperiment:
    data_config: TrainingExperimentDataConfig
    training_config: TrainingExperimentTrainingConfig
    compute_config: TrainingExperimentComputeConfig
    evaluation_config: TrainingExperimentEvaluationConfig | None = None

    def resolve(
        self,
        *,
        name: str,
        compute_type: str,
        compute_profile: TrainingLaunchComputeProfile,
    ) -> ResolvedTrainingExperiment:
        compute_family = self.compute_config.family(experiment=name, compute_type=compute_type)
        gradient_accumulation_steps = self.training_config.training.gradient_accumulation_steps
        global_batch_size = self.training_config.global_batch_size
        denominator = compute_family.per_device_batch_size * gradient_accumulation_steps
        if denominator <= 0:
            raise ValueError(f"Experiment {name} has invalid batch denominator {denominator}")
        if global_batch_size % denominator != 0:
            raise ValueError(
                f"Experiment {name} global batch size {global_batch_size} "
                f"must be divisible by {compute_type} per-device batch size "
                f"{compute_family.per_device_batch_size} times gradient accumulation "
                f"{gradient_accumulation_steps}"
            )
        num_gpus = global_batch_size // denominator
        return ResolvedTrainingExperiment(
            name=name,
            compute_type=compute_type,
            data_section=self.data_section(),
            data_cli_section=self.data_cli_section(),
            training_section=self.training_section(global_batch_size),
            model_section=self.model_section(),
            evaluation_section=self.evaluation_section(),
            num_nodes=compute_profile.num_nodes_for_num_gpus(num_gpus),
            num_gpus=num_gpus,
            batch_size=compute_family.per_device_batch_size,
            global_batch_size=global_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            env_vars=self.compute_config.env_section(),
            skypilot_queue=compute_family.skypilot_queue,
            osmo_exec_timeout=compute_family.osmo_exec_timeout,
        )

    def data_section(self) -> Mapping[str, Any]:
        return self.serializable_value(self.data_config.data)

    def data_cli_section(self) -> Mapping[str, Any]:
        return self.serializable_value(self.data_config.launcher_data)

    def training_section(self, global_batch_size: int) -> Mapping[str, Any]:
        return self.merged_mapping(
            self.dataclass_overrides(self.training_config.training),
            {"global_batch_size": global_batch_size},
        )

    def model_section(self) -> Mapping[str, Any]:
        return self.merged_mapping(
            {"name": self.training_config.model_name},
            self.training_config.model,
        )

    def evaluation_section(self) -> Mapping[str, Any] | None:
        if self.evaluation_config is None:
            return None
        return {
            key: value
            for key, value in self.serializable_value(self.evaluation_config).items()
            if value is not None
        }

    def dataclass_overrides(self, instance: Any) -> Mapping[str, Any]:
        if not dataclasses.is_dataclass(instance):
            raise TypeError(f"Expected dataclass, got {type(instance).__name__}")
        default = type(instance)()
        current_body = dataclasses.asdict(instance)
        default_body = dataclasses.asdict(default)
        return {
            key: self.serializable_value(value)
            for key, value in current_body.items()
            if value != default_body.get(key)
        }

    def merged_mapping(
        self,
        base: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        merged = {str(key): self.serializable_value(value) for key, value in base.items()}
        for raw_key, value in overrides.items():
            key = str(raw_key)
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = self.merged_mapping(merged[key], value)
            else:
                merged[key] = self.serializable_value(value)
        return merged

    def serializable_value(self, value: Any) -> Any:
        if dataclasses.is_dataclass(value):
            return self.serializable_value(dataclasses.asdict(value))
        if isinstance(value, Mapping):
            return {
                self.serializable_key(key): self.serializable_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.serializable_value(item) for item in value]
        if isinstance(value, tuple):
            return [self.serializable_value(item) for item in value]
        return value

    def serializable_key(self, key: Any) -> Any:
        if key is None or isinstance(key, str | int | float | bool):
            return key
        return str(key)
