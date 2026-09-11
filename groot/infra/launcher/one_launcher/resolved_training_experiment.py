# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Resolved Python experiment ready for one-launcher submission."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
import copy
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

import yaml

from groot.core.training.configs.training.training_config import TrainingConfig
from groot.infra.launcher.one_launcher.resolved_training_experiment_files import (
    ResolvedTrainingExperimentFiles,
)
from groot.infra.launcher.one_launcher.training_experiment_evaluation_config import (
    TrainingExperimentEvaluationConfig,
)
from groot.infra.launcher.one_launcher.training_experiment_override import (
    TrainingExperimentOverride,
)
from groot.infra.launcher.one_launcher.training_launch_compute_profile import (
    TrainingLaunchComputeProfile,
)
from groot.infra.launcher.one_launcher.workflow_env import AUTHORITATIVE_WORKFLOW_ENV_NAMES

_OVERRIDE_ROOTS = {"compute", "data", "evaluation", "training"}
_COMPUTE_SIZE_TARGETS = {
    "num_gpus": "num_gpus",
    "num_nodes": "num_nodes",
    "per_device_batch_size": "batch_size",
}
_COMPUTE_SCALAR_TARGETS = {
    **_COMPUTE_SIZE_TARGETS,
    "osmo_exec_timeout": "osmo_exec_timeout",
    "skypilot_queue": "skypilot_queue",
}
_TRAINING_PATHS_FOR_COMPUTE_FIELDS = {
    "batch_size": "per_device_batch_size",
    "env_vars": "env_vars.NAME",
    "num_gpus": "num_gpus",
    "num_nodes": "num_nodes",
    "per_device_batch_size": "per_device_batch_size",
}
_DERIVED_GPU_INPUTS = {
    "batch_size",
    "global_batch_size",
    "gradient_accumulation_steps",
}
_PUBLIC_SIZE_PATHS = {
    "batch_size": "compute.per_device_batch_size",
    "global_batch_size": "training.global_batch_size",
    "gradient_accumulation_steps": "training.gradient_accumulation_steps",
    "num_gpus": "compute.num_gpus",
    "num_nodes": "compute.num_nodes",
}
_LAUNCHER_MANAGED_ENV_VARS = {*AUTHORITATIVE_WORKFLOW_ENV_NAMES, "GLOBAL_BATCH_SIZE"}


@dataclass(frozen=True)
class ResolvedTrainingExperiment:
    name: str
    compute_type: str
    data_section: Mapping[str, Any]
    training_section: Mapping[str, Any]
    model_section: Mapping[str, Any]
    num_nodes: int
    num_gpus: int
    batch_size: int
    global_batch_size: int
    gradient_accumulation_steps: int
    osmo_exec_timeout: str
    data_cli_section: Mapping[str, Any] = field(default_factory=dict)
    env_vars: Mapping[str, str] = field(default_factory=dict)
    skypilot_queue: str | None = None
    evaluation_section: Mapping[str, Any] | None = None

    def config_fingerprint(self, *, include_evaluation: bool = True) -> str:
        override_payload = self.override_payload()
        if override_payload["skypilot_queue"] is None:
            del override_payload["skypilot_queue"]
        if not include_evaluation or override_payload["evaluation"] is None:
            del override_payload["evaluation"]
        payload = {
            "name": self.name,
            "compute_type": self.compute_type,
            **override_payload,
        }
        return hashlib.sha256(yaml.safe_dump(payload, sort_keys=True).encode()).hexdigest()

    def with_overrides(
        self,
        overrides: Sequence[str],
        *,
        compute_profile: TrainingLaunchComputeProfile,
    ) -> ResolvedTrainingExperiment:
        if not overrides:
            return self
        payload = self.override_payload()
        explicit_size_fields: set[str] = set()
        for expression in overrides:
            override = TrainingExperimentOverride.init_from_expression(expression)
            self.validate_override_path(override)
            target_path = self.override_target_path(override)
            override.apply(
                payload,
                alias_aware=override.path[:2] in {("data", "launcher_data"), ("training", "model")},
                path=target_path,
            )
            size_field = self.synchronize_size_override(payload, target_path)
            if size_field is not None:
                explicit_size_fields.add(size_field)

        num_nodes, num_gpus, batch_size, global_batch_size, gradient_accumulation_steps = (
            self.reconcile_size_fields(
                payload,
                explicit_size_fields,
                compute_profile=compute_profile,
            )
        )
        training_section = self.mapping_value(payload, "training")
        training_section["global_batch_size"] = global_batch_size
        if (
            "gradient_accumulation_steps" in explicit_size_fields
            or "gradient_accumulation_steps" in training_section
        ):
            training_section["gradient_accumulation_steps"] = gradient_accumulation_steps

        return ResolvedTrainingExperiment(
            name=self.name,
            compute_type=self.compute_type,
            data_section=self.mapping_value(payload, "data"),
            data_cli_section=self.mapping_value(payload, "data_cli"),
            training_section=training_section,
            model_section=self.mapping_value(payload, "model"),
            evaluation_section=self.evaluation_values(payload),
            num_nodes=num_nodes,
            num_gpus=num_gpus,
            batch_size=batch_size,
            global_batch_size=global_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            env_vars=self.environment_values(payload),
            skypilot_queue=self.optional_string(payload, "skypilot_queue"),
            osmo_exec_timeout=self.required_string(payload, "osmo_exec_timeout"),
        )

    def override_payload(self) -> MutableMapping[str, Any]:
        return {
            "data": copy.deepcopy(dict(self.data_section)),
            "data_cli": copy.deepcopy(dict(self.data_cli_section)),
            "training": copy.deepcopy(dict(self.training_section)),
            "model": copy.deepcopy(dict(self.model_section)),
            "evaluation": (
                copy.deepcopy(dict(self.evaluation_section))
                if self.evaluation_section is not None
                else None
            ),
            "env_vars": copy.deepcopy(dict(self.env_vars)),
            "num_nodes": self.num_nodes,
            "num_gpus": self.num_gpus,
            "batch_size": self.batch_size,
            "global_batch_size": self.global_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "osmo_exec_timeout": self.osmo_exec_timeout,
            "skypilot_queue": self.skypilot_queue,
        }

    def validate_override_path(self, override: TrainingExperimentOverride) -> None:
        root = override.path[0]
        if root not in _OVERRIDE_ROOTS:
            supported = ", ".join(sorted(_OVERRIDE_ROOTS))
            raise ValueError(
                f"Unknown experiment override root {root!r}; supported roots: {supported}"
            )
        if len(override.path) == 1:
            raise ValueError(f"--set {override.expression!r} must target a value below {root}.")

        if root == "data":
            if override.path[1] == "launcher_data":
                if len(override.path) == 2:
                    raise ValueError(
                        "--set data.launcher_data must target a nested launcher data value"
                    )
                override.validate_cli_value()
            return

        if root == "evaluation":
            if self.evaluation_section is None:
                raise ValueError(f"Experiment {self.name!r} does not define an evaluation_config")
            if len(override.path) != 2 or override.path[1] not in {
                "local_checkpoint",
                "task",
                "num_trials",
                "num_parallel_trials",
                "inference_steps",
                "remote_checkpoint",
                "seed",
                "persist",
            }:
                raise ValueError(
                    "Evaluation overrides must target evaluation.task, evaluation.num_trials, "
                    "evaluation.num_parallel_trials, evaluation.inference_steps, "
                    "evaluation.seed, evaluation.persist, evaluation.local_checkpoint, or "
                    "evaluation.remote_checkpoint"
                )
            return

        if root == "training":
            if override.path[1] == "model" and len(override.path) == 2:
                raise ValueError("--set training.model must target a nested model value")
            if override.path[1] == "model_name":
                raise ValueError("Use --set training.model.name=... instead of training.model_name")
            if override.path[1] in _TRAINING_PATHS_FOR_COMPUTE_FIELDS:
                compute_path = _TRAINING_PATHS_FOR_COMPUTE_FIELDS[override.path[1]]
                raise ValueError(
                    f"Use --set compute.{compute_path}=... instead of training.{override.path[1]}"
                )
            override.validate_cli_value()
            return

        compute_key = override.path[1]
        if compute_key in _COMPUTE_SCALAR_TARGETS:
            if len(override.path) != 2:
                raise ValueError(f"--set compute.{compute_key} is a scalar")
            return
        if compute_key != "env_vars":
            supported = ", ".join([*_COMPUTE_SCALAR_TARGETS, "env_vars.NAME"])
            raise ValueError(
                f"Unknown compute override {compute_key!r}; supported values: {supported}"
            )
        if len(override.path) != 3:
            raise ValueError(
                "--set compute.env_vars overrides must use compute.env_vars.NAME=VALUE"
            )
        if override.path[2] in _LAUNCHER_MANAGED_ENV_VARS:
            raise ValueError(
                f"--set cannot override launcher-managed environment variable {override.path[2]}"
            )

    def override_target_path(self, override: TrainingExperimentOverride) -> tuple[str, ...]:
        root = override.path[0]
        if root == "data":
            if override.path[1] == "launcher_data":
                return ("data_cli", *override.path[2:])
            return ("data", *override.path[1:])
        if root == "training":
            if override.path[1] == "model":
                return ("model", *override.path[2:])
            return ("training", *override.path[1:])
        if root == "evaluation":
            return override.path
        compute_key = override.path[1]
        if compute_key == "env_vars":
            return ("env_vars", *override.path[2:])
        return (_COMPUTE_SCALAR_TARGETS[compute_key],)

    def synchronize_size_override(
        self,
        payload: MutableMapping[str, Any],
        target_path: tuple[str, ...],
    ) -> str | None:
        training_section = self.mapping_value(payload, "training")
        if target_path == ("training", "global_batch_size"):
            payload["global_batch_size"] = training_section["global_batch_size"]
            return "global_batch_size"
        if target_path == ("training", "gradient_accumulation_steps"):
            payload["gradient_accumulation_steps"] = training_section["gradient_accumulation_steps"]
            return "gradient_accumulation_steps"
        root = target_path[0]
        if root in {"batch_size", "num_gpus", "num_nodes"}:
            return root
        return None

    def reconcile_size_fields(
        self,
        payload: MutableMapping[str, Any],
        explicit_size_fields: set[str],
        *,
        compute_profile: TrainingLaunchComputeProfile,
    ) -> tuple[int, int, int, int, int]:
        num_nodes = self.positive_integer(payload, "num_nodes")
        num_gpus = self.positive_integer(payload, "num_gpus")
        batch_size = self.positive_integer(payload, "batch_size")
        global_batch_size = self.positive_integer(payload, "global_batch_size")
        gradient_accumulation_steps = self.positive_integer(
            payload,
            "gradient_accumulation_steps",
        )
        batch_denominator = batch_size * gradient_accumulation_steps

        if "num_gpus" in explicit_size_fields:
            effective_global_batch_size = num_gpus * batch_denominator
            if (
                "global_batch_size" in explicit_size_fields
                and global_batch_size != effective_global_batch_size
            ):
                raise ValueError(
                    "Experiment overrides require training.global_batch_size == "
                    "compute.num_gpus * compute.per_device_batch_size * "
                    "training.gradient_accumulation_steps; got "
                    f"{global_batch_size} != {num_gpus} * {batch_size} * "
                    f"{gradient_accumulation_steps}"
                )
            global_batch_size = effective_global_batch_size
        elif explicit_size_fields & _DERIVED_GPU_INPUTS:
            if global_batch_size % batch_denominator != 0:
                raise ValueError(
                    f"Experiment training.global_batch_size {global_batch_size} must be "
                    f"divisible by compute.per_device_batch_size {batch_size} times "
                    "training.gradient_accumulation_steps "
                    f"{gradient_accumulation_steps}"
                )
            num_gpus = global_batch_size // batch_denominator

        if "num_nodes" not in explicit_size_fields and (
            "num_gpus" in explicit_size_fields or bool(explicit_size_fields & _DERIVED_GPU_INPUTS)
        ):
            num_nodes = compute_profile.num_nodes_for_num_gpus(num_gpus)
        gpu_capacity = num_nodes * compute_profile.gpus_per_node(num_gpus)
        if gpu_capacity < num_gpus:
            raise ValueError(
                f"Experiment compute.num_nodes {num_nodes} provides capacity for "
                f"{gpu_capacity} GPUs, fewer than compute.num_gpus {num_gpus}"
            )
        return (
            num_nodes,
            num_gpus,
            batch_size,
            global_batch_size,
            gradient_accumulation_steps,
        )

    def positive_integer(self, payload: Mapping[str, Any], key: str) -> int:
        value = payload.get(key)
        public_path = _PUBLIC_SIZE_PATHS[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(  # noqa: TRY004 - invalid user-supplied --set value
                f"Experiment override {public_path} must be an integer, got {value!r}"
            )
        if value <= 0:
            raise ValueError(
                f"Experiment override {public_path} must be greater than 0, got {value}"
            )
        return value

    def mapping_value(
        self,
        payload: Mapping[str, Any],
        key: str,
    ) -> MutableMapping[str, Any]:
        value = payload.get(key)
        if not isinstance(value, MutableMapping):
            raise ValueError(  # noqa: TRY004 - invalid user-supplied --set value
                f"Experiment override {key} must remain a mapping"
            )
        return value

    def environment_values(self, payload: Mapping[str, Any]) -> Mapping[str, str]:
        values = self.mapping_value(payload, "env_vars")
        environment: dict[str, str] = {}
        for key, value in values.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Experiment env_vars override names must be non-empty strings")
            if isinstance(value, Mapping | list):
                raise ValueError(  # noqa: TRY004 - invalid user-supplied --set value
                    "Experiment env_vars override values must be YAML scalars"
                )
            if value is not None:
                environment[key] = str(value)
        return environment

    def evaluation_values(self, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        value = payload.get("evaluation")
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError(  # noqa: TRY004 - invalid user-supplied --set value
                "Experiment override evaluation must remain a mapping"
            )
        unknown = set(value) - {
            "local_checkpoint",
            "task",
            "num_trials",
            "num_parallel_trials",
            "inference_steps",
            "remote_checkpoint",
            "seed",
            "persist",
        }
        missing = {"task", "num_trials"} - set(value)
        if unknown:
            raise ValueError(
                f"Experiment evaluation contains unknown fields: {', '.join(sorted(unknown))}"
            )
        if missing:
            raise ValueError(
                f"Experiment evaluation is missing fields: {', '.join(sorted(missing))}"
            )
        try:
            config = TrainingExperimentEvaluationConfig(
                task=value["task"],
                num_trials=value["num_trials"],
                num_parallel_trials=value.get("num_parallel_trials", 1),
                inference_steps=value.get("inference_steps"),
                seed=value.get("seed", 0),
                persist=value.get("persist", False),
                local_checkpoint=value.get("local_checkpoint"),
                remote_checkpoint=value.get("remote_checkpoint"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid experiment evaluation override: {exc}") from exc
        return {
            key: item
            for key, item in {
                "task": config.task,
                "num_trials": config.num_trials,
                "num_parallel_trials": config.num_parallel_trials,
                "inference_steps": config.inference_steps,
                "seed": config.seed,
                "persist": config.persist,
                "local_checkpoint": config.local_checkpoint,
                "remote_checkpoint": config.remote_checkpoint,
            }.items()
            if item is not None
            and not (key == "persist" and item is False)
            and not (key == "num_parallel_trials" and item == 1)
        }

    def optional_string(self, payload: Mapping[str, Any], key: str) -> str | None:
        value = payload.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Experiment override compute.{key} must be a non-empty string")
        return value

    def required_string(self, payload: Mapping[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Experiment override compute.{key} must be a non-empty string")
        return value

    def write_configs(self, output_root: Path) -> ResolvedTrainingExperimentFiles:
        output_dir = output_root / self.name / self.compute_type
        output_dir.mkdir(parents=True, exist_ok=True)
        files = ResolvedTrainingExperimentFiles(
            data_config=output_dir / "data.yaml",
            training_config=output_dir / "training.yaml",
            full_config=output_dir / "resolved_config.yaml",
        )
        data_payload: dict[str, Any] = {"data": dict(self.data_section)}
        training_discriminator = self.training_section.get("_config")
        if training_discriminator is not None:
            # The config loader needs subtype-only fields in data.yaml to construct
            # the registered training dataclass. Base fields stay in training.yaml
            # and on the CLI, preserving their existing override precedence;
            # global_batch_size stays launcher-only and is expanded there as well.
            base_fields = TrainingConfig.__dataclass_fields__
            data_payload["training"] = {
                key: value
                for key, value in self.training_section.items()
                if key == "_config" or (key not in base_fields and key != "global_batch_size")
            }
        files.data_config.write_text(yaml.safe_dump(data_payload, sort_keys=False))
        training_payload: dict[str, Any] = {}
        if self.data_cli_section:
            training_payload["data"] = dict(self.data_cli_section)
        training_payload["training"] = dict(self.training_section)
        training_payload["model"] = dict(self.model_section)
        files.training_config.write_text(yaml.safe_dump(training_payload, sort_keys=False))
        files.full_config.write_text(yaml.safe_dump(dict(self.summary(files)), sort_keys=False))
        return files

    def summary(self, files: ResolvedTrainingExperimentFiles) -> Mapping[str, Any]:
        summary = {
            "experiment": self.name,
            "compute_type": self.compute_type,
            "num_nodes": self.num_nodes,
            "num_gpus": self.num_gpus,
            "batch_size": self.batch_size,
            "global_batch_size": self.global_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "osmo_exec_timeout": self.osmo_exec_timeout,
            "data_config": str(files.data_config),
            "training_config": str(files.training_config),
            "full_config": str(files.full_config),
            "data": dict(self.data_section),
            "data_cli": dict(self.data_cli_section),
            "training": dict(self.training_section),
            "model": dict(self.model_section),
            "env_vars": dict(self.env_vars),
            "skypilot_queue": self.skypilot_queue,
        }
        if self.evaluation_section is not None:
            summary["evaluation"] = dict(self.evaluation_section)
        return summary
