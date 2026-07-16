# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""YAML-backed golden-run variant registry.

The YAML map key selects a concrete training/eval variant. Each variant declares
the canonical ``run_type`` value persisted in YAM requests and emitted as the
Grafana metric label. Multiple variants can share the same ``run_type`` when the
task name or evaluation type already provides the needed metric split. Scheduled
golden-run fan-out runs a variant on the scheduler's ref by default, or on every
ref listed in the variant's optional ``branches`` field. Adding a new
golden-run variant should only require adding one YAML entry, plus either a
legacy launcher script, structured data/training configs, or a one-launcher
Python experiment if the training recipe itself is new. One-launcher experiment
variants derive compute sizing from the Python ``EXPERIMENT`` object rather than
duplicating per-platform training shape in YAML.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml

from groot.infra.ci.compute_type import ComputeType
from groot.infra.ci.evaluation_type import EvaluationType

CONFIG_PATH = Path(__file__).with_name("golden_runs.yaml")
CONFIG_BACKED_LAUNCHER = "python -m groot.internal.infra.job.main.launch_trainer"
_RUN_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_VARIANT_FIELDS = {
    "label",
    "launcher",
    "experiment",
    "data_config",
    "training_config",
    "run_type",
    "branches",
    "eval",
    "platforms",
    "artifacts",
    "uses_deltacat",
}
_EVAL_FIELDS = {
    "type",
    "evaluation_type",
    "tasks",
    "flags",
    "policy_type",
    "checkpoint_step",
    "num_eval_nodes",
    "sim_eval_pool",
}
_PLATFORM_FIELDS = {"num_gpus", "num_nodes", "batch_size", "gradient_accumulation_steps", "pool"}
_ARTIFACT_FIELDS = {"base_model", "dataset", "final_model"}
_HF_MODEL_ARTIFACT_FIELDS = {"repo_id", "revision"}
_HF_DATASET_ARTIFACT_FIELDS = {"repo_id", "revision", "local_name"}


@dataclass(frozen=True)
class GoldenRunEvalConfig:
    """Evaluation settings owned by a run variant."""

    evaluation_type: EvaluationType
    tasks: tuple[str, ...]
    flags: tuple[str, ...]
    policy_type: str
    # The training step whose checkpoint will be evaluated. The golden-run
    # train job writes <checkpoint_root>/checkpoint-<step> to ckpts_path.txt,
    # which the sim_eval job then reads. Defaults to 10000 (the historical
    # n17es / n2-minwam max_steps); variants that train for a different number
    # of steps must override to the matching value.
    checkpoint_step: int = 10000
    # Number of L40 eval-cluster nodes (8 GPUs each) to use for sim eval.
    # Default 1 matches the historical golden-run behaviour. Variants whose
    # eval is bottlenecked on serial task time (e.g. gr00t_n2 policies which
    # force num_envs=1 and thus need more parallel evaluator actors to keep
    # the 50-task robocasa365_target50 suite under the 24h sim_eval timeout)
    # may bump this up.
    num_eval_nodes: int = 1
    # OSMO pool for the sim-eval cluster. Empty ("") means use the CI job default
    # (groot-l40-ci-02). Set per-variant to override -- e.g. n2-robocasa365 uses
    # groot-h100-ci-02, which runs that eval ~1.65x faster at the same success rate.
    sim_eval_pool: str = ""

    @property
    def flags_string(self) -> str:
        return " ".join(self.flags)


@dataclass(frozen=True)
class GoldenRunPlatformConfig:
    """Compute-platform settings consumed by golden-run CI cluster orchestration."""

    num_gpus: int
    num_nodes: int
    batch_size: int
    gradient_accumulation_steps: int = 1
    pool: str | None = None


@dataclass(frozen=True)
class GoldenRunHFModelArtifactConfig:
    """Hugging Face model artifact location owned by a run variant."""

    repo_id: str
    revision: str


@dataclass(frozen=True)
class GoldenRunHFDatasetArtifactConfig:
    """Hugging Face dataset artifact location owned by a run variant."""

    repo_id: str
    revision: str
    local_name: str


@dataclass(frozen=True)
class GoldenRunArtifactsConfig:
    """Canonical input and output artifact locations for a run variant."""

    base_model: GoldenRunHFModelArtifactConfig
    dataset: GoldenRunHFDatasetArtifactConfig
    final_model: GoldenRunHFModelArtifactConfig


@dataclass(frozen=True)
class GoldenRunVariant:
    """One golden-run variant loaded from YAML."""

    run_type: str
    label: str
    launcher: str
    experiment: str | None
    data_config: str | None
    training_config: str | None
    branches: tuple[str, ...]
    eval: GoldenRunEvalConfig | None
    platforms: dict[ComputeType, GoldenRunPlatformConfig]
    artifacts: GoldenRunArtifactsConfig | None
    uses_deltacat: bool = False

    @property
    def compute_types(self) -> tuple[ComputeType, ...]:
        if self.platforms:
            return _ordered_compute_types(platform.value for platform in self.platforms)
        if self.experiment:
            return _experiment_compute_types(self.experiment)
        return ()

    def platform(self, compute_type: str | ComputeType) -> GoldenRunPlatformConfig:
        compute = ComputeType(compute_type)
        if not self.platforms and self.experiment:
            if compute not in self.compute_types:
                supported = ", ".join(c.value for c in self.compute_types)
                raise ValueError(
                    f"Run type {self.run_type!r} does not support compute_type "
                    f"{compute.value!r}. Supported: {supported}"
                )
            return _experiment_platform(self.experiment, compute)
        try:
            return self.platforms[compute]
        except KeyError as exc:
            supported = ", ".join(c.value for c in self.compute_types)
            raise ValueError(
                f"Run type {self.run_type!r} does not support compute_type "
                f"{compute.value!r}. Supported: {supported}"
            ) from exc


def load_variants(path: Path | None = None) -> dict[str, GoldenRunVariant]:
    """Load and validate the golden-run variant registry."""
    config_path = path or CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: top-level YAML must be a mapping")
    raw_variants = raw.get("variants")
    if not isinstance(raw_variants, dict) or not raw_variants:
        raise ValueError(f"{config_path}: expected non-empty 'variants' mapping")

    variants: dict[str, GoldenRunVariant] = {}
    for variant_name, body in raw_variants.items():
        if not isinstance(variant_name, str) or not _RUN_TYPE_RE.fullmatch(variant_name):
            raise ValueError(
                f"{config_path}: invalid variant key {variant_name!r}; use lowercase "
                "letters, numbers, '.', '_' or '-'"
            )
        if variant_name in variants:
            raise ValueError(f"{config_path}: duplicate variant {variant_name!r}")
        variants[variant_name] = _parse_variant(config_path, variant_name, body)
    return variants


def get_variant(run_type: str, path: Path | None = None) -> GoldenRunVariant:
    variants = load_variants(path)
    try:
        return variants[run_type]
    except KeyError as exc:
        valid = ", ".join(sorted(variants))
        raise ValueError(
            f"Unknown golden-run variant {run_type!r}. Valid variants: {valid}"
        ) from exc


def validate_run_type(run_type: str, path: Path | None = None) -> str:
    """Validate and return the metric-label run type from the variant selector."""
    return get_variant(run_type, path).run_type


def valid_run_types(path: Path | None = None) -> tuple[str, ...]:
    return tuple(sorted(load_variants(path)))


def _parse_variant(config_path: Path, run_type: str, body: Any) -> GoldenRunVariant:
    if not isinstance(body, dict):
        raise TypeError(f"{config_path}: variants.{run_type} must be a mapping")
    unknown = set(body) - _VARIANT_FIELDS
    if unknown:
        raise ValueError(
            f"{config_path}: variants.{run_type} has unknown field(s): {', '.join(sorted(unknown))}"
        )
    label = _required_str(config_path, body, f"variants.{run_type}.label")
    metric_run_type = _required_str(config_path, body, f"variants.{run_type}.run_type")
    if not isinstance(metric_run_type, str) or not _RUN_TYPE_RE.fullmatch(metric_run_type):
        raise ValueError(
            f"{config_path}: variants.{run_type}.run_type must use lowercase letters, numbers, '.', '_' or '-'"
        )
    launcher = _optional_str(config_path, body, f"variants.{run_type}.launcher") or None
    experiment = _optional_str(config_path, body, f"variants.{run_type}.experiment") or None
    data_config = _optional_str(config_path, body, f"variants.{run_type}.data_config") or None
    training_config = (
        _optional_str(config_path, body, f"variants.{run_type}.training_config") or None
    )
    if bool(data_config) != bool(training_config):
        raise ValueError(
            f"{config_path}: variants.{run_type} must set both data_config and training_config, or neither"
        )
    if experiment and data_config:
        raise ValueError(
            f"{config_path}: variants.{run_type}.experiment cannot be combined with "
            "data_config and training_config"
        )
    if experiment:
        if launcher:
            raise ValueError(
                f"{config_path}: variants.{run_type}.launcher must be omitted when experiment is set"
            )
        launcher = CONFIG_BACKED_LAUNCHER
    elif data_config:
        if launcher:
            raise ValueError(
                f"{config_path}: variants.{run_type}.launcher must be omitted when "
                "data_config and training_config are set"
            )
        launcher = CONFIG_BACKED_LAUNCHER
    elif not launcher:
        raise ValueError(
            f"{config_path}: variants.{run_type}.launcher is required unless experiment "
            "or data_config and training_config are set"
        )
    branches = _optional_branch_list(config_path, body, f"variants.{run_type}.branches")
    duplicate_branches = _duplicates(branches)
    if duplicate_branches:
        raise ValueError(
            f"{config_path}: variants.{run_type}.branches contains duplicate "
            f"branch(es): {', '.join(duplicate_branches)}"
        )
    eval_cfg = _parse_eval(config_path, run_type, body.get("eval"))
    platforms = _parse_platforms(
        config_path,
        run_type,
        body.get("platforms"),
        experiment_backed=bool(experiment),
    )
    artifacts = _parse_artifacts(config_path, run_type, body.get("artifacts"))
    uses_deltacat = _optional_bool(config_path, body, f"variants.{run_type}.uses_deltacat")
    return GoldenRunVariant(
        run_type=metric_run_type,
        label=label,
        launcher=launcher,
        experiment=experiment,
        data_config=data_config,
        training_config=training_config,
        branches=tuple(branches),
        eval=eval_cfg,
        platforms=platforms,
        artifacts=artifacts,
        uses_deltacat=uses_deltacat,
    )


def _parse_eval(config_path: Path, run_type: str, body: Any) -> GoldenRunEvalConfig | None:
    if body is None:
        return None
    if not isinstance(body, dict):
        raise TypeError(f"{config_path}: variants.{run_type}.eval must be a mapping")
    if not body:
        return None
    unknown = set(body) - _EVAL_FIELDS
    if unknown:
        raise ValueError(
            f"{config_path}: variants.{run_type}.eval has unknown field(s): {', '.join(sorted(unknown))}"
        )
    evaluation_type = _parse_evaluation_type(config_path, body, f"variants.{run_type}.eval")
    tasks = _required_str_list(config_path, body, f"variants.{run_type}.eval.tasks")
    flags = _optional_str_list(config_path, body, f"variants.{run_type}.eval.flags")
    policy_type = _optional_str(config_path, body, f"variants.{run_type}.eval.policy_type")
    checkpoint_step = _optional_positive_int(
        config_path, body, f"variants.{run_type}.eval.checkpoint_step", default=10000
    )
    num_eval_nodes = _optional_positive_int(
        config_path, body, f"variants.{run_type}.eval.num_eval_nodes", default=1
    )
    sim_eval_pool = _optional_str(config_path, body, f"variants.{run_type}.eval.sim_eval_pool")
    return GoldenRunEvalConfig(
        evaluation_type=evaluation_type,
        tasks=tuple(tasks),
        flags=tuple(flags),
        policy_type=policy_type,
        checkpoint_step=checkpoint_step,
        num_eval_nodes=num_eval_nodes,
        sim_eval_pool=sim_eval_pool,
    )


def _parse_evaluation_type(
    config_path: Path, body: dict[str, Any], field_prefix: str
) -> EvaluationType:
    raw = body.get("type", body.get("evaluation_type"))
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{config_path}: {field_prefix}.type must be one of: yam, yam_simulation")
    normalized = raw.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "yam": EvaluationType.YAM,
        "real_yam": EvaluationType.YAM,
        "yam_real": EvaluationType.YAM,
        "yam_sim": EvaluationType.YAM_SIMULATION,
        "yam_simulation": EvaluationType.YAM_SIMULATION,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        valid = ", ".join(sorted(aliases))
        raise ValueError(
            f"{config_path}: {field_prefix}.type has unknown evaluation type {raw!r}. Valid aliases: {valid}"
        ) from exc


def _parse_platforms(
    config_path: Path,
    run_type: str,
    body: Any,
    *,
    experiment_backed: bool = False,
) -> dict[ComputeType, GoldenRunPlatformConfig]:
    if body is None and experiment_backed:
        return {}
    if not isinstance(body, dict):
        raise TypeError(f"{config_path}: variants.{run_type}.platforms must be a mapping")
    platforms: dict[ComputeType, GoldenRunPlatformConfig] = {}
    for raw_name, raw_platform in body.items():
        try:
            compute = ComputeType(raw_name)
        except ValueError as exc:
            valid = ", ".join(c.value for c in ComputeType)
            raise ValueError(
                f"{config_path}: variants.{run_type}.platforms has unknown platform "
                f"{raw_name!r}. Valid platforms: {valid}"
            ) from exc
        if not isinstance(raw_platform, dict):
            raise TypeError(
                f"{config_path}: variants.{run_type}.platforms.{raw_name} must be a mapping"
            )
        unknown = set(raw_platform) - _PLATFORM_FIELDS
        if unknown:
            raise ValueError(
                f"{config_path}: variants.{run_type}.platforms.{raw_name} has "
                f"unknown field(s): {', '.join(sorted(unknown))}"
            )
        platforms[compute] = GoldenRunPlatformConfig(
            num_gpus=_required_int(
                config_path, raw_platform, f"variants.{run_type}.platforms.{raw_name}.num_gpus"
            ),
            num_nodes=_required_int(
                config_path, raw_platform, f"variants.{run_type}.platforms.{raw_name}.num_nodes"
            ),
            batch_size=_required_int(
                config_path, raw_platform, f"variants.{run_type}.platforms.{raw_name}.batch_size"
            ),
            gradient_accumulation_steps=_optional_positive_int(
                config_path,
                raw_platform,
                f"variants.{run_type}.platforms.{raw_name}.gradient_accumulation_steps",
                default=1,
            ),
            pool=_optional_str(
                config_path, raw_platform, f"variants.{run_type}.platforms.{raw_name}.pool"
            )
            or None,
        )
    return platforms


def _experiment_compute_types(experiment: str) -> tuple[ComputeType, ...]:
    experiment_obj = _load_experiment(experiment)
    return _ordered_compute_types(experiment_obj.compute_config.families)


def _experiment_platform(
    experiment: str,
    compute_type: ComputeType,
) -> GoldenRunPlatformConfig:
    try:
        from groot.infra.launcher.one_launcher.training_experiment_resolver import (
            TrainingExperimentResolver,
        )
    except ImportError as exc:
        raise ValueError(
            f"Run type backed by experiment {experiment!r} has no YAML platforms and "
            "this package cannot resolve one-launcher experiments."
        ) from exc

    resolved = TrainingExperimentResolver.init_default().resolve(
        name=experiment,
        compute_type=compute_type.value,
    )
    return GoldenRunPlatformConfig(
        num_gpus=resolved.num_gpus,
        num_nodes=resolved.num_nodes,
        batch_size=resolved.batch_size,
        gradient_accumulation_steps=resolved.gradient_accumulation_steps,
    )


def _load_experiment(experiment: str) -> Any:
    try:
        from groot.infra.launcher.one_launcher.training_experiment_resolver import (
            TrainingExperimentResolver,
        )
    except ImportError as exc:
        raise ValueError(
            f"Run type backed by experiment {experiment!r} has no YAML platforms and "
            "this package cannot resolve one-launcher experiments."
        ) from exc

    return TrainingExperimentResolver.init_default().load(experiment)


def _ordered_compute_types(raw_names: Iterable[str | ComputeType]) -> tuple[ComputeType, ...]:
    compute_types: set[ComputeType] = set()
    for raw_name in raw_names:
        compute_types.add(ComputeType(raw_name))
    return tuple(compute_type for compute_type in ComputeType if compute_type in compute_types)


def _parse_artifacts(
    config_path: Path, run_type: str, body: Any
) -> GoldenRunArtifactsConfig | None:
    if body is None:
        return None
    if not isinstance(body, dict):
        raise TypeError(f"{config_path}: variants.{run_type}.artifacts must be a mapping")
    unknown = set(body) - _ARTIFACT_FIELDS
    if unknown:
        raise ValueError(
            f"{config_path}: variants.{run_type}.artifacts has unknown field(s): {', '.join(sorted(unknown))}"
        )
    base_model = _parse_hf_model_artifact(
        config_path,
        _required_mapping(config_path, body, f"variants.{run_type}.artifacts.base_model"),
        f"variants.{run_type}.artifacts.base_model",
    )
    dataset = _parse_hf_dataset_artifact(
        config_path,
        _required_mapping(config_path, body, f"variants.{run_type}.artifacts.dataset"),
        f"variants.{run_type}.artifacts.dataset",
    )
    final_model = _parse_hf_model_artifact(
        config_path,
        _required_mapping(config_path, body, f"variants.{run_type}.artifacts.final_model"),
        f"variants.{run_type}.artifacts.final_model",
    )
    return GoldenRunArtifactsConfig(
        base_model=base_model,
        dataset=dataset,
        final_model=final_model,
    )


def _parse_hf_model_artifact(
    config_path: Path, body: dict[str, Any], field_prefix: str
) -> GoldenRunHFModelArtifactConfig:
    unknown = set(body) - _HF_MODEL_ARTIFACT_FIELDS
    if unknown:
        raise ValueError(
            f"{config_path}: {field_prefix} has unknown field(s): {', '.join(sorted(unknown))}"
        )
    return GoldenRunHFModelArtifactConfig(
        repo_id=_required_str(config_path, body, f"{field_prefix}.repo_id"),
        revision=_required_str(config_path, body, f"{field_prefix}.revision"),
    )


def _parse_hf_dataset_artifact(
    config_path: Path, body: dict[str, Any], field_prefix: str
) -> GoldenRunHFDatasetArtifactConfig:
    unknown = set(body) - _HF_DATASET_ARTIFACT_FIELDS
    if unknown:
        raise ValueError(
            f"{config_path}: {field_prefix} has unknown field(s): {', '.join(sorted(unknown))}"
        )
    return GoldenRunHFDatasetArtifactConfig(
        repo_id=_required_str(config_path, body, f"{field_prefix}.repo_id"),
        revision=_required_str(config_path, body, f"{field_prefix}.revision"),
        local_name=_required_str(config_path, body, f"{field_prefix}.local_name"),
    )


def _required_mapping(config_path: Path, body: dict[str, Any], field: str) -> dict[str, Any]:
    key = field.rsplit(".", 1)[-1]
    value = body.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{config_path}: {field} must be a non-empty mapping")
    return value


def _required_str(config_path: Path, body: dict[str, Any], field: str) -> str:
    key = field.rsplit(".", 1)[-1]
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{config_path}: {field} must be a non-empty string")
    return value


def _required_int(config_path: Path, body: dict[str, Any], field: str) -> int:
    key = field.rsplit(".", 1)[-1]
    value = body.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{config_path}: {field} must be a positive integer")
    return value


def _optional_positive_int(
    config_path: Path, body: dict[str, Any], field: str, *, default: int
) -> int:
    key = field.rsplit(".", 1)[-1]
    if key not in body:
        return default
    value = body[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{config_path}: {field} must be a positive integer")
    return value


def _optional_bool(config_path: Path, body: dict[str, Any], field: str) -> bool:
    key = field.rsplit(".", 1)[-1]
    if key not in body:
        return False
    value = body[key]
    if not isinstance(value, bool):
        raise TypeError(f"{config_path}: {field} must be a boolean")
    return value


def _required_str_list(config_path: Path, body: dict[str, Any], field: str) -> list[str]:
    key = field.rsplit(".", 1)[-1]
    value = body.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{config_path}: {field} must be a non-empty list of strings")
    return value


def _optional_str_list(config_path: Path, body: dict[str, Any], field: str) -> list[str]:
    key = field.rsplit(".", 1)[-1]
    value = body.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{config_path}: {field} must be a list of strings")
    return value


def _optional_str(config_path: Path, body: dict[str, Any], field: str) -> str:
    key = field.rsplit(".", 1)[-1]
    value = body.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{config_path}: {field} must be a string")
    return value.strip()


def _optional_branch_list(config_path: Path, body: dict[str, Any], field: str) -> list[str]:
    key = field.rsplit(".", 1)[-1]
    value = body.get(key)
    if value is None:
        return []
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{config_path}: {field} must be a non-empty list of strings")
    return [item.strip() for item in value]


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates
