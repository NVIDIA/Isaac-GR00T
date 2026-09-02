# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Resolved AutoModel experiment ready for One Launcher submission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml

from groot.infra.launcher.one_launcher.automodel_topology import AutoModelTopology
from groot.infra.launcher.one_launcher.resolved_automodel_experiment_files import (
    ResolvedAutoModelExperimentFiles,
)
from groot.infra.launcher.one_launcher.training_experiment_override import (
    TrainingExperimentOverride,
)
from groot.infra.launcher.one_launcher.training_launch_compute_profile import (
    TrainingLaunchComputeProfile,
)

_OVERRIDE_ROOTS = {"compute", "recipe", "topology"}
_TOPOLOGY_FIELDS = frozenset(AutoModelTopology.__dataclass_fields__)


@dataclass(frozen=True)
class ResolvedAutoModelTrainingExperiment:
    name: str
    compute_type: str
    recipe: Mapping[str, Any]
    topology: AutoModelTopology
    num_nodes: int
    spare_nodes: int
    num_gpus: int
    gpus_per_node: int
    batch_size: int
    global_batch_size: int
    gradient_accumulation_steps: int
    checkpoint_step: int
    output_dir: str
    osmo_exec_timeout: str
    env_vars: Mapping[str, str]
    migration_contract: Mapping[str, Any]
    forbidden_recipe_substrings: tuple[str, ...]
    skypilot_queue: str | None = None
    launch_blocker: str = ""

    def config_fingerprint(self, *, include_evaluation: bool = True) -> str:
        del include_evaluation
        payload = self.override_payload()
        if payload["compute"]["skypilot_queue"] is None:
            del payload["compute"]["skypilot_queue"]
        return hashlib.sha256(yaml.safe_dump(payload, sort_keys=True).encode()).hexdigest()

    def with_overrides(
        self,
        overrides: Sequence[str],
        *,
        compute_profile: TrainingLaunchComputeProfile,
    ) -> ResolvedAutoModelTrainingExperiment:
        if not overrides:
            return self
        payload = self.override_payload()
        for expression in overrides:
            override = TrainingExperimentOverride.init_from_expression(expression)
            self.validate_override_path(override)
            override.apply(payload)

        topology_body = payload["topology"]
        assert isinstance(topology_body, dict)
        topology = AutoModelTopology(**topology_body)
        gpus_per_node = compute_profile.gpus_per_node(topology.world_size)
        topology.validate(experiment=self.name, gpus_per_node=gpus_per_node)
        recipe = payload["recipe"]
        assert isinstance(recipe, dict)
        self.validate_recipe_topology(recipe=recipe, topology=topology)
        self.validate_forbidden_recipe_values(recipe)
        compute = payload["compute"]
        assert isinstance(compute, dict)
        spare_nodes = self.nonnegative_integer(compute, "spare_nodes")
        training_nodes = topology.training_nodes(gpus_per_node=gpus_per_node)
        contract = payload["contract"]
        assert isinstance(contract, dict)
        return ResolvedAutoModelTrainingExperiment(
            name=self.name,
            compute_type=self.compute_type,
            recipe=recipe,
            topology=topology,
            num_nodes=training_nodes + spare_nodes,
            spare_nodes=spare_nodes,
            num_gpus=topology.world_size,
            gpus_per_node=gpus_per_node,
            batch_size=topology.local_batch_size,
            global_batch_size=topology.global_batch_size,
            gradient_accumulation_steps=topology.gradient_accumulation_steps,
            checkpoint_step=self.positive_integer(compute, "checkpoint_step"),
            output_dir=self.required_string(compute, "output_dir"),
            osmo_exec_timeout=self.required_string(compute, "osmo_exec_timeout"),
            env_vars=self.environment_values(compute),
            migration_contract=copy.deepcopy(contract["migration"]),
            forbidden_recipe_substrings=tuple(contract["forbidden_recipe_substrings"]),
            skypilot_queue=self.optional_string(compute, "skypilot_queue"),
            launch_blocker=self.launch_blocker,
        )

    def override_payload(self) -> dict[str, Any]:
        return {
            "recipe": copy.deepcopy(dict(self.recipe)),
            "topology": self.topology.as_dict(),
            "compute": {
                "checkpoint_step": self.checkpoint_step,
                "env_vars": copy.deepcopy(dict(self.env_vars)),
                "osmo_exec_timeout": self.osmo_exec_timeout,
                "output_dir": self.output_dir,
                "spare_nodes": self.spare_nodes,
                "skypilot_queue": self.skypilot_queue,
            },
            "contract": {
                "migration": copy.deepcopy(dict(self.migration_contract)),
                "forbidden_recipe_substrings": list(self.forbidden_recipe_substrings),
            },
        }

    def validate_override_path(self, override: TrainingExperimentOverride) -> None:
        root = override.path[0]
        if root not in _OVERRIDE_ROOTS:
            raise ValueError(
                f"Unknown AutoModel override root {root!r}; supported roots: "
                f"{', '.join(sorted(_OVERRIDE_ROOTS))}"
            )
        if len(override.path) == 1:
            raise ValueError(f"--set {override.expression!r} must target a nested value")
        if root == "topology":
            if len(override.path) != 2 or override.path[1] not in _TOPOLOGY_FIELDS:
                raise ValueError(
                    "AutoModel topology overrides must target one of: "
                    f"{', '.join(sorted(_TOPOLOGY_FIELDS))}"
                )
            return
        if root == "compute":
            if override.path[1] == "env_vars":
                if len(override.path) != 3:
                    raise ValueError(
                        "AutoModel compute.env_vars overrides require compute.env_vars.NAME"
                    )
                return
            if len(override.path) != 2 or override.path[1] not in {
                "checkpoint_step",
                "osmo_exec_timeout",
                "output_dir",
                "spare_nodes",
                "skypilot_queue",
            }:
                raise ValueError("Unknown AutoModel compute override")
            return
        override.validate_cli_value()

    def validate_recipe_topology(
        self,
        *,
        recipe: Mapping[str, Any],
        topology: AutoModelTopology,
    ) -> None:
        scheduler = recipe.get("step_scheduler")
        distributed = recipe.get("distributed")
        if not isinstance(scheduler, Mapping) or not isinstance(distributed, Mapping):
            raise TypeError("AutoModel recipe requires step_scheduler and distributed mappings")
        expected = {
            "step_scheduler.global_batch_size": topology.global_batch_size,
            "step_scheduler.local_batch_size": topology.local_batch_size,
            "distributed.dp_size": topology.data_parallel,
            "distributed.cp_size": topology.context_parallel,
            "distributed.tp_size": topology.tensor_parallel,
            "distributed.pp_size": topology.pipeline_parallel,
        }
        actual = {
            "step_scheduler.global_batch_size": scheduler.get("global_batch_size"),
            "step_scheduler.local_batch_size": scheduler.get("local_batch_size"),
            "distributed.dp_size": distributed.get("dp_size"),
            "distributed.cp_size": distributed.get("cp_size"),
            "distributed.tp_size": distributed.get("tp_size"),
            "distributed.pp_size": distributed.get("pp_size"),
        }
        mismatches = [
            f"{path}={actual[path]!r} (expected {value})"
            for path, value in expected.items()
            if actual[path] != value
        ]
        if mismatches:
            raise ValueError(f"AutoModel recipe/topology mismatch: {', '.join(mismatches)}")
        self.validate_dp_replicate_size(
            distributed=distributed,
            data_parallel=topology.data_parallel,
        )

    def validate_dp_replicate_size(
        self,
        *,
        distributed: Mapping[str, Any],
        data_parallel: int,
    ) -> None:
        value = distributed.get("dp_replicate_size", 1)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("AutoModel recipe.distributed.dp_replicate_size must be positive")
        if data_parallel % value:
            raise ValueError(
                f"AutoModel data_parallel={data_parallel} must be divisible by "
                f"dp_replicate_size={value}"
            )
        if value >= data_parallel and value != 1:
            raise ValueError(
                f"AutoModel dp_replicate_size={value} must be less than "
                f"data_parallel={data_parallel}; FSDP2 does not support the pure-DDP case"
            )

    def validate_forbidden_recipe_values(self, recipe: Mapping[str, Any]) -> None:
        def strings(value: Any):
            if isinstance(value, str):
                yield value
            elif isinstance(value, Mapping):
                for nested in value.values():
                    yield from strings(nested)
            elif isinstance(value, list | tuple):
                for nested in value:
                    yield from strings(nested)

        for forbidden in self.forbidden_recipe_substrings:
            if not forbidden or any(forbidden in value for value in strings(recipe)):
                raise ValueError(f"AutoModel recipe references forbidden lineage {forbidden!r}")

    def positive_integer(self, body: Mapping[str, Any], key: str) -> int:
        value = body.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"AutoModel compute.{key} must be a positive integer")
        return value

    def nonnegative_integer(self, body: Mapping[str, Any], key: str) -> int:
        value = body.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"AutoModel compute.{key} must be a nonnegative integer")
        return value

    def required_string(self, body: Mapping[str, Any], key: str) -> str:
        value = body.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"AutoModel compute.{key} must be a non-empty string")
        return value

    def optional_string(self, body: Mapping[str, Any], key: str) -> str | None:
        value = body.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"AutoModel compute.{key} must be a non-empty string or null")
        return value

    def environment_values(self, body: Mapping[str, Any]) -> Mapping[str, str]:
        raw = body.get("env_vars", {})
        if not isinstance(raw, Mapping):
            raise TypeError("AutoModel compute.env_vars must remain a mapping")
        return {str(key): str(value) for key, value in raw.items() if value is not None}

    def write_configs(self, output_root: Path) -> ResolvedAutoModelExperimentFiles:
        output_dir = output_root / self.name / self.compute_type
        output_dir.mkdir(parents=True, exist_ok=True)
        files = ResolvedAutoModelExperimentFiles(
            recipe_config=output_dir / "automodel_recipe.yaml",
            full_config=output_dir / "resolved_config.yaml",
        )
        files.recipe_config.write_text(yaml.safe_dump(dict(self.recipe), sort_keys=False))
        files.full_config.write_text(yaml.safe_dump(dict(self.summary(files)), sort_keys=False))
        return files

    def summary(self, files: ResolvedAutoModelExperimentFiles) -> Mapping[str, Any]:
        return {
            "experiment": self.name,
            "backend": "automodel",
            "compute_type": self.compute_type,
            "num_nodes": self.num_nodes,
            "training_nodes": self.topology.training_nodes(gpus_per_node=self.gpus_per_node),
            "spare_nodes": self.spare_nodes,
            "num_gpus": self.num_gpus,
            "gpus_per_node": self.gpus_per_node,
            "batch_size": self.batch_size,
            "global_batch_size": self.global_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "checkpoint_step": self.checkpoint_step,
            "output_dir": self.output_dir,
            "osmo_exec_timeout": self.osmo_exec_timeout,
            "skypilot_queue": self.skypilot_queue,
            "launch_blocker": self.launch_blocker,
            "env_vars": dict(self.env_vars),
            "migration_contract": copy.deepcopy(dict(self.migration_contract)),
            "forbidden_recipe_substrings": list(self.forbidden_recipe_substrings),
            "topology": self.topology.as_dict(),
            "recipe_config": str(files.recipe_config),
            "full_config": str(files.full_config),
            "recipe": dict(self.recipe),
        }
