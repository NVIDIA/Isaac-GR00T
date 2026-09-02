# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Python-authored AutoModel training experiment definition."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass, field
from typing import Any

from groot.infra.launcher.one_launcher.automodel_compute_variant import (
    AutoModelComputeVariant,
)
from groot.infra.launcher.one_launcher.automodel_topology import AutoModelTopology
from groot.infra.launcher.one_launcher.resolved_automodel_training_experiment import (
    ResolvedAutoModelTrainingExperiment,
)
from groot.infra.launcher.one_launcher.training_experiment_compute_config import (
    TrainingExperimentComputeConfig,
)
from groot.infra.launcher.one_launcher.training_launch_compute_profile import (
    TrainingLaunchComputeProfile,
)


@dataclass(frozen=True)
class AutoModelTrainingExperiment:
    recipe: Mapping[str, Any]
    topology: AutoModelTopology
    compute_config: TrainingExperimentComputeConfig
    checkpoint_step: int
    # None preserves the platform default (one H100 runtime-preflight spare).
    # Bounded studies can pin this to zero so the experiment definition itself
    # resolves to the exact requested training gang without a CLI-only contract.
    spare_nodes: int | None = None
    output_dir: str = "{PATHS.checkpoints}/{USER_SLUG}/{JOB_NAME}"
    migration_contract: Mapping[str, Any] = field(default_factory=dict)
    forbidden_recipe_substrings: tuple[str, ...] = ()
    compute_variants: Mapping[str, AutoModelComputeVariant] = field(default_factory=dict)

    def resolve(
        self,
        *,
        name: str,
        compute_type: str,
        compute_profile: TrainingLaunchComputeProfile,
    ) -> ResolvedAutoModelTrainingExperiment:
        compute_family = self.compute_config.family(
            experiment=name,
            compute_type=compute_type,
        )
        variant = self.compute_variants.get(compute_type)
        if compute_type != "h100" and variant is None:
            valid = ", ".join(sorted({"h100", *self.compute_variants}))
            raise ValueError(
                f"Experiment {name} has no AutoModel variant for {compute_type}; "
                f"valid compute types: {valid}"
            )
        topology = variant.topology if variant is not None else self.topology
        recipe = (
            variant.recipe_from(self.recipe)
            if variant is not None
            else copy.deepcopy(dict(self.recipe))
        )
        env_vars = self.compute_config.env_section()
        if variant is not None:
            env_vars = variant.environment_from(env_vars)
        gpus_per_node = compute_profile.gpus_per_node(topology.world_size)
        topology.validate(experiment=name, gpus_per_node=gpus_per_node)
        if compute_family.per_device_batch_size != topology.local_batch_size:
            raise ValueError(
                f"Experiment {name} compute per-device batch size "
                f"{compute_family.per_device_batch_size} does not match topology local batch "
                f"size {topology.local_batch_size}"
            )
        if self.checkpoint_step <= 0:
            raise ValueError(f"Experiment {name} checkpoint_step must be positive")
        self.validate_recipe_topology(name=name, recipe=recipe, topology=topology)
        self.validate_forbidden_recipe_values(name=name, recipe=recipe)
        training_nodes = topology.training_nodes(gpus_per_node=gpus_per_node)
        if self.spare_nodes is None:
            num_nodes = compute_profile.num_nodes_for_num_gpus(topology.world_size)
            spare_nodes = num_nodes - training_nodes
        else:
            if (
                isinstance(self.spare_nodes, bool)
                or not isinstance(self.spare_nodes, int)
                or self.spare_nodes < 0
            ):
                raise ValueError(f"Experiment {name} spare_nodes must be a nonnegative integer")
            spare_nodes = self.spare_nodes
            num_nodes = training_nodes + spare_nodes
        return ResolvedAutoModelTrainingExperiment(
            name=name,
            compute_type=compute_type,
            recipe=recipe,
            topology=topology,
            num_nodes=num_nodes,
            spare_nodes=spare_nodes,
            num_gpus=topology.world_size,
            gpus_per_node=gpus_per_node,
            batch_size=topology.local_batch_size,
            global_batch_size=topology.global_batch_size,
            gradient_accumulation_steps=topology.gradient_accumulation_steps,
            checkpoint_step=self.checkpoint_step,
            output_dir=self.output_dir,
            env_vars=env_vars,
            migration_contract=copy.deepcopy(dict(self.migration_contract)),
            forbidden_recipe_substrings=self.forbidden_recipe_substrings,
            skypilot_queue=compute_family.skypilot_queue,
            osmo_exec_timeout=compute_family.osmo_exec_timeout,
            launch_blocker=variant.launch_blocker if variant is not None else "",
        )

    def validate_recipe_topology(
        self,
        *,
        name: str,
        recipe: Mapping[str, Any],
        topology: AutoModelTopology | None = None,
    ) -> None:
        topology = topology or self.topology
        scheduler = self.required_mapping(name=name, recipe=recipe, section="step_scheduler")
        distributed = self.required_mapping(name=name, recipe=recipe, section="distributed")
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
            raise ValueError(f"Experiment {name} recipe/topology mismatch: {', '.join(mismatches)}")
        self.validate_dp_replicate_size(
            name=name,
            distributed=distributed,
            data_parallel=topology.data_parallel,
        )

    def validate_dp_replicate_size(
        self,
        *,
        name: str,
        distributed: Mapping[str, Any],
        data_parallel: int,
    ) -> None:
        value = distributed.get("dp_replicate_size", 1)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"Experiment {name} recipe.distributed.dp_replicate_size must be positive"
            )
        if data_parallel % value:
            raise ValueError(
                f"Experiment {name} data_parallel={data_parallel} must be divisible by "
                f"dp_replicate_size={value}"
            )
        if value >= data_parallel and value != 1:
            raise ValueError(
                f"Experiment {name} dp_replicate_size={value} must be less than "
                f"data_parallel={data_parallel}; FSDP2 does not support the pure-DDP case"
            )

    def required_mapping(
        self,
        *,
        name: str,
        recipe: Mapping[str, Any],
        section: str,
    ) -> Mapping[str, Any]:
        value = recipe.get(section)
        if not isinstance(value, Mapping):
            raise TypeError(f"Experiment {name} recipe.{section} must be a mapping")
        return value

    def validate_forbidden_recipe_values(
        self,
        *,
        name: str,
        recipe: Mapping[str, Any],
    ) -> None:
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
                raise ValueError(
                    f"Experiment {name} recipe references forbidden lineage {forbidden!r}"
                )
