# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Resolve Python training experiments for one-launcher."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys

from groot.infra.launcher.one_launcher.resolved_training_experiment import ResolvedTrainingExperiment
from groot.infra.launcher.one_launcher.training_experiment import TrainingExperiment
from groot.infra.launcher.one_launcher.training_launch_compute_profile import TrainingLaunchComputeProfile

DEFAULT_EXPERIMENT_DIR = Path("groot/infra/experiments/training/configs")


@dataclass(frozen=True)
class TrainingExperimentResolver:
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR

    @classmethod
    def init_default(
        cls,
        *,
        experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
    ) -> TrainingExperimentResolver:
        return cls(experiment_dir=experiment_dir)

    def resolve(
        self,
        *,
        name: str,
        compute_type: str,
        config_overrides: Sequence[str] = (),
    ) -> ResolvedTrainingExperiment:
        experiment = self.load(name)
        compute_profile = TrainingLaunchComputeProfile(compute_type=compute_type)
        resolved = experiment.resolve(
            name=name,
            compute_type=compute_type,
            compute_profile=compute_profile,
        )
        return resolved.with_overrides(
            config_overrides,
            compute_profile=compute_profile,
        )

    def load(self, name: str) -> TrainingExperiment:
        module_path = self.experiment_dir / f"{name}.py"
        if not module_path.is_file():
            raise ValueError(f"Unknown experiment {name!r}; expected file {module_path}")
        digest = hashlib.sha1(str(module_path.resolve()).encode()).hexdigest()[:12]
        module_name = f"_ol_experiment_{name}_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load experiment {name!r} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        experiment = getattr(module, "EXPERIMENT", None)
        if not isinstance(experiment, TrainingExperiment):
            raise TypeError(f"{module_path} must define EXPERIMENT as a TrainingExperiment")
        return experiment
