# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""RoboDojo plugin for shared closed-loop evaluation."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from groot.infra.closed_loop.backends.gymnasium.vector_env import RaySingleVectorEnv

TASK_PREFIX = "robodojo/"


class RoboDojoBackend:
    """Create isolated ARX X5 environments from approved RoboDojo suites."""

    def __init__(self, *, suite: str, options: Mapping[str, Any]) -> None:
        unknown = sorted(
            set(options) - {"simulator_py_executable", "simulator_env", "robodojo_root"}
        )
        if unknown:
            raise ValueError(f"unsupported RoboDojo backend options: {', '.join(unknown)}")
        self._suite = suite
        self._options = dict(options)

    @classmethod
    def suite_task_names(cls, suite: str) -> tuple[str, ...]:
        """Load one approved shared RoboDojo task list."""
        path = _task_config_path(suite)
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, list) or not document:
            raise ValueError(f"{path} must contain a non-empty task list")
        tasks = tuple(str(task) for task in document)
        if any(not task.startswith(TASK_PREFIX) for task in tasks):
            raise ValueError(f"{suite!r} is not a RoboDojo task configuration")
        return tasks

    def create_vector_env(
        self,
        *,
        task: str,
        num_envs: int,
        seed: int,
        max_episode_steps: int | None,
        record_video: bool,
        video_dir: str,
    ) -> RaySingleVectorEnv:
        """Create a one-slot facade over the isolated Isaac Sim runtime."""
        del max_episode_steps
        if num_envs != 1:
            raise ValueError("RoboDojo portable workers currently require num_envs=1")
        executable = str(
            self._options.get("simulator_py_executable")
            or Path(__file__).parent / ".venv/bin/python"
        )
        if not Path(executable).is_file():
            raise FileNotFoundError(
                f"RoboDojo runtime not found at {executable}; run "
                "groot/infra/closed_loop/backends/robodojo/setup.sh"
            )
        root = str(self._options.get("robodojo_root") or Path(__file__).parent / ".robodojo")
        environment = {
            "ROBODOJO_ROOT": root,
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
        }
        configured_environment = self._options.get("simulator_env", {})
        if not isinstance(configured_environment, Mapping):
            raise TypeError("RoboDojo simulator_env must be a mapping")
        environment.update({str(key): str(value) for key, value in configured_environment.items()})
        return RaySingleVectorEnv(
            backend_path=f"{type(self).__module__}:{type(self).__name__}",
            suite=self._suite,
            backend_options=self._options,
            task=task,
            seed=seed,
            # RoboDojo owns each task's horizon through its official ``step_lim``.
            max_episode_steps=None,
            record_video=record_video,
            video_dir=video_dir,
            py_executable=executable,
            env_vars=environment,
        )

    @classmethod
    def make_env(cls, *, task: str, seed: int) -> Any:
        """Instantiate one RoboDojo environment inside the Isaac Sim actor."""
        del seed
        if not task.startswith(TASK_PREFIX):
            raise ValueError(f"RoboDojo task must start with {TASK_PREFIX!r}")
        import gymnasium as gym

        os.environ.setdefault("ROBODOJO_ROOT", str(Path(__file__).parent / ".robodojo"))

        from groot.infra.closed_loop.backends.robodojo.environment import (
            register_robodojo_envs,
        )

        register_robodojo_envs()
        return gym.make(task)

    @classmethod
    def render_frame(cls, env: Any, observation: Mapping[str, Any]) -> np.ndarray:
        """Select RoboDojo's canonical high camera for video artifacts."""
        del env
        return np.asarray(observation["video.cam_high"])


def _task_config_path(suite: str) -> Path:
    if Path(suite).name != suite:
        raise ValueError("suite must be a task configuration name, not a path")
    path = Path(__file__).parents[3] / "eval" / "configs" / "tasks" / f"{suite}.yaml"
    if not path.is_file():
        raise ValueError(f"unknown RoboDojo suite {suite!r}")
    return path
