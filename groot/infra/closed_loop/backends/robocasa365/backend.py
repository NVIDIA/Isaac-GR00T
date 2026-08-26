# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""RoboCasa365 plugin for shared closed-loop evaluation."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

from groot.infra.closed_loop.backends.gymnasium.vector_env import RaySingleVectorEnv

TASK_PREFIX = "robocasa365_panda_omron/"


class RoboCasa365Backend:
    """Create isolated RoboCasa365 environments from shared suite definitions."""

    def __init__(self, *, suite: str, options: Mapping[str, Any]) -> None:
        unknown = sorted(
            set(options) - {"simulator_py_executable", "simulator_env", "robocasa_root"}
        )
        if unknown:
            raise ValueError(f"unsupported RoboCasa365 backend options: {', '.join(unknown)}")
        self._suite = suite
        self._options = dict(options)

    @classmethod
    def suite_task_names(cls, suite: str) -> tuple[str, ...]:
        """Load one canonical shared RoboCasa365 task list."""
        path = _task_config_path(suite)
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, list) or not document:
            raise ValueError(f"{path} must contain a non-empty task list")
        tasks = tuple(str(task) for task in document)
        if any(not task.startswith(TASK_PREFIX) for task in tasks):
            raise ValueError(f"{suite!r} is not a RoboCasa365 task configuration")
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
        """Create a one-slot facade over the isolated RoboCasa365 runtime."""
        if num_envs != 1:
            raise ValueError("RoboCasa365 portable workers currently require num_envs=1")
        executable = str(
            self._options.get("simulator_py_executable")
            or Path(__file__).parent / ".venv/bin/python"
        )
        if not Path(executable).is_file():
            raise FileNotFoundError(
                f"RoboCasa365 runtime not found at {executable}; run "
                "groot/infra/closed_loop/backends/robocasa365/setup.sh"
            )
        root = str(self._options.get("robocasa_root") or Path(__file__).parent / ".robocasa365")
        environment = {"MUJOCO_GL": "egl", "ROBOCASA_ROOT": root}
        configured_environment = self._options.get("simulator_env", {})
        if not isinstance(configured_environment, Mapping):
            raise TypeError("RoboCasa365 simulator_env must be a mapping")
        environment.update({str(key): str(value) for key, value in configured_environment.items()})
        return RaySingleVectorEnv(
            backend_path=f"{type(self).__module__}:{type(self).__name__}",
            suite=self._suite,
            backend_options=self._options,
            task=task,
            seed=seed,
            max_episode_steps=max_episode_steps or task_horizon(task),
            record_video=record_video,
            video_dir=video_dir,
            py_executable=executable,
            env_vars=environment,
        )

    @classmethod
    def make_env(cls, *, task: str, seed: int) -> Any:
        """Instantiate one RoboCasa task inside the simulator actor."""
        if not task.startswith(TASK_PREFIX):
            raise ValueError(f"RoboCasa365 task must start with {TASK_PREFIX!r}")
        root = Path(os.environ["ROBOCASA_ROOT"])
        if root.is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import gymnasium as gym
        import robocasa  # noqa: F401
        import robosuite  # noqa: F401

        return gym.make(f"robocasa/{task.removeprefix(TASK_PREFIX)}", seed=seed, split="pretrain")

    @classmethod
    def render_frame(cls, env: Any, observation: Mapping[str, Any]) -> np.ndarray:
        """Select the first policy camera, falling back to the simulator renderer."""
        video_keys = sorted(key for key in observation if key.startswith("video."))
        if video_keys:
            return np.asarray(observation[video_keys[0]])
        return np.asarray(env.render())


def task_horizon(task: str) -> int:
    """Return the shared canonical episode horizon for one RoboCasa365 task."""
    document = yaml.safe_load((Path(__file__).parent / "horizons.yaml").read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("RoboCasa365 horizons.yaml must contain a mapping")
    assert task in document, f"missing RoboCasa365 horizon for {task!r}"
    horizon = int(document[task])
    if horizon <= 0:
        raise ValueError(f"RoboCasa365 horizon must be positive for {task!r}")
    return horizon


def _task_config_path(suite: str) -> Path:
    if Path(suite).name != suite:
        raise ValueError("suite must be a task configuration name, not a path")
    path = Path(__file__).parents[3] / "eval" / "configs" / "tasks" / f"{suite}.yaml"
    if not path.is_file():
        raise ValueError(f"unknown RoboCasa365 suite {suite!r}")
    return path
