# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""LIBERO-PRO plugin for shared closed-loop evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from groot.infra.closed_loop.backends.gymnasium.vector_env import RaySingleVectorEnv


class LiberoProBackend:
    """Create isolated LIBERO-PRO environments without loading them in the model runtime."""

    def __init__(self, *, suite: str, options: Mapping[str, Any]) -> None:
        unknown = sorted(set(options) - {"simulator_py_executable", "simulator_env"})
        if unknown:
            raise ValueError(f"unsupported LIBERO-PRO backend options: {', '.join(unknown)}")
        self._suite = suite
        self._options = dict(options)

    @classmethod
    def suite_task_names(cls, suite: str) -> tuple[str, ...]:
        """Load the canonical shared task list without importing LIBERO-PRO."""
        path = _task_config_path(suite)
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, list) or not document:
            raise ValueError(f"{path} must contain a non-empty task list")
        tasks = tuple(str(task) for task in document)
        if any(not task.startswith("libero_pro_sim/") for task in tasks):
            raise ValueError(f"{suite!r} is not a LIBERO-PRO task configuration")
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
        """Create the one-slot vector facade used by each policy worker."""
        if num_envs != 1:
            raise ValueError("LIBERO-PRO portable workers currently require num_envs=1")
        executable = str(
            self._options.get("simulator_py_executable")
            or Path(__file__).parent / ".venv/bin/python"
        )
        if not Path(executable).is_file():
            raise FileNotFoundError(
                f"LIBERO-PRO runtime not found at {executable}; run "
                "groot/infra/closed_loop/backends/libero_pro/setup.sh"
            )
        environment = {
            "MUJOCO_GL": "egl",
            "LIBERO_CONFIG_PATH": str(Path(__file__).parent / ".libero-config"),
        }
        configured_environment = self._options.get("simulator_env", {})
        if not isinstance(configured_environment, Mapping):
            raise TypeError("LIBERO-PRO simulator_env must be a mapping")
        environment.update({str(key): str(value) for key, value in configured_environment.items()})
        return RaySingleVectorEnv(
            backend_path=f"{type(self).__module__}:{type(self).__name__}",
            suite=self._suite,
            backend_options=self._options,
            task=task,
            seed=seed,
            max_episode_steps=max_episode_steps or 520,
            record_video=record_video,
            video_dir=video_dir,
            py_executable=executable,
            env_vars=environment,
        )

    @classmethod
    def make_env(cls, *, task: str, seed: int) -> Any:
        """Instantiate one environment inside the isolated simulator actor."""
        del seed
        import gymnasium as gym

        from groot.infra.closed_loop.backends.libero_pro.environment import register_libero_pro_envs

        register_libero_pro_envs()
        return gym.make(task)

    @classmethod
    def render_frame(cls, env: Any, observation: Mapping[str, Any]) -> np.ndarray:
        """Select the canonical agent-view frame for video artifacts."""
        del env
        return np.asarray(observation["video.image"])


def _task_config_path(suite: str) -> Path:
    if Path(suite).name != suite:
        raise ValueError("suite must be a task configuration name, not a path")
    path = Path(__file__).parents[3] / "eval" / "configs" / "tasks" / f"{suite}.yaml"
    if not path.is_file():
        raise ValueError(f"unknown LIBERO-PRO suite {suite!r}")
    return path
