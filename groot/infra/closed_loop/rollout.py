# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Dependency-light closed-loop rollout primitives shared by internal and OSS evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np


class RolloutPolicy(Protocol):
    """Policy lifecycle consumed by the shared vectorized rollout loop."""

    def on_env_init(self, env: Any) -> None:
        """Initialize policy state that depends on the vector environment."""

    def on_env_reset(self, env_indices: np.ndarray) -> None:
        """Reset policy state for completed vector-environment slots."""

    def action(self, observations: Any) -> Any:
        """Return one batched action for the current observations."""


class RolloutProgress(Protocol):
    """Minimal progress interface accepted without depending on a UI library."""

    def update(self, count: int = 1) -> None:
        """Advance progress by ``count`` completed episodes."""

    def close(self) -> None:
        """Release progress-rendering resources."""


@dataclass(frozen=True)
class RolloutResult:
    """Results produced for one task by the shared rollout loop."""

    successes: tuple[bool, ...]
    episode_infos: dict[str, tuple[Any, ...]]


@dataclass(frozen=True)
class TaskAssignment:
    """One task invocation assigned to a policy evaluator actor."""

    task: str
    num_episodes: int


def build_task_assignments(
    tasks: list[str], *, num_workers: int, num_episodes: int
) -> tuple[TaskAssignment, ...]:
    """Reproduce the established internal task-to-evaluator scheduling policy."""
    if not tasks:
        raise ValueError("at least one task is required")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if len(tasks) >= num_workers:
        return tuple(TaskAssignment(task, num_episodes) for task in tasks)

    workers_per_task = num_workers // len(tasks)
    episodes_per_worker = math.ceil(num_episodes / workers_per_task)
    assignments: list[TaskAssignment] = []
    for task in tasks:
        remaining = num_episodes
        for _ in range(workers_per_task):
            assigned = min(remaining, episodes_per_worker)
            remaining -= assigned
            assignments.append(TaskAssignment(task, assigned))
    return tuple(assignments)


def run_vectorized_rollout(
    *,
    env: Any,
    policy: RolloutPolicy,
    num_envs: int,
    num_episodes: int,
    progress: RolloutProgress | None = None,
) -> RolloutResult:
    """Run the established batched environment loop without owning setup or reporting."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if num_episodes < num_envs:
        raise ValueError("num_episodes must be at least num_envs")

    successes: list[bool] = []
    current_successes = [False] * num_envs
    completed_episodes = 0
    episode_infos: defaultdict[str, list[Any]] = defaultdict(list)

    policy.on_env_init(env)
    observations, _ = env.reset()
    policy.on_env_reset(np.arange(num_envs))
    try:
        while completed_episodes < num_episodes:
            actions = policy.action(observations)
            observations, _, terminations, truncations, infos = env.step(actions)
            reset_indices = np.arange(num_envs)[terminations | truncations]
            policy.on_env_reset(reset_indices)

            for env_index in range(num_envs):
                if "success" in infos:
                    current_successes[env_index] |= _coerce_success(infos["success"][env_index])
                else:
                    current_successes[env_index] = False
                final_info = _final_info_at(infos, env_index)
                if final_info is not None and "success" in final_info:
                    current_successes[env_index] |= _coerce_success(final_info["success"])
                if not (terminations[env_index] or truncations[env_index]):
                    continue
                successes.append(current_successes[env_index])
                current_successes[env_index] = False
                completed_episodes += 1
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()
        env.reset()
        env.close()

    if len(successes) < num_episodes:
        raise RuntimeError(f"expected at least {num_episodes} episodes, got {len(successes)}")
    checked_infos = {name: tuple(values) for name, values in episode_infos.items()}
    for name, values in checked_infos.items():
        if len(values) != len(successes):
            raise RuntimeError(
                f"episode metric {name!r} has {len(values)} values for {len(successes)} episodes"
            )
    return RolloutResult(successes=tuple(successes), episode_infos=checked_infos)


def _final_info_at(infos: Mapping[str, Any], env_index: int) -> Mapping[str, Any] | None:
    if "final_info" not in infos:
        return None
    final_info = infos["final_info"][env_index]
    if final_info is None:
        return None
    if not isinstance(final_info, Mapping):
        raise TypeError(f"final_info must be a mapping, got {type(final_info)}")
    return final_info


def _coerce_success(value: Any) -> bool:
    if isinstance(value, list | np.ndarray):
        return bool(np.any(value))
    if isinstance(value, bool | int | np.bool_ | np.integer):
        return bool(value)
    raise TypeError(f"unsupported success value type: {type(value)}")
