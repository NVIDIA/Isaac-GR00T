# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""One-slot vector facade over an isolated Ray Gymnasium actor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from groot.infra.closed_loop.backends.gymnasium.environment_actor import (
    GymnasiumEnvironmentActor,
    decode_from_wire,
    encode_for_wire,
)


class RaySingleVectorEnv:
    """Expose an isolated environment through the shared vector rollout API."""

    def __init__(
        self,
        *,
        backend_path: str,
        suite: str,
        backend_options: dict[str, Any],
        task: str,
        seed: int,
        max_episode_steps: int | None,
        record_video: bool,
        video_dir: str,
        py_executable: str,
        env_vars: Mapping[str, str],
    ) -> None:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        actor_type = ray.remote(
            num_cpus=0,
            runtime_env={"py_executable": py_executable, "env_vars": dict(env_vars)},
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(), soft=False
            ),
        )(GymnasiumEnvironmentActor)
        self._ray = ray
        self._actor = actor_type.remote(
            backend_path=backend_path,
            suite=suite,
            backend_options=backend_options,
            task=task,
            seed=seed,
            max_episode_steps=max_episode_steps,
            record_video=record_video,
            video_dir=video_dir,
        )

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = decode_from_wire(self._ray.get(self._actor.reset.remote()))
        return _batch_mapping(observation), info

    def step(
        self, actions: Mapping[str, Any]
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        action = {key: _unbatch(value) for key, value in actions.items()}
        observation, reward, terminated, truncated, info, final_info = decode_from_wire(
            self._ray.get(self._actor.step.remote(encode_for_wire(action)))
        )
        infos = {
            "success": np.array([bool(info.get("success", False))]),  # [B]
            "final_info": np.array([final_info], dtype=object),  # [B]
        }
        return (
            _batch_mapping(observation),
            np.array([reward], dtype=np.float32),  # [B]
            np.array([terminated], dtype=bool),  # [B]
            np.array([truncated], dtype=bool),  # [B]
            infos,
        )

    def close(self) -> None:
        # Some simulator shutdown hooks terminate their own worker process, which
        # makes waiting on close surface a spurious ActorDiedError after a
        # successful rollout. Ray owns the isolated process, so terminate it
        # directly once all terminal artifacts have been written.
        self._ray.kill(self._actor, no_restart=True)


def _batch_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _batch(item) for key, item in value.items()}


def _batch(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.expand_dims(value, axis=0)  # [B=1,...]
    if isinstance(value, str):
        return [value]
    return np.expand_dims(np.asarray(value), axis=0)  # [B=1,...]


def _unbatch(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape[0] != 1:
        raise ValueError("RaySingleVectorEnv requires exactly one action batch element")
    return array[0]
