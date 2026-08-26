# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Ray actor that owns one Gymnasium environment in an isolated runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArrayPayload:
    """NumPy-independent array representation safe across simulator environments."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes


def encode_for_wire(value: Any) -> Any:
    """Remove NumPy-specific pickle references before crossing runtime boundaries."""
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object-dtype arrays cannot cross the simulator boundary")
        contiguous = np.ascontiguousarray(value)  # [...]
        return ArrayPayload(contiguous.dtype.str, contiguous.shape, contiguous.tobytes())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: encode_for_wire(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(encode_for_wire(item) for item in value)
    if isinstance(value, list):
        return [encode_for_wire(item) for item in value]
    return value


def decode_from_wire(value: Any) -> Any:
    """Rebuild arrays using the NumPy installed in the receiving runtime."""
    import numpy as np

    if isinstance(value, ArrayPayload):
        return np.frombuffer(value.data, dtype=np.dtype(value.dtype)).reshape(value.shape).copy()
    if isinstance(value, Mapping):
        return {key: decode_from_wire(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(decode_from_wire(item) for item in value)
    if isinstance(value, list):
        return [decode_from_wire(item) for item in value]
    return value


class GymnasiumEnvironmentActor:
    """Create, step, and optionally record one simulator environment."""

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
    ) -> None:
        module_name, _, class_name = backend_path.partition(":")
        backend_class = getattr(importlib.import_module(module_name), class_name)
        self._backend = backend_class(suite=suite, options=backend_options)
        self._env = self._backend.make_env(task=task, seed=seed)
        self._task = task
        self._seed = seed
        self._max_episode_steps = max_episode_steps
        self._record_video = record_video
        self._video_dir = Path(video_dir)
        self._episode_index = 0
        self._steps = 0
        self._frames: list[Any] = []

    def reset(self) -> Any:
        observation, info = self._env.reset(seed=self._seed + self._episode_index)
        self._steps = 0
        self._frames = []
        self._append_frame(observation)
        return encode_for_wire((observation, info))

    def step(self, action: Any) -> Any:
        observation, reward, terminated, truncated, info = self._env.step(decode_from_wire(action))
        self._steps += 1
        if self._max_episode_steps is not None and self._steps >= self._max_episode_steps:
            truncated = True
        self._append_frame(observation)
        done = bool(terminated or truncated)
        final_info = dict(info) if done else None
        if done:
            self._write_video()
            self._episode_index += 1
            observation, _ = self._env.reset(seed=self._seed + self._episode_index)
            self._steps = 0
            self._frames = []
            self._append_frame(observation)
        return encode_for_wire(
            (observation, reward, bool(terminated), bool(truncated), info, final_info)
        )

    def close(self) -> None:
        self._env.close()

    def _append_frame(self, observation: Mapping[str, Any]) -> None:
        if self._record_video:
            self._frames.append(self._backend.render_frame(self._env, observation))

    def _write_video(self) -> None:
        if not self._record_video or not self._frames:
            return
        import imageio.v2 as imageio

        safe_task = "".join(character if character.isalnum() else "-" for character in self._task)
        path = self._video_dir / safe_task / f"episode-{self._episode_index:03d}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, self._frames, fps=20, codec="libx264")
