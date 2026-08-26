# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Canonical Gymnasium wrapper and registration for LIBERO-PRO."""

from __future__ import annotations

import logging
import math
import os
import re

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path
import numpy as np

logger = logging.getLogger(__name__)


def sanitize_gym_id(name: str) -> str:
    r"""Sanitize a name to Gymnasium's ``[\w:.-]+`` environment ID alphabet."""
    safe = re.sub(r"[^\w:.\-]", "_", name)
    safe = re.sub(r"_+", "_", safe)
    return safe.strip("_")


def quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Convert an ``(x, y, z, w)`` quaternion to axis-angle coordinates."""
    if quaternion[3] > 1.0:
        quaternion[3] = 1.0
    elif quaternion[3] < -1.0:
        quaternion[3] = -1.0
    denominator = np.sqrt(1.0 - quaternion[3] * quaternion[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quaternion[:3] * 2.0 * math.acos(quaternion[3]) / denominator


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """Map the standardized gripper action from ``[0, 1]`` to ``[-1, 1]``."""
    action[..., -1] = 2 * action[..., -1] - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Match LIBERO's gripper sign convention."""
    action[..., -1] *= -1.0
    return action


class LiberoProEnv(gym.Env):
    """Expose a LIBERO-PRO BDDL task through the shared flat policy schema."""

    def __init__(self, task_bddl_file: str) -> None:
        self._env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=256,
            camera_widths=256,
        )
        self._task_description = self._env.language_instruction.strip()
        if not self._task_description:
            raise ValueError(f"Missing language instruction in LIBERO-PRO BDDL: {task_bddl_file}")
        self.observation_space = spaces.Dict(
            {
                "video.image": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
                "video.wrist_image": spaces.Box(
                    low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
                ),
                "state.x": spaces.Box(low=-1, high=1, shape=(1,)),
                "state.y": spaces.Box(low=-1, high=1, shape=(1,)),
                "state.z": spaces.Box(low=-1, high=1, shape=(1,)),
                "state.roll": spaces.Box(low=-1, high=1, shape=(1,)),
                "state.pitch": spaces.Box(low=-1, high=1, shape=(1,)),
                "state.yaw": spaces.Box(low=-1, high=1, shape=(1,)),
                "state.gripper": spaces.Box(low=-1, high=1, shape=(2,)),
                "annotation.human.action.task_description": spaces.Text(max_length=512),
            }
        )
        self.action_space = spaces.Dict(
            {
                "action.x": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.y": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.z": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.roll": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.pitch": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.yaw": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.gripper": spaces.Box(low=-1, high=1, shape=(1,)),
            }
        )

    def reset(self, seed=None, options=None):  # noqa: ARG002
        observation = self._process_observation(self._env.reset())
        return observation, {"success": self._env.check_success()}

    def step(self, action):
        action_vector = np.concatenate(
            [
                action["action.x"],
                action["action.y"],
                action["action.z"],
                action["action.roll"],
                action["action.pitch"],
                action["action.yaw"],
                action["action.gripper"],
            ],
            axis=0,
        )
        action_vector = invert_gripper_action(normalize_gripper_action(action_vector))
        observation, reward, done, info = self._env.step(action_vector)
        info["success"] = self._env.check_success()
        return self._process_observation(observation), reward, done, False, info

    def _process_observation(self, observation):
        position = observation["robot0_eef_pos"]
        rotation = quat_to_axis_angle(observation["robot0_eef_quat"])
        return {
            "video.image": observation["agentview_image"][::-1, ::-1],
            "video.wrist_image": observation["robot0_eye_in_hand_image"][::-1, ::-1],
            "state.x": [position[0]],
            "state.y": [position[1]],
            "state.z": [position[2]],
            "state.roll": [rotation[0]],
            "state.pitch": [rotation[1]],
            "state.yaw": [rotation[2]],
            "state.gripper": observation["robot0_gripper_qpos"],
            "annotation.human.action.task_description": self._task_description,
        }


def register_libero_pro_envs() -> None:
    """Register installed LIBERO-PRO tasks with collision-free Gymnasium IDs."""
    bddl_root = get_libero_path("bddl_files")
    for suite_name, suite_factory in benchmark.get_benchmark_dict().items():
        try:
            suite = suite_factory()
        except Exception as error:  # noqa: BLE001 - one broken optional suite must not hide others.
            logger.warning("Skipping LIBERO-PRO suite %s: %s", suite_name, error)
            continue
        for task_index in range(suite.get_num_tasks()):
            task = suite.get_task(task_index)
            bddl_file = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
            if not os.path.exists(bddl_file):
                continue
            environment_id = (
                f"libero_pro_sim/{sanitize_gym_id(f'{task.problem_folder}__{task.name}')}"
            )
            if environment_id not in gym.envs.registry:
                register(
                    id=environment_id,
                    entry_point="groot.infra.closed_loop.backends.libero_pro.environment:LiberoProEnv",
                    kwargs={"task_bddl_file": bddl_file},
                )


# Preserve established imports while internal callers migrate to descriptive names.
_sanitize_gym_id = sanitize_gym_id
quat2axisangle = quat_to_axis_angle
