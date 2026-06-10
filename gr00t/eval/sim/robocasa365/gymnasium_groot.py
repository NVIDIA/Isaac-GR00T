# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GR00T-compatible Gymnasium wrapper for upstream RoboCasa365.

The existing RoboCasa benchmark uses a fork that registers
``robocasa_panda_omron/<Task>_PandaOmron_Env`` and emits the Panda Omron
observation/action keys used by the ROBOCASA_PANDA_OMRON checkpoint. Upstream
RoboCasa365 has a newer task registry and different wrapper keys, so this
module registers a separate namespace while preserving the checkpoint schema.
"""

from __future__ import annotations

import sys
from typing import Any

import cv2
from gymnasium import Env, spaces
from gymnasium.envs.registration import register, registry
import mujoco
import numpy as np
import robocasa  # noqa: F401 - imports register upstream RoboCasa env classes
from robocasa.utils.env_utils import create_env
from robosuite.controllers.composite.composite_controller import HybridMobileBase
from robosuite.environments.base import REGISTERED_ENVS


ALLOWED_LANGUAGE_CHARSET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\n\t[]{}()!?'_:"
)
CAMERA_NAMES = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]
MAPPED_CAMERA_NAMES = [
    "video.res256_image_side_0",
    "video.res256_image_side_1",
    "video.res256_image_wrist_0",
]
CAMERA_RESOLUTION = 512
FINAL_IMAGE_RESOLUTION = (256, 256)
DEFAULT_SPLIT = "target"
DEFAULT_OBJ_REGISTRIES = None


def _gather_robot_observations(env) -> dict[str, np.ndarray]:
    observations = {}

    for robot_id, robot in enumerate(env.robots):
        sim = robot.sim
        gripper_names = {robot.get_gripper_name(arm): robot.gripper[arm] for arm in robot.arms}
        for part_name, indexes in robot._ref_joints_indexes_dict.items():
            qpos_values = []
            for joint_id in indexes:
                qpos_addr = sim.model.jnt_qposadr[joint_id]
                joint_type = sim.model.jnt_type[joint_id]
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    qpos_size = 7
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    qpos_size = 4
                else:
                    qpos_size = 1
                qpos_values = np.append(
                    qpos_values, sim.data.qpos[qpos_addr : qpos_addr + qpos_size]
                )

            if part_name in gripper_names:
                qpos_values = np.asarray(qpos_values)[::-1]
            if len(qpos_values) > 0:
                observations[f"robot{robot_id}_{part_name}"] = qpos_values

    return observations


def _map_obs(input_obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "state.gripper_qpos": input_obs["robot0_gripper_qpos"],
        "state.base_position": input_obs["robot0_base_pos"],
        "state.base_rotation": input_obs["robot0_base_quat"],
        "state.end_effector_position_relative": input_obs["robot0_base_to_eef_pos"],
        "state.end_effector_rotation_relative": input_obs["robot0_base_to_eef_quat"],
        "state.gripper_qvel": input_obs["robot0_gripper_qvel"],
        "state.end_effector_position_absolute": input_obs["robot0_eef_pos"],
        "state.end_effector_rotation_absolute": input_obs["robot0_eef_quat"],
        "state.joint_position": input_obs["robot0_joint_pos"],
        "state.joint_position_cos": input_obs["robot0_joint_pos_cos"],
        "state.joint_position_sin": input_obs["robot0_joint_pos_sin"],
        "state.joint_velocity": input_obs["robot0_joint_vel"],
    }


def _unmap_action(input_action: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    return {
        "robot0_right_gripper": -1.0 if input_action["action.gripper_close"] < 0.5 else 1.0,
        "robot0_right": np.concatenate(
            (
                input_action["action.end_effector_position"],
                input_action["action.end_effector_rotation"],
            ),
            axis=-1,
        ),
        "robot0_base": input_action["action.base_motion"][..., 0:3],
        "robot0_torso": input_action["action.base_motion"][..., 3:4],
        "robot0_base_mode": -1.0 if input_action["action.control_mode"] < 0.5 else 1.0,
    }


class GrootRoboCasa365Env(Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        env_name: str,
        enable_render: bool = True,
        split: str = DEFAULT_SPLIT,
        obj_registries: tuple[str, ...] | list[str] | None = DEFAULT_OBJ_REGISTRIES,
        **kwargs: Any,
    ):
        self.env_name = env_name
        self.enable_render = enable_render
        if obj_registries is not None:
            kwargs = {**kwargs, "obj_registries": tuple(obj_registries)}
        self.env = create_env(
            env_name=env_name,
            robots="PandaOmron",
            camera_names=CAMERA_NAMES,
            camera_widths=CAMERA_RESOLUTION,
            camera_heights=CAMERA_RESOLUTION,
            split=split,
            render_onscreen=False,
            **kwargs,
        )
        self.camera_names = CAMERA_NAMES
        self.render_obs_key = f"{self.camera_names[0]}_image"
        self.render_cache = None
        self._create_spaces()

    @staticmethod
    def _process_img(img: np.ndarray) -> np.ndarray:
        h, w, _ = img.shape
        if h != w:
            dim = max(h, w)
            y_offset = (dim - h) // 2
            x_offset = (dim - w) // 2
            img = np.pad(img, ((y_offset, y_offset), (x_offset, x_offset), (0, 0)))
            h, w = dim, dim
        if (h, w) != FINAL_IMAGE_RESOLUTION:
            img = cv2.resize(img, FINAL_IMAGE_RESOLUTION, cv2.INTER_AREA)
        return np.copy(img)

    def _create_spaces(self) -> None:
        raw_obs = self.env.reset()
        raw_obs = self._get_basic_observation(raw_obs)
        mapped_obs = _map_obs(raw_obs)

        observation_space = spaces.Dict()
        for key, value in mapped_obs.items():
            observation_space[key] = spaces.Box(
                low=-1, high=1, shape=(len(value),), dtype=np.float32
            )
        for mapped_name in MAPPED_CAMERA_NAMES:
            observation_space[mapped_name] = spaces.Box(
                low=0, high=255, shape=(*FINAL_IMAGE_RESOLUTION, 3), dtype=np.uint8
            )
            observation_space[mapped_name.replace("256", "512")] = spaces.Box(
                low=0, high=255, shape=(CAMERA_RESOLUTION, CAMERA_RESOLUTION, 3), dtype=np.uint8
            )
        observation_space["annotation.human.action.task_description"] = spaces.Text(
            max_length=256, charset=ALLOWED_LANGUAGE_CHARSET
        )
        self.observation_space = observation_space

        self.action_space = spaces.Dict(
            {
                "action.gripper_close": spaces.Discrete(2),
                "action.end_effector_position": spaces.Box(
                    low=-1, high=1, shape=(3,), dtype=np.float32
                ),
                "action.end_effector_rotation": spaces.Box(
                    low=-1, high=1, shape=(3,), dtype=np.float32
                ),
                "action.base_motion": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                "action.control_mode": spaces.Discrete(2),
            }
        )

    def _get_basic_observation(self, raw_obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        raw_obs.update(_gather_robot_observations(self.env))
        for obs_name, obs_value in list(raw_obs.items()):
            if obs_name.endswith("_image"):
                raw_obs[obs_name] = np.copy(obs_value[::-1, :, :])
            elif obs_name.endswith("_depth"):
                raw_obs[obs_name] = np.copy(obs_value[::-1, :, :]).astype(np.float32)
            elif isinstance(obs_value, np.ndarray):
                raw_obs[obs_name] = obs_value.astype(np.float32)

        if not self.enable_render:
            for name in self.camera_names:
                raw_obs[f"{name}_image"] = np.zeros(
                    (CAMERA_RESOLUTION, CAMERA_RESOLUTION, 3), dtype=np.uint8
                )

        self.render_cache = raw_obs[self.render_obs_key]
        raw_obs["language"] = self.env.get_ep_meta().get("lang", "")
        return raw_obs

    def _get_groot_observation(self, raw_obs: dict[str, np.ndarray]) -> dict[str, Any]:
        basic_obs = self._get_basic_observation(raw_obs)
        obs: dict[str, Any] = _map_obs(basic_obs)
        for mapped_name, camera_name in zip(MAPPED_CAMERA_NAMES, CAMERA_NAMES):
            image = self._process_img(basic_obs[f"{camera_name}_image"])
            obs[mapped_name] = image
            obs[mapped_name.replace("256", "512")] = np.copy(basic_obs[f"{camera_name}_image"])
        obs["annotation.human.action.task_description"] = basic_obs["language"]
        return obs

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)

        raw_obs = self.env.reset()
        obs = self._get_groot_observation(raw_obs)
        return obs, {"success": False}

    def step(self, action: dict[str, np.ndarray]):
        action_dict = _unmap_action(action)

        env_action = []
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            robot_action = np.zeros(cc.action_limits[0].shape)
            for part_name in cc.part_controllers:
                start_idx, end_idx = cc._action_split_indexes[part_name]
                robot_action[start_idx:end_idx] = action_dict.pop(f"{pf}{part_name}")
            if isinstance(cc, HybridMobileBase):
                robot_action[-1] = action_dict.pop(f"{pf}base_mode")
            env_action.append(robot_action)

        if action_dict:
            raise RuntimeError(f"Unprocessed RoboCasa365 actions: {sorted(action_dict)}")

        raw_obs, _, done, info = self.env.step(np.concatenate(env_action))
        is_success = bool(self.env._check_success())
        reward = 1.0 if is_success else 0.0
        obs = self._get_groot_observation(raw_obs)
        info["success"] = is_success
        return obs, reward, done, False, info

    def render(self):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        return self.render_cache

    def close(self):
        self.env.close()

    def __getattr__(self, name: str):
        return getattr(self.env, name)


def _create_groot_robocasa365_env_class(env_name: str) -> None:
    class_name = f"{env_name}_PandaOmron_Env"
    id_name = f"robocasa365_panda_omron/{class_name}"
    if id_name in registry:
        return

    env_class_type = type(
        class_name,
        (GrootRoboCasa365Env,),
        {
            "__init__": lambda self, **kwargs: super(self.__class__, self).__init__(
                env_name=env_name,
                **kwargs,
            )
        },
    )

    current_module = sys.modules["gr00t.eval.sim.robocasa365.gymnasium_groot"]
    setattr(current_module, class_name, env_class_type)
    register(
        id=id_name,
        entry_point=f"gr00t.eval.sim.robocasa365.gymnasium_groot:{class_name}",
    )


for _ENV in REGISTERED_ENVS:
    _create_groot_robocasa365_env_class(_ENV)
