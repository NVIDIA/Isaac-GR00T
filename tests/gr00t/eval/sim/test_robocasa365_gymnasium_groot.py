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

from __future__ import annotations

import importlib
import sys
import types

import numpy as np


def _install_robocasa365_import_stubs(monkeypatch):
    robocasa = types.ModuleType("robocasa")
    robocasa_utils = types.ModuleType("robocasa.utils")
    robocasa_env_utils = types.ModuleType("robocasa.utils.env_utils")
    robocasa_env_utils.create_env = lambda **kwargs: None

    mujoco = types.ModuleType("mujoco")
    mujoco.mjtJoint = types.SimpleNamespace(mjJNT_FREE=0, mjJNT_BALL=1)

    robosuite = types.ModuleType("robosuite")
    robosuite_controllers = types.ModuleType("robosuite.controllers")
    robosuite_composite = types.ModuleType("robosuite.controllers.composite")
    robosuite_composite_controller = types.ModuleType(
        "robosuite.controllers.composite.composite_controller"
    )
    robosuite_composite_controller.HybridMobileBase = type("HybridMobileBase", (), {})
    robosuite_environments = types.ModuleType("robosuite.environments")
    robosuite_base = types.ModuleType("robosuite.environments.base")
    robosuite_base.REGISTERED_ENVS = []

    for name, module in {
        "robocasa": robocasa,
        "robocasa.utils": robocasa_utils,
        "robocasa.utils.env_utils": robocasa_env_utils,
        "mujoco": mujoco,
        "robosuite": robosuite,
        "robosuite.controllers": robosuite_controllers,
        "robosuite.controllers.composite": robosuite_composite,
        "robosuite.controllers.composite.composite_controller": robosuite_composite_controller,
        "robosuite.environments": robosuite_environments,
        "robosuite.environments.base": robosuite_base,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_robocasa365_observation_keys_match_embodiment_config(monkeypatch):
    _install_robocasa365_import_stubs(monkeypatch)
    module = importlib.import_module("gr00t.eval.sim.robocasa365.gymnasium_groot")

    env = module.GrootRoboCasa365Env.__new__(module.GrootRoboCasa365Env)
    env.env = types.SimpleNamespace(robots=[], get_ep_meta=lambda: {"lang": "open the fridge"})
    env.enable_render = True
    env.camera_names = module.CAMERA_NAMES
    env.render_obs_key = "robot0_agentview_left_image"
    env.render_cache = None

    raw_obs = {
        "robot0_gripper_qpos": np.zeros(1, dtype=np.float32),
        "robot0_base_pos": np.zeros(3, dtype=np.float32),
        "robot0_base_quat": np.zeros(4, dtype=np.float32),
        "robot0_base_to_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_base_to_eef_quat": np.zeros(4, dtype=np.float32),
        "robot0_gripper_qvel": np.zeros(1, dtype=np.float32),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.zeros(4, dtype=np.float32),
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_joint_pos_cos": np.zeros(7, dtype=np.float32),
        "robot0_joint_pos_sin": np.zeros(7, dtype=np.float32),
        "robot0_joint_vel": np.zeros(7, dtype=np.float32),
        "robot0_agentview_left_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_agentview_right_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
    }

    obs = env._get_groot_observation(raw_obs)

    assert set(module.VIDEO_OBSERVATION_KEYS).issubset(obs)
    assert set(module.ROBOCASA_PANDA_VIDEO_OBSERVATION_KEYS).issubset(obs)
    assert obs["video.robot0_agentview_left"].shape == (256, 256, 3)
    assert obs["video.res256_image_side_0"].shape == (256, 256, 3)
    assert obs["annotation.human.task_description"] == "open the fridge"
    assert obs["annotation.human.action.task_description"] == "open the fridge"
