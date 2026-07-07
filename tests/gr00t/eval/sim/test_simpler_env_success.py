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

"""SimplerEnv success signal must come from the task predicate, not termination.

The step wrappers previously set ``info["success"] = done``. Under
``terminate_on_success`` the multistep wrapper both terminates on and counts
``info["success"]``, so conflating it with the raw termination flag miscounts
episodes (a timeout/other termination reads as a success, and a success that has
not yet terminated reads as a failure). These CPU-only tests drive ``step`` with a
fake inner env and assert the two signals are decoupled. Heavy sim deps are stubbed
so the test runs without cv2 / simpler_env / transforms3d installed.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


_ACTION_KEYS = (
    "action.x",
    "action.y",
    "action.z",
    "action.roll",
    "action.pitch",
    "action.yaw",
    "action.gripper",
)


def _install_simpler_env_import_stubs(monkeypatch):
    """Register lightweight stand-ins for the module's heavy sim imports."""
    cv2 = types.ModuleType("cv2")
    cv2.resize = lambda img, size: img

    gymnasium = types.ModuleType("gymnasium")
    gymnasium.Env = type("Env", (), {})
    gymnasium.spaces = types.SimpleNamespace(Box=object, Dict=object, Text=object)
    gym_envs = types.ModuleType("gymnasium.envs")
    gym_registration = types.ModuleType("gymnasium.envs.registration")
    gym_registration.register = lambda **kwargs: None

    simpler_env = types.ModuleType("simpler_env")
    simpler_env.make = lambda name: None
    se_utils = types.ModuleType("simpler_env.utils")
    se_env = types.ModuleType("simpler_env.utils.env")
    se_obs = types.ModuleType("simpler_env.utils.env.observation_utils")
    se_obs.get_image_from_maniskill2_obs_dict = lambda env, obs: np.zeros((4, 4, 3), np.uint8)

    transforms3d = types.ModuleType("transforms3d")
    t3d_euler = types.ModuleType("transforms3d.euler")
    t3d_quaternions = types.ModuleType("transforms3d.quaternions")
    transforms3d.euler = t3d_euler
    transforms3d.quaternions = t3d_quaternions

    for name, module in {
        "cv2": cv2,
        "gymnasium": gymnasium,
        "gymnasium.envs": gym_envs,
        "gymnasium.envs.registration": gym_registration,
        "simpler_env": simpler_env,
        "simpler_env.utils": se_utils,
        "simpler_env.utils.env": se_env,
        "simpler_env.utils.env.observation_utils": se_obs,
        "transforms3d": transforms3d,
        "transforms3d.euler": t3d_euler,
        "transforms3d.quaternions": t3d_quaternions,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


class _FakeInner:
    """Minimal inner env returning a controlled (done, info["success"]) pair."""

    def __init__(self, done: bool, success: bool):
        self._done = done
        self._success = success

    def step(self, action_vector):
        info = {"success": self._success, "elapsed_steps": 1}
        return {"agent": {"eef_pos": np.zeros(8)}}, 0.0, self._done, False, info


def _make_env(cls, inner):
    env = cls.__new__(cls)
    env.env = inner
    env.image_size = (256, 256)
    # Bypass the heavy image/proprio processing; step only needs the info flag.
    env._process_observation = lambda obs: {}
    # Gripper state used by GoogleFractalEnv._postprocess_gripper (harmless for WidowX).
    env.previous_gripper_action = None
    env.sticky_action_is_on = False
    env.sticky_gripper_action = 0.0
    env.gripper_action_repeat = 0
    env.sticky_gripper_num_repeat = 15
    return env


def _action():
    return {key: np.zeros(1, dtype=np.float32) for key in _ACTION_KEYS}


@pytest.mark.parametrize("env_cls_name", ["GoogleFractalEnv", "WidowXBridgeEnv"])
@pytest.mark.parametrize(
    "done, success",
    [
        (True, False),  # termination without task success (e.g. timeout) -> not success
        (False, True),  # task solved before termination -> success
        (True, True),
        (False, False),
    ],
)
def test_step_success_from_predicate_not_termination(monkeypatch, env_cls_name, done, success):
    _install_simpler_env_import_stubs(monkeypatch)
    module = importlib.import_module("gr00t.eval.sim.SimplerEnv.simpler_env")
    env = _make_env(getattr(module, env_cls_name), _FakeInner(done, success))

    _obs, _reward, ret_done, _truncated, info = env.step(_action())

    assert ret_done == done, "termination flag must be passed through unchanged"
    assert info["success"] is success, "success must reflect the task predicate, not done"


@pytest.mark.parametrize("env_cls_name", ["GoogleFractalEnv", "WidowXBridgeEnv"])
def test_step_success_defaults_false_when_predicate_absent(monkeypatch, env_cls_name):
    _install_simpler_env_import_stubs(monkeypatch)
    module = importlib.import_module("gr00t.eval.sim.SimplerEnv.simpler_env")

    class _NoSuccessInner(_FakeInner):
        def step(self, action_vector):
            return {"agent": {"eef_pos": np.zeros(8)}}, 0.0, True, False, {"elapsed_steps": 1}

    env = _make_env(getattr(module, env_cls_name), _NoSuccessInner(True, False))

    _obs, _reward, _done, _truncated, info = env.step(_action())

    assert info["success"] is False, "missing predicate must default to failure, not done"
