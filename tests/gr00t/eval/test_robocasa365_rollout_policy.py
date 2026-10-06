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

import sys
import types

from gr00t.eval import rollout_policy


def test_robocasa365_env_fn_passes_split_to_gym_make(monkeypatch):
    fake_robocasa365 = types.ModuleType("gr00t.eval.sim.robocasa365.gymnasium_groot")
    monkeypatch.setitem(
        sys.modules,
        "gr00t.eval.sim.robocasa365.gymnasium_groot",
        fake_robocasa365,
    )

    calls = []

    def fake_make(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(rollout_policy.gym, "make", fake_make)

    env_fn = rollout_policy.get_robocasa_env_fn(
        "robocasa365_panda_omron/CloseFridge_PandaOmron_Env",
        robocasa_split="pretrain",
    )
    env_fn()

    assert calls == [
        (
            ("robocasa365_panda_omron/CloseFridge_PandaOmron_Env",),
            {"enable_render": True, "split": "pretrain"},
        )
    ]


def test_robocasa365_record_video_keys_match_observation_keys():
    assert rollout_policy.ROBOCASA365_PANDA_RECORD_VIDEO_KEYS == (
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    )
