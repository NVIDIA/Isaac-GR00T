# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gymnasium as gym
import numpy as np


class ObsIndexSelectionWrapper(gym.Wrapper):
    def __init__(self, env, video_delta_indices, state_delta_indices):
        super().__init__(env)
        self.video_delta_indices = video_delta_indices
        self.video_horizon = len(video_delta_indices)
        self.assert_delta_indices(self.video_delta_indices, self.video_horizon)

        if state_delta_indices is not None:
            self.state_delta_indices = state_delta_indices
            self.state_horizon = len(state_delta_indices)
            self.assert_delta_indices(self.state_delta_indices, self.state_horizon)
        else:
            self.state_horizon = None
            self.state_delta_indices = None

        self._observation_space = self.convert_observation_space(
            self.observation_space,
            self.video_horizon,
            self.state_horizon,
        )

    def assert_delta_indices(self, delta_indices: np.ndarray, horizon: int):
        # Check the length
        # (In this wrapper, this seems redundant because we get the horizon from the delta indices. But in the policy, the horizon is not derived from the delta indices but we need to make it consistent. To make the function consistent, we keep the check here.)
        assert len(delta_indices) == horizon, f"{delta_indices=}, {horizon=}"
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent (because in real robot experiments, we actually use the dt to get the observations, which requires the step to be consistent)
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"

    def select_steps_for_values(self, data_value, delta_indices):
        """
        data_value: [L, ...]
        delta_indices: np.ndarray[int], please check `assert_delta_indices` to see the requirements
        """
        L = data_value.shape[0]
        assert L >= len(delta_indices), f"{L=}, {len(delta_indices)=}"
        selected_indices = (L - 1) + delta_indices
        assert selected_indices[0] >= 0, f"{L=}, {selected_indices=}"
        return data_value[selected_indices]

    def select_steps_for_obs(self, obs):
        new_obs = {}
        for k in obs.keys():
            if k.startswith("video"):
                new_obs[k] = self.select_steps_for_values(obs[k], self.video_delta_indices)
            elif k.startswith("state"):
                if self.state_delta_indices is not None:
                    new_obs[k] = self.select_steps_for_values(obs[k], self.state_delta_indices)
                else:
                    # Don't include the state in the observation
                    continue
            else:
                raise ValueError(f"Unknown key: {k}")
        return new_obs

    def convert_observation_space(self, observation_space, video_horizon, state_horizon):
        new_observation_space = {}
        for k in observation_space.keys():
            box = observation_space[k]
            if k.startswith("video"):
                horizon = video_horizon
            elif k.startswith("state"):
                if state_horizon is not None:
                    horizon = state_horizon
                else:
                    # Don't include the state in the observation space
                    continue
            else:
                raise ValueError(f"Unknown key: {k}")

            new_observation_space[k] = gym.spaces.Box(
                low=box.low[:horizon],
                high=box.high[:horizon],
                shape=(horizon, *box.shape[1:]),
                dtype=box.dtype,
            )
        return gym.spaces.Dict(new_observation_space)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        obs = self.select_steps_for_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self.select_steps_for_obs(obs)
        return obs, reward, terminated, truncated, info
