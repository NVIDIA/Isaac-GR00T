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

import time
from collections import deque
from typing import Any, Dict

import numpy as np

from gr00t.data.dataset import ModalityConfig
from gr00t.model.policy import BasePolicy


class RTCPolicyWrapper(BasePolicy):
    """
    A real-time chunking (RTC) policy wrapper for streaming robot control.

    This class wraps a standard policy's `get_action(obs)` method to enable real-time
    chunked inference and action denoising, adapting to system latency and control frequency.

    Args:
        policy (HttpClientPolicy): The underlying policy client to query for actions.
        control_freq (int): Control frequency of the robot (steps per second).
        denoising_steps (int): Number of steps to use for action denoising.
        max_rtc_overlap_factor (float): Maximum overlap factor for RTC, between 0 and 1.
            1.0 means full overlap (entire overlap chunk used in rtc), 0.0 means no overlap.
        latency_queue_size (int): Size of the latency queue.
        systematic_latency_offset (float): Systematic latency offset in seconds.
    """

    def __init__(
        self,
        policy: BasePolicy,
        control_freq: int,
        denoising_steps: int = 8,
        max_rtc_overlap_factor: float = 0.75,
        latency_queue_size: int = 10,
        systematic_latency_offset: float = 0.02,
    ):
        self.policy = policy
        self.control_freq = control_freq
        self.latency_queue = deque(maxlen=latency_queue_size)
        self.latency_queue.append(0)  # initialize with 0
        self.denoising_steps = denoising_steps
        self._previous_action = None
        self._last_inference_completion_time = 0
        self._action_horizon = None
        self._max_rtc_overlap_factor = max_rtc_overlap_factor
        self.systematic_latency_offset = systematic_latency_offset

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._previous_action is not None:
            observation = {**observation, **self._previous_action}
            config = self._get_config()
        else:
            config = None
        start_time = time.monotonic()
        action = self.policy.get_action(observation, config)
        self._previous_action = action.copy()

        # get the action horizon
        if self._action_horizon is None:
            self._set_action_horizon(action)

        # update latency
        latency = time.monotonic() - start_time + self.systematic_latency_offset
        if latency > (1 / self.control_freq) * self._action_horizon:
            print(f"Latency is too high: {latency} seconds, skipping")
        else:
            self.latency_queue.append(latency)

        self._last_inference_completion_time = time.monotonic()
        return action

    def _set_action_horizon(self, action: dict[str, Any]):
        # get the first key of the action, and get the shape of the value
        for k in action.keys():
            # check if it is a numpy array
            if isinstance(action[k], np.ndarray):
                self._action_horizon = action[k].shape[0]
                print(f"action_horizon: {self._action_horizon}")
                return
        raise ValueError("No numpy array found in action")

    def _get_config(self) -> dict[str, Any]:
        # get avg latency
        assert self._action_horizon is not None, "action_horizon is not set"
        avg_latency = sum(self.latency_queue) / len(self.latency_queue)
        frozen_steps = int(avg_latency * self.control_freq)

        between_inference_time = time.monotonic() - self._last_inference_completion_time
        executed_steps = int(self.control_freq * between_inference_time)
        max_rtc_steps = self._action_horizon * self._max_rtc_overlap_factor

        frozen_steps = int(max(min(frozen_steps, max_rtc_steps), 0))
        overlap_steps = int(
            max(
                min(self._action_horizon - executed_steps + frozen_steps, max_rtc_steps),
                frozen_steps,
            )
        )

        assert (
            self._action_horizon >= overlap_steps >= 0
        ), f"overlap_steps is not within bounds, {overlap_steps}"
        assert (
            self._action_horizon >= frozen_steps >= 0
        ), f"rtc_frozen_steps is not within bounds, {frozen_steps}"

        return {
            "denoising_steps": self.denoising_steps,
            "rtc_overlap_steps": overlap_steps,
            "rtc_frozen_steps": frozen_steps,
        }

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        return self.policy.get_modality_config()
