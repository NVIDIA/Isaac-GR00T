"""
This is an example of how to use the HttpClientPolicy to get the action from the robot.

pip install requests json-numpy
"""

import time
from collections import deque
from typing import Any

import numpy as np

try:
    import json_numpy
    import numpy as np
    import requests

    json_numpy.patch()
except ImportError:
    print("json_numpy or requests not found")
    pass


class HttpClientPolicy:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def get_action(
        self, observation: dict[str, Any], config: dict[str, Any] = None
    ) -> dict[str, Any]:
        print(f" -> GET ACTION CONFIG: {config}")
        response = requests.post(
            f"http://{self.host}:{self.port}/act",
            json={"observation": observation, "config": config},
        )
        return response.json()


class RTCPolicyWrapper:
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
    """

    def __init__(
        self,
        policy: HttpClientPolicy,
        control_freq: int,
        denoising_steps: int = 8,
        max_rtc_overlap_factor: float = 0.75,
        latency_queue_size: int = 10,
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

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._previous_action is not None:
            observation = {**observation, **self._previous_action}
            config = self._get_config()
        else:
            config = None
        start_time = time.monotonic()
        action = self.policy.get_action(observation, config)
        self._previous_action = action

        # get the action horizon
        if self._action_horizon is None:
            self._set_action_horizon(action)

        # update latency
        latency = time.monotonic() - start_time
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

        frozen_steps = max(min(frozen_steps, max_rtc_steps), 0)
        overlap_steps = max(
            min(self._action_horizon - executed_steps + frozen_steps, max_rtc_steps), frozen_steps
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument("--denoising_steps", type=int, default=4)
    parser.add_argument("--max_rtc_overlap_factor", type=float, default=0.75)
    args = parser.parse_args()

    policy = HttpClientPolicy(args.host, args.port)
    policy = RTCPolicyWrapper(
        policy, args.control_freq, args.denoising_steps, args.max_rtc_overlap_factor
    )

    for i in range(20):
        t = time.time()
        obs = {
            "video.test": np.zeros((1, 480, 640, 3), dtype=np.uint8),
            "video.ego_view": np.zeros((1, 256, 256, 3), dtype=np.uint8),
            "state.left_arm": np.random.rand(1, 7),
            "state.right_arm": np.random.rand(1, 7),
            "state.left_hand": np.random.rand(1, 6),
            "state.right_hand": np.random.rand(1, 6),
            "state.waist": np.random.rand(1, 3),
            "annotation.human.action.task_description": ["do your thing!"],
        }
        action = policy.get_action(obs)

        # NOTE(youliang): you will execute your action here in the real robot
        # env.step(action)

        # lazy sleep to simulate the inference frequency
        time.sleep(0.5)
        print(f"{i}:  used time {time.time() - t}")
