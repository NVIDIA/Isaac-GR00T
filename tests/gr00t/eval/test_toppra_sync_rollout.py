from __future__ import annotations

import inspect
import threading
import time
from typing import Any

from gr00t.eval.real_robot.TOPPRA.eval_toppra_bimanual import (
    DEFAULT_TASK,
    BimanualToppraRolloutConfig,
    CartesianLimits,
    GR00TAgent,
    GR00TBimanualToppraAgent,
    GR00TPolicyClient,
)
import numpy as np


class _BlockingPolicy:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del observation, options
        self.calls += 1
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test did not release the fake policy")

        batch, horizon = 1, 32
        left = np.zeros((batch, horizon, 6), dtype=np.float32)
        right = np.zeros_like(left)
        left[..., 0] = 0.004
        left[..., 5] = 0.008
        right[..., 1] = -0.004
        right[..., 3] = -0.008
        pinch = np.zeros((batch, horizon, 1), dtype=np.float32)
        return (
            {
                "left_delta_tcp": left,
                "right_delta_tcp": right,
                "left_pinch": pinch,
                "right_pinch": pinch.copy(),
            },
            {},
        )

    def close(self) -> None:
        self.release.set()


def _limits() -> CartesianLimits:
    return CartesianLimits(
        max_linear_velocity=(1.0, 1.0, 1.0),
        max_angular_velocity=(3.0, 3.0, 3.0),
        max_linear_acceleration=(10.0, 10.0, 10.0),
        max_angular_acceleration=(30.0, 30.0, 30.0),
        safety_margin=0.9,
    )


def _observation(left_x: float = 0.4) -> dict[str, np.ndarray]:
    return {
        "observation.state.left_tcp": np.array(
            [left_x, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
        "observation.state.right_tcp": np.array(
            [0.4, -0.2, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
    }


def _wait_for_generation(agent: GR00TBimanualToppraAgent, generation: int) -> None:
    deadline = time.perf_counter() + 2.0
    while agent.diagnostics()["completed_generation"] < generation:
        if time.perf_counter() >= deadline:
            raise TimeoutError(f"trajectory generation {generation} did not complete")
        time.sleep(0.002)


def test_example_agent_interface_remains_drop_in_compatible() -> None:
    expected_prefix = [
        "self",
        "host",
        "port",
        "fps",
        "refill_threshold",
        "max_latency_s",
        "buffer_ttl_s",
        "ignore_first_latency",
        "task_default",
        "timeout_ms",
        "scaling",
    ]
    assert list(inspect.signature(GR00TAgent.__init__).parameters)[: len(expected_prefix)] == (
        expected_prefix
    )
    assert callable(GR00TPolicyClient.get_action)
    assert callable(GR00TPolicyClient.get_modality_config)
    assert callable(GR00TPolicyClient.ping)
    assert callable(GR00TPolicyClient.close)

    policy = _BlockingPolicy()
    agent = GR00TAgent(policy_client=policy)
    try:
        assert agent.config.inference_mode == "async"
        assert agent.task_default == DEFAULT_TASK == "move parcel onto conveyor belt one by one"
        assert agent.config.left_limits.max_linear_velocity == (1.0, 1.0, 1.0)
        assert callable(agent.act)
        assert callable(agent.teardown)
        assert callable(agent.setup_gui)
        assert callable(agent.update_gui)
        assert set(agent.act(_observation())) == {
            "action.left_tcp",
            "action.right_tcp",
            "action.left_pinch",
            "action.right_pinch",
        }
    finally:
        policy.release.set()
        agent.teardown()


def test_sync_rollout_holds_pose_and_executes_full_zero_boundary_trajectory() -> None:
    limits = _limits()
    config = BimanualToppraRolloutConfig(left_limits=limits, right_limits=limits)
    assert config.inference_mode == "sync"
    assert config.terminal_path_velocity_min == 0.0
    assert config.terminal_path_velocity_max == 0.0

    policy = _BlockingPolicy()
    agent = GR00TBimanualToppraAgent(config, policy_client=policy)
    initial_obs = _observation()
    try:
        first = agent.act(initial_obs)
        np.testing.assert_allclose(
            first["action.left_tcp"], initial_obs["observation.state.left_tcp"]
        )
        assert policy.started.wait(timeout=1.0)

        changed_during_inference = _observation(left_x=0.5)
        held = agent.act(changed_during_inference)
        np.testing.assert_allclose(
            held["action.left_tcp"], initial_obs["observation.state.left_tcp"]
        )
        assert agent.diagnostics()["execution_state"] == "holding_for_inference"

        policy.release.set()
        _wait_for_generation(agent, 1)
        trajectory = agent.poll_trajectory()
        assert trajectory is not None
        assert trajectory.path_speeds[0] == 0.0
        assert trajectory.path_speeds[-1] == 0.0

        start = trajectory.sample(0.0)
        end = trajectory.sample(trajectory.duration)
        np.testing.assert_allclose(
            start["action.left_tcp"], initial_obs["observation.state.left_tcp"], atol=1e-6
        )
        for arm in ("left", "right"):
            np.testing.assert_allclose(start[f"action.{arm}_tcp_velocity"], 0.0)
            np.testing.assert_allclose(start[f"action.{arm}_tcp_acceleration"], 0.0)
            np.testing.assert_allclose(end[f"action.{arm}_tcp_velocity"], 0.0)
            np.testing.assert_allclose(end[f"action.{arm}_tcp_acceleration"], 0.0)

        activated = agent.act(initial_obs)
        np.testing.assert_allclose(
            activated["action.left_tcp"], initial_obs["observation.state.left_tcp"], atol=1e-5
        )
        diagnostics = agent.diagnostics()
        assert diagnostics["execution_state"] == "executing"
        assert diagnostics["latest_activation_offset_s"] == 0.0

        policy.started.clear()
        policy.release.clear()
        next_obs = _observation(left_x=0.45)
        with agent._cv:
            assert agent._active_trajectory is not None
            endpoint_sample = agent._active_action_buf[-1]
            endpoint_pose = endpoint_sample.action["action.left_tcp"].copy()
            agent._active_last_sample = endpoint_sample
            agent._active_action_buf.clear()
        next_hold = agent.act(next_obs)
        np.testing.assert_allclose(next_hold["action.left_tcp"], endpoint_pose)
        assert policy.started.wait(timeout=1.0)

        changed_again = _observation(left_x=0.55)
        held_again = agent.act(changed_again)
        np.testing.assert_allclose(held_again["action.left_tcp"], endpoint_pose)

        policy.release.set()
        _wait_for_generation(agent, 2)
        next_trajectory = agent.poll_trajectory(after_sequence_id=1)
        assert next_trajectory is not None
        np.testing.assert_allclose(
            next_trajectory.sample(0.0)["action.left_tcp"],
            endpoint_pose,
            atol=1e-6,
        )
    finally:
        policy.release.set()
        agent.teardown()
