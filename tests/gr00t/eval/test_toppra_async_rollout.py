from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
from typing import Any

from gr00t.eval.real_robot.TOPPRA.eval_toppra_bimanual import (
    BimanualCartesianToppraPlanner,
    BimanualToppraRolloutConfig,
    CartesianLimits,
    GR00TBimanualToppraAgent,
)
import numpy as np
import pytest


class _SequencedPolicy:
    def __init__(self, request_count: int = 3) -> None:
        self.started = [threading.Event() for _ in range(request_count)]
        self.release = [threading.Event() for _ in range(request_count)]
        self.calls = 0
        self.observations: list[dict[str, Any]] = []

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        call_index = self.calls
        self.calls += 1
        self.observations.append(observation)
        self.started[call_index].set()
        if not self.release[call_index].wait(timeout=3.0):
            raise TimeoutError("test did not release the fake policy")

        batch, horizon = 1, 16
        left = np.zeros((batch, horizon, 6), dtype=np.float32)
        right = np.zeros_like(left)
        left[..., 0] = 0.003
        right[..., 1] = -0.003
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
        for event in self.release:
            event.set()


def _limits() -> CartesianLimits:
    return CartesianLimits(
        max_linear_velocity=(0.1, 0.1, 0.1),
        max_angular_velocity=(3.0, 3.0, 3.0),
        max_linear_acceleration=(100.0, 100.0, 100.0),
        max_angular_acceleration=(300.0, 300.0, 300.0),
        safety_margin=0.9,
    )


def _observation(
    left_pose: np.ndarray | None = None,
    right_pose: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    return {
        "observation.state.left_tcp": np.array(
            [0.4, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0] if left_pose is None else left_pose,
            dtype=np.float32,
        ),
        "observation.state.right_tcp": np.array(
            [0.4, -0.2, 0.5, 1.0, 0.0, 0.0, 0.0] if right_pose is None else right_pose,
            dtype=np.float32,
        ),
    }


def _wait_for_generation(agent: GR00TBimanualToppraAgent, generation: int) -> None:
    deadline = time.perf_counter() + 3.0
    while agent.diagnostics()["completed_generation"] < generation:
        if time.perf_counter() >= deadline:
            raise TimeoutError(f"trajectory generation {generation} did not complete")
        time.sleep(0.002)


def _advance_active_progress(
    agent: GR00TBimanualToppraAgent,
    completed_waypoints: int,
    observation: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    for _ in range(1000):
        if agent.diagnostics()["total_completed_waypoints"] >= completed_waypoints:
            return observation
        action = agent.act(observation)
        observation = _observation(action["action.left_tcp"], action["action.right_tcp"])
    raise AssertionError(f"active buffer did not complete {completed_waypoints} waypoints")


def _handoff_test_config(*, handoff_margin_s: float = 0.05) -> BimanualToppraRolloutConfig:
    limits = _limits()
    return BimanualToppraRolloutConfig(
        left_limits=limits,
        right_limits=limits,
        inference_mode="async",
        refill_lead_s=0.0,
        max_latency_s=0.0,
        scheduling_margin_s=0.0,
        handoff_margin_s=handoff_margin_s,
        open_loop_horizon=100,
        async_terminal_path_velocity=2.0,
    )


def _start_first_trajectory(
    agent: GR00TBimanualToppraAgent,
    policy: _SequencedPolicy,
    observation: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    held = agent.act(observation)
    np.testing.assert_allclose(held["action.left_tcp"], observation["observation.state.left_tcp"])
    assert policy.started[0].wait(timeout=1.0)
    policy.release[0].set()
    _wait_for_generation(agent, 1)
    action = agent.act(observation)
    assert agent.diagnostics()["active_sequence_id"] == 1
    return _observation(action["action.left_tcp"], action["action.right_tcp"])


def test_async_trigger_uses_fixed_latency_and_open_loop_budgets() -> None:
    limits = _limits()
    config = BimanualToppraRolloutConfig(
        left_limits=limits,
        right_limits=limits,
        inference_mode="async",
        refill_lead_s=0.0,
        max_latency_s=0.12,
        scheduling_margin_s=0.05,
        handoff_margin_s=0.05,
        open_loop_horizon=8,
    )
    policy = _SequencedPolicy(request_count=1)
    agent = GR00TBimanualToppraAgent(config, policy_client=policy)
    try:
        observation = _start_first_trajectory(agent, policy, _observation())
        del observation
        with agent._cv:
            assert agent._async_refill_lead_locked() == pytest.approx(0.22)
            assert agent._should_request_async_refill_locked(0.219)
            agent._active_policy_discarded_waypoints = config.open_loop_horizon
            assert agent._should_request_async_refill_locked(10.0)
    finally:
        policy.close()
        agent.teardown()


def test_async_rollout_switches_at_reserved_command_handoff(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.INFO,
        logger="gr00t.eval.real_robot.TOPPRA.eval_toppra_bimanual",
    )
    config = _handoff_test_config()
    policy = _SequencedPolicy(request_count=2)
    agent = GR00TBimanualToppraAgent(config, policy_client=policy)
    original_plan = agent.planner.plan
    captured_pose: dict[str, np.ndarray] = {}
    captured_twist: dict[str, np.ndarray] = {}
    plan_calls = 0

    def capture_second_plan(*args: Any, **kwargs: Any):
        nonlocal plan_calls
        plan_calls += 1
        if plan_calls == 2:
            captured_pose.update({arm: value.copy() for arm, value in args[1].items()})
            captured_twist.update({arm: value.copy() for arm, value in args[2].items()})
        return original_plan(*args, **kwargs)

    agent.planner.plan = capture_second_plan  # type: ignore[method-assign]
    observation = _observation()
    try:
        observation = _start_first_trajectory(agent, policy, observation)
        agent.request_trajectory(observation, block=False)
        assert policy.started[1].wait(timeout=1.0)

        started_at = time.perf_counter()
        previous = agent.act(observation)
        assert time.perf_counter() - started_at < 0.05
        observation = _observation(previous["action.left_tcp"], previous["action.right_tcp"])
        policy.release[1].set()
        _wait_for_generation(agent, 2)

        with agent._cv:
            ready = agent._ready_trajectory
            assert ready is not None
            assert ready.handoff_sample is not None
            assert ready.handoff_control_count is not None
            handoff_count = ready.handoff_control_count
            handoff_action = ready.handoff_sample.action
            new_start = ready.samples[0].action
            assert agent._total_consumed_control_samples < handoff_count

        for arm in ("left", "right"):
            np.testing.assert_allclose(
                captured_pose[arm],
                handoff_action[config.pose_output_keys[arm]],
            )
            np.testing.assert_allclose(
                captured_twist[arm],
                handoff_action[config.velocity_output_keys[arm]],
            )
            np.testing.assert_allclose(
                new_start[config.pose_output_keys[arm]],
                handoff_action[config.pose_output_keys[arm]],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                new_start[config.velocity_output_keys[arm]],
                handoff_action[config.velocity_output_keys[arm]],
                atol=1e-6,
            )

        while agent.diagnostics()["total_consumed_control_samples"] < handoff_count:
            previous = agent.act(observation)
            observation = _observation(previous["action.left_tcp"], previous["action.right_tcp"])
            assert agent.diagnostics()["active_sequence_id"] == 1

        switched = agent.act(observation)
        diagnostics = agent.diagnostics()
        assert diagnostics["active_sequence_id"] == 2
        assert diagnostics["latest_activation_offset_s"] == 0.0
        assert diagnostics["planning_discarded_control_samples"] == 0
        assert diagnostics["planning_twist_source"] == "scheduled_command"
        assert diagnostics["handoff_position_error_m"] <= 1e-6
        assert diagnostics["handoff_rotation_error_rad"] <= 1e-6
        assert diagnostics["handoff_twist_error"] <= 1e-6
        np.testing.assert_allclose(
            switched["action.left_tcp"],
            handoff_action[config.left_pose_output_key],
            atol=1e-6,
        )
        messages = [record.getMessage() for record in caplog.records]
        assert any("[TOPPRA][handoff_reserved]" in message for message in messages)
        assert any("[TOPPRA][plan_success]" in message for message in messages)
        assert any("[TOPPRA][handoff_activated]" in message for message in messages)
    finally:
        policy.close()
        agent.teardown()


def test_async_handoff_uses_old_command_even_when_measured_pose_lags() -> None:
    limits = _limits()
    config = BimanualToppraRolloutConfig(
        left_limits=limits,
        right_limits=limits,
        inference_mode="async",
        refill_lead_s=0.0,
        max_latency_s=0.0,
        scheduling_margin_s=0.0,
        handoff_margin_s=0.05,
        open_loop_horizon=100,
        left_velocity_state_key="observation.state.left_tcp_twist",
        right_velocity_state_key="observation.state.right_tcp_twist",
        velocity_filter=1.0,
    )
    policy = _SequencedPolicy(request_count=2)
    agent = GR00TBimanualToppraAgent(config, policy_client=policy)
    captured_pose: dict[str, np.ndarray] = {}
    captured_twist: dict[str, np.ndarray] = {}
    original_plan = agent.planner.plan
    plan_calls = 0

    def capture_plan(*args: Any, **kwargs: Any):
        nonlocal plan_calls
        plan_calls += 1
        if plan_calls == 2:
            captured_pose.update({arm: value.copy() for arm, value in args[1].items()})
            captured_twist.update({arm: value.copy() for arm, value in args[2].items()})
        return original_plan(*args, **kwargs)

    agent.planner.plan = capture_plan  # type: ignore[method-assign]
    initial_obs = _observation()
    initial_obs["observation.state.left_tcp_twist"] = np.zeros(6, dtype=np.float32)
    initial_obs["observation.state.right_tcp_twist"] = np.zeros(6, dtype=np.float32)
    latest_obs = _observation(
        left_pose=np.array([0.41, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0]),
        right_pose=np.array([0.4, -0.21, 0.5, 1.0, 0.0, 0.0, 0.0]),
    )
    latest_obs["observation.state.left_tcp_twist"] = np.array(
        [0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    latest_obs["observation.state.right_tcp_twist"] = np.array(
        [0.0, -0.01, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    try:
        _start_first_trajectory(agent, policy, initial_obs)
        agent.request_trajectory(latest_obs, block=False)
        assert policy.started[1].wait(timeout=1.0)
        agent.act(latest_obs)
        policy.release[1].set()
        _wait_for_generation(agent, 2)

        with agent._cv:
            ready = agent._ready_trajectory
            assert ready is not None
            assert ready.handoff_sample is not None
            handoff_action = ready.handoff_sample.action
        for arm in ("left", "right"):
            np.testing.assert_allclose(
                captured_pose[arm],
                handoff_action[config.pose_output_keys[arm]],
            )
            np.testing.assert_allclose(
                captured_twist[arm],
                handoff_action[config.velocity_output_keys[arm]],
            )
        assert not np.allclose(captured_pose["left"], latest_obs[config.left_pose_state_key])
        assert not np.allclose(captured_pose["right"], latest_obs[config.right_pose_state_key])
        assert "left_tcp_twist" not in policy.observations[1]["state"]
        assert "right_tcp_twist" not in policy.observations[1]["state"]
    finally:
        policy.close()
        agent.teardown()


def test_async_rollout_discards_a_trajectory_that_misses_its_handoff(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.INFO,
        logger="gr00t.eval.real_robot.TOPPRA.eval_toppra_bimanual",
    )
    config = _handoff_test_config(handoff_margin_s=0.02)
    policy = _SequencedPolicy(request_count=2)
    agent = GR00TBimanualToppraAgent(config, policy_client=policy)
    original_plan = agent.planner.plan
    second_plan_started = threading.Event()
    release_second_plan = threading.Event()
    plan_calls = 0

    def controlled_plan(*args: Any, **kwargs: Any):
        nonlocal plan_calls
        plan_calls += 1
        plan_args = list(args)
        if plan_calls == 2:
            plan_args[2] = {arm: np.zeros(6, dtype=np.float64) for arm in ("left", "right")}
        trajectory = original_plan(*plan_args, **kwargs)
        if plan_calls == 2:
            second_plan_started.set()
            if not release_second_plan.wait(timeout=3.0):
                raise TimeoutError("test did not release the second TOPPRA plan")
        return trajectory

    agent.planner.plan = controlled_plan  # type: ignore[method-assign]
    observation = _observation()
    try:
        observation = _start_first_trajectory(agent, policy, observation)
        agent.request_trajectory(observation, block=False)
        assert policy.started[1].wait(timeout=1.0)
        policy.release[1].set()
        assert second_plan_started.wait(timeout=1.0)

        lead_samples = int(np.ceil(config.handoff_margin_s / config.control_dt))
        for _ in range(lead_samples + 2):
            action = agent.act(observation)
            observation = _observation(action["action.left_tcp"], action["action.right_tcp"])

        release_second_plan.set()
        _wait_for_generation(agent, 2)
        previous_sequence = agent.diagnostics()["active_sequence_id"]
        agent.act(observation)
        diagnostics = agent.diagnostics()
        assert previous_sequence == 1
        assert diagnostics["active_sequence_id"] == 1
        assert diagnostics["ready_sequence_id"] is None
        assert "missed the reserved handoff" in diagnostics["latest_error"]
        messages = [record.getMessage() for record in caplog.records]
        assert any("[TOPPRA][handoff_missed]" in message for message in messages)
    finally:
        policy.close()
        release_second_plan.set()
        agent.teardown()


def test_toppra_planning_failure_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(
        logging.INFO,
        logger="gr00t.eval.real_robot.TOPPRA.eval_toppra_bimanual",
    )
    limits = _limits()
    config = BimanualToppraRolloutConfig(
        left_limits=limits,
        right_limits=limits,
        inference_mode="sync",
    )
    policy = _SequencedPolicy(request_count=1)
    agent = GR00TBimanualToppraAgent(config, policy_client=policy)

    def reject_plan(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("forced TOPPRA failure")

    agent.planner.plan = reject_plan  # type: ignore[method-assign]
    try:
        agent.act(_observation())
        assert policy.started[0].wait(timeout=1.0)
        policy.release[0].set()
        _wait_for_generation(agent, 1)

        diagnostics = agent.diagnostics()
        assert "forced TOPPRA failure" in diagnostics["latest_error"]
        messages = [record.getMessage() for record in caplog.records]
        assert any("[TOPPRA][candidate_failed]" in message for message in messages)
        assert any("[TOPPRA][planning_failed]" in message for message in messages)
    finally:
        policy.close()
        agent.teardown()


def test_unmeasured_command_twist_can_be_relaxed_to_the_controllable_set() -> None:
    limits = CartesianLimits(
        max_linear_velocity=(0.1, 0.1, 0.1),
        max_angular_velocity=(3.0, 3.0, 3.0),
        max_linear_acceleration=(0.2, 0.2, 0.2),
        max_angular_acceleration=(30.0, 30.0, 30.0),
        safety_margin=0.9,
    )
    config = BimanualToppraRolloutConfig(
        left_limits=limits,
        right_limits=limits,
        inference_mode="async",
        control_frequency=30.0,
    )
    planner = BimanualCartesianToppraPlanner(config)
    poses = {
        "left": np.array([0.4, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0]),
        "right": np.array([0.4, -0.2, 0.5, 1.0, 0.0, 0.0, 0.0]),
    }
    twists = {
        "left": np.array([0.09, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "right": np.array([0.0, -0.08, 0.0, 0.0, 0.0, 0.0]),
    }
    horizon = 16
    left_actions = np.zeros((horizon, 6))
    right_actions = np.zeros((horizon, 6))
    left_actions[:, 0] = 0.003
    right_actions[:, 1] = -0.003
    actions = {
        "action.left_delta_tcp": left_actions,
        "action.right_delta_tcp": right_actions,
        "action.left_pinch": np.zeros((horizon, 1)),
        "action.right_pinch": np.zeros((horizon, 1)),
    }

    with pytest.raises(RuntimeError, match="outside the controllable interval"):
        planner.plan(actions, poses, twists, relax_initial_twist=False)

    trajectory = planner.plan(actions, poses, twists, relax_initial_twist=True)
    assert 0.0 < trajectory.initial_twist_scale < 1.0
    start = trajectory.sample(0.0)
    np.testing.assert_allclose(
        start[config.left_velocity_output_key],
        twists["left"] * trajectory.initial_twist_scale,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        start[config.right_velocity_output_key],
        twists["right"] * trajectory.initial_twist_scale,
        atol=1e-6,
    )


def test_command_twist_fallback_is_aligned_with_the_observed_pose() -> None:
    limits = _limits()
    config = BimanualToppraRolloutConfig(
        left_limits=limits,
        right_limits=limits,
        inference_mode="async",
    )
    policy = _SequencedPolicy(request_count=1)
    agent = GR00TBimanualToppraAgent(config, policy_client=policy)
    observed_twists = {
        "left": np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "right": np.array([0.0, -0.01, 0.0, 0.0, 0.0, 0.0]),
    }
    next_command_twists = {
        "left": np.array([0.08, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "right": np.array([0.0, -0.08, 0.0, 0.0, 0.0, 0.0]),
    }

    def sample_with_twists(twists: dict[str, np.ndarray]) -> SimpleNamespace:
        return SimpleNamespace(
            action={config.velocity_output_keys[arm]: twist.copy() for arm, twist in twists.items()}
        )

    try:
        with agent._cv:
            agent._active_trajectory = object()  # type: ignore[assignment]
            agent._active_last_sample = sample_with_twists(observed_twists)  # type: ignore[assignment]
            agent._record_observation_locked(_observation(), "")
            agent._active_last_sample = sample_with_twists(  # type: ignore[assignment]
                next_command_twists
            )
            planning_twists, source = agent._planning_twist_locked()
            agent._active_trajectory = None
            agent._active_last_sample = None

        assert source == "commanded"
        for arm in ("left", "right"):
            np.testing.assert_allclose(planning_twists[arm], observed_twists[arm])
    finally:
        policy.close()
        agent.teardown()
