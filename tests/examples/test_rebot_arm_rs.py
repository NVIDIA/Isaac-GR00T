# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import msgpack_numpy as mnp
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "examples" / "rebot-arm" / "eval_rebot_arm_rs.py"
SERVER_CLIENT = REPO_ROOT / "gr00t" / "policy" / "server_client.py"


@pytest.fixture(scope="module")
def rebot_rs():
    """Load the client and real serializer without importing Torch or robot packages."""
    gr00t = ModuleType("gr00t")
    gr00t.__path__ = []
    gr00t_policy = ModuleType("gr00t.policy")
    gr00t_policy.__path__ = []
    gr00t_data = ModuleType("gr00t.data")
    gr00t_data.__path__ = []
    gr00t_data_types = ModuleType("gr00t.data.types")
    gr00t_data_types.ModalityConfig = type("ModalityConfig", (), {})
    gr00t_data_utils = ModuleType("gr00t.data.utils")
    gr00t_policy_base = ModuleType("gr00t.policy.policy")
    gr00t_policy_base.BasePolicy = type("BasePolicy", (), {})

    def to_json_serializable(value):
        return value

    gr00t_data_utils.to_json_serializable = to_json_serializable

    lerobot = ModuleType("lerobot")
    lerobot_cameras = ModuleType("lerobot.cameras")
    lerobot_opencv = ModuleType("lerobot.cameras.opencv")
    lerobot_opencv.OpenCVCameraConfig = type("OpenCVCameraConfig", (), {})
    lerobot_utils = ModuleType("lerobot.utils")
    lerobot_robot_utils = ModuleType("lerobot.utils.robot_utils")
    lerobot_robot_utils.precise_sleep = lambda _: None

    plugin = ModuleType("lerobot_robot_seeed_b601")
    plugin.SeeedB601RSFollower = type("SeeedB601RSFollower", (), {})
    plugin.SeeedB601RSFollowerConfig = type("SeeedB601RSFollowerConfig", (), {})

    serializer_spec = importlib.util.spec_from_file_location(
        "gr00t.policy.server_client", SERVER_CLIENT
    )
    assert serializer_spec is not None and serializer_spec.loader is not None
    serializer_module = importlib.util.module_from_spec(serializer_spec)

    eval_spec = importlib.util.spec_from_file_location("test_eval_rebot_arm_rs_target", EVAL_SCRIPT)
    assert eval_spec is not None and eval_spec.loader is not None
    eval_module = importlib.util.module_from_spec(eval_spec)

    stub_modules = {
        "gr00t": gr00t,
        "gr00t.data": gr00t_data,
        "gr00t.data.types": gr00t_data_types,
        "gr00t.data.utils": gr00t_data_utils,
        "gr00t.policy": gr00t_policy,
        "gr00t.policy.policy": gr00t_policy_base,
        "gr00t.policy.server_client": serializer_module,
        "lerobot": lerobot,
        "lerobot.cameras": lerobot_cameras,
        "lerobot.cameras.opencv": lerobot_opencv,
        "lerobot.utils": lerobot_utils,
        "lerobot.utils.robot_utils": lerobot_robot_utils,
        "lerobot_robot_seeed_b601": plugin,
    }
    with patch.dict(sys.modules, stub_modules):
        serializer_spec.loader.exec_module(serializer_module)
        eval_spec.loader.exec_module(eval_module)
    return eval_module


def _action_chunk(*, horizon: int = 2) -> dict[str, np.ndarray]:
    return {
        "single_arm": np.zeros((1, horizon, 6), dtype=np.float32),
        "gripper": np.zeros((1, horizon, 1), dtype=np.float32),
    }


def _physical_state(module, command: np.ndarray) -> np.ndarray:
    return np.asarray(command, dtype=np.float32) * module.JOINT_DIRECTIONS


def _observation(module, physical_state: np.ndarray) -> dict[str, float]:
    return {key: float(value) for key, value in zip(module.JOINT_KEYS, physical_state, strict=True)}


class FakeRobot:
    def __init__(self, module, physical_state: np.ndarray) -> None:
        self.module = module
        self.physical_state = np.asarray(physical_state, dtype=np.float32)
        self.is_connected = True
        self.sent_actions: list[dict[str, float]] = []
        self.disable_torque_calls = 0

    def get_observation(self) -> dict[str, float]:
        return _observation(self.module, self.physical_state)

    def send_action(self, action: dict[str, float]) -> None:
        self.sent_actions.append(action)

    def disable_torque(self) -> None:
        self.disable_torque_calls += 1


def test_policy_serializer_round_trips_numeric_arrays(rebot_rs):
    payload = {"action": np.array([1.0, 2.0, 3.0], dtype=np.float32)}

    decoded = rebot_rs.PolicyClient._unpack(rebot_rs.PolicyClient._pack(payload))

    np.testing.assert_array_equal(decoded["action"], payload["action"])


def test_policy_serializer_rejects_object_dtype_on_encode(rebot_rs):
    payload = np.array([{"unsafe": True}], dtype=object)

    with pytest.raises(TypeError, match="object-dtype"):
        rebot_rs.PolicyClient._pack(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {b"nd": True, b"kind": b"O", b"type": "|O", b"shape": (1,), b"data": b""},
        {"nd": True, "kind": "O", "type": "|O", "shape": (1,), "data": b""},
        {b"nd": 1, b"kind": b"O", b"type": "|O", b"shape": (1,), b"data": b""},
    ],
)
def test_policy_serializer_rejects_forged_object_arrays(rebot_rs, payload):
    forged = mnp.packb(payload)

    with pytest.raises(ValueError, match="object-dtype"):
        rebot_rs.PolicyClient._unpack(forged)


def test_decode_action_chunk_requires_all_modalities(rebot_rs):
    with pytest.raises(KeyError, match="gripper"):
        rebot_rs.decode_action_chunk({"single_arm": _action_chunk()["single_arm"]})


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_decode_action_chunk_rejects_non_finite_values(rebot_rs, bad_value):
    chunk = _action_chunk()
    chunk["single_arm"][0, 0, 0] = bad_value

    with pytest.raises(ValueError, match="NaN or infinity"):
        rebot_rs.decode_action_chunk(chunk)


def test_decode_action_chunk_clamps_to_command_limits(rebot_rs):
    chunk = _action_chunk(horizon=1)
    chunk["single_arm"][0, 0] = np.array([999.0, -999.0, 999.0, -999.0, 999.0, -999.0])
    chunk["gripper"][0, 0, 0] = 999.0

    decoded = rebot_rs.decode_action_chunk(chunk)

    np.testing.assert_array_equal(
        decoded[0],
        rebot_rs.COMMAND_UPPER * np.array([1, 0, 1, 0, 1, 0, 1])
        + rebot_rs.COMMAND_LOWER * np.array([0, 1, 0, 1, 0, 1, 0]),
    )


def test_smooth_action_segment_clamps_arm_and_passes_gripper_through(rebot_rs):
    requested = np.array(
        [
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 30.0],
        ],
        dtype=np.float32,
    )

    smoothed, final_state = rebot_rs.smooth_action_segment(
        requested,
        np.zeros(7, dtype=np.float32),
        arm_alpha=0.5,
        max_delta=2.0,
    )

    np.testing.assert_allclose(smoothed[:, :6], [[1.0] * 6, [2.0] * 6])
    np.testing.assert_array_equal(smoothed[:, 6], [20.0, 30.0])
    np.testing.assert_array_equal(final_state, smoothed[-1])


def test_trajectory_limiter_enforces_velocity_and_acceleration(rebot_rs):
    targets = np.tile(
        np.array([100.0, 100.0, -100.0, 60.0, 80.0, 80.0, 45.0], dtype=np.float32),
        (20, 1),
    )

    commands, _, _ = rebot_rs.limit_action_trajectory(
        targets,
        np.zeros(7, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
    )

    arm_steps = np.diff(
        np.vstack([np.zeros((1, 6), dtype=np.float32), commands[:, :6]]),
        axis=0,
    )
    arm_acceleration = np.diff(
        np.vstack([np.zeros((1, 6), dtype=np.float32), arm_steps]),
        axis=0,
    )
    assert np.all(np.abs(arm_steps) <= rebot_rs.TRAJECTORY_MAX_STEP + 1e-6)
    assert np.all(np.abs(arm_acceleration) <= rebot_rs.TRAJECTORY_MAX_STEP_CHANGE + 1e-6)
    np.testing.assert_array_equal(commands[:, 6], np.full(20, 45.0, dtype=np.float32))


@pytest.mark.parametrize("field", ["targets", "position", "step"])
def test_trajectory_limiter_rejects_non_finite_state(rebot_rs, field):
    targets = np.zeros((2, 7), dtype=np.float32)
    position = np.zeros(7, dtype=np.float32)
    step = np.zeros(7, dtype=np.float32)
    if field == "targets":
        targets[0, 0] = np.nan
    elif field == "position":
        position[0] = np.inf
    else:
        step[0] = -np.inf

    with pytest.raises(ValueError, match="NaN or infinity"):
        rebot_rs.limit_action_trajectory(targets, position, step)


def test_blend_overlapping_actions_aligns_arm_and_uses_new_gripper(rebot_rs):
    previous = np.zeros((6, 7), dtype=np.float32)
    previous[:, 6] = 5.0
    new = np.full((3, 7), 10.0, dtype=np.float32)
    new[:, 6] = np.array([20.0, 30.0, 40.0])

    blended = rebot_rs.blend_overlapping_actions(
        new,
        execution_start_step=2,
        previous_chunk=previous,
        previous_observation_step=0,
        max_old_weight=0.5,
        arm_disagreement_scale=1e6,
    )

    assert np.all(blended[0, :6] < new[0, :6])
    np.testing.assert_allclose(blended[-1, :6], new[-1, :6])
    np.testing.assert_array_equal(blended[:, 6], new[:, 6])


def test_blend_without_previous_chunk_returns_copy(rebot_rs):
    new = np.arange(21, dtype=np.float32).reshape(3, 7)

    blended = rebot_rs.blend_overlapping_actions(
        new,
        execution_start_step=0,
        previous_chunk=None,
        previous_observation_step=None,
    )

    np.testing.assert_array_equal(blended, new)
    assert blended is not new


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("invalid_source", ["current", "startup"])
def test_safe_return_rejects_non_finite_state_before_sending(
    rebot_rs,
    monkeypatch,
    bad_value,
    invalid_source,
):
    monkeypatch.setattr(rebot_rs, "precise_sleep", lambda _: None)
    command = np.array([0.0, 20.0, -30.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float32)
    current = _physical_state(rebot_rs, command)
    startup = current.copy()
    if invalid_source == "current":
        current[0] = bad_value
    else:
        startup[0] = bad_value
    robot = FakeRobot(rebot_rs, current)

    with pytest.raises(rebot_rs.UnsafeReturnStateError, match="Unsafe return state"):
        rebot_rs.return_to_start_pose(
            robot,
            startup,
            control_hz=30.0,
            max_arm_speed=20.0,
            max_gripper_speed=8.0,
            hold_s=0.0,
        )

    assert robot.sent_actions == []
    assert robot.disable_torque_calls == 1


@pytest.mark.parametrize("invalid_source", ["current", "startup"])
def test_safe_return_rejects_implausible_state_before_sending(
    rebot_rs,
    monkeypatch,
    invalid_source,
):
    monkeypatch.setattr(rebot_rs, "precise_sleep", lambda _: None)
    command = np.array([0.0, 20.0, -30.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float32)
    current = _physical_state(rebot_rs, command)
    startup = current.copy()
    if invalid_source == "current":
        current[0] = 999.0
    else:
        startup[0] = 999.0
    robot = FakeRobot(rebot_rs, current)

    with pytest.raises(rebot_rs.UnsafeReturnStateError, match="outside B601-RS command limits"):
        rebot_rs.return_to_start_pose(
            robot,
            startup,
            control_hz=30.0,
            max_arm_speed=20.0,
            max_gripper_speed=8.0,
            hold_s=0.0,
        )

    assert robot.sent_actions == []
    assert robot.disable_torque_calls == 1


def test_safe_return_valid_state_reaches_startup_without_disabling_torque(
    rebot_rs,
    monkeypatch,
):
    monkeypatch.setattr(rebot_rs, "precise_sleep", lambda _: None)
    current_command = np.array([0.0, 20.0, -30.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float32)
    startup_command = np.array([5.0, 25.0, -35.0, 5.0, 5.0, 5.0, 12.0], dtype=np.float32)
    robot = FakeRobot(rebot_rs, _physical_state(rebot_rs, current_command))

    rebot_rs.return_to_start_pose(
        robot,
        _physical_state(rebot_rs, startup_command),
        control_hz=2.0,
        max_arm_speed=1000.0,
        max_gripper_speed=1000.0,
        hold_s=0.0,
    )

    assert robot.sent_actions
    np.testing.assert_allclose(
        [robot.sent_actions[-1][key] for key in rebot_rs.JOINT_KEYS],
        startup_command,
    )
    assert robot.disable_torque_calls == 0
