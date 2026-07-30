#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Closed-loop GR00T N1.7 evaluation client for the Seeed reBot B601-RS arm.

Data flow:

    B601-RS joints/cameras
        -> build a GR00T observation
        -> send it to the GPU policy server over ZeroMQ
        -> receive a 16-step action chunk
        -> time-align and blend overlapping predictions
        -> execute 8 aligned steps sequentially at 30 Hz
        -> infer the next chunk asynchronously during the last 4 steps
        -> apply delta clamping and EMA to the blended arm targets
        -> generate a velocity- and acceleration-continuous trajectory
        -> call LeRobot ``robot.send_action()``

Important:

1. The server returns denormalized 7-D actions in dataset command space.
2. Do not flip elbow/wrist signs or multiply the gripper by 6 in this file.
   ``SeeedB601RSFollower.send_action()`` applies ``joint_directions``:
   ``[1, 1, -1, -1, -1, 1, 6]``.
3. Preview mode performs inference without sending actions. Pass ``--execute`` to
   control the real robot.
4. Exceptions and Ctrl+C always reach ``finally``, which requests torque disable
   and releases the hardware.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import sys
import time
from typing import Any

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot_robot_seeed_b601 import SeeedB601RSFollower, SeeedB601RSFollowerConfig
import msgpack
import msgpack_numpy as msgpack_numpy
import numpy as np
import zmq


# This order must exactly match the observation.state/action names in dataset info.json.
# It is also used to concatenate single_arm(6) and gripper(1) into a 7-D action.
JOINT_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_yaw.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
CAMERA_KEYS = ("front", "side")  # Must match the checkpoint modality configuration.

# The driver reports physical motor angles, while dataset actions are command angles.
# Divide physical feedback by these direction/scale factors to recover command space.
# This array must match SeeedB601RSFollowerConfig.joint_directions exactly.
JOINT_DIRECTIONS = np.array([1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 6.0], dtype=np.float32)

# Hard limits in dataset command space.
#
# These are command limits before joint_directions, not final motor limits.
# For example, elbow_flex direction=-1 maps physical [0, 200] to command [-200, 0],
# and gripper direction=6 maps physical [0, 270] to command [0, 45].
# The driver applies the physical limits again as a second safety boundary.
COMMAND_LOWER = np.array([-145.0, 0.0, -200.0, -90.0, -90.0, -90.0, 0.0], dtype=np.float32)
COMMAND_UPPER = np.array([145.0, 170.0, 0.0, 80.0, 90.0, 90.0, 45.0], dtype=np.float32)

# Fixed control parameters validated on the real robot.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_FOURCC = "YUYV"
POLICY_TIMEOUT_MS = 120_000
# Consume 8 actions per chunk. Capture a new observation at step 4 and infer the
# next chunk in the background so synchronous inference does not stall RS MIT control.
ACTION_HORIZON = 8
POLICY_LATENCY_STEPS = 4
CONTROL_HZ = 30.0
# First stage: conservative delta clamping and EMA for the high-stiffness RS MIT controller.
ACTION_SMOOTHING_ALPHA = 0.5
ACTION_SMOOTHING_MAX_DELTA = 3.0
# Blend predictions for the same control timestamp across adjacent chunks.
# Old predictions receive at most 50% weight and decay faster as disagreement grows.
CHUNK_BLEND_MAX_WEIGHT = 0.5
CHUNK_BLEND_ARM_DISAGREEMENT_SCALE = 8.0
# Second stage: per-joint velocity and acceleration limits at 30 Hz.
# Units are command units/frame and command units/frame²:
#
# - shoulder_lift/elbow/wrist_flex: 1.0°/frame (30°/s);
# - other arm joints: 1.2°/frame (36°/s);
# - velocity may change by at most 0.10-0.12° per frame.
TRAJECTORY_MAX_STEP = np.array([1.2, 1.0, 1.0, 1.0, 1.2, 1.2], dtype=np.float32)
TRAJECTORY_MAX_STEP_CHANGE = np.array(
    [0.12, 0.10, 0.10, 0.10, 0.12, 0.12],
    dtype=np.float32,
)
# Keep driver max_relative_target disabled because it depends on delayed CAN feedback.
# Firmware limits, MIT gains, and COMMAND_LOWER/UPPER remain active.
RETURN_CONTROL_HZ = 30.0
RETURN_ARM_SPEED = 20.0
RETURN_GRIPPER_SPEED = 8.0
RETURN_HOLD_S = 2.0


class PolicyClient:
    """Lightweight GR00T ZeroMQ client.

    This client intentionally avoids importing ``gr00t.policy.PolicyClient`` so the
    robot-side LeRobot environment only needs msgpack and pyzmq, not the complete
    GPU inference dependency stack.

    The wire protocol matches GR00T ``PolicyServer``:
    - request: {"endpoint": ..., "data": ...}
    - get_action response: [action_dict, info_dict]
    - NumPy arrays are encoded with msgpack-numpy
    """

    def __init__(self, host: str, port: int, timeout_ms: int) -> None:
        self.endpoint = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket: zmq.Socket | None = None
        self._open_socket()

    def _open_socket(self) -> None:
        """Create a REQ socket or recreate it after a timeout.

        A timed-out ZeroMQ REQ socket remains in the receive state, so it must be
        closed and recreated before another request can be sent.
        """
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.endpoint)

    @staticmethod
    def _pack(value: Any) -> bytes:
        return msgpack.packb(value, default=msgpack_numpy.encode, use_bin_type=True)

    @staticmethod
    def _unpack(value: bytes) -> Any:
        return msgpack.unpackb(value, object_hook=msgpack_numpy.decode, raw=False)

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        """Call a server endpoint and normalize timeout/server error handling."""
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        assert self.socket is not None
        try:
            self.socket.send(self._pack(request))
            response = self._unpack(self.socket.recv())
        except zmq.error.Again as exc:
            self._open_socket()
            raise TimeoutError(
                f"Policy server {self.endpoint} did not respond within {self.timeout_ms} ms"
            ) from exc
        if response == "ERROR":
            raise RuntimeError("Policy server returned an unspecified error")
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Policy server error: {response['error']}")
        return response

    def ping(self) -> None:
        """Verify that the policy server and ZeroMQ protocol are available."""
        self.call("ping")

    def get_action(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Send a GR00T observation and return its action chunk and metadata."""
        response = self.call(
            "get_action",
            {"observation": observation, "options": None},
        )
        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise TypeError(f"Unexpected policy response: {type(response).__name__}")
        return response[0], response[1]

    def close(self) -> None:
        """Release the socket and context without blocking at process exit."""
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.context.term()


class AsyncPolicyWorker:
    """Compute the next action chunk in a background thread.

    ZeroMQ sockets cannot cross thread boundaries, so the worker creates and owns
    its own PolicyClient. The control thread only submits observations and reads Futures.
    """

    def __init__(self, host: str, port: int, timeout_ms: int) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gr00t-policy")
        self.client: PolicyClient | None = None

    def _infer(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self.client is None:
            self.client = PolicyClient(self.host, self.port, self.timeout_ms)
        return self.client.get_action(observation)

    def submit(
        self, observation: dict[str, Any]
    ) -> Future[tuple[dict[str, np.ndarray], dict[str, Any]]]:
        return self.executor.submit(self._infer, observation)

    def close(self) -> None:
        """Finish any in-flight request and close the socket in its owning thread."""

        def close_in_worker() -> None:
            if self.client is not None:
                self.client.close()
                self.client = None

        self.executor.submit(close_in_worker).result()
        self.executor.shutdown(wait=True, cancel_futures=True)


def parse_camera_source(value: str) -> int | Path:
    """Parse a numeric camera index or return a /dev/videoX/file path."""
    return int(value) if value.isdecimal() else Path(value)


def build_policy_observation(robot_observation: dict[str, Any], instruction: str) -> dict[str, Any]:
    """Convert a raw LeRobot observation to the GR00T Policy API format.

    LeRobot provides:
        joint_name.pos -> float
        front/side     -> (H, W, 3) uint8 RGB

    GR00T expects:
        video.front/side -> (B=1, T=1, H, W, 3)
        state.single_arm -> (B=1, T=1, 6)
        state.gripper    -> (B=1, T=1, 1)
        language         -> [[str]]

    This model only uses the current frame, so T=1.
    """
    # Validate fields before the network request to produce a useful local error.
    missing = [key for key in (*JOINT_KEYS, *CAMERA_KEYS) if key not in robot_observation]
    if missing:
        raise KeyError(f"Robot observation is missing keys: {missing}")

    # Preserve the dataset joint order. Driver feedback is expressed in degrees.
    state = np.asarray([robot_observation[key] for key in JOINT_KEYS], dtype=np.float32)
    if not np.all(np.isfinite(state)):
        raise ValueError(f"Robot returned a non-finite state: {state}")
    if np.all(np.abs(state) < 1e-6):
        raise RuntimeError(
            "All seven joint readings are zero; no valid motor feedback was received"
        )

    # The Policy API requires contiguous RGB uint8 arrays; [None, None] adds B and T.
    videos: dict[str, np.ndarray] = {}
    for key in CAMERA_KEYS:
        frame = np.asarray(robot_observation[key])
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Camera {key!r} returned invalid shape {frame.shape}; expected HxWx3")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        videos[key] = np.ascontiguousarray(frame[None, None, ...])

    return {
        "video": videos,
        "state": {
            "single_arm": state[None, None, :6],
            "gripper": state[None, None, 6:7],
        },
        "language": {
            "annotation.human.task_description": [[instruction]],
        },
    }


def decode_action_chunk(action_chunk: dict[str, Any]) -> np.ndarray:
    """Validate, concatenate, and safely clamp a server action chunk.

    The server returns:
        single_arm: (1, T, 6)
        gripper:    (1, T, 1)

    The result has shape (T, 7) in dataset command space. ``joint_directions``
    and the gripper ×6 scaling are applied only once inside the B601 driver.
    """
    # Missing modalities must fail closed to prevent joint misalignment.
    missing = {"single_arm", "gripper"} - set(action_chunk)
    if missing:
        raise KeyError(f"Policy action is missing modalities: {sorted(missing)}")

    arm = np.asarray(action_chunk["single_arm"], dtype=np.float32)
    gripper = np.asarray(action_chunk["gripper"], dtype=np.float32)
    if arm.ndim != 3 or arm.shape[0] != 1 or arm.shape[2] != 6:
        raise ValueError(f"single_arm action must have shape (1, T, 6), got {arm.shape}")
    if gripper.ndim != 3 or gripper.shape[:2] != arm.shape[:2] or gripper.shape[2] != 1:
        raise ValueError(f"gripper action must have shape (1, T, 1), got {gripper.shape}")

    # Remove the batch dimension and concatenate [6 arm joints, 1 gripper].
    actions = np.concatenate((arm[0], gripper[0]), axis=1)
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy returned NaN or infinity; refusing to control the robot")

    # The server already denormalizes actions. The client only enforces hardware
    # command limits and does not depend on training-dataset statistics.
    clipped = np.clip(actions, COMMAND_LOWER, COMMAND_UPPER)
    if not np.array_equal(actions, clipped):
        clipped_by_joint = np.count_nonzero(actions != clipped, axis=0)
        count = int(clipped_by_joint.sum())
        details = ", ".join(
            f"{key}={int(joint_count)}"
            for key, joint_count in zip(JOINT_KEYS, clipped_by_joint, strict=True)
            if joint_count
        )
        print(f"WARNING: clipped {count} predicted values to B601-RS command limits ({details})")
    return clipped


def action_dict(action: np.ndarray) -> dict[str, float]:
    """Convert a 7-D NumPy action to a LeRobot ``send_action()`` dictionary."""
    return {key: float(value) for key, value in zip(JOINT_KEYS, action, strict=True)}


def physical_state_to_command_state(physical_state: np.ndarray) -> np.ndarray:
    """Convert physical motor angles to dataset action command space.

    ``get_observation()`` reports physical angles, while ``send_action()`` applies
    ``joint_directions`` to dataset commands. For example:

    - an elbow physical angle of +30° corresponds to command -30°;
    - a gripper physical angle of +60° corresponds to command +10.

    Trajectory limiting must compare positions in the same coordinate system.
    """
    return np.asarray(physical_state, dtype=np.float32) / JOINT_DIRECTIONS


def smooth_action_segment(
    requested_segment: np.ndarray,
    previous_filtered_action: np.ndarray,
    *,
    arm_alpha: float = ACTION_SMOOTHING_ALPHA,
    max_delta: float = ACTION_SMOOTHING_MAX_DELTA,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply delta clamping and EMA to six arm joints; pass the gripper through."""
    requested_segment = np.asarray(requested_segment, dtype=np.float32)
    previous = np.asarray(previous_filtered_action, dtype=np.float32).copy()
    if requested_segment.ndim != 2 or requested_segment.shape[1] != len(JOINT_KEYS):
        raise ValueError(f"requested_segment must have shape (T, 7), got {requested_segment.shape}")
    if previous.shape != (len(JOINT_KEYS),):
        raise ValueError(f"previous_filtered_action must have shape (7,), got {previous.shape}")
    if not 0.0 < arm_alpha <= 1.0:
        raise ValueError(f"arm_alpha must be in (0, 1], got {arm_alpha}")
    if max_delta <= 0.0:
        raise ValueError(f"max_delta must be positive, got {max_delta}")

    smoothed = np.empty_like(requested_segment)
    for index, requested in enumerate(requested_segment):
        clamped_target = previous + np.clip(
            requested - previous,
            -max_delta,
            max_delta,
        )
        previous = arm_alpha * clamped_target + (1.0 - arm_alpha) * previous
        previous[6] = requested[6]
        smoothed[index] = previous
    return smoothed, previous.copy()


def limit_action_trajectory(
    filtered_segment: np.ndarray,
    previous_action: np.ndarray,
    previous_step: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert filtered targets into a velocity- and acceleration-continuous trajectory.

    EMA can still reverse velocity abruptly between adjacent chunks. The RS arm's
    high-stiffness MIT control exposes such discontinuities as vibration and impact.

    This function tracks the previous command velocity in command units/frame and limits:

    1. maximum per-frame joint displacement;
    2. change in displacement between frames (discrete acceleration);
    3. approach speed based on remaining braking distance.

    These limits apply only to the six arm joints; the gripper target passes through.

    The limiter depends only on sent commands, not delayed CAN feedback.
    """
    targets = np.asarray(filtered_segment, dtype=np.float32)
    position = np.asarray(previous_action, dtype=np.float32).copy()
    step = np.asarray(previous_step, dtype=np.float32).copy()
    expected_shape = (len(JOINT_KEYS),)
    if targets.ndim != 2 or targets.shape[1] != len(JOINT_KEYS):
        raise ValueError(f"filtered_segment must have shape (T, 7), got {targets.shape}")
    if position.shape != expected_shape:
        raise ValueError(f"previous_action must have shape (7,), got {position.shape}")
    if step.shape != expected_shape:
        raise ValueError(f"previous_step must have shape (7,), got {step.shape}")
    if not np.all(np.isfinite(targets)):
        raise ValueError("filtered_segment contains NaN or infinity")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(step)):
        raise ValueError("trajectory state contains NaN or infinity")
    commands = np.empty_like(targets)

    for index, target in enumerate(targets):
        error = target[:6] - position[:6]

        # sqrt(2*a*d) is the maximum speed that can stop within the remaining distance.
        # Proportional slowdown prevents repeated overshoot around small errors.
        braking_step = np.sqrt(2.0 * TRAJECTORY_MAX_STEP_CHANGE * np.abs(error))
        proportional_step = 0.35 * np.abs(error)
        desired_step = np.sign(error) * np.minimum(
            TRAJECTORY_MAX_STEP,
            np.minimum(braking_step, proportional_step),
        )

        step[:6] += np.clip(
            desired_step - step[:6],
            -TRAJECTORY_MAX_STEP_CHANGE,
            TRAJECTORY_MAX_STEP_CHANGE,
        )
        clipped = position.copy()
        clipped[:6] = np.clip(
            position[:6] + step[:6],
            COMMAND_LOWER[:6],
            COMMAND_UPPER[:6],
        )
        clipped[6] = np.clip(target[6], COMMAND_LOWER[6], COMMAND_UPPER[6])
        step[:6] = clipped[:6] - position[:6]
        step[6] = 0.0
        position = clipped
        commands[index] = position

    return commands, position.copy(), step.copy()


def blend_overlapping_actions(
    new_segment: np.ndarray,
    *,
    execution_start_step: int,
    previous_chunk: np.ndarray | None,
    previous_observation_step: int | None,
    max_old_weight: float = CHUNK_BLEND_MAX_WEIGHT,
    arm_disagreement_scale: float = CHUNK_BLEND_ARM_DISAGREEMENT_SCALE,
) -> np.ndarray:
    """Blend overlapping action chunks by absolute control time.

    Each GR00T chunk treats its observation frame as step 0. In asynchronous control,
    the previous chunk tail and the new chunk middle often describe the same future
    control frames. The old chunk is aligned using
    ``execution_start_step - previous_observation_step`` and adaptively blended:

    - old weight is highest at the overlap boundary and decays linearly to zero;
    - larger disagreement reduces the old weight;
    - the gripper always uses the new prediction.

    This reduces diffusion-induced chunk-boundary jitter without treating stale visual
    information as a hard constraint.
    """
    new_segment = np.asarray(new_segment, dtype=np.float32)
    if new_segment.ndim != 2 or new_segment.shape[1] != len(JOINT_KEYS):
        raise ValueError(f"new_segment must have shape (T, 7), got {new_segment.shape}")
    if not 0.0 <= max_old_weight < 1.0:
        raise ValueError(f"max_old_weight must be in [0, 1), got {max_old_weight}")
    if arm_disagreement_scale <= 0.0:
        raise ValueError("chunk disagreement scale must be positive")
    if previous_chunk is None or previous_observation_step is None or max_old_weight == 0.0:
        return new_segment.copy()

    previous_chunk = np.asarray(previous_chunk, dtype=np.float32)
    if previous_chunk.ndim != 2 or previous_chunk.shape[1] != len(JOINT_KEYS):
        raise ValueError(f"previous_chunk must have shape (T, 7), got {previous_chunk.shape}")

    previous_start = execution_start_step - previous_observation_step
    if previous_start < 0 or previous_start >= len(previous_chunk):
        return new_segment.copy()

    overlap = min(len(new_segment), len(previous_chunk) - previous_start)
    if overlap <= 0:
        return new_segment.copy()

    old = previous_chunk[previous_start : previous_start + overlap]
    new = new_segment[:overlap]

    # Smooth even a one-frame overlap. For longer overlaps, phase out the old chunk.
    age_weight = (
        np.array([max_old_weight], dtype=np.float32)
        if overlap == 1
        else max_old_weight * np.linspace(1.0, 0.0, overlap, dtype=np.float32)
    )
    agreement_weight = np.exp(-np.abs(new[:, :6] - old[:, :6]) / arm_disagreement_scale)
    old_weight = age_weight[:, None] * agreement_weight

    blended = new_segment.copy()
    blended[:overlap, :6] = old_weight * old[:, :6] + (1.0 - old_weight) * new[:, :6]
    return blended


def observation_state(robot_observation: dict[str, Any]) -> np.ndarray:
    """Extract the 7-D physical state from a LeRobot observation in fixed joint order."""
    return np.asarray([robot_observation[key] for key in JOINT_KEYS], dtype=np.float32)


def return_to_start_pose(
    robot: SeeedB601RSFollower,
    start_physical_state: np.ndarray,
    *,
    control_hz: float,
    max_arm_speed: float,
    max_gripper_speed: float,
    hold_s: float,
) -> None:
    """Return smoothly to the startup pose while keeping torque enabled.

    The startup pose is captured from live feedback before the first GR00T inference.
    A smoothstep curve provides gradual acceleration and deceleration:

        s(u) = 3u² - 2u³,  u ∈ [0, 1]

    The maximum smoothstep derivative is 1.5, so return duration includes the same
    factor to keep peak velocity below the configured limits.

    This is a joint-space return, not Cartesian path planning. Keep the space between
    the current and startup poses clear.
    """
    if not robot.is_connected:
        raise RuntimeError("Cannot return home because the robot is not connected")

    # Start the return from current CAN feedback, not the last model command.
    current_observation = robot.get_observation()
    current_command = physical_state_to_command_state(observation_state(current_observation))
    target_command = physical_state_to_command_state(start_physical_state)
    target_command = np.clip(target_command, COMMAND_LOWER, COMMAND_UPPER)

    delta = target_command - current_command
    # Gripper command space is multiplied by 6 in the driver, so use a separate limit.
    duration_arm = float(np.max(1.5 * np.abs(delta[:6]) / max_arm_speed))
    duration_gripper = float(1.5 * abs(delta[6]) / max_gripper_speed)
    duration_s = max(duration_arm, duration_gripper, 1.0)
    steps = max(2, int(np.ceil(duration_s * control_hz)))
    period = 1.0 / control_hz

    print(
        f"\nReturning safely to the startup pose: {steps} frames, "
        f"approximately {duration_s:.2f} seconds."
    )
    for index in range(1, steps + 1):
        tick = time.monotonic()
        u = index / steps
        blend = 3.0 * u * u - 2.0 * u * u * u
        command = current_command + blend * delta
        robot.send_action(action_dict(command))
        precise_sleep(max(period - (time.monotonic() - tick), 0.0))

    # Hold the target briefly so slower joints can settle before torque is disabled.
    hold_steps = int(np.ceil(hold_s * control_hz))
    for _ in range(hold_steps):
        tick = time.monotonic()
        robot.send_action(action_dict(target_command))
        precise_sleep(max(period - (time.monotonic() - tick), 0.0))

    print("Safe return completed.")


def make_robot(args: argparse.Namespace) -> SeeedB601RSFollower:
    """Create the B601-RS and two OpenCV cameras from command-line arguments.

    ``max_relative_target=None`` matches the working ``lerobot-replay`` configuration.
    Normal GR00T execution uses the client trajectory limiter; safe return uses its
    separate 20°/s arm speed.
    """
    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=parse_camera_source(args.front_camera),
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            # Match the dataset's 30 FPS; action frequency is controlled separately.
            fps=CAMERA_FPS,
            fourcc=CAMERA_FOURCC,
        ),
        "side": OpenCVCameraConfig(
            index_or_path=parse_camera_source(args.side_camera),
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            fourcc=CAMERA_FOURCC,
        ),
    }
    config = SeeedB601RSFollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        can_adapter="socketcan",
        cameras=cameras,
        # Match lerobot-replay: do not apply a driver limit based on stale CAN feedback.
        max_relative_target=None,
    )
    return SeeedB601RSFollower(config)


def safe_disconnect(robot: SeeedB601RSFollower) -> None:
    """Disable torque and release fully or partially connected hardware.

    Normally this calls the plugin's ``disconnect()``. If camera or CAN setup fails
    midway, ``robot.is_connected`` may be false even though the bus exists and motors
    may be enabled, so the fallback directly disables and closes every resource.
    """
    if robot.is_connected:
        try:
            robot.disconnect()
            print("Robot disconnected; torque disabled.")
            return
        except Exception as exc:
            print(f"WARNING: normal robot disconnect failed ({exc}); forcing torque off.")

    bus = getattr(robot, "bus", None)
    if bus is not None:
        try:
            bus.disable_all()
        except Exception as exc:
            print(f"WARNING: could not disable all motors: {exc}")
        for motor in getattr(robot, "motors", {}).values():
            try:
                motor.close()
            except Exception:
                pass
        try:
            bus.close()
        except Exception:
            pass
        robot.bus = None

    for camera in getattr(robot, "cameras", {}).values():
        if camera.is_connected:
            try:
                camera.disconnect()
            except Exception:
                pass
    print("Robot resources released; torque-disable was requested.")


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a GR00T N1.7 policy on a Seeed reBot B601-RS arm."
    )
    parser.add_argument("--robot-port", default="can0")
    parser.add_argument("--robot-id", default="follower1")
    parser.add_argument("--front-camera", default="/dev/video0")
    parser.add_argument("--side-camera", default="/dev/video6")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--instruction", default="Organize test tube")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=25.0,
        help="Maximum policy run time; default 25 seconds, 0 runs until Ctrl+C.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send commands. Without this flag, one prediction is printed only.",
    )
    args = parser.parse_args()
    if args.duration_s < 0:
        parser.error("--duration-s must be non-negative (0 means unlimited)")
    return args


def main() -> int:
    """Connect the server and robot, then run receding-horizon control."""
    args = parse_args()

    print(f"Control: {CONTROL_HZ:g} Hz, horizon={ACTION_HORIZON}")

    # Create policy clients first; this does not connect or enable the robot.
    client = PolicyClient(args.policy_host, args.policy_port, POLICY_TIMEOUT_MS)
    async_policy = AsyncPolicyWorker(args.policy_host, args.policy_port, POLICY_TIMEOUT_MS)
    robot: SeeedB601RSFollower | None = None
    startup_state: np.ndarray | None = None
    try:
        # Check the GPU server before connecting and enabling the real robot.
        print(f"Checking policy server at tcp://{args.policy_host}:{args.policy_port} ...")
        client.ping()
        print("Policy server is ready.")

        # connect() opens CAN and cameras, configures control mode, and enables motors.
        robot = make_robot(args)
        robot.connect()
        print("Robot and cameras connected.")

        # Discard initial frames while asynchronous cameras and CAN feedback warm up.
        #
        # Preview captures one torque-enabled observation to match training conditions,
        # then disables torque before the potentially slow inference request.
        observation: dict[str, Any] | None = None
        for _ in range(3):
            observation = robot.get_observation()
        assert observation is not None
        # Capture the startup pose before GR00T sends any action. Normal completion or
        # the first Ctrl+C returns smoothly to this pose.
        startup_state = observation_state(observation).copy()

        preview_observation = observation
        if not args.execute:
            robot.disable_torque()
            print(
                "PREVIEW MODE: captured a torque-enabled observation, then disabled "
                "motor torque; no policy action will be sent."
            )

        executed_actions = 0
        period = 1.0 / CONTROL_HZ
        target_action_count = (
            None if args.duration_s == 0 else int(round(args.duration_s * CONTROL_HZ))
        )
        first_chunk = True
        pending_future: Future[tuple[dict[str, np.ndarray], dict[str, Any]]] | None = None
        pending_observation_step: int | None = None
        # Convert physical RS feedback to dataset/send_action command space before
        # initializing the smoothing state.
        initial_command_state = physical_state_to_command_state(startup_state)
        previous_filtered_action = initial_command_state.copy()
        previous_sent_action = initial_command_state.copy()
        previous_sent_step = np.zeros(len(JOINT_KEYS), dtype=np.float32)
        previous_chunk_actions: np.ndarray | None = None
        previous_chunk_observation_step: int | None = None

        # Use an 8-frame receding horizon and prefetch the next chunk:
        #
        #   infer the first chunk synchronously and execute steps 0..7;
        #   capture an observation before step 4 and start background inference;
        #   execute steps 4..11 from the next chunk to compensate for elapsed time.
        #
        # Capture and takeover are four control periods apart, so the next chunk's
        # step 4 aligns with real control time.
        while target_action_count is None or executed_actions < target_action_count:
            if first_chunk:
                observation = preview_observation if not args.execute else robot.get_observation()
                policy_observation = build_policy_observation(observation, args.instruction)
                chunk_observation_step = executed_actions
                action_chunk, _ = client.get_action(policy_observation)
                first_chunk = False
            else:
                if pending_future is None or pending_observation_step is None:
                    raise RuntimeError("Next policy chunk was not prefetched")
                action_chunk, _ = pending_future.result()
                chunk_observation_step = pending_observation_step
                pending_future = None
                pending_observation_step = None

            actions = decode_action_chunk(action_chunk)
            # Derive latency compensation from absolute control frames instead of
            # hard-coding step 4.
            action_start = executed_actions - chunk_observation_step
            if action_start < 0:
                raise RuntimeError(
                    "Policy observation is newer than the current control step: "
                    f"observation_step={chunk_observation_step}, "
                    f"execution_step={executed_actions}"
                )
            if not args.execute:
                print("first action:", action_dict(actions[action_start]))
                print("Preview complete. Re-run with --execute to control the arm.")
                break

            required_actions = action_start + ACTION_HORIZON
            if len(actions) < required_actions:
                raise ValueError(
                    f"Policy returned only {len(actions)} steps, but "
                    f"{required_actions} are required for latency-aligned execution"
                )

            # Blend time-aligned arm predictions, then apply EMA and trajectory limits.
            # The gripper target passes through all three stages unchanged.
            raw_segment = actions[action_start : action_start + ACTION_HORIZON].copy()
            planned_segment = blend_overlapping_actions(
                raw_segment,
                execution_start_step=executed_actions,
                previous_chunk=previous_chunk_actions,
                previous_observation_step=previous_chunk_observation_step,
            )

            # Preserve the full current chunk for absolute-time alignment with the next one.
            previous_chunk_actions = actions
            previous_chunk_observation_step = chunk_observation_step

            if target_action_count is not None:
                remaining = target_action_count - executed_actions
                planned_segment = planned_segment[:remaining]

            # Truncate the last segment before advancing filter state so predictions
            # beyond the requested duration do not affect future state.
            filtered_segment, previous_filtered_action = smooth_action_segment(
                planned_segment,
                previous_filtered_action,
            )
            segment, previous_sent_action, previous_sent_step = limit_action_trajectory(
                filtered_segment,
                previous_sent_action,
                previous_sent_step,
            )
            need_next_chunk = (
                target_action_count is None or executed_actions + len(segment) < target_action_count
            )

            # Send actions at the dataset frequency. The driver still applies
            # joint_directions and physical limits.
            for local_step, action_to_send in enumerate(segment):
                tick = time.monotonic()

                # Capture before step 4; background inference overlaps steps 4..7.
                if need_next_chunk and local_step == POLICY_LATENCY_STEPS:
                    observation = robot.get_observation()
                    pending_observation_step = executed_actions
                    policy_observation = build_policy_observation(
                        observation,
                        args.instruction,
                    )
                    pending_future = async_policy.submit(policy_observation)

                robot.send_action(action_dict(action_to_send))
                executed_actions += 1
                # precise_sleep keeps the action stream close to 30 Hz.
                precise_sleep(max(period - (time.monotonic() - tick), 0.0))

            # Fail explicitly if prefetch was not submitted; never fall back silently
            # to synchronous inference and reintroduce periodic stalls.
            if need_next_chunk and pending_future is None:
                raise RuntimeError("Policy prefetch was not submitted during the action segment")

        # Return safely after the requested duration. Ctrl+C and exceptions use the
        # same return path below.
        if args.execute:
            assert startup_state is not None
            return_to_start_pose(
                robot,
                startup_state,
                control_hz=RETURN_CONTROL_HZ,
                max_arm_speed=RETURN_ARM_SPEED,
                max_gripper_speed=RETURN_GRIPPER_SPEED,
                hold_s=RETURN_HOLD_S,
            )

        print("\nEvaluation finished.")
        return 0
    except KeyboardInterrupt:
        print("\nCtrl+C received; no new policy actions will be requested.")
        # The first Ctrl+C returns smoothly. A second Ctrl+C interrupts the return
        # and proceeds directly to torque disable in finally.
        if args.execute and robot is not None and robot.is_connected and startup_state is not None:
            print("Returning to the startup pose. Press Ctrl+C again to disable torque immediately.")
            try:
                return_to_start_pose(
                    robot,
                    startup_state,
                    control_hz=RETURN_CONTROL_HZ,
                    max_arm_speed=RETURN_ARM_SPEED,
                    max_gripper_speed=RETURN_GRIPPER_SPEED,
                    hold_s=RETURN_HOLD_S,
                )
            except KeyboardInterrupt:
                print("\nSecond Ctrl+C received; aborting return and requesting torque disable.")
        return 130
    except Exception as execution_error:
        # If hardware is still available, return before disabling torque. Re-raise the
        # original execution error even if the return attempt also fails.
        print(f"\nEvaluation failed: {execution_error}")
        if args.execute and robot is not None and robot.is_connected and startup_state is not None:
            print("Robot is still connected; attempting a safe return first.")
            try:
                return_to_start_pose(
                    robot,
                    startup_state,
                    control_hz=RETURN_CONTROL_HZ,
                    max_arm_speed=RETURN_ARM_SPEED,
                    max_gripper_speed=RETURN_GRIPPER_SPEED,
                    hold_s=RETURN_HOLD_S,
                )
            except Exception as return_error:
                print(f"WARNING: safe return after failure also failed: {return_error}")
        raise
    finally:
        # Always release hardware before closing policy clients.
        try:
            if robot is not None:
                safe_disconnect(robot)
        finally:
            try:
                async_policy.close()
            finally:
                client.close()


if __name__ == "__main__":
    sys.exit(main())
