# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bimanual GR00T rollout that outputs continuous TOPPRA trajectories.

The verified action convention for ``parcel_4f_v5.7`` is:

* TCP state: ``[x, y, z, qw, qx, qy, qz]`` in the robot base frame.
* TCP action: base-frame translation and spatial ``xyz`` Euler rotation increments.
* Composition: ``p_next = p + dp`` and ``R_next = dR * R``.

In synchronous mode the robot holds the measured TCP poses while GR00T inference and TOPPRA
planning run. The completed trajectory is pre-sampled from time zero at the configured control
frequency, and every :meth:`GR00TAgent.act` call consumes exactly one sample.

In asynchronous mode the old trajectory keeps executing during policy inference and planning.
After policy inference, a future sample on the old command trajectory is reserved as the handoff.
Policy candidates are trimmed to that handoff, and the new TOPPRA trajectory starts from exactly
the reserved command pose and twist before the control thread installs it atomically.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import io
import logging
import math
import threading
import time
from typing import Any, Callable, Literal

import msgpack
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint
import zmq


LOGGER = logging.getLogger(__name__)


try:
    # These classes belong to the external real-robot application used by example.py.
    from . import Agent

    FRAMEWORK_AGENT_AVAILABLE = True
except ImportError:
    FRAMEWORK_AGENT_AVAILABLE = False

    class Agent:  # type: ignore[no-redef]
        """Standalone fallback when the external rollout framework is not installed."""


try:
    from gui.module import GUIModule

    FRAMEWORK_GUI_AVAILABLE = True
except ImportError:
    FRAMEWORK_GUI_AVAILABLE = False

    class GUIModule:  # type: ignore[no-redef]
        """No-op base that keeps the TOPPRA rollout usable without the external GUI."""

        def __init__(self, name: str, placement: str = "top") -> None:
            self.gui_name = name
            self.gui_placement = placement


FloatArray = NDArray[np.float64]
ActionDict = dict[str, NDArray[Any]]
ArmName = str
TrajectoryCallback = Callable[["BimanualContinuousTrajectory"], None]
DEFAULT_TASK = "move parcel onto conveyor belt one by one"


def _pack_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "O":
            raise TypeError("Object-dtype arrays are not allowed in policy requests")
        array = np.ascontiguousarray(obj)
        return {
            b"nd": True,
            b"type": array.dtype.str,
            b"kind": b"",
            b"shape": array.shape,
            b"data": array.tobytes(),
        }
    if isinstance(obj, np.generic):
        array = np.asarray(obj)
        return {b"nd": False, b"type": array.dtype.str, b"data": array.tobytes()}
    return obj


def _unpack_numpy(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    if b"nd" in obj:
        if obj.get(b"kind") == b"O":
            raise ValueError("Object-dtype arrays are not allowed in policy responses")
        dtype = np.dtype(obj[b"type"])
        if obj[b"nd"] is True:
            return np.ndarray(buffer=obj[b"data"], dtype=dtype, shape=obj[b"shape"])
        return np.frombuffer(obj[b"data"], dtype=dtype)[0]
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    if "__ModalityConfig__" in obj:
        return obj["as_json"]
    return obj


class GR00TPolicyClient:
    """Minimal ZMQ policy client used only by the inference worker thread."""

    def __init__(self, host: str, port: int, timeout_ms: int = 15000) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.context = zmq.Context()
        self.socket: zmq.Socket[Any] | None = None
        self._init_socket()

    def _init_socket(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        requires_input: bool = True,
    ) -> Any:
        if self.socket is None:
            self._init_socket()
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        try:
            assert self.socket is not None
            self.socket.send(msgpack.packb(request, default=_pack_numpy, use_bin_type=True))
            message = self.socket.recv()
        except zmq.error.Again as exc:
            self._init_socket()
            raise TimeoutError(f"GR00T policy server timed out calling {endpoint!r}") from exc
        if message == b"ERROR":
            raise RuntimeError("GR00T policy server returned ERROR")
        response = msgpack.unpackb(message, object_hook=_unpack_numpy, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T policy server error: {response['error']}")
        return response

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        if not isinstance(response, (tuple, list)) or len(response) != 2:
            raise TypeError("GR00T get_action must return (action, info)")
        return response[0], response[1]

    def get_modality_config(self) -> dict[str, Any]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def ping(self) -> bool:
        self.call_endpoint("ping", requires_input=False)
        return True

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.context.term()


@dataclass(frozen=True)
class CartesianLimits:
    """Component-wise TCP limits in metres/radians and seconds."""

    max_linear_velocity: tuple[float, float, float]
    max_angular_velocity: tuple[float, float, float]
    max_linear_acceleration: tuple[float, float, float]
    max_angular_acceleration: tuple[float, float, float]
    safety_margin: float = 0.9

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                *self.max_linear_velocity,
                *self.max_angular_velocity,
                *self.max_linear_acceleration,
                *self.max_angular_acceleration,
            ),
            dtype=np.float64,
        )
        if values.shape != (12,) or np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("All Cartesian velocity and acceleration limits must be positive")
        if not 0 < self.safety_margin <= 1:
            raise ValueError("safety_margin must be in (0, 1]")

    @property
    def velocity(self) -> FloatArray:
        return self.safety_margin * np.asarray(
            (*self.max_linear_velocity, *self.max_angular_velocity), dtype=np.float64
        )

    @property
    def acceleration(self) -> FloatArray:
        return self.safety_margin * np.asarray(
            (*self.max_linear_acceleration, *self.max_angular_acceleration),
            dtype=np.float64,
        )


DEFAULT_CARTESIAN_LIMITS = CartesianLimits(
    max_linear_velocity=(1.0, 1.0, 1.0),
    max_angular_velocity=(3.0, 3.0, 3.0),
    max_linear_acceleration=(5.0, 5.0, 5.0),
    max_angular_acceleration=(15.0, 15.0, 15.0),
)


def _coerce_cartesian_limits(
    value: CartesianLimits | Mapping[str, Any],
    name: str,
) -> CartesianLimits:
    if isinstance(value, CartesianLimits):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be CartesianLimits or a JSON-style mapping")

    vector_fields = (
        "max_linear_velocity",
        "max_angular_velocity",
        "max_linear_acceleration",
        "max_angular_acceleration",
    )
    missing = [field_name for field_name in vector_fields if field_name not in value]
    if missing:
        raise ValueError(f"{name} is missing fields: {missing}")
    allowed = {*vector_fields, "safety_margin"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{name} has unknown fields: {unknown}")

    try:
        vectors = {
            field_name: tuple(float(component) for component in value[field_name])
            for field_name in vector_fields
        }
        return CartesianLimits(
            max_linear_velocity=vectors["max_linear_velocity"],
            max_angular_velocity=vectors["max_angular_velocity"],
            max_linear_acceleration=vectors["max_linear_acceleration"],
            max_angular_acceleration=vectors["max_angular_acceleration"],
            safety_margin=float(value.get("safety_margin", 0.9)),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc


@dataclass(frozen=True)
class BimanualToppraRolloutConfig:
    left_limits: CartesianLimits
    right_limits: CartesianLimits
    left_pose_state_key: str = "observation.state.left_tcp"
    right_pose_state_key: str = "observation.state.right_tcp"
    left_velocity_state_key: str | None = None
    right_velocity_state_key: str | None = None
    left_action_key: str = "action.left_delta_tcp"
    right_action_key: str = "action.right_delta_tcp"
    left_pinch_key: str = "action.left_pinch"
    right_pinch_key: str = "action.right_pinch"
    left_pose_output_key: str = "action.left_tcp"
    right_pose_output_key: str = "action.right_tcp"
    left_velocity_output_key: str = "action.left_tcp_velocity"
    right_velocity_output_key: str = "action.right_tcp_velocity"
    left_acceleration_output_key: str = "action.left_tcp_acceleration"
    right_acceleration_output_key: str = "action.right_tcp_acceleration"
    time_output_key: str = "trajectory.time_from_start"
    duration_output_key: str = "trajectory.duration"
    sequence_output_key: str = "trajectory.sequence_id"
    euler_order: str = "xyz"
    degrees: bool = False
    policy_frequency: float = 30.0
    control_frequency: float = 250.0
    inference_mode: Literal["async", "sync"] = "sync"
    refill_lead_s: float = 0.0
    max_latency_s: float = 0.12
    scheduling_margin_s: float = 0.05
    handoff_margin_s: float = 0.05
    open_loop_horizon: int = 8
    act_timeout_s: float = 20.0
    ignore_first_latency: bool = True
    velocity_filter: float = 0.35
    include_velocity_in_policy_observation: bool = False
    gridpoints_per_segment: int = 16
    min_gridpoints: int = 100
    terminal_path_velocity_min: float = 0.0
    terminal_path_velocity_max: float = 0.0
    async_terminal_path_velocity: float | None = 1.0
    tts_samples: int = 1
    tts_waypoint_count: int | None = 5
    allow_timing_fallback: bool = False
    task_default: str = DEFAULT_TASK
    scaling: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.policy_frequency <= 0 or self.control_frequency <= 0 or self.act_timeout_s <= 0:
            raise ValueError(
                "policy_frequency, control_frequency, and act_timeout_s must be positive"
            )
        if self.inference_mode not in ("async", "sync"):
            raise ValueError("inference_mode must be 'async' or 'sync'")
        if self.refill_lead_s < 0:
            raise ValueError("refill_lead_s must be non-negative")
        if self.max_latency_s < 0:
            raise ValueError("max_latency_s must be non-negative")
        if self.scheduling_margin_s < 0 or self.handoff_margin_s < 0:
            raise ValueError("scheduling_margin_s and handoff_margin_s must be non-negative")
        if self.open_loop_horizon < 1:
            raise ValueError("open_loop_horizon must be at least one")
        if not 0 < self.velocity_filter <= 1:
            raise ValueError("velocity_filter must be in (0, 1]")
        if self.gridpoints_per_segment < 2 or self.min_gridpoints < 2:
            raise ValueError("TOPPRA gridpoint counts must be at least two")
        if not 0 <= self.terminal_path_velocity_min <= self.terminal_path_velocity_max:
            raise ValueError("Invalid terminal path velocity interval")
        if self.async_terminal_path_velocity is not None and self.async_terminal_path_velocity < 0:
            raise ValueError("async_terminal_path_velocity must be non-negative or None")
        if self.tts_samples < 1:
            raise ValueError("tts_samples must be at least one")
        if self.tts_waypoint_count is not None and self.tts_waypoint_count < 1:
            raise ValueError("tts_waypoint_count must be at least one when configured")
        if len(set((*self.pose_state_keys.values(), *self.action_keys.values()))) != 4:
            raise ValueError("Left and right state/action keys must be distinct")
        try:
            Rotation.from_euler(self.euler_order, [0.0, 0.0, 0.0], degrees=self.degrees)
        except ValueError as exc:
            raise ValueError(f"Invalid scipy Euler order {self.euler_order!r}") from exc

    @property
    def policy_dt(self) -> float:
        return 1.0 / self.policy_frequency

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_frequency

    def _per_arm(self, suffix: str) -> dict[ArmName, Any]:
        return {arm: getattr(self, f"{arm}_{suffix}") for arm in ("left", "right")}

    @property
    def pose_state_keys(self) -> dict[ArmName, str]:
        return self._per_arm("pose_state_key")

    @property
    def velocity_state_keys(self) -> dict[ArmName, str | None]:
        return self._per_arm("velocity_state_key")

    @property
    def action_keys(self) -> dict[ArmName, str]:
        return self._per_arm("action_key")

    @property
    def pose_output_keys(self) -> dict[ArmName, str]:
        return self._per_arm("pose_output_key")

    @property
    def velocity_output_keys(self) -> dict[ArmName, str]:
        return self._per_arm("velocity_output_key")

    @property
    def acceleration_output_keys(self) -> dict[ArmName, str]:
        return self._per_arm("acceleration_output_key")

    @property
    def limits(self) -> dict[ArmName, CartesianLimits]:
        return self._per_arm("limits")


@dataclass(frozen=True)
class BimanualContinuousTrajectory:
    """Continuous dual-arm trajectory returned by the rollout layer.

    Arrays held by this object define the geometric spline and TOPPRA time law. Call ``sample`` at
    times selected by the downstream controller. Do not mutate the stored arrays.
    """

    config: BimanualToppraRolloutConfig
    sequence_id: int
    path: Any
    start_rotations: dict[ArmName, Rotation]
    gridpoints: FloatArray
    path_speeds: FloatArray
    source_times: FloatArray
    time_scale: float
    waypoint_s: FloatArray
    auxiliary_actions: dict[str, FloatArray]
    inference_started_at_s: float
    created_at_monotonic_s: float
    used_toppra: bool
    fallback_reason: str | None = None
    tts_selected_index: int = 0
    tts_candidate_scores: tuple[float, ...] = ()
    initial_twist_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.time_scale <= 0 or not np.isfinite(self.time_scale):
            raise ValueError("time_scale must be finite and positive")
        if len(self.gridpoints) < 2 or len(self.path_speeds) != len(self.gridpoints):
            raise ValueError("TOPPRA gridpoints and path speeds must have the same nonzero length")
        if len(self.source_times) != len(self.gridpoints):
            raise ValueError("TOPPRA time grid must match the path grid")
        if np.any(np.diff(self.source_times) <= 0):
            raise ValueError("TOPPRA source times must be strictly increasing")

    @property
    def duration(self) -> float:
        """Dynamically retimed duration of this action chunk in seconds."""

        return float(self.source_times[-1] * self.time_scale)

    @property
    def waypoint_times(self) -> FloatArray:
        """TOPPRA arrival times for action waypoints, excluding the path start."""

        return (
            _times_at_path_positions(
                self.gridpoints,
                self.path_speeds,
                self.source_times,
                self.waypoint_s[1:],
            )
            * self.time_scale
        )

    @property
    def terminal_path_speed(self) -> float:
        return float(self.path_speeds[-1])

    def sample(
        self,
        time_from_start: float | FloatArray,
        *,
        clamp: bool = False,
    ) -> ActionDict:
        """Evaluate pose, spatial velocity, acceleration, and auxiliary actions at arbitrary times."""

        scalar = np.asarray(time_from_start).ndim == 0
        times = np.atleast_1d(np.asarray(time_from_start, dtype=np.float64))
        if times.ndim != 1 or np.any(~np.isfinite(times)):
            raise ValueError("time_from_start must be a finite scalar or one-dimensional array")
        if len(times) > 1 and np.any(np.diff(times) < 0):
            raise ValueError("Vector sample times must be nondecreasing")
        if clamp:
            times = np.clip(times, 0.0, self.duration)
        elif np.any(times < 0) or np.any(times > self.duration):
            raise ValueError(f"Sample time must lie in [0, {self.duration:.9f}]")
        result = _sample_continuous_trajectory(self, times)
        if scalar:
            return {key: np.asarray(value[0]).copy() for key, value in result.items()}
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "duration": self.duration,
            "inference_started_at_s": self.inference_started_at_s,
            "created_at_monotonic_s": self.created_at_monotonic_s,
            "pipeline_latency_s": self.created_at_monotonic_s - self.inference_started_at_s,
            "used_toppra": self.used_toppra,
            "fallback_reason": self.fallback_reason,
            "tts_selected_index": self.tts_selected_index,
            "tts_candidate_scores": self.tts_candidate_scores,
            "initial_twist_scale": self.initial_twist_scale,
            "terminal_path_speed": self.terminal_path_speed,
        }


class BimanualCartesianToppraPlanner:
    """Build and retime one synchronized, piecewise-cubic 12D Cartesian path."""

    ARMS = ("left", "right")

    def __init__(self, config: BimanualToppraRolloutConfig) -> None:
        self.config = config

    def plan(
        self,
        action_arrays: dict[str, FloatArray],
        current_pose: dict[ArmName, FloatArray],
        current_twist: dict[ArmName, FloatArray] | None = None,
        *,
        sequence_id: int = 0,
        inference_started_at_s: float | None = None,
        tts_selected_index: int = 0,
        tts_candidate_scores: tuple[float, ...] = (),
        relax_initial_twist: bool = False,
    ) -> BimanualContinuousTrajectory:
        poses = self._validated_poses(current_pose)
        twists = self._validated_twists(current_twist)
        pose_actions, _ = self._pose_actions(action_arrays)
        path, start_rotations, waypoint_s, initial_path_speed = self._build_path(
            poses, twists, pose_actions
        )

        used_toppra = True
        fallback_reason = None
        initial_twist_scale = 1.0
        try:
            gridpoints, path_speeds, initial_twist_scale = self._solve_toppra(
                path,
                waypoint_s,
                initial_path_speed,
                relax_initial_twist=relax_initial_twist,
            )
        except Exception as exc:
            if not self.config.allow_timing_fallback:
                raise RuntimeError(
                    f"TOPPRA could not parameterize the bimanual path: {type(exc).__name__}: {exc}"
                ) from exc
            used_toppra = False
            fallback_reason = f"{type(exc).__name__}: {exc}"
            gridpoints = waypoint_s.copy()
            path_speeds = np.ones_like(gridpoints)
            path_speeds[0] = initial_path_speed

        source_times = self._time_grid(gridpoints, path_speeds)
        pose_keys = set(self.config.action_keys.values())
        auxiliary = {
            key: np.asarray(values, dtype=np.float64).copy()
            for key, values in action_arrays.items()
            if key not in pose_keys
        }
        now = time.perf_counter()
        return BimanualContinuousTrajectory(
            config=self.config,
            sequence_id=sequence_id,
            path=path,
            start_rotations=start_rotations,
            gridpoints=gridpoints,
            path_speeds=path_speeds,
            source_times=source_times,
            time_scale=1.0,
            waypoint_s=waypoint_s,
            auxiliary_actions=auxiliary,
            inference_started_at_s=(
                now if inference_started_at_s is None else inference_started_at_s
            ),
            created_at_monotonic_s=now,
            used_toppra=used_toppra,
            fallback_reason=fallback_reason,
            tts_selected_index=tts_selected_index,
            tts_candidate_scores=tts_candidate_scores,
            initial_twist_scale=initial_twist_scale,
        )

    def select_tts_candidate(
        self,
        candidates: list[dict[str, FloatArray]],
        current_pose: dict[ArmName, FloatArray],
        current_twist: dict[ArmName, FloatArray],
    ) -> tuple[int, tuple[float, ...]]:
        if not candidates:
            raise ValueError("TTS requires at least one action candidate")
        poses = self._validated_poses(current_pose)
        twists = self._validated_twists(current_twist)
        scores: list[float] = []
        for candidate in candidates:
            try:
                pose_actions, horizon = self._pose_actions(candidate)
                waypoint_count = horizon
                if self.config.tts_waypoint_count is not None:
                    waypoint_count = min(waypoint_count, self.config.tts_waypoint_count)
                truncated = {arm: values[:waypoint_count] for arm, values in pose_actions.items()}
                path, _rotations, waypoint_s, _speed = self._build_path(poses, twists, truncated)
                count = max(
                    self.config.min_gridpoints,
                    self.config.gridpoints_per_segment * waypoint_count + 1,
                )
                sample_s = np.linspace(waypoint_s[0], waypoint_s[-1], count)
                curvature_squared = np.sum(
                    np.asarray(path(sample_s, 2), dtype=np.float64) ** 2,
                    axis=1,
                )
                integral = float(np.trapz(curvature_squared, sample_s))
                scores.append(float(waypoint_s[-1]) / max(integral, np.finfo(float).eps))
            except (KeyError, TypeError, ValueError):
                scores.append(float("-inf"))
        score_array = np.asarray(scores, dtype=np.float64)
        if not np.any(np.isfinite(score_array)):
            raise RuntimeError("All TTS candidates have invalid bimanual Cartesian paths")
        return int(np.argmax(score_array)), tuple(scores)

    def _build_path(
        self,
        current_pose: dict[ArmName, FloatArray],
        current_twist: dict[ArmName, FloatArray],
        pose_actions: dict[ArmName, FloatArray],
    ) -> tuple[Any, dict[ArmName, Rotation], FloatArray, float]:
        horizon = len(pose_actions["left"])
        start_rotations: dict[ArmName, Rotation] = {}
        arm_waypoints: dict[ArmName, FloatArray] = {}
        for arm in self.ARMS:
            pose = current_pose[arm]
            start_rotation = _pose_rotation(pose)
            positions, rotations = self._absolute_waypoints(
                pose[:3], start_rotation, pose_actions[arm]
            )
            rotation_vectors = self._continuous_spatial_rotation_vectors(start_rotation, rotations)
            start_rotations[arm] = start_rotation
            arm_waypoints[arm] = np.vstack(
                [
                    np.concatenate((pose[:3], np.zeros(3))),
                    np.column_stack((positions, rotation_vectors)),
                ]
            )
        waypoint_s = np.arange(horizon + 1, dtype=np.float64) * self.config.policy_dt
        waypoints = np.column_stack([arm_waypoints[arm] for arm in self.ARMS])
        physical_twist = np.concatenate([current_twist[arm] for arm in self.ARMS])
        if np.max(np.abs(physical_twist)) < 1e-6:
            initial_path_speed = 0.0
            spline_derivative = np.zeros_like(physical_twist)
        else:
            nominal_derivative = (waypoints[1] - waypoints[0]) / self.config.policy_dt
            velocity_scale = np.concatenate([self.config.limits[arm].velocity for arm in self.ARMS])
            normalized_twist = physical_twist / velocity_scale
            normalized_nominal = nominal_derivative / velocity_scale
            alignment = float(np.dot(normalized_twist, normalized_nominal))
            if alignment > 1e-9:
                initial_path_speed = float(np.dot(normalized_twist, normalized_twist) / alignment)
            else:
                initial_path_speed = 1.0
            spline_derivative = physical_twist / initial_path_speed
        path = ta.SplineInterpolator(
            waypoint_s,
            waypoints,
            bc_type=((1, spline_derivative), "natural"),
        )
        return path, start_rotations, waypoint_s, initial_path_speed

    def _absolute_waypoints(
        self,
        start_position: FloatArray,
        start_rotation: Rotation,
        pose_actions: FloatArray,
    ) -> tuple[FloatArray, Rotation]:
        positions: list[FloatArray] = []
        rotations: list[Rotation] = []
        position = start_position.copy()
        rotation = start_rotation
        for action in pose_actions:
            position = position + action[:3]
            delta_rotation = Rotation.from_euler(
                self.config.euler_order,
                action[3:],
                degrees=self.config.degrees,
            )
            rotation = delta_rotation * rotation
            positions.append(position.copy())
            rotations.append(rotation)
        return np.asarray(positions), Rotation.concatenate(rotations)

    @staticmethod
    def _continuous_spatial_rotation_vectors(
        start_rotation: Rotation, rotations: Rotation
    ) -> FloatArray:
        relative = rotations * start_rotation.inv()
        continuous: list[FloatArray] = []
        previous = np.zeros(3, dtype=np.float64)
        for principal in relative.as_rotvec():
            angle = float(np.linalg.norm(principal))
            if angle < 1e-12:
                selected = np.zeros(3, dtype=np.float64)
            else:
                axis = principal / angle
                candidates = np.asarray(
                    [axis * (angle + 2 * np.pi * turn) for turn in range(-2, 3)]
                )
                selected = candidates[np.argmin(np.linalg.norm(candidates - previous, axis=1))]
            continuous.append(selected)
            previous = selected
        return np.asarray(continuous)

    def _solve_toppra(
        self,
        path: Any,
        waypoint_s: FloatArray,
        initial_path_speed: float,
        *,
        relax_initial_twist: bool = False,
    ) -> tuple[FloatArray, FloatArray, float]:
        velocity = np.concatenate([self.config.limits[arm].velocity for arm in self.ARMS])
        acceleration = np.concatenate([self.config.limits[arm].acceleration for arm in self.ARMS])
        constraints = [
            constraint.JointVelocityConstraint(np.column_stack((-velocity, velocity))),
            constraint.JointAccelerationConstraint(
                np.column_stack((-acceleration, acceleration)),
                discretization_scheme=constraint.DiscretizationType.Interpolation,
            ),
        ]
        count = max(
            self.config.min_gridpoints,
            self.config.gridpoints_per_segment * (len(waypoint_s) - 1) + 1,
        )
        gridpoints = np.linspace(waypoint_s[0], waypoint_s[-1], count)
        instance = algo.TOPPRA(
            constraints,
            path,
            gridpoints=gridpoints,
            solver_wrapper="seidel",
            parametrizer="ParametrizeConstAccel",
        )
        path_speeds, initial_twist_scale = self._parameterize_with_terminal_range(
            instance,
            initial_path_speed,
            relax_initial_twist=relax_initial_twist,
        )
        if path_speeds is None or np.any(~np.isfinite(path_speeds)):
            raise RuntimeError(
                "The bimanual path is not controllable from the requested TCP twists"
            )
        return gridpoints, path_speeds, initial_twist_scale

    def _parameterize_with_terminal_range(
        self,
        instance: Any,
        initial_path_speed: float,
        *,
        relax_initial_twist: bool = False,
    ) -> tuple[FloatArray | None, float]:
        terminal_min = self.config.terminal_path_velocity_min
        terminal_max = self.config.terminal_path_velocity_max
        if (
            self.config.inference_mode == "async"
            and self.config.async_terminal_path_velocity is not None
        ):
            terminal_min = 0.0
            terminal_max = self.config.async_terminal_path_velocity
        controllable = instance.compute_controllable_sets(
            terminal_min,
            terminal_max,
        )
        if np.any(~np.isfinite(controllable)):
            return None, 1.0
        start_squared_speed = initial_path_speed**2
        start_min = float(controllable[0, 0])
        start_max = float(controllable[0, 1])
        initial_twist_scale = 1.0
        numerical_tolerance = 1e-6 * max(
            1.0,
            abs(start_squared_speed),
            abs(start_min),
            abs(start_max),
        )
        if start_min - numerical_tolerance <= start_squared_speed < start_min:
            start_squared_speed = start_min
        elif start_max < start_squared_speed <= start_max + numerical_tolerance:
            start_squared_speed = start_max
        if initial_path_speed > 1e-12:
            initial_twist_scale = math.sqrt(max(start_squared_speed, 0.0)) / initial_path_speed
        if not start_min - 1e-8 <= start_squared_speed <= start_max + 1e-8:
            if relax_initial_twist and start_squared_speed > start_max >= 0.0:
                relaxed_squared_speed = start_max
                if relaxed_squared_speed > start_min:
                    relaxed_squared_speed = max(start_min, 0.999 * relaxed_squared_speed)
                start_squared_speed = relaxed_squared_speed
                if initial_path_speed > 1e-12:
                    initial_twist_scale = math.sqrt(start_squared_speed) / initial_path_speed
            else:
                raise RuntimeError(
                    f"initial path speed {initial_path_speed:.6f} is outside the controllable "
                    f"interval [{math.sqrt(max(start_min, 0.0)):.6f}, "
                    f"{math.sqrt(max(start_max, 0.0)):.6f}]"
                )
        solver = instance.solver_wrapper
        deltas = np.asarray(solver.get_deltas(), dtype=np.float64)
        squared_speeds = np.zeros(len(instance.gridpoints), dtype=np.float64)
        squared_speeds[0] = start_squared_speed
        solver.setup_solver()
        try:
            index = 0
            retries = 0
            while index < len(deltas):
                result = instance._forward_step(  # noqa: SLF001 - terminal speed interval
                    index, squared_speeds[index], controllable[index + 1]
                )
                if result is None or not np.isfinite(result[0]):
                    if retries >= 10:
                        return None, initial_twist_scale
                    squared_speeds[index] = max(
                        squared_speeds[index] - 1e-9,
                        0.999 * squared_speeds[index],
                    )
                    retries += 1
                    continue
                retries = 0
                next_speed = squared_speeds[index] + 2 * deltas[index] * result[0]
                next_speed = max(next_speed - 1e-9, 0.9999 * next_speed)
                squared_speeds[index + 1] = np.clip(
                    next_speed,
                    controllable[index + 1, 0],
                    controllable[index + 1, 1],
                )
                index += 1
        finally:
            solver.close_solver()
        return np.sqrt(np.maximum(squared_speeds, 0.0)), initial_twist_scale

    @staticmethod
    def _time_grid(gridpoints: FloatArray, path_speeds: FloatArray) -> FloatArray:
        delta_s = np.diff(gridpoints)
        speed_sums = path_speeds[:-1] + path_speeds[1:]
        if np.any(speed_sums <= 1e-12):
            raise RuntimeError("A bimanual path segment has zero average path velocity")
        return np.concatenate(([0.0], np.cumsum(2 * delta_s / speed_sums)))

    def _validated_poses(self, poses: dict[ArmName, FloatArray]) -> dict[ArmName, FloatArray]:
        result: dict[ArmName, FloatArray] = {}
        for arm in self.ARMS:
            pose = np.asarray(poses[arm], dtype=np.float64).reshape(-1)
            if pose.shape != (7,) or np.any(~np.isfinite(pose)):
                raise ValueError(f"{arm} TCP pose must be a finite seven-dimensional vector")
            quaternion_norm = float(np.linalg.norm(pose[3:]))
            if quaternion_norm < 1e-8:
                raise ValueError(f"{arm} TCP quaternion has zero norm")
            pose = pose.copy()
            pose[3:] /= quaternion_norm
            result[arm] = pose
        return result

    def _validated_twists(
        self, twists: dict[ArmName, FloatArray] | None
    ) -> dict[ArmName, FloatArray]:
        if twists is None:
            return {arm: np.zeros(6, dtype=np.float64) for arm in self.ARMS}
        result: dict[ArmName, FloatArray] = {}
        for arm in self.ARMS:
            twist = np.asarray(twists[arm], dtype=np.float64).reshape(-1)
            if twist.shape != (6,) or np.any(~np.isfinite(twist)):
                raise ValueError(f"{arm} TCP twist must be a finite six-dimensional vector")
            result[arm] = twist
        return result

    def _pose_actions(
        self, action_arrays: dict[str, FloatArray]
    ) -> tuple[dict[ArmName, FloatArray], int]:
        result: dict[ArmName, FloatArray] = {}
        horizon: int | None = None
        for arm, configured_key in self.config.action_keys.items():
            key = configured_key
            if key not in action_arrays:
                alternate = key[7:] if key.startswith("action.") else f"action.{key}"
                if alternate not in action_arrays:
                    raise KeyError(f"Missing {arm} TCP action {configured_key!r}")
                key = alternate
            values = np.asarray(action_arrays[key], dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != 6 or len(values) == 0:
                raise ValueError(f"{arm} TCP action must have shape (chunk, 6), got {values.shape}")
            if np.any(~np.isfinite(values)):
                raise ValueError(f"{arm} TCP action contains non-finite values")
            if horizon is None:
                horizon = len(values)
            elif len(values) != horizon:
                raise ValueError("Left and right TCP actions must have the same horizon")
            result[arm] = values
        assert horizon is not None
        return result, horizon


@dataclass(frozen=True)
class _InferenceRequest:
    generation: int
    observation: dict[str, NDArray[Any]]
    task: str
    pose: dict[ArmName, FloatArray]
    twist: dict[ArmName, FloatArray]
    requested_at_s: float
    source_sequence_id: int | None
    completed_waypoints_at_start: int


@dataclass(frozen=True)
class _ReadyTrajectory:
    trajectory: BimanualContinuousTrajectory
    samples: tuple[_TrajectorySample, ...]
    prepared_at_s: float
    source_sequence_id: int | None
    handoff_control_count: int | None
    handoff_completed_waypoints: int
    handoff_sample: _TrajectorySample | None
    policy_discarded_waypoints: int
    twist_source: str


@dataclass(frozen=True)
class _TrajectorySample:
    time_from_start: float
    action: ActionDict


class GR00TBimanualToppraAgent:
    """Maintain continuous trajectories and return one sampled action per ``act`` call."""

    _HANDOFF_POSITION_TOLERANCE_M = 1e-5
    _HANDOFF_ROTATION_TOLERANCE_RAD = 1e-5
    _HANDOFF_TWIST_TOLERANCE = 1e-4

    def __init__(
        self,
        config: BimanualToppraRolloutConfig,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = 15000,
        trajectory_callback: TrajectoryCallback | None = None,
        policy_client: Any | None = None,
    ) -> None:
        self.config = config
        self.policy = policy_client
        self._policy_client_owned = policy_client is None
        self._policy_host = host
        self._policy_port = int(port)
        self._policy_timeout_ms = int(timeout_ms)
        self.planner = BimanualCartesianToppraPlanner(config)
        self.trajectory_callback = trajectory_callback

        self._shutdown = threading.Event()
        self._cv = threading.Condition()
        self._latest_obs: dict[str, NDArray[Any]] | None = None
        self._latest_task = config.task_default
        self._latest_pose: dict[ArmName, FloatArray] | None = None
        self._latest_measured_twist: dict[ArmName, FloatArray | None] = {
            arm: None for arm in self.planner.ARMS
        }
        self._latest_commanded_twist_at_observation = {
            arm: np.zeros(6, dtype=np.float64) for arm in self.planner.ARMS
        }
        self._pending_request: _InferenceRequest | None = None
        self._latest_trajectory: BimanualContinuousTrajectory | None = None
        self._ready_trajectory: _ReadyTrajectory | None = None
        self._active_trajectory: BimanualContinuousTrajectory | None = None
        self._active_action_buf: deque[_TrajectorySample] = deque()
        self._active_last_sample: _TrajectorySample | None = None
        self._active_initial_completed_waypoints = 0
        self._active_policy_discarded_waypoints = 0
        self._completed_waypoints_before_active = 0
        self._total_consumed_control_samples = 0
        self._hold_action: ActionDict | None = None
        self._hold_sequence_id: int | None = None
        self._last_pinch = {
            self.config.left_pinch_key: np.zeros(1, dtype=np.float32),
            self.config.right_pinch_key: np.zeros(1, dtype=np.float32),
        }
        self._refill_requested_sequence_id: int | None = None
        self._trajectory_generation = 0
        self._request_generation = 0
        self._completed_generation = 0
        self._latest_latency_s: float | None = None
        self._latest_planning_latency_s: float | None = None
        self._latest_pipeline_latency_s: float | None = None
        self._latest_activation_offset_s: float | None = None
        self._latest_policy_discarded_waypoints = 0
        self._latest_planning_discarded_waypoints = 0
        self._latest_planning_discarded_control_samples = 0
        self._latest_twist_source = "zero"
        self._latest_handoff_control_count: int | None = None
        self._latest_handoff_position_error_m: float | None = None
        self._latest_handoff_rotation_error_rad: float | None = None
        self._latest_handoff_twist_error: float | None = None
        self._latest_error: str | None = None
        self._fatal_error: str | None = None
        self._inference_count = 0
        self._sequence_id = 0

        self._producer = threading.Thread(
            target=self._producer_loop,
            daemon=True,
            name="GR00T_BIMANUAL_TOPPRA_Producer",
        )
        self._producer.start()

    def act(self, obs: dict[str, NDArray[Any]], task: str = "") -> ActionDict:
        """Return one absolute dual-arm TCP target sampled from the active TOPPRA trajectory."""

        with self._cv:
            self._record_observation_locked(obs, task)
            self._activate_latest_trajectory_locked()

            if self.config.inference_mode == "sync":
                return self._act_sync_locked()
            if self._active_trajectory is None:
                if self._hold_action is None:
                    self._set_hold_from_current_pose_locked()
                self._ensure_inference_requested_locked()
                return self._copy_action(self._hold_action)

            trajectory = self._active_trajectory
            if not self._active_action_buf:
                if self._hold_sequence_id != trajectory.sequence_id:
                    LOGGER.warning(
                        "[TOPPRA][buffer_underflow] sequence=%d consumed_control_samples=%d "
                        "action=hold_last_target",
                        trajectory.sequence_id,
                        self._total_consumed_control_samples,
                    )
                self._set_hold_from_trajectory_end_locked(trajectory)
                self._ensure_inference_requested_locked()
                return self._copy_action(self._hold_action)

            action = self._pop_active_action_locked()
            remaining_s = len(self._active_action_buf) * self.config.control_dt
            should_refill = self._should_request_async_refill_locked(remaining_s)
            if should_refill and self._refill_requested_sequence_id != trajectory.sequence_id:
                reason = (
                    "remaining_time"
                    if remaining_s <= self._async_refill_lead_locked()
                    else "open_loop_horizon"
                )
                LOGGER.info(
                    "[TOPPRA][refill_trigger] source_sequence=%d reason=%s remaining_s=%.4f "
                    "lead_s=%.4f completed_waypoints=%d",
                    trajectory.sequence_id,
                    reason,
                    remaining_s,
                    self._async_refill_lead_locked(),
                    self._total_completed_waypoints_locked(),
                )
                self._ensure_inference_requested_locked()
                self._refill_requested_sequence_id = trajectory.sequence_id
            return action

    def _act_sync_locked(self) -> ActionDict:
        """Hold during planning, then execute one complete trajectory without blocking."""

        if self._active_trajectory is not None and self._active_action_buf:
            return self._pop_active_action_locked()
        if not self._inference_pending_locked():
            if self._active_trajectory is None:
                self._set_hold_from_current_pose_locked()
            else:
                self._set_hold_from_trajectory_end_locked(self._active_trajectory)
            self._ensure_inference_requested_locked()
        return self._copy_action(self._hold_action)

    def _pop_active_action_locked(self) -> ActionDict:
        sample = self._active_action_buf.popleft()
        self._active_last_sample = sample
        self._total_consumed_control_samples += 1
        action = self._public_action(sample.action)
        for key in self._last_pinch:
            self._last_pinch[key] = np.array(action[key], copy=True)
        return action

    def _public_action(self, sample: ActionDict) -> ActionDict:
        required_keys = (
            self.config.left_pose_output_key,
            self.config.right_pose_output_key,
            self.config.left_pinch_key,
            self.config.right_pinch_key,
        )
        missing = [key for key in required_keys if key not in sample]
        if missing:
            raise KeyError(f"Sampled TOPPRA action is missing fields: {missing}")
        return {key: np.array(sample[key], copy=True) for key in required_keys}

    @staticmethod
    def _copy_action(action: ActionDict | None) -> ActionDict:
        if action is None:
            raise RuntimeError("Hold pose has not been initialized")
        return {key: np.array(value, copy=True) for key, value in action.items()}

    def _set_hold_from_current_pose_locked(self) -> None:
        if self._latest_pose is None:
            raise RuntimeError("Bimanual TCP state has not been observed")
        self._hold_action = {
            self.config.left_pose_output_key: self._latest_pose["left"].astype(np.float32),
            self.config.right_pose_output_key: self._latest_pose["right"].astype(np.float32),
            **{key: np.array(value, copy=True) for key, value in self._last_pinch.items()},
        }
        self._hold_sequence_id = None

    def _set_hold_from_trajectory_end_locked(
        self, trajectory: BimanualContinuousTrajectory
    ) -> None:
        if self._hold_sequence_id == trajectory.sequence_id:
            return
        if self._active_last_sample is None:
            sample = trajectory.sample(trajectory.duration)
        else:
            sample = self._active_last_sample.action
        self._hold_action = self._public_action(sample)
        self._hold_sequence_id = trajectory.sequence_id

    def _active_logical_time_locked(self) -> float:
        if self._active_last_sample is None:
            return 0.0
        return self._active_last_sample.time_from_start

    def _total_completed_waypoints_locked(self) -> int:
        if self._active_trajectory is None:
            return self._completed_waypoints_before_active
        elapsed_s = self._active_logical_time_locked()
        completed_in_trajectory = int(
            np.searchsorted(self._active_trajectory.waypoint_times, elapsed_s, side="right")
        )
        completed_since_activation = max(
            completed_in_trajectory - self._active_initial_completed_waypoints,
            0,
        )
        return self._completed_waypoints_before_active + completed_since_activation

    def _async_refill_lead_locked(self) -> float:
        fixed_budget_s = (
            self.config.max_latency_s
            + self.config.scheduling_margin_s
            + self.config.handoff_margin_s
        )
        return max(self.config.refill_lead_s, fixed_budget_s)

    def _should_request_async_refill_locked(self, remaining_s: float) -> bool:
        lead_s = self._async_refill_lead_locked()
        if remaining_s <= lead_s:
            return True
        if self._active_trajectory is None:
            return False
        projected_time_s = min(
            self._active_logical_time_locked() + lead_s,
            self._active_trajectory.duration,
        )
        projected_waypoints = int(
            np.searchsorted(
                self._active_trajectory.waypoint_times,
                projected_time_s,
                side="right",
            )
        )
        executed_after_activation = max(
            projected_waypoints - self._active_initial_completed_waypoints,
            0,
        )
        predicted_open_loop = self._active_policy_discarded_waypoints + executed_after_activation
        return predicted_open_loop >= self.config.open_loop_horizon

    def _inference_pending_locked(self) -> bool:
        return self._request_generation > self._completed_generation

    def request_trajectory(
        self,
        obs: dict[str, NDArray[Any]],
        task: str = "",
        *,
        block: bool = False,
    ) -> BimanualContinuousTrajectory | None:
        """Start inference; optionally wait while a lower layer continues executing its old path."""

        now = time.perf_counter()
        with self._cv:
            self._record_observation_locked(obs, task)
            requested_generation = self._queue_inference_locked(force=True)
            if not block:
                return None
            self._wait_for_generation_locked(
                requested_generation,
                now + self.config.act_timeout_s,
            )
            return self._latest_trajectory

    def observe(
        self,
        obs: dict[str, NDArray[Any]],
        task: str = "",
        *,
        request_inference: bool = False,
    ) -> None:
        """Update measured state and optionally trigger non-blocking asynchronous inference."""

        with self._cv:
            self._record_observation_locked(obs, task)
            if request_inference:
                self._queue_inference_locked(force=True)

    def poll_trajectory(
        self, after_sequence_id: int | None = None
    ) -> BimanualContinuousTrajectory | None:
        """Return a newly completed trajectory without blocking."""

        with self._cv:
            trajectory = self._latest_trajectory
            if trajectory is None:
                return None
            if after_sequence_id is not None and trajectory.sequence_id <= after_sequence_id:
                return None
            return trajectory

    def _ensure_inference_requested_locked(self) -> int:
        return self._queue_inference_locked(force=False)

    def _queue_inference_locked(self, *, force: bool) -> int:
        if not force and self._inference_pending_locked():
            return self._request_generation
        if self._latest_obs is None or self._latest_pose is None:
            raise RuntimeError("Cannot request inference before observing bimanual TCP state")

        self._request_generation += 1
        generation = self._request_generation
        requested_at_s = time.perf_counter()
        if self.config.inference_mode == "sync":
            request_twist = {arm: np.zeros(6, dtype=np.float64) for arm in self.planner.ARMS}
            request_pose = self._sync_planning_pose_locked()
        else:
            request_twist = {
                arm: twist.copy()
                for arm, twist in self._latest_commanded_twist_at_observation.items()
            }
            request_pose = {arm: pose.copy() for arm, pose in self._latest_pose.items()}
        self._pending_request = _InferenceRequest(
            generation=generation,
            observation={
                key: np.array(value, copy=True) for key, value in self._latest_obs.items()
            },
            task=self._latest_task,
            pose=request_pose,
            twist=request_twist,
            requested_at_s=requested_at_s,
            source_sequence_id=(
                None if self._active_trajectory is None else self._active_trajectory.sequence_id
            ),
            completed_waypoints_at_start=self._total_completed_waypoints_locked(),
        )
        LOGGER.info(
            "[TOPPRA][inference_request] generation=%d mode=%s source_sequence=%s "
            "completed_waypoints=%d active_buffer=%d",
            generation,
            self.config.inference_mode,
            self._pending_request.source_sequence_id,
            self._pending_request.completed_waypoints_at_start,
            len(self._active_action_buf),
        )
        self._cv.notify_all()
        return generation

    def _sync_planning_pose_locked(self) -> dict[ArmName, FloatArray]:
        if self._hold_action is None:
            return {arm: pose.copy() for arm, pose in self._latest_pose.items()}
        return {
            arm: np.asarray(
                self._hold_action[self.config.pose_output_keys[arm]], dtype=np.float64
            ).copy()
            for arm in self.planner.ARMS
        }

    def _wait_for_generation_locked(self, generation: int, deadline_s: float) -> None:
        while self._completed_generation < generation and not self._shutdown.is_set():
            if self._fatal_error is not None:
                raise RuntimeError(f"Bimanual TOPPRA producer stopped: {self._fatal_error}")
            remaining_s = deadline_s - time.perf_counter()
            if remaining_s <= 0:
                detail = f": {self._latest_error}" if self._latest_error else ""
                raise TimeoutError(f"Timed out waiting for a TOPPRA action trajectory{detail}")
            self._cv.wait(timeout=remaining_s)
        if self._latest_trajectory is None or self._trajectory_generation < generation:
            detail = self._latest_error or "trajectory generation failed"
            raise RuntimeError(f"TOPPRA action trajectory is unavailable: {detail}")

    def _activate_latest_trajectory_locked(
        self,
        *,
        min_generation: int = 0,
    ) -> bool:
        ready = self._ready_trajectory
        if ready is None or self._trajectory_generation < min_generation:
            return False
        trajectory = ready.trajectory
        if (
            self._active_trajectory is not None
            and trajectory.sequence_id <= self._active_trajectory.sequence_id
        ):
            self._ready_trajectory = None
            return False

        completed_before_switch = self._total_completed_waypoints_locked()
        activated_completed_waypoints = 0
        if self.config.inference_mode == "async":
            active_sequence_id = (
                None if self._active_trajectory is None else self._active_trajectory.sequence_id
            )
            if active_sequence_id != ready.source_sequence_id:
                self._latest_error = (
                    "Discarded an asynchronous trajectory because its source trajectory changed"
                )
                LOGGER.warning(
                    "[TOPPRA][handoff_source_changed] new_sequence=%d expected_source=%s "
                    "active_source=%s action=discard",
                    trajectory.sequence_id,
                    ready.source_sequence_id,
                    active_sequence_id,
                )
                self._ready_trajectory = None
                self._refill_requested_sequence_id = None
                return False
            if ready.handoff_control_count is not None:
                if self._total_consumed_control_samples < ready.handoff_control_count:
                    return False
                if self._total_consumed_control_samples > ready.handoff_control_count:
                    self._latest_error = (
                        "Discarded an asynchronous trajectory because it missed the reserved "
                        "handoff control sample"
                    )
                    LOGGER.warning(
                        "[TOPPRA][handoff_missed] new_sequence=%d source_sequence=%s "
                        "reserved_control_count=%d current_control_count=%d late_samples=%d "
                        "action=discard_and_continue_old",
                        trajectory.sequence_id,
                        ready.source_sequence_id,
                        ready.handoff_control_count,
                        self._total_consumed_control_samples,
                        self._total_consumed_control_samples - ready.handoff_control_count,
                    )
                    self._ready_trajectory = None
                    self._refill_requested_sequence_id = None
                    return False
                if not self._handoff_is_continuous_locked(ready):
                    self._latest_error = (
                        "Discarded an asynchronous trajectory because its command pose or twist "
                        "is discontinuous at the reserved handoff"
                    )
                    LOGGER.error(
                        "[TOPPRA][handoff_discontinuous] new_sequence=%d source_sequence=%s "
                        "position_error_m=%.9f rotation_error_rad=%.9f twist_error=%.9f "
                        "action=discard_and_continue_old",
                        trajectory.sequence_id,
                        ready.source_sequence_id,
                        self._latest_handoff_position_error_m,
                        self._latest_handoff_rotation_error_rad,
                        self._latest_handoff_twist_error,
                    )
                    self._ready_trajectory = None
                    self._refill_requested_sequence_id = None
                    return False
                completed_before_switch = ready.handoff_completed_waypoints
            elif self._active_trajectory is not None:
                self._latest_error = (
                    "Discarded an asynchronous trajectory without a handoff for its active source"
                )
                LOGGER.warning(
                    "[TOPPRA][handoff_missing] new_sequence=%d source_sequence=%s action=discard",
                    trajectory.sequence_id,
                    ready.source_sequence_id,
                )
                self._ready_trajectory = None
                self._refill_requested_sequence_id = None
                return False
            activated_completed_waypoints = int(
                np.searchsorted(
                    trajectory.waypoint_times,
                    0.0,
                    side="right",
                )
            )

        self._active_trajectory = trajectory
        self._active_action_buf = deque(ready.samples)
        self._active_last_sample = None
        self._completed_waypoints_before_active = completed_before_switch
        self._active_initial_completed_waypoints = activated_completed_waypoints
        self._active_policy_discarded_waypoints = ready.policy_discarded_waypoints
        self._latest_activation_offset_s = 0.0
        self._latest_policy_discarded_waypoints = ready.policy_discarded_waypoints
        self._latest_planning_discarded_waypoints = 0
        self._latest_planning_discarded_control_samples = 0
        self._latest_twist_source = ready.twist_source
        self._latest_handoff_control_count = ready.handoff_control_count
        self._ready_trajectory = None
        self._hold_action = None
        self._hold_sequence_id = None
        self._refill_requested_sequence_id = None
        if ready.handoff_control_count is None:
            LOGGER.info(
                "[TOPPRA][trajectory_activated] sequence=%d mode=%s samples=%d duration_s=%.4f",
                trajectory.sequence_id,
                self.config.inference_mode,
                len(ready.samples),
                trajectory.duration,
            )
        else:
            LOGGER.info(
                "[TOPPRA][handoff_activated] new_sequence=%d source_sequence=%s "
                "control_count=%d position_error_m=%.9f rotation_error_rad=%.9f "
                "twist_error=%.9f samples=%d duration_s=%.4f",
                trajectory.sequence_id,
                ready.source_sequence_id,
                ready.handoff_control_count,
                self._latest_handoff_position_error_m,
                self._latest_handoff_rotation_error_rad,
                self._latest_handoff_twist_error,
                len(ready.samples),
                trajectory.duration,
            )
        return True

    def _handoff_is_continuous_locked(self, ready: _ReadyTrajectory) -> bool:
        if ready.handoff_sample is None or not ready.samples:
            return False
        old_action = ready.handoff_sample.action
        new_action = ready.samples[0].action
        position_error = 0.0
        rotation_error = 0.0
        twist_error = 0.0
        for arm in self.planner.ARMS:
            old_pose = np.asarray(old_action[self.config.pose_output_keys[arm]], dtype=np.float64)
            new_pose = np.asarray(new_action[self.config.pose_output_keys[arm]], dtype=np.float64)
            position_error = max(
                position_error,
                float(np.linalg.norm(old_pose[:3] - new_pose[:3])),
            )
            relative_rotation = _pose_rotation(new_pose) * _pose_rotation(old_pose).inv()
            rotation_error = max(rotation_error, float(relative_rotation.magnitude()))
            old_twist = np.asarray(
                old_action[self.config.velocity_output_keys[arm]],
                dtype=np.float64,
            )
            new_twist = np.asarray(
                new_action[self.config.velocity_output_keys[arm]],
                dtype=np.float64,
            )
            twist_error = max(
                twist_error,
                float(np.max(np.abs(old_twist - new_twist))),
            )
        self._latest_handoff_position_error_m = position_error
        self._latest_handoff_rotation_error_rad = rotation_error
        self._latest_handoff_twist_error = twist_error
        return (
            position_error <= self._HANDOFF_POSITION_TOLERANCE_M
            and rotation_error <= self._HANDOFF_ROTATION_TOLERANCE_RAD
            and twist_error <= self._HANDOFF_TWIST_TOLERANCE
        )

    def diagnostics(self) -> dict[str, Any]:
        with self._cv:
            trajectory = self._latest_trajectory
            active = self._active_trajectory
            active_elapsed_s: float | None = None
            active_remaining_s: float | None = None
            if active is not None:
                active_elapsed_s = self._active_logical_time_locked()
                active_remaining_s = len(self._active_action_buf) * self.config.control_dt
            inference_pending = self._inference_pending_locked()
            if self.config.inference_mode == "sync":
                if inference_pending:
                    execution_state = "holding_for_inference"
                elif active_remaining_s is not None and active_remaining_s > 0.0:
                    execution_state = "executing"
                elif self._latest_trajectory is not None:
                    execution_state = "ready"
                else:
                    execution_state = "holding"
            else:
                if active is None:
                    execution_state = "waiting"
                elif active_remaining_s == 0.0:
                    execution_state = "holding_after_underflow"
                elif inference_pending:
                    execution_state = "executing_async_refilling"
                else:
                    execution_state = "executing_async"
            return {
                "execution_state": execution_state,
                "request_generation": self._request_generation,
                "completed_generation": self._completed_generation,
                "inference_pending": inference_pending,
                "latest_latency_s": self._latest_latency_s,
                "latest_planning_latency_s": self._latest_planning_latency_s,
                "latest_pipeline_latency_s": self._latest_pipeline_latency_s,
                "latest_activation_offset_s": self._latest_activation_offset_s,
                "policy_discarded_waypoints": self._latest_policy_discarded_waypoints,
                "planning_discarded_waypoints": self._latest_planning_discarded_waypoints,
                "planning_discarded_control_samples": (
                    self._latest_planning_discarded_control_samples
                ),
                "planning_twist_source": self._latest_twist_source,
                "async_refill_lead_s": self._async_refill_lead_locked(),
                "open_loop_horizon": self.config.open_loop_horizon,
                "handoff_control_count": self._latest_handoff_control_count,
                "handoff_position_error_m": self._latest_handoff_position_error_m,
                "handoff_rotation_error_rad": self._latest_handoff_rotation_error_rad,
                "handoff_twist_error": self._latest_handoff_twist_error,
                "total_completed_waypoints": self._total_completed_waypoints_locked(),
                "inference_count": self._inference_count,
                "latest_error": self._latest_error,
                "fatal_error": self._fatal_error,
                "trajectory_duration_s": _attribute_or_none(trajectory, "duration"),
                "active_sequence_id": _attribute_or_none(active, "sequence_id"),
                "ready_sequence_id": (
                    None
                    if self._ready_trajectory is None
                    else self._ready_trajectory.trajectory.sequence_id
                ),
                "active_duration_s": _attribute_or_none(active, "duration"),
                "active_elapsed_s": active_elapsed_s,
                "active_remaining_s": active_remaining_s,
                "active_buffer_size": len(self._active_action_buf),
                "total_consumed_control_samples": self._total_consumed_control_samples,
                "control_frequency": self.config.control_frequency,
                "active_expired_at_activation": (
                    None
                    if active is None or self._latest_activation_offset_s is None
                    else self._latest_activation_offset_s >= active.duration
                ),
                "used_toppra": _attribute_or_none(trajectory, "used_toppra"),
                "initial_twist_scale": _attribute_or_none(trajectory, "initial_twist_scale"),
                "active_initial_twist_scale": _attribute_or_none(active, "initial_twist_scale"),
                "terminal_path_speed": _attribute_or_none(trajectory, "terminal_path_speed"),
                "active_terminal_path_speed": _attribute_or_none(active, "terminal_path_speed"),
                "tts_selected_index": _attribute_or_none(trajectory, "tts_selected_index"),
                "tts_candidate_scores": _attribute_or_none(trajectory, "tts_candidate_scores"),
            }

    def teardown(self) -> None:
        self._shutdown.set()
        with self._cv:
            self._cv.notify_all()
        if self._producer.is_alive():
            self._producer.join(timeout=self._policy_timeout_ms / 1000.0 + 1.0)
        if self._producer.is_alive():
            raise RuntimeError("GR00T bimanual TOPPRA producer did not stop")

    def _producer_loop(self) -> None:
        try:
            if self._policy_client_owned:
                self.policy = GR00TPolicyClient(
                    host=self._policy_host,
                    port=self._policy_port,
                    timeout_ms=self._policy_timeout_ms,
                )
            self._run_producer_loop()
        except Exception as exc:
            with self._cv:
                self._fatal_error = f"{type(exc).__name__}: {exc}"
                self._latest_error = self._fatal_error
                self._cv.notify_all()
            LOGGER.exception("[TOPPRA][producer_fatal] error=%s", self._fatal_error)
        finally:
            if self._policy_client_owned and self.policy is not None:
                self.policy.close()
                self.policy = None

    def _run_producer_loop(self) -> None:
        processed_generation = 0
        while not self._shutdown.is_set():
            with self._cv:
                self._cv.wait_for(
                    lambda: (
                        self._shutdown.is_set()
                        or (
                            self._pending_request is not None
                            and self._pending_request.generation > processed_generation
                        )
                    )
                )
                if self._shutdown.is_set():
                    return
                request = self._pending_request
                assert request is not None
                generation = request.generation
            try:
                ready = self._infer_trajectory(request)
                trajectory = ready.trajectory
                processed_generation = generation
                with self._cv:
                    if generation != self._request_generation:
                        self._cv.notify_all()
                        continue
                    self._latest_trajectory = trajectory
                    self._ready_trajectory = ready
                    self._trajectory_generation = generation
                    self._completed_generation = generation
                    self._latest_pipeline_latency_s = ready.prepared_at_s - request.requested_at_s
                    self._latest_error = None
                    self._cv.notify_all()
                LOGGER.info(
                    "[TOPPRA][trajectory_ready] generation=%d sequence=%d source_sequence=%s "
                    "pipeline_latency_s=%.4f samples=%d handoff_control_count=%s",
                    generation,
                    trajectory.sequence_id,
                    ready.source_sequence_id,
                    ready.prepared_at_s - request.requested_at_s,
                    len(ready.samples),
                    ready.handoff_control_count,
                )
                if self.trajectory_callback is not None:
                    self.trajectory_callback(trajectory)
            except Exception as exc:
                processed_generation = generation
                with self._cv:
                    self._latest_error = f"{type(exc).__name__}: {exc}"
                    self._completed_generation = generation
                    self._refill_requested_sequence_id = None
                    self._cv.notify_all()
                LOGGER.exception(
                    "[TOPPRA][planning_failed] generation=%d source_sequence=%s error=%s: %s",
                    generation,
                    request.source_sequence_id,
                    type(exc).__name__,
                    exc,
                )

    def _infer_trajectory(
        self,
        request: _InferenceRequest,
    ) -> _ReadyTrajectory:
        if self.policy is None:
            raise RuntimeError("GR00T policy client is not initialized")
        inference_started_at_s = time.perf_counter()
        response = self.policy.get_action(
            self._convert_obs(
                request.observation,
                request.task,
                batch_size=self.config.tts_samples,
            )
        )
        latency_s = time.perf_counter() - inference_started_at_s
        candidates = self._response_to_action_candidates(response)
        if not candidates:
            raise RuntimeError("GR00T returned an empty action chunk")
        if len(candidates) != self.config.tts_samples:
            raise ValueError(
                f"GR00T returned {len(candidates)} TTS candidates, "
                f"expected {self.config.tts_samples}"
            )
        candidates = [
            {key: self._scaled_action(key, values) for key, values in candidate.items()}
            for candidate in candidates
        ]
        candidate_horizons = tuple(len(next(iter(candidate.values()))) for candidate in candidates)
        LOGGER.info(
            "[TOPPRA][policy_response] generation=%d latency_s=%.4f candidates=%d horizons=%s",
            request.generation,
            latency_s,
            len(candidates),
            candidate_horizons,
        )
        planning_started_at_s = time.perf_counter()

        if self.config.inference_mode == "sync":
            inference_pose = request.pose
            inference_twist = request.twist
            source_sequence_id = request.source_sequence_id
            policy_discarded_waypoints = 0
            handoff_control_count = None
            handoff_completed_waypoints = request.completed_waypoints_at_start
            handoff_sample = None
            twist_source = "zero"
        else:
            with self._cv:
                (
                    inference_pose,
                    inference_twist,
                    handoff_control_count,
                    handoff_completed_waypoints,
                    handoff_sample,
                    twist_source,
                ) = self._scheduled_handoff_locked(request)
                policy_discarded_waypoints = max(
                    handoff_completed_waypoints - request.completed_waypoints_at_start,
                    0,
                )
                candidates = self._trim_action_candidates(
                    candidates,
                    policy_discarded_waypoints,
                )
                source_sequence_id = request.source_sequence_id
                LOGGER.info(
                    "[TOPPRA][handoff_reserved] generation=%d source_sequence=%s "
                    "handoff_control_count=%s current_control_count=%d wait_samples=%s "
                    "discarded_policy_waypoints=%d twist_source=%s",
                    request.generation,
                    source_sequence_id,
                    handoff_control_count,
                    self._total_consumed_control_samples,
                    (
                        None
                        if handoff_control_count is None
                        else handoff_control_count - self._total_consumed_control_samples
                    ),
                    policy_discarded_waypoints,
                    twist_source,
                )

        with self._cv:
            self._sequence_id += 1
            sequence_id = self._sequence_id
            self._latest_latency_s = latency_s
            self._inference_count += 1
        selected_index, scores = self.planner.select_tts_candidate(
            candidates,
            inference_pose,
            inference_twist,
        )
        LOGGER.info(
            "[TOPPRA][tts_selection] generation=%d sequence=%d selected_candidate=%d scores=%s",
            request.generation,
            sequence_id,
            selected_index,
            scores,
        )
        ranked_indices = [
            selected_index,
            *sorted(
                (index for index in range(len(candidates)) if index != selected_index),
                key=lambda index: scores[index],
                reverse=True,
            ),
        ]
        planning_errors: list[str] = []
        trajectory: BimanualContinuousTrajectory | None = None
        for candidate_index in ranked_indices:
            if not np.isfinite(scores[candidate_index]):
                continue
            try:
                trajectory = self.planner.plan(
                    candidates[candidate_index],
                    inference_pose,
                    inference_twist,
                    sequence_id=sequence_id,
                    inference_started_at_s=request.requested_at_s,
                    tts_selected_index=candidate_index,
                    tts_candidate_scores=scores,
                    relax_initial_twist=False,
                )
                break
            except RuntimeError as exc:
                planning_errors.append(f"candidate {candidate_index}: {exc}")
                LOGGER.warning(
                    "[TOPPRA][candidate_failed] generation=%d sequence=%d candidate=%d "
                    "score=%s error=%s",
                    request.generation,
                    sequence_id,
                    candidate_index,
                    scores[candidate_index],
                    exc,
                )
        if trajectory is None:
            detail = "; ".join(planning_errors) or "no finite TTS candidates"
            raise RuntimeError(f"TOPPRA rejected every TTS candidate: {detail}")
        samples = self._sample_trajectory_for_control(trajectory)
        prepared_at_s = time.perf_counter()
        planning_latency_s = prepared_at_s - planning_started_at_s
        with self._cv:
            self._latest_planning_latency_s = planning_latency_s
        log = LOGGER.info if trajectory.used_toppra else LOGGER.warning
        event = "plan_success" if trajectory.used_toppra else "timing_fallback"
        log(
            "[TOPPRA][%s] generation=%d sequence=%d candidate=%d duration_s=%.4f "
            "planning_latency_s=%.4f initial_twist_scale=%.6f terminal_path_speed=%.6f "
            "samples=%d fallback_reason=%s",
            event,
            request.generation,
            sequence_id,
            trajectory.tts_selected_index,
            trajectory.duration,
            planning_latency_s,
            trajectory.initial_twist_scale,
            trajectory.terminal_path_speed,
            len(samples),
            trajectory.fallback_reason,
        )
        return _ReadyTrajectory(
            trajectory=trajectory,
            samples=samples,
            prepared_at_s=prepared_at_s,
            source_sequence_id=source_sequence_id,
            handoff_control_count=handoff_control_count,
            handoff_completed_waypoints=handoff_completed_waypoints,
            handoff_sample=handoff_sample,
            policy_discarded_waypoints=policy_discarded_waypoints,
            twist_source=twist_source,
        )

    def _scheduled_handoff_locked(
        self,
        request: _InferenceRequest,
    ) -> tuple[
        dict[ArmName, FloatArray],
        dict[ArmName, FloatArray],
        int | None,
        int,
        _TrajectorySample | None,
        str,
    ]:
        active_sequence_id = (
            None if self._active_trajectory is None else self._active_trajectory.sequence_id
        )
        if active_sequence_id != request.source_sequence_id:
            raise RuntimeError("The active trajectory changed while GR00T inference was running")
        if self._active_trajectory is None:
            return (
                {arm: pose.copy() for arm, pose in request.pose.items()},
                {arm: twist.copy() for arm, twist in request.twist.items()},
                None,
                request.completed_waypoints_at_start,
                None,
                "zero",
            )

        lead_samples = int(math.ceil(self.config.handoff_margin_s / self.config.control_dt))
        if lead_samples >= len(self._active_action_buf):
            raise RuntimeError(
                "The active trajectory has insufficient samples for the reserved handoff: "
                f"need index {lead_samples}, have {len(self._active_action_buf)} samples"
            )
        handoff_sample = self._active_action_buf[lead_samples]
        handoff_completed_waypoints = self._completed_waypoints_at_active_time_locked(
            handoff_sample.time_from_start
        )
        pose = {
            arm: np.asarray(
                handoff_sample.action[self.config.pose_output_keys[arm]],
                dtype=np.float64,
            ).copy()
            for arm in self.planner.ARMS
        }
        twist = {
            arm: np.asarray(
                handoff_sample.action[self.config.velocity_output_keys[arm]],
                dtype=np.float64,
            ).copy()
            for arm in self.planner.ARMS
        }
        return (
            pose,
            twist,
            self._total_consumed_control_samples + lead_samples,
            handoff_completed_waypoints,
            handoff_sample,
            "scheduled_command",
        )

    def _completed_waypoints_at_active_time_locked(self, time_from_start: float) -> int:
        if self._active_trajectory is None:
            return self._completed_waypoints_before_active
        completed_in_trajectory = int(
            np.searchsorted(
                self._active_trajectory.waypoint_times,
                time_from_start,
                side="right",
            )
        )
        return self._completed_waypoints_before_active + max(
            completed_in_trajectory - self._active_initial_completed_waypoints,
            0,
        )

    def _sample_trajectory_for_control(
        self, trajectory: BimanualContinuousTrajectory
    ) -> tuple[_TrajectorySample, ...]:
        sample_count = int(math.ceil(trajectory.duration / self.config.control_dt)) + 1
        times = np.minimum(
            np.arange(sample_count, dtype=np.float64) * self.config.control_dt,
            trajectory.duration,
        )
        sampled_actions = trajectory.sample(times)
        return tuple(
            _TrajectorySample(
                time_from_start=float(times[index]),
                action={
                    key: np.array(values[index], copy=True)
                    for key, values in sampled_actions.items()
                },
            )
            for index in range(sample_count)
        )

    def _record_observation_locked(
        self,
        obs: dict[str, NDArray[Any]],
        task: str,
    ) -> None:
        commanded_twist_at_observation = self._commanded_twist_locked()
        # The full camera/state payload is copied only when an inference request is queued.
        # Copying every image on every 250 Hz act call adds avoidable control-loop latency.
        self._latest_obs = obs
        self._latest_task = task or self.config.task_default
        raw_poses: dict[ArmName, FloatArray] = {}
        for arm, key in self.config.pose_state_keys.items():
            if key not in obs:
                raise KeyError(f"Missing measured {arm} TCP pose {key!r}")
            raw_poses[arm] = np.asarray(obs[key])
        poses = self.planner._validated_poses(raw_poses)
        for arm in self.planner.ARMS:
            velocity_key = self.config.velocity_state_keys[arm]
            if velocity_key is not None and velocity_key in obs:
                twist = np.asarray(obs[velocity_key], dtype=np.float64).reshape(-1)
                if twist.shape != (6,) or np.any(~np.isfinite(twist)):
                    raise ValueError(f"{arm} TCP velocity must be a finite six-dimensional vector")
                previous = self._latest_measured_twist[arm]
                if previous is None:
                    self._latest_measured_twist[arm] = twist.copy()
                else:
                    alpha = self.config.velocity_filter
                    self._latest_measured_twist[arm] = alpha * twist + (1 - alpha) * previous
            else:
                self._latest_measured_twist[arm] = None
        self._latest_pose = poses
        self._latest_commanded_twist_at_observation = {
            arm: twist.copy() for arm, twist in commanded_twist_at_observation.items()
        }

    def _copy_latest_pose_locked(self) -> dict[ArmName, FloatArray]:
        if self._latest_pose is None:
            raise RuntimeError("Bimanual TCP state has not been observed")
        return {arm: pose.copy() for arm, pose in self._latest_pose.items()}

    def _commanded_twist_locked(self) -> dict[ArmName, FloatArray]:
        if self._active_trajectory is None or self._active_last_sample is None:
            return {arm: np.zeros(6, dtype=np.float64) for arm in self.planner.ARMS}
        sample = self._active_last_sample.action
        return {
            arm: np.asarray(sample[self.config.velocity_output_keys[arm]], dtype=np.float64)
            for arm in self.planner.ARMS
        }

    def _planning_twist_locked(self) -> tuple[dict[ArmName, FloatArray], str]:
        commanded = self._latest_commanded_twist_at_observation
        twists: dict[ArmName, FloatArray] = {}
        sources: set[str] = set()
        for arm in self.planner.ARMS:
            measured = self._latest_measured_twist[arm]
            if measured is None:
                if self._active_trajectory is None:
                    twists[arm] = commanded[arm]
                    sources.add("zero")
                else:
                    limit = 0.999 * self.config.limits[arm].velocity
                    bounded = np.clip(commanded[arm], -limit, limit)
                    twists[arm] = bounded
                    sources.add(
                        "commanded_clipped"
                        if np.any(np.abs(bounded - commanded[arm]) > 1e-12)
                        else "commanded"
                    )
            else:
                twists[arm] = measured.copy()
                sources.add("measured")
        return twists, "+".join(sorted(sources))

    def _trim_action_candidates(
        self,
        candidates: list[dict[str, FloatArray]],
        discarded_waypoints: int,
    ) -> list[dict[str, FloatArray]]:
        if discarded_waypoints <= 0:
            return candidates
        trimmed: list[dict[str, FloatArray]] = []
        for candidate in candidates:
            _pose_actions, horizon = self.planner._pose_actions(candidate)
            if discarded_waypoints >= horizon:
                raise RuntimeError(
                    "GR00T action chunk became stale before asynchronous planning started: "
                    f"discard={discarded_waypoints}, horizon={horizon}"
                )
            trimmed.append(
                {
                    key: np.asarray(values[discarded_waypoints:], dtype=np.float64).copy()
                    for key, values in candidate.items()
                }
            )
        return trimmed

    def _scaled_action(self, key: str, values: FloatArray) -> FloatArray:
        factor = self.config.scaling.get(key)
        if factor is None:
            alternate = key[7:] if key.startswith("action.") else f"action.{key}"
            factor = self.config.scaling.get(alternate, 1.0)
        result = np.asarray(values, dtype=np.float64) * float(factor)
        if np.any(~np.isfinite(result)):
            raise ValueError(f"Scaled action {key!r} contains non-finite values")
        return result

    def _convert_obs(
        self,
        obs: dict[str, NDArray[Any]],
        task: str,
        batch_size: int = 1,
    ) -> defaultdict[str, dict[str, Any]]:
        converted: defaultdict[str, dict[str, Any]] = defaultdict(dict)
        local_velocity_keys = {
            key for key in self.config.velocity_state_keys.values() if key is not None
        }
        for key, value in obs.items():
            if key.startswith("observation.images"):
                image = self._convert_image(value)[np.newaxis, np.newaxis, :]
                converted["video"][key[19:]] = np.repeat(image, batch_size, axis=0)
            elif key.startswith("observation.state"):
                if (
                    not self.config.include_velocity_in_policy_observation
                    and key in local_velocity_keys
                ):
                    continue
                state = np.asarray(value, dtype=np.float32)[np.newaxis, np.newaxis, :]
                converted["state"][key[18:]] = np.repeat(state, batch_size, axis=0)
        converted["language"]["annotation.human.task_description"] = [
            [task] for _ in range(batch_size)
        ]
        return converted

    @staticmethod
    def _convert_image(value: NDArray[Any]) -> NDArray[np.uint8]:
        array = np.asarray(value)
        if array.ndim != 3:
            raise ValueError(f"Expected a three-dimensional image, got {array.shape}")
        if array.shape[0] in (1, 3, 4):
            array = array.transpose(1, 2, 0)
        elif array.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Expected a CHW or HWC image, got {array.shape}")
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.size:
                if float(np.nanmax(array)) <= 1.0:
                    array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    @classmethod
    def _response_to_action_candidates(cls, response: Any) -> list[dict[str, FloatArray]]:
        if isinstance(response, (list, tuple)):
            if not response:
                return []
            response = response[0]
        if not isinstance(response, dict):
            raise TypeError(f"Expected a GR00T action dictionary, got {type(response)!r}")
        arrays: dict[str, FloatArray] = {}
        batch_size: int | None = None
        horizon: int | None = None
        for key, value in response.items():
            array = cls._as_numpy(value)
            if array.ndim < 2:
                raise ValueError(
                    f"Expected action {key!r} with shape (batch, chunk, ...), got {array.shape}"
                )
            if batch_size is None:
                batch_size = int(array.shape[0])
            elif array.shape[0] != batch_size:
                raise ValueError(f"Mismatched action batch size for {key!r}")
            if horizon is None:
                horizon = int(array.shape[1])
            elif array.shape[1] != horizon:
                raise ValueError(f"Mismatched action horizon for {key!r}")
            action_key = key if key.startswith("action.") else f"action.{key}"
            arrays[action_key] = np.asarray(array, dtype=np.float64)
        if batch_size is None:
            return []
        return [
            {key: values[index].copy() for key, values in arrays.items()}
            for index in range(batch_size)
        ]

    @staticmethod
    def _as_numpy(value: Any) -> NDArray[Any]:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)


def _sample_continuous_trajectory(
    trajectory: BimanualContinuousTrajectory,
    times: FloatArray,
) -> ActionDict:
    config = trajectory.config
    raw_times = times / trajectory.time_scale
    path_s, raw_path_velocity, raw_path_acceleration = _path_parameter_at_time(
        trajectory.gridpoints,
        trajectory.path_speeds,
        trajectory.source_times,
        raw_times,
    )
    path_velocity = raw_path_velocity / trajectory.time_scale
    path_acceleration = raw_path_acceleration / trajectory.time_scale**2
    coordinates = np.asarray(trajectory.path(path_s, 0), dtype=np.float64)
    coordinate_velocity = (
        np.asarray(trajectory.path(path_s, 1), dtype=np.float64) * path_velocity[:, np.newaxis]
    )
    coordinate_acceleration = (
        np.asarray(trajectory.path(path_s, 2), dtype=np.float64) * path_velocity[:, np.newaxis] ** 2
        + np.asarray(trajectory.path(path_s, 1), dtype=np.float64)
        * path_acceleration[:, np.newaxis]
    )
    result: ActionDict = {
        config.time_output_key: times.copy(),
        config.duration_output_key: np.full(len(times), trajectory.duration, dtype=np.float64),
        config.sequence_output_key: np.full(len(times), trajectory.sequence_id, dtype=np.int64),
    }
    boundary = (times <= 1e-12) | (times >= trajectory.duration - 1e-12)
    for arm_index, arm in enumerate(("left", "right")):
        arm_slice = slice(6 * arm_index, 6 * (arm_index + 1))
        arm_coordinates = coordinates[:, arm_slice]
        arm_velocity = coordinate_velocity[:, arm_slice]
        arm_acceleration = coordinate_acceleration[:, arm_slice]
        rotations, angular_velocity, angular_acceleration = _rotation_kinematics(
            trajectory.start_rotations[arm],
            arm_coordinates[:, 3:],
            arm_velocity[:, 3:],
            arm_acceleration[:, 3:],
        )
        quaternion = _continuous_wxyz_quaternion(rotations)
        result[config.pose_output_keys[arm]] = np.column_stack(
            (arm_coordinates[:, :3], quaternion)
        ).astype(np.float32)
        result[config.velocity_output_keys[arm]] = np.column_stack(
            (arm_velocity[:, :3], angular_velocity)
        ).astype(np.float32)
        result[config.acceleration_output_keys[arm]] = np.column_stack(
            (arm_acceleration[:, :3], angular_acceleration)
        ).astype(np.float32)
        if config.inference_mode == "sync":
            result[config.acceleration_output_keys[arm]][boundary] = 0.0
    action_s = trajectory.waypoint_s[1:]
    indices = np.searchsorted(action_s, path_s, side="left")
    indices = np.clip(indices, 0, len(action_s) - 1)
    for key, values in trajectory.auxiliary_actions.items():
        result[key] = np.asarray(values[indices], dtype=np.float32)
    return result


def _path_parameter_at_time(
    gridpoints: FloatArray,
    path_speeds: FloatArray,
    time_grid: FloatArray,
    times: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    indices = np.searchsorted(time_grid, times, side="right") - 1
    indices = np.clip(indices, 0, len(gridpoints) - 2)
    delta_s = np.diff(gridpoints)
    path_accelerations = 0.5 * np.diff(path_speeds**2) / delta_s
    local_time = times - time_grid[indices]
    velocity = path_speeds[indices] + local_time * path_accelerations[indices]
    position = (
        gridpoints[indices]
        + local_time * path_speeds[indices]
        + 0.5 * local_time**2 * path_accelerations[indices]
    )
    reached_end = times >= time_grid[-1]
    position[reached_end] = gridpoints[-1]
    velocity[reached_end] = path_speeds[-1]
    return position, velocity, path_accelerations[indices]


def _times_at_path_positions(
    gridpoints: FloatArray,
    path_speeds: FloatArray,
    time_grid: FloatArray,
    path_positions: FloatArray,
) -> FloatArray:
    """Invert the constant-acceleration TOPPRA time law at selected path positions."""

    positions = np.asarray(path_positions, dtype=np.float64)
    indices = np.searchsorted(gridpoints, positions, side="right") - 1
    indices = np.clip(indices, 0, len(gridpoints) - 2)
    local_distance = positions - gridpoints[indices]
    segment_distance = np.diff(gridpoints)[indices]
    start_speed = path_speeds[indices]
    end_speed = path_speeds[indices + 1]
    acceleration = 0.5 * (end_speed**2 - start_speed**2) / segment_distance
    local_squared_speed = np.maximum(
        start_speed**2 + 2 * acceleration * local_distance,
        0.0,
    )
    local_end_speed = np.sqrt(local_squared_speed)
    speed_sum = start_speed + local_end_speed
    if np.any((local_distance > 1e-12) & (speed_sum <= 1e-12)):
        raise RuntimeError("Cannot map a zero-speed TOPPRA segment to waypoint time")
    local_time = np.divide(
        2 * local_distance,
        speed_sum,
        out=np.zeros_like(local_distance),
        where=speed_sum > 1e-12,
    )
    return time_grid[indices] + local_time


def _rotation_kinematics(
    start_rotation: Rotation,
    rotation_vectors: FloatArray,
    rotation_vector_rates: FloatArray,
    rotation_vector_accelerations: FloatArray,
) -> tuple[Rotation, FloatArray, FloatArray]:
    relative = Rotation.from_rotvec(rotation_vectors)
    angular_velocity = np.empty_like(rotation_vector_rates)
    angular_acceleration = np.empty_like(rotation_vector_accelerations)
    for index, (phi, phi_dot, phi_ddot) in enumerate(
        zip(rotation_vectors, rotation_vector_rates, rotation_vector_accelerations, strict=True)
    ):
        jacobian, jacobian_dot = _so3_left_jacobian_and_derivative(phi, phi_dot)
        angular_velocity[index] = jacobian @ phi_dot
        angular_acceleration[index] = jacobian @ phi_ddot + jacobian_dot @ phi_dot
    return relative * start_rotation, angular_velocity, angular_acceleration


def _so3_left_jacobian_and_derivative(
    rotation_vector: FloatArray,
    rotation_vector_rate: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    angle = float(np.linalg.norm(rotation_vector))
    skew = _skew(rotation_vector)
    skew_rate = _skew(rotation_vector_rate)
    if angle < 1e-6:
        angle_rate_product = float(np.dot(rotation_vector, rotation_vector_rate))
        coefficient_a = 0.5 - angle**2 / 24 + angle**4 / 720
        coefficient_b = 1 / 6 - angle**2 / 120 + angle**4 / 5040
        coefficient_a_dot = angle_rate_product * (-1 / 12 + angle**2 / 180)
        coefficient_b_dot = angle_rate_product * (-1 / 60 + angle**2 / 1260)
    else:
        angle_rate = float(np.dot(rotation_vector, rotation_vector_rate)) / angle
        coefficient_a = (1 - math.cos(angle)) / angle**2
        coefficient_b = (angle - math.sin(angle)) / angle**3
        coefficient_a_dot = (
            (angle * math.sin(angle) - 2 * (1 - math.cos(angle))) / angle**3
        ) * angle_rate
        coefficient_b_dot = (
            (angle * (1 - math.cos(angle)) - 3 * (angle - math.sin(angle))) / angle**4
        ) * angle_rate
    jacobian = np.eye(3) + coefficient_a * skew + coefficient_b * skew @ skew
    jacobian_dot = (
        coefficient_a_dot * skew
        + coefficient_a * skew_rate
        + coefficient_b_dot * skew @ skew
        + coefficient_b * (skew_rate @ skew + skew @ skew_rate)
    )
    return jacobian, jacobian_dot


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _pose_rotation(pose: FloatArray) -> Rotation:
    return Rotation.from_quat(pose[[4, 5, 6, 3]])


def _attribute_or_none(instance: Any | None, attribute: str) -> Any | None:
    return None if instance is None else getattr(instance, attribute)


def _continuous_wxyz_quaternion(rotations: Rotation) -> FloatArray:
    xyzw = rotations.as_quat()
    for index in range(1, len(xyzw)):
        if np.dot(xyzw[index - 1], xyzw[index]) < 0:
            xyzw[index] *= -1
    return xyzw[:, [3, 0, 1, 2]]


class GR00TAgent(GR00TBimanualToppraAgent, Agent, GUIModule):
    """Adapter exposing the same framework lifecycle as the provided ``example.py``.

    The external ``Agent`` and ``GUIModule`` implementations are not part of Isaac-GR00T. When
    they are installed, this class inherits the real implementations. Otherwise the same rollout
    remains usable through standalone fallback bases.
    """

    _GUI_TEXT = (
        ("sequence", "Sequence: ---"),
        ("latency", "Latency: ---"),
        ("duration", "Duration: ---"),
        ("delay", "Offset: ---"),
        ("refills", "Refills: 0"),
        ("status", "Status: waiting"),
    )

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        fps: float = 30.0,
        refill_threshold: int = 20,
        max_latency_s: float = 0.12,
        buffer_ttl_s: float = 5.0,
        ignore_first_latency: bool = True,
        task_default: str = DEFAULT_TASK,
        timeout_ms: int = 15000,
        scaling: dict[str, float] | None = None,
        *,
        toppra_config: BimanualToppraRolloutConfig | None = None,
        left_limits: CartesianLimits | Mapping[str, Any] | None = None,
        right_limits: CartesianLimits | Mapping[str, Any] | None = None,
        inference_mode: Literal["async", "sync"] = "async",
        control_frequency: float = 250.0,
        refill_lead_s: float | None = None,
        scheduling_margin_s: float | None = None,
        handoff_margin_s: float | None = None,
        open_loop_horizon: int | None = None,
        tts_samples: int | None = None,
        tts_waypoint_count: int | None = None,
        async_terminal_path_velocity: float | None = None,
        left_velocity_state_key: str | None = None,
        right_velocity_state_key: str | None = None,
        velocity_filter: float | None = None,
        include_velocity_in_policy_observation: bool | None = None,
        trajectory_callback: TrajectoryCallback | None = None,
        policy_client: Any | None = None,
    ) -> None:
        GUIModule.__init__(self, "GR00T TOPPRA RTC", placement="top")

        self.buffer_ttl_s = float(buffer_ttl_s)
        self._gui_update_interval_s = 0.1
        self._last_gui_update_s = 0.0
        self._gui_prefix = f"gr00t_toppra_rtc_{id(self)}"
        self._gui_tags = {
            key: f"{self._gui_prefix}_{key}"
            for key in ("progress", *(key for key, _label in self._GUI_TEXT))
        }

        if toppra_config is None:
            parsed_left_limits = _coerce_cartesian_limits(
                DEFAULT_CARTESIAN_LIMITS if left_limits is None else left_limits,
                "left_limits",
            )
            parsed_right_limits = _coerce_cartesian_limits(
                DEFAULT_CARTESIAN_LIMITS if right_limits is None else right_limits,
                "right_limits",
            )
            frequency = float(fps)
            if frequency <= 0:
                raise ValueError("fps must be positive")
            control_hz = float(control_frequency)
            if control_hz <= 0:
                raise ValueError("control_frequency must be positive")
            threshold = max(0, int(refill_threshold))
            lead_s = threshold / control_hz if refill_lead_s is None else float(refill_lead_s)
            config = BimanualToppraRolloutConfig(
                left_limits=parsed_left_limits,
                right_limits=parsed_right_limits,
                policy_frequency=frequency,
                control_frequency=control_hz,
                inference_mode=inference_mode,
                refill_lead_s=lead_s,
                max_latency_s=float(max_latency_s),
                scheduling_margin_s=(
                    0.05 if scheduling_margin_s is None else float(scheduling_margin_s)
                ),
                handoff_margin_s=(0.05 if handoff_margin_s is None else float(handoff_margin_s)),
                open_loop_horizon=(8 if open_loop_horizon is None else int(open_loop_horizon)),
                ignore_first_latency=bool(ignore_first_latency),
                left_velocity_state_key=left_velocity_state_key,
                right_velocity_state_key=right_velocity_state_key,
                velocity_filter=0.35 if velocity_filter is None else float(velocity_filter),
                include_velocity_in_policy_observation=(
                    False
                    if include_velocity_in_policy_observation is None
                    else bool(include_velocity_in_policy_observation)
                ),
                tts_samples=1 if tts_samples is None else int(tts_samples),
                tts_waypoint_count=(5 if tts_waypoint_count is None else int(tts_waypoint_count)),
                async_terminal_path_velocity=(
                    1.0
                    if async_terminal_path_velocity is None
                    else float(async_terminal_path_velocity)
                ),
                task_default=str(task_default),
                scaling=dict(scaling or {}),
            )
        else:
            if left_limits is not None or right_limits is not None:
                raise ValueError("Pass either toppra_config or left_limits/right_limits, not both")
            config = toppra_config
            overrides = {
                "scaling": None if scaling is None else dict(scaling),
                "tts_samples": None if tts_samples is None else int(tts_samples),
                "scheduling_margin_s": (
                    None if scheduling_margin_s is None else float(scheduling_margin_s)
                ),
                "handoff_margin_s": (None if handoff_margin_s is None else float(handoff_margin_s)),
                "open_loop_horizon": (
                    None if open_loop_horizon is None else int(open_loop_horizon)
                ),
                "tts_waypoint_count": (
                    None if tts_waypoint_count is None else int(tts_waypoint_count)
                ),
                "async_terminal_path_velocity": (
                    None
                    if async_terminal_path_velocity is None
                    else float(async_terminal_path_velocity)
                ),
                "left_velocity_state_key": left_velocity_state_key,
                "right_velocity_state_key": right_velocity_state_key,
                "velocity_filter": None if velocity_filter is None else float(velocity_filter),
                "include_velocity_in_policy_observation": (
                    None
                    if include_velocity_in_policy_observation is None
                    else bool(include_velocity_in_policy_observation)
                ),
            }
            config = replace(
                config,
                **{name: value for name, value in overrides.items() if value is not None},
            )

        self.fps = config.policy_frequency
        self.dt = config.policy_dt
        self.control_frequency = config.control_frequency
        self.control_dt = config.control_dt
        self.refill_threshold = max(
            0,
            int(round(config.refill_lead_s * self.control_frequency)),
        )
        self.max_latency_s = config.max_latency_s
        self.ignore_first_latency = config.ignore_first_latency
        self.task_default = config.task_default
        self.scaling = config.scaling
        GR00TBimanualToppraAgent.__init__(
            self,
            config=config,
            host=host,
            port=port,
            timeout_ms=timeout_ms,
            trajectory_callback=trajectory_callback,
            policy_client=policy_client,
        )

    def diagnostics(self) -> dict[str, Any]:
        values = super().diagnostics()
        values.update(
            {
                "framework_agent_available": FRAMEWORK_AGENT_AVAILABLE,
                "framework_gui_available": FRAMEWORK_GUI_AVAILABLE,
            }
        )
        return values

    def setup_gui(self) -> None:
        import dearpygui.dearpygui as dpg

        with dpg.group(horizontal=True):
            dpg.add_text("TOPPRA trajectory:")
            dpg.add_progress_bar(
                default_value=0.0,
                width=300,
                overlay="",
                tag=self._gui_tags["progress"],
            )
            for index, (key, label) in enumerate(self._GUI_TEXT):
                if index:
                    dpg.add_spacer(width=12)
                dpg.add_text(label, tag=self._gui_tags[key])

    def update_gui(self) -> None:
        now_s = time.perf_counter()
        if now_s - self._last_gui_update_s < self._gui_update_interval_s:
            return
        self._last_gui_update_s = now_s

        values = self.diagnostics()
        elapsed_s = values["active_elapsed_s"]
        duration_s = values["active_duration_s"]
        progress = 0.0
        if elapsed_s is not None and duration_s is not None and duration_s > 0:
            progress = min(max(elapsed_s / duration_s, 0.0), 1.0)
        gui_values = {
            "progress": progress,
            "sequence": f"Sequence: {self._fmt(values['active_sequence_id'])}",
            "latency": f"Latency: {self._fmt(values['latest_latency_s'], milliseconds=True)}",
            "duration": f"Duration: {self._fmt(duration_s, seconds=True)}",
            "delay": (
                f"Offset: {self._fmt(values['latest_activation_offset_s'], milliseconds=True)}"
            ),
            "refills": f"Refills: {values['inference_count']}",
            "status": f"Status: {self._format_status(values)}",
        }
        for key, value in gui_values.items():
            self._set_gui_value(key, value)

    def _set_gui_value(self, key: str, value: Any) -> None:
        import dearpygui.dearpygui as dpg

        tag = self._gui_tags[key]
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, value)

    @staticmethod
    def _fmt(value: Any | None, *, milliseconds: bool = False, seconds: bool = False) -> str:
        if value is None:
            return "---"
        if milliseconds:
            return f"{float(value) * 1000.0:.1f} ms"
        if seconds:
            return f"{float(value):.3f} s"
        return str(value)

    @staticmethod
    def _format_status(values: dict[str, Any]) -> str:
        error = values["latest_error"]
        if error:
            one_line = str(error).replace("\n", " ")
            return one_line if len(one_line) <= 80 else one_line[:77] + "..."
        return str(values["execution_state"])


__all__ = [
    "BimanualCartesianToppraPlanner",
    "BimanualContinuousTrajectory",
    "BimanualToppraRolloutConfig",
    "CartesianLimits",
    "FRAMEWORK_AGENT_AVAILABLE",
    "FRAMEWORK_GUI_AVAILABLE",
    "GR00TAgent",
    "GR00TBimanualToppraAgent",
    "GR00TPolicyClient",
]
