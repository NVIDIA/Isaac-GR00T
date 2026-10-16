from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..eval_toppra_bimanual import BimanualCartesianToppraPlanner, BimanualToppraRolloutConfig
from .agent import SpeedRLAgent
from .config import BASELINE_SPEED_VALUES, epsilon_for_decisions
from .contract import pool_candidate_feature
from .network import validate_action_mask


FloatArray = NDArray[np.float64]


def accelerated_sample_phases(
    horizon: int,
    speed: float,
    *,
    sample_count: int | None = None,
) -> FloatArray:
    """Map SpeedTuning action indices to phases of an incremental TCP path.

    SpeedTuning samples an absolute action chunk at indices ``0, speed, 2 * speed, ...``.  Our
    source phase zero is instead the measured TCP pose before the first delta, so action index zero
    maps to path phase one.  Clamping matches SpeedTuning's repeated final action when ``speed < 1``.
    """

    if horizon < 1:
        raise ValueError("action horizon must be positive")
    value = float(speed)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("speed must be finite and positive")
    if sample_count is None:
        action_indices = np.arange(0.0, float(horizon), value, dtype=np.float64)
        phases = np.minimum(action_indices + 1.0, float(horizon))
    else:
        count = int(sample_count)
        if count < 1:
            raise ValueError("sample_count must be positive")
        action_indices = np.arange(count, dtype=np.float64) * value
        if action_indices[-1] > horizon - 1 + 1e-12:
            raise ValueError(
                "speed and k_skip exceed the policy chunk: "
                f"last_index={action_indices[-1]:.6f}, horizon={horizon}"
            )
        phases = action_indices + 1.0
    if np.any(np.diff(phases) < 0.0):
        raise RuntimeError("accelerated sample phases must be nondecreasing")
    return phases


def _interpolate_rotations(
    rotations: list[Rotation],
    lower: NDArray[np.int64],
    upper: NDArray[np.int64],
    fraction: FloatArray,
) -> list[Rotation]:
    result: list[Rotation] = []
    for low, high, alpha in zip(lower, upper, fraction, strict=True):
        start = rotations[int(low)]
        if low == high:
            result.append(start)
            continue
        relative = rotations[int(high)] * start.inv()
        result.append(Rotation.from_rotvec(float(alpha) * relative.as_rotvec()) * start)
    return result


def resample_euler_delta_chunk(
    delta_actions: NDArray[Any],
    speed: float,
    *,
    euler_order: str = "xyz",
    degrees: bool = False,
    sample_count: int | None = None,
) -> FloatArray:
    """Temporally resample incremental TCP actions without linearly blending Euler angles.

    The input is ``[dx, dy, dz, droll, dpitch, dyaw]``.  Deltas are first accumulated into a
    relative SE(3) path.  Translation is interpolated linearly and orientation follows the
    shortest rotation-space arc.  The sampled path is then converted back to Euler increments so
    it retains the checkpoint's action representation.
    """

    actions = np.asarray(delta_actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 6 or not len(actions):
        raise ValueError(f"delta actions must have shape (horizon, 6), got {actions.shape}")
    if np.any(~np.isfinite(actions)):
        raise ValueError("delta actions must be finite")
    try:
        Rotation.from_euler(euler_order, np.zeros(3), degrees=degrees)
    except ValueError as exc:
        raise ValueError(f"invalid Euler order {euler_order!r}") from exc

    position_knots = np.vstack(
        (np.zeros((1, 3), dtype=np.float64), np.cumsum(actions[:, :3], axis=0))
    )
    rotation_knots = [Rotation.identity()]
    rotation = Rotation.identity()
    for values in actions[:, 3:]:
        rotation = Rotation.from_euler(euler_order, values, degrees=degrees) * rotation
        rotation_knots.append(rotation)

    phases = accelerated_sample_phases(len(actions), speed, sample_count=sample_count)
    lower = np.floor(phases).astype(np.int64)
    upper = np.ceil(phases).astype(np.int64)
    fraction = phases - lower
    sampled_positions = (
        position_knots[lower] * (1.0 - fraction[:, np.newaxis])
        + position_knots[upper] * fraction[:, np.newaxis]
    )
    sampled_rotations = _interpolate_rotations(rotation_knots, lower, upper, fraction)

    result = np.empty((len(phases), 6), dtype=np.float64)
    previous_position = np.zeros(3, dtype=np.float64)
    previous_rotation = Rotation.identity()
    for index, (position, sampled_rotation) in enumerate(
        zip(sampled_positions, sampled_rotations, strict=True)
    ):
        result[index, :3] = position - previous_position
        delta_rotation = sampled_rotation * previous_rotation.inv()
        result[index, 3:] = delta_rotation.as_euler(euler_order, degrees=degrees)
        previous_position = position
        previous_rotation = sampled_rotation
    return result


def _resample_auxiliary(
    values: NDArray[Any],
    phases: FloatArray,
    initial_value: NDArray[Any],
) -> FloatArray:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim < 1 or len(source) < 1 or np.any(~np.isfinite(source)):
        raise ValueError("auxiliary action must have a finite nonempty horizon")
    initial = np.asarray(initial_value, dtype=np.float64)
    if initial.shape != source.shape[1:] or np.any(~np.isfinite(initial)):
        raise ValueError("initial auxiliary action shape does not match the chunk")
    knots = np.concatenate((initial[np.newaxis], source), axis=0)
    lower = np.floor(phases).astype(np.int64)
    upper = np.ceil(phases).astype(np.int64)
    fraction = phases - lower
    reshape = (len(fraction),) + (1,) * (source.ndim - 1)
    alpha = fraction.reshape(reshape)
    return knots[lower] * (1.0 - alpha) + knots[upper] * alpha


def _integrate_from_pose(
    start_pose: FloatArray,
    delta_actions: FloatArray,
    *,
    euler_order: str,
    degrees: bool,
) -> FloatArray:
    position = np.asarray(start_pose[:3], dtype=np.float64).copy()
    rotation = Rotation.from_quat(np.asarray(start_pose[[4, 5, 6, 3]], dtype=np.float64))
    previous_wxyz = np.asarray(start_pose[3:], dtype=np.float64).copy()
    poses = np.empty((len(delta_actions), 7), dtype=np.float64)
    for index, action in enumerate(delta_actions):
        position = position + action[:3]
        rotation = Rotation.from_euler(euler_order, action[3:], degrees=degrees) * rotation
        xyzw = rotation.as_quat()
        wxyz = xyzw[[3, 0, 1, 2]]
        if float(np.dot(previous_wxyz, wxyz)) < 0.0:
            wxyz = -wxyz
        poses[index] = np.concatenate((position, wxyz))
        previous_wxyz = wxyz
    return poses


@dataclass(frozen=True)
class InterpolatedBimanualTrajectory:
    """Piecewise-constant 30 Hz target chunk republished by the 250 Hz Servo loop."""

    config: BimanualToppraRolloutConfig
    sequence_id: int
    target_poses: dict[str, FloatArray]
    auxiliary_actions: dict[str, FloatArray]
    action_frequency: float
    source_horizon: int
    speed_scale: float
    inference_started_at_s: float
    created_at_monotonic_s: float
    tts_selected_index: int = 0
    tts_candidate_scores: tuple[float, ...] = (0.0,)
    used_toppra: bool = False
    fallback_reason: str = "speedtuning_interpolation_baseline"
    initial_twist_scale: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.action_frequency) or self.action_frequency <= 0.0:
            raise ValueError("action_frequency must be finite and positive")
        if self.source_horizon < 1:
            raise ValueError("source_horizon must be positive")
        lengths = {len(values) for values in self.target_poses.values()}
        lengths.update(len(values) for values in self.auxiliary_actions.values())
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
            raise ValueError("all interpolated trajectory fields must share a nonempty horizon")
        for arm, values in self.target_poses.items():
            array = np.asarray(values)
            if array.ndim != 2 or array.shape[1] != 7 or np.any(~np.isfinite(array)):
                raise ValueError(f"{arm} target poses must have finite shape (horizon, 7)")

    @property
    def resampled_horizon(self) -> int:
        return len(next(iter(self.target_poses.values())))

    @property
    def duration(self) -> float:
        return self.resampled_horizon / self.action_frequency

    @property
    def waypoint_times(self) -> FloatArray:
        return np.arange(1, self.resampled_horizon + 1, dtype=np.float64) / self.action_frequency

    @property
    def terminal_path_speed(self) -> float:
        return 0.0

    def sample(
        self,
        time_from_start: float | FloatArray,
        *,
        clamp: bool = False,
    ) -> dict[str, NDArray[Any]]:
        scalar = np.asarray(time_from_start).ndim == 0
        times = np.atleast_1d(np.asarray(time_from_start, dtype=np.float64))
        if times.ndim != 1 or np.any(~np.isfinite(times)):
            raise ValueError("time_from_start must be a finite scalar or one-dimensional array")
        if len(times) > 1 and np.any(np.diff(times) < 0.0):
            raise ValueError("vector sample times must be nondecreasing")
        if clamp:
            times = np.clip(times, 0.0, self.duration)
        elif np.any(times < 0.0) or np.any(times > self.duration):
            raise ValueError(f"sample time must lie in [0, {self.duration:.9f}]")
        indices = np.floor(times * self.action_frequency + 1e-12).astype(np.int64)
        indices = np.clip(indices, 0, self.resampled_horizon - 1)
        result: dict[str, NDArray[Any]] = {
            self.config.time_output_key: times.copy(),
            self.config.duration_output_key: np.full(len(times), self.duration),
            self.config.sequence_output_key: np.full(len(times), self.sequence_id, dtype=np.int64),
        }
        for arm in ("left", "right"):
            pose_key = self.config.pose_output_keys[arm]
            result[pose_key] = np.asarray(self.target_poses[arm][indices], dtype=np.float32)
            result[self.config.velocity_output_keys[arm]] = np.zeros(
                (len(times), 6), dtype=np.float32
            )
            result[self.config.acceleration_output_keys[arm]] = np.zeros(
                (len(times), 6), dtype=np.float32
            )
        for key, values in self.auxiliary_actions.items():
            result[key] = np.asarray(values[indices], dtype=np.float32)
        if scalar:
            return {key: np.asarray(value[0]).copy() for key, value in result.items()}
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "duration": self.duration,
            "source_horizon": self.source_horizon,
            "resampled_horizon": self.resampled_horizon,
            "speed_scale": self.speed_scale,
            "action_frequency": self.action_frequency,
            "inference_started_at_s": self.inference_started_at_s,
            "created_at_monotonic_s": self.created_at_monotonic_s,
            "pipeline_latency_s": self.created_at_monotonic_s - self.inference_started_at_s,
            "used_toppra": self.used_toppra,
            "fallback_reason": self.fallback_reason,
            "tts_selected_index": self.tts_selected_index,
            "tts_candidate_scores": self.tts_candidate_scores,
        }


class _InterpolationPlannerRouter:
    ARMS = ("left", "right")

    def __init__(self, agent: InterpolationSpeedRLAgent) -> None:
        self.agent = agent
        self.validation_planner = BimanualCartesianToppraPlanner(agent.config)
        self.config = agent.config

    def __getattr__(self, name: str) -> Any:
        return getattr(self.validation_planner, name)

    def select_tts_candidate(
        self,
        candidates: list[dict[str, FloatArray]],
        _current_pose: dict[str, FloatArray],
        _current_twist: dict[str, FloatArray],
    ) -> tuple[int, tuple[float, ...]]:
        if len(candidates) != 1:
            raise ValueError("the interpolation baseline requires exactly one policy candidate")
        return 0, (0.0,)

    def plan(
        self,
        action_arrays: dict[str, FloatArray],
        current_pose: dict[str, FloatArray],
        _current_twist: dict[str, FloatArray] | None = None,
        **kwargs: Any,
    ) -> InterpolatedBimanualTrajectory:
        context = self.agent._feature_context
        if context is None or context.features is None:
            raise RuntimeError("Speed-RL feature context is unavailable during interpolation")
        candidate_index = int(kwargs.get("tts_selected_index", -1))
        if not 0 <= candidate_index < len(context.features):
            raise RuntimeError(f"candidate index {candidate_index} has no aligned Speed-RL feature")
        feature = pool_candidate_feature(context.features[candidate_index])
        action_mask = context.action_mask.copy()
        with self.agent._speed_lock:
            action_mask &= self.agent._action_mask
            epsilon = (
                0.0
                if self.agent._greedy
                else epsilon_for_decisions(self.agent._activated_decision_count)
            )
            fixed_action = self.agent._fixed_action
        validate_action_mask(action_mask, len(self.agent.speed_values))
        action_index = self.agent.speed_actor.select(
            feature,
            action_mask,
            epsilon=epsilon,
            fixed_action=fixed_action,
        )
        speed = self.agent.speed_values[action_index]
        poses = self.validation_planner._validated_poses(current_pose)
        pose_actions, source_horizon = self.validation_planner._pose_actions(action_arrays)
        phases = accelerated_sample_phases(
            source_horizon,
            speed,
            sample_count=self.agent.baseline_k_skip,
        )
        resampled_deltas = {
            arm: resample_euler_delta_chunk(
                pose_actions[arm],
                speed,
                euler_order=self.config.euler_order,
                degrees=self.config.degrees,
                sample_count=self.agent.baseline_k_skip,
            )
            for arm in self.ARMS
        }
        target_poses = {
            arm: _integrate_from_pose(
                poses[arm],
                resampled_deltas[arm],
                euler_order=self.config.euler_order,
                degrees=self.config.degrees,
            )
            for arm in self.ARMS
        }
        pose_keys = set(self.config.action_keys.values())
        auxiliary: dict[str, FloatArray] = {}
        for key, values in action_arrays.items():
            if key in pose_keys:
                continue
            source = np.asarray(values, dtype=np.float64)
            if len(source) != source_horizon:
                raise ValueError(f"auxiliary action {key!r} does not match the pose horizon")
            initial = self.agent._last_pinch.get(key)
            if initial is None:
                initial = source[0]
            auxiliary[key] = _resample_auxiliary(source, phases, initial)

        now = time.perf_counter()
        trajectory = InterpolatedBimanualTrajectory(
            config=self.config,
            sequence_id=int(kwargs.get("sequence_id", 0)),
            target_poses=target_poses,
            auxiliary_actions=auxiliary,
            action_frequency=self.agent.baseline_action_frequency,
            source_horizon=source_horizon,
            speed_scale=speed,
            inference_started_at_s=float(kwargs.get("inference_started_at_s", now)),
            created_at_monotonic_s=now,
            tts_selected_index=candidate_index,
            tts_candidate_scores=tuple(kwargs.get("tts_candidate_scores", (0.0,))),
        )
        with self.agent._cv:
            if context.episode_epoch == self.agent._episode_epoch:
                self.agent.trajectory_decisions[trajectory.sequence_id] = {
                    "sequence_id": trajectory.sequence_id,
                    "request_generation": context.request_generation,
                    "feature": feature.copy(),
                    "action_index": action_index,
                    "speed_scale": speed,
                    "action_mask": action_mask.copy(),
                    "policy_version": self.agent.speed_actor.policy_version,
                    "episode_epoch": context.episode_epoch,
                    "tts_candidate_index": candidate_index,
                    "trajectory_duration_s": trajectory.duration,
                    "used_toppra": False,
                    "execution_backend": "interpolation",
                    "feature_horizon": len(context.features[candidate_index]),
                    "source_horizon": trajectory.source_horizon,
                    "execution_horizon": trajectory.resampled_horizon,
                    "full_resampled_horizon": len(
                        accelerated_sample_phases(trajectory.source_horizon, speed)
                    ),
                    "discarded_resampled_actions": len(
                        accelerated_sample_phases(trajectory.source_horizon, speed)
                    )
                    - trajectory.resampled_horizon,
                    "last_source_action_index": (trajectory.resampled_horizon - 1) * speed,
                    "resampled_horizon": trajectory.resampled_horizon,
                    "action_frequency_hz": trajectory.action_frequency,
                }
                assert context.planned_sequences is not None
                context.planned_sequences.append(trajectory.sequence_id)
        return trajectory


class InterpolationSpeedRLAgent(SpeedRLAgent):
    """SpeedTuning-style comparison backend using 30 Hz SE(3) chunk interpolation."""

    execution_backend = "interpolation"

    def __init__(
        self,
        *args: Any,
        baseline_action_frequency: float = 30.0,
        baseline_k_skip: int = 10,
        **kwargs: Any,
    ) -> None:
        frequency = float(baseline_action_frequency)
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("baseline_action_frequency must be finite and positive")
        if int(baseline_k_skip) < 1:
            raise ValueError("baseline_k_skip must be positive")
        self.baseline_action_frequency = frequency
        self.baseline_k_skip = int(baseline_k_skip)
        kwargs.setdefault("speed_values", BASELINE_SPEED_VALUES)
        super().__init__(*args, **kwargs)
        if self.config.inference_mode != "sync":
            self.teardown()
            raise ValueError("the interpolation baseline currently supports sync inference only")
        if self.config.tts_samples != 1:
            self.teardown()
            raise ValueError("the interpolation baseline requires tts_samples=1")
        self.planner = _InterpolationPlannerRouter(self)

    def diagnostics(self) -> dict[str, Any]:
        values = super().diagnostics()
        values.update(
            {
                "speed_rl_execution_backend": self.execution_backend,
                "baseline_action_frequency_hz": self.baseline_action_frequency,
                "baseline_k_skip": self.baseline_k_skip,
            }
        )
        return values
