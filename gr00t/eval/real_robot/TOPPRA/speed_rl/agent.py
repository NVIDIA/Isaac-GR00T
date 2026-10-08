from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import logging
import threading
import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ..eval_toppra_bimanual import (
    BimanualCartesianToppraPlanner,
    BimanualContinuousTrajectory,
    BimanualToppraRolloutConfig,
    CartesianLimits,
    GR00TAgent,
    GR00TBimanualToppraAgent,
    _InferenceRequest,
    _ReadyTrajectory,
)
from .config import TOPPRA_SPEED_VALUES, epsilon_for_decisions, validate_speed_values
from .contract import (
    FeatureContract,
    extract_candidate_features,
    pool_candidate_feature,
    trim_candidate_features,
)
from .learner import SpeedActor
from .network import validate_action_mask
from .phases import EpisodeOutcome
from .replay import Transition


LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
SPEED_REWARD_WEIGHT = 0.01
SPEED_REWARD_FREQUENCY_HZ = 30.0
TERMINAL_SUCCESS_BONUS = 100.0


def speedtuning_reward(
    speed: float,
    executed_motion_s: float,
    *,
    terminal: bool,
    safe_success: bool,
) -> tuple[float, float, float, float]:
    """Return total reward, speed reward, terminal bonus and 30 Hz-equivalent steps."""

    speed = float(speed)
    motion_s = float(executed_motion_s)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("reward speed must be finite and positive")
    if not np.isfinite(motion_s) or motion_s < 0.0:
        raise ValueError("executed motion time must be finite and non-negative")
    executed_steps = SPEED_REWARD_FREQUENCY_HZ * motion_s
    speed_reward = SPEED_REWARD_WEIGHT * executed_steps * speed**2
    terminal_bonus = TERMINAL_SUCCESS_BONUS if terminal and safe_success else 0.0
    return speed_reward + terminal_bonus, speed_reward, terminal_bonus, executed_steps


@dataclass
class _InferenceFeatureContext:
    request_generation: int
    episode_epoch: int
    action_mask: BoolArray
    features: list[FloatArray] | None = None
    planned_sequences: list[int] | None = None


class _SpeedPlannerRouter:
    """Delegate each candidate to the planner for its one fixed speed decision."""

    ARMS = ("left", "right")

    def __init__(self, agent: SpeedRLAgent) -> None:
        self.agent = agent
        self.planners = {
            index: BimanualCartesianToppraPlanner(
                replace(
                    agent.config,
                    left_limits=_limits_for_scale(scale),
                    right_limits=_limits_for_scale(scale),
                )
            )
            for index, scale in enumerate(agent.speed_values)
        }
        self.selection_planner = next(iter(self.planners.values()))
        self.config = agent.config

    def __getattr__(self, name: str) -> Any:
        return getattr(self.selection_planner, name)

    def select_tts_candidate(
        self,
        candidates: list[dict[str, NDArray[np.float64]]],
        current_pose: dict[str, NDArray[np.float64]],
        current_twist: dict[str, NDArray[np.float64]],
    ) -> tuple[int, tuple[float, ...]]:
        return self.selection_planner.select_tts_candidate(
            candidates,
            current_pose,
            current_twist,
        )

    def plan(
        self,
        action_arrays: dict[str, NDArray[np.float64]],
        current_pose: dict[str, NDArray[np.float64]],
        current_twist: dict[str, NDArray[np.float64]] | None = None,
        **kwargs: Any,
    ) -> BimanualContinuousTrajectory:
        context = self.agent._feature_context
        if context is None or context.features is None:
            raise RuntimeError("Speed-RL feature context is unavailable during TOPPRA planning")
        candidate_index = int(kwargs.get("tts_selected_index", -1))
        if not 0 <= candidate_index < len(context.features):
            raise RuntimeError(
                f"TTS candidate index {candidate_index} has no aligned Speed-RL feature"
            )
        candidate_feature = context.features[candidate_index]
        feature = pool_candidate_feature(candidate_feature)
        action_mask = context.action_mask.copy()
        action_mask &= self._handoff_feasibility_mask(current_twist)
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
        planner = self.planners[action_index]
        source_horizons = {len(np.asarray(values)) for values in action_arrays.values()}
        if len(source_horizons) != 1:
            raise ValueError("TOPPRA action fields must share one source horizon")
        source_horizon = next(iter(source_horizons))
        execution_horizon = self.agent.toppra_execution_horizon
        if source_horizon < execution_horizon:
            raise RuntimeError(
                "TOPPRA candidate is shorter than the configured execution horizon: "
                f"source={source_horizon}, execution={execution_horizon}"
            )
        executed_actions = {
            key: np.asarray(values)[:execution_horizon].copy()
            for key, values in action_arrays.items()
        }
        trajectory = planner.plan(
            executed_actions,
            current_pose,
            current_twist,
            **kwargs,
        )
        with self.agent._cv:
            if context.episode_epoch == self.agent._episode_epoch:
                self.agent.trajectory_decisions[trajectory.sequence_id] = {
                    "sequence_id": trajectory.sequence_id,
                    "request_generation": context.request_generation,
                    "feature": feature.copy(),
                    "action_index": action_index,
                    "speed_scale": self.agent.speed_values[action_index],
                    "action_mask": action_mask.copy(),
                    "policy_version": self.agent.speed_actor.policy_version,
                    "episode_epoch": context.episode_epoch,
                    "tts_candidate_index": candidate_index,
                    "trajectory_duration_s": float(trajectory.duration),
                    "used_toppra": bool(trajectory.used_toppra),
                    "execution_backend": "toppra",
                    "feature_horizon": len(candidate_feature),
                    "source_horizon": source_horizon,
                    "execution_horizon": execution_horizon,
                    "discarded_source_actions": source_horizon - execution_horizon,
                }
                assert context.planned_sequences is not None
                context.planned_sequences.append(trajectory.sequence_id)
        return trajectory

    def _handoff_feasibility_mask(
        self,
        current_twist: dict[str, NDArray[np.float64]] | None,
    ) -> BoolArray:
        mask = np.ones(len(self.agent.speed_values), dtype=np.bool_)
        if self.agent.config.inference_mode != "async" or current_twist is None:
            return mask
        for action_index, planner in self.planners.items():
            for arm in self.ARMS:
                twist = np.asarray(current_twist[arm], dtype=np.float64)
                velocity_limit = planner.config.limits[arm].velocity
                if twist.shape != (6,) or np.any(np.abs(twist) > velocity_limit + 1e-9):
                    mask[action_index] = False
                    break
        return mask


def _limits_for_scale(scale: float) -> CartesianLimits:
    scale = float(scale)
    return CartesianLimits(
        max_linear_velocity=(scale,) * 3,
        max_angular_velocity=(3.0 * scale,) * 3,
        max_linear_acceleration=(5.0 * scale**2,) * 3,
        max_angular_acceleration=(15.0 * scale**2,) * 3,
        safety_margin=1.0,
    )


class SpeedRLAgent(GR00TAgent):
    """Parallel Speed-RL agent that leaves the existing rollout implementation frozen."""

    execution_backend = "toppra"

    def __init__(
        self,
        *args: Any,
        feature_contract: FeatureContract,
        speed_actor: SpeedActor,
        greedy: bool = False,
        activated_decisions: int = 0,
        speed_values: Sequence[float] = TOPPRA_SPEED_VALUES,
        policy_chunk_horizon: int = 40,
        toppra_execution_horizon: int = 30,
        **kwargs: Any,
    ) -> None:
        if speed_actor.config.feature_dim != feature_contract.feature_dim:
            raise ValueError("Actor feature dimension does not match the server feature contract")
        self.feature_contract = feature_contract
        self.speed_actor = speed_actor
        self.speed_values = validate_speed_values(speed_values)
        if len(self.speed_values) != speed_actor.config.action_count:
            raise ValueError("Actor action_count does not match the configured speed values")
        self.policy_chunk_horizon = int(policy_chunk_horizon)
        self.toppra_execution_horizon = int(toppra_execution_horizon)
        if self.policy_chunk_horizon < 1:
            raise ValueError("policy_chunk_horizon must be positive")
        if not 1 <= self.toppra_execution_horizon <= self.policy_chunk_horizon:
            raise ValueError("TOPPRA execution horizon must lie inside the policy chunk")
        self._speed_lock = threading.Lock()
        self._feature_local = threading.local()
        self._episode_epoch = 0
        self._activated_decision_count = max(0, int(activated_decisions))
        self._action_mask = np.ones(len(self.speed_values), dtype=np.bool_)
        self._fixed_action: int | None = None
        self._greedy = bool(greedy)
        self.trajectory_decisions: dict[int, dict[str, Any]] = {}
        self._activation_events: list[dict[str, Any]] = []
        self._finished_episode_decisions: list[dict[str, Any]] = []
        self._discarded_decision_count = 0
        self.speed_actor.set_greedy(self._greedy)
        super().__init__(*args, **kwargs)
        self.config = replace(
            self.config,
            left_limits=_limits_for_scale(1.0),
            right_limits=_limits_for_scale(1.0),
        )
        self.planner = _SpeedPlannerRouter(self)

    @property
    def _feature_context(self) -> _InferenceFeatureContext | None:
        return getattr(self._feature_local, "context", None)

    def set_action_mask(self, action_mask: Sequence[bool]) -> None:
        mask = validate_action_mask(np.asarray(action_mask), len(self.speed_values))
        with self._speed_lock:
            self._action_mask = mask

    def set_fixed_action(self, action_index: int | None) -> None:
        if action_index is not None and not 0 <= int(action_index) < len(self.speed_values):
            raise ValueError("fixed Speed-RL action is out of range")
        with self._speed_lock:
            self._fixed_action = None if action_index is None else int(action_index)

    def set_greedy(self, greedy: bool) -> None:
        with self._speed_lock:
            self._greedy = bool(greedy)
            self.speed_actor.set_greedy(self._greedy)

    def _infer_trajectory(self, request: _InferenceRequest) -> _ReadyTrajectory:
        with self._speed_lock:
            action_mask = self._action_mask.copy()
        self._feature_local.context = _InferenceFeatureContext(
            request_generation=request.generation,
            episode_epoch=self._episode_epoch,
            action_mask=action_mask,
            planned_sequences=[],
        )
        try:
            ready = super()._infer_trajectory(request)
            context = self._feature_context
            assert context is not None
            with self._cv:
                if (
                    request.generation != self._request_generation
                    or context.episode_epoch != self._episode_epoch
                ):
                    discarded = self.trajectory_decisions.pop(
                        ready.trajectory.sequence_id,
                        None,
                    )
                    if discarded is not None:
                        self._discarded_decision_count += 1
            return ready
        except BaseException:
            context = self._feature_context
            if context is not None:
                with self._cv:
                    for sequence_id in context.planned_sequences or ():
                        discarded = self.trajectory_decisions.pop(sequence_id, None)
                        if discarded is not None:
                            self._discarded_decision_count += 1
            raise
        finally:
            self._feature_local.context = None

    def _response_to_action_candidates(
        self,
        response: Any,
    ) -> list[dict[str, NDArray[np.float64]]]:
        candidates = GR00TBimanualToppraAgent._response_to_action_candidates(response)
        horizons = tuple(len(next(iter(candidate.values()))) for candidate in candidates)
        if any(horizon != self.policy_chunk_horizon for horizon in horizons):
            raise ValueError(
                "Speed-RL policy chunk horizon does not match the configured contract: "
                f"expected={self.policy_chunk_horizon}, received={horizons}"
            )
        context = self._feature_context
        if context is None:
            raise RuntimeError("Policy response arrived without a Speed-RL request context")
        context.features = extract_candidate_features(
            response,
            self.feature_contract,
            horizons,
        )
        return candidates

    def _trim_action_candidates(
        self,
        candidates: list[dict[str, NDArray[np.float64]]],
        discarded_waypoints: int,
    ) -> list[dict[str, NDArray[np.float64]]]:
        trimmed = super()._trim_action_candidates(candidates, discarded_waypoints)
        context = self._feature_context
        if context is None or context.features is None:
            raise RuntimeError("Cannot trim actions without aligned Speed-RL features")
        context.features = trim_candidate_features(context.features, discarded_waypoints)
        for candidate, feature in zip(trimmed, context.features, strict=True):
            horizon = len(next(iter(candidate.values())))
            if len(feature) != horizon:
                raise RuntimeError(
                    "Speed-RL feature/action trim lost waypoint alignment: "
                    f"feature={len(feature)}, action={horizon}"
                )
        return trimmed

    def _activate_latest_trajectory_locked(self, *, min_generation: int = 0) -> bool:
        ready = self._ready_trajectory
        sequence_id = None if ready is None else ready.trajectory.sequence_id
        activated = super()._activate_latest_trajectory_locked(min_generation=min_generation)
        if sequence_id is None:
            return activated
        if activated:
            decision = self.trajectory_decisions.pop(sequence_id, None)
            if decision is not None and int(decision["episode_epoch"]) == self._episode_epoch:
                decision["activation_index"] = len(self._activation_events)
                decision["activated_monotonic_s"] = time.monotonic()
                decision["executed_motion_s"] = 0.0
                decision["executed_control_samples"] = 0
                self._activation_events.append(decision)
                self._activated_decision_count += 1
        elif self._ready_trajectory is None:
            discarded = self.trajectory_decisions.pop(sequence_id, None)
            if discarded is not None:
                self._discarded_decision_count += 1
        return activated

    def _pop_active_action_locked(self) -> dict[str, NDArray[Any]]:
        action = super()._pop_active_action_locked()
        if self._active_trajectory is None or self._active_last_sample is None:
            return action
        sequence_id = self._active_trajectory.sequence_id
        event = next(
            (
                item
                for item in reversed(self._activation_events)
                if item.get("sequence_id") == sequence_id
                and item.get("episode_epoch") == self._episode_epoch
            ),
            None,
        )
        if event is not None:
            event["executed_control_samples"] = int(event["executed_control_samples"]) + 1
            event["executed_motion_s"] = max(
                float(event["executed_motion_s"]),
                float(self._active_last_sample.time_from_start),
            )
        return action

    def finish_episode(self, outcome: EpisodeOutcome) -> list[Transition]:
        with self._cv:
            events = list(self._activation_events)
            self._activation_events.clear()
            self._finished_episode_decisions.clear()
            self._discarded_decision_count += len(self.trajectory_decisions)
            self.trajectory_decisions.clear()
            self._episode_epoch += 1
            self._request_generation += 1
            self._completed_generation = self._request_generation
            self._pending_request = None
            self._ready_trajectory = None
            self._refill_requested_sequence_id = None
            self._cv.notify_all()
        if not events:
            return []
        transitions: list[Transition] = []
        decision_records: list[dict[str, Any]] = []
        for index, event in enumerate(events):
            terminal = index == len(events) - 1
            next_event = events[index] if terminal else events[index + 1]
            reward, speed_reward, terminal_bonus, executed_30hz_steps = speedtuning_reward(
                float(event["speed_scale"]),
                float(event.get("executed_motion_s", 0.0)),
                terminal=terminal,
                safe_success=outcome.safe_success,
            )
            if not outcome.control_fault:
                transitions.append(
                    Transition(
                        state=np.asarray(event["feature"], dtype=np.float32).copy(),
                        action=int(event["action_index"]),
                        reward=reward,
                        next_state=np.asarray(next_event["feature"], dtype=np.float32).copy(),
                        done=terminal,
                        action_mask=np.asarray(event["action_mask"], dtype=np.bool_).copy(),
                        next_action_mask=np.asarray(
                            next_event["action_mask"],
                            dtype=np.bool_,
                        ).copy(),
                    )
                )
            feature = np.asarray(event["feature"], dtype=np.float32)
            decision_records.append(
                {
                    key: value
                    for key, value in event.items()
                    if key not in {"feature", "action_mask"}
                }
                | {
                    "action_mask": np.asarray(event["action_mask"], dtype=np.bool_).tolist(),
                    "feature_mean": float(np.mean(feature)),
                    "feature_std": float(np.std(feature)),
                    "feature_l2_norm": float(np.linalg.norm(feature)),
                    "executed_30hz_steps": executed_30hz_steps,
                    "speed_reward": speed_reward,
                    "terminal_bonus": terminal_bonus,
                    "reward": reward,
                    "terminal": terminal,
                    "safe_success": outcome.safe_success,
                    "excluded_from_replay": outcome.control_fault,
                }
            )
        with self._cv:
            self._finished_episode_decisions = decision_records
        return transitions

    def last_episode_decisions(self) -> list[dict[str, Any]]:
        with self._cv:
            return [dict(item) for item in self._finished_episode_decisions]

    def reset_execution(self) -> None:
        """Start a new epoch and invalidate producer results from the previous episode."""

        with self._cv:
            self._episode_epoch += 1
            self._discarded_decision_count += len(self.trajectory_decisions)
            self.trajectory_decisions.clear()
            self._activation_events.clear()
            self._finished_episode_decisions.clear()
            self._request_generation += 1
            self._completed_generation = self._request_generation
            self._pending_request = None
            self._ready_trajectory = None
            self._latest_trajectory = None
            self._active_trajectory = None
            self._active_action_buf = deque()
            self._active_last_sample = None
            self._active_initial_completed_waypoints = 0
            self._active_policy_discarded_waypoints = 0
            self._completed_waypoints_before_active = 0
            self._total_consumed_control_samples = 0
            self._hold_action = None
            self._hold_sequence_id = None
            self._refill_requested_sequence_id = None
            self._last_pinch = {
                self.config.left_pinch_key: np.zeros(1, dtype=np.float32),
                self.config.right_pinch_key: np.zeros(1, dtype=np.float32),
            }
            self._latest_obs = None
            self._latest_task = self.config.task_default
            self._latest_pose = None
            self._latest_measured_twist = {arm: None for arm in self.planner.ARMS}
            self._latest_commanded_twist_at_observation = {
                arm: np.zeros(6, dtype=np.float64) for arm in self.planner.ARMS
            }
            self._latest_latency_s = None
            self._latest_planning_latency_s = None
            self._latest_pipeline_latency_s = None
            self._latest_activation_offset_s = None
            self._latest_policy_discarded_waypoints = 0
            self._latest_planning_discarded_waypoints = 0
            self._latest_planning_discarded_control_samples = 0
            self._latest_twist_source = "zero"
            self._latest_handoff_control_count = None
            self._latest_handoff_position_error_m = None
            self._latest_handoff_rotation_error_rad = None
            self._latest_handoff_twist_error = None
            self._latest_error = None
            self._cv.notify_all()

    def diagnostics(self) -> dict[str, Any]:
        values = super().diagnostics()
        with self._speed_lock:
            action_mask = self._action_mask.copy()
            fixed_action = self._fixed_action
            greedy = self._greedy
        with self._cv:
            active_sequence = values.get("active_sequence_id")
            active_decision = (
                None
                if active_sequence is None
                else next(
                    (
                        item
                        for item in reversed(self._activation_events)
                        if item.get("episode_epoch") == self._episode_epoch
                        and item.get("sequence_id") == active_sequence
                    ),
                    None,
                )
            )
        values.update(
            {
                "speed_rl_execution_backend": self.execution_backend,
                "speed_rl_speed_values": self.speed_values,
                "speed_rl_episode_epoch": self._episode_epoch,
                "speed_rl_action_mask": action_mask.tolist(),
                "speed_rl_fixed_action": fixed_action,
                "speed_rl_greedy": greedy,
                "speed_rl_policy_version": self.speed_actor.policy_version,
                "speed_rl_activated_decisions": self._activated_decision_count,
                "speed_rl_pending_decisions": len(self.trajectory_decisions),
                "speed_rl_discarded_decisions": self._discarded_decision_count,
                "speed_rl_active_action_index": (
                    None if active_decision is None else active_decision["action_index"]
                ),
                "speed_rl_active_scale": (
                    None if active_decision is None else active_decision["speed_scale"]
                ),
            }
        )
        return values


def speed_planner_configs(
    base_config: BimanualToppraRolloutConfig,
    speed_values: Sequence[float] = TOPPRA_SPEED_VALUES,
) -> tuple[BimanualToppraRolloutConfig, ...]:
    return tuple(
        replace(
            base_config,
            left_limits=_limits_for_scale(scale),
            right_limits=_limits_for_scale(scale),
        )
        for scale in validate_speed_values(speed_values)
    )
