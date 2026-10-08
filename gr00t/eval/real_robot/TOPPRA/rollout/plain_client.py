from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from enum import Enum
import json
import logging
import math
from pathlib import Path
from queue import Empty, Queue
import shlex
import threading
import time
from typing import Any

import numpy as np

from ..eval_toppra_bimanual import DEFAULT_TASK, CartesianLimits, GR00TAgent, GR00TPolicyClient
from ..kion_client.client import (
    LEFT_TARGET_KEY,
    LEFT_TWIST_KEY,
    RIGHT_TARGET_KEY,
    RIGHT_TWIST_KEY,
    KionDualArmServo,
    KionTopics,
    RosTypes,
    load_ros_types,
)
from ..kion_client.tracking import TrackingRecorder, make_tracking_row
from .hardware_safety import (
    TargetSafetyError,
    TargetSafetyGuard,
    parse_workspace_bounds,
    require_real_policy_cameras,
)
from .observation import RolloutObservationBuffer, policy_observation, rollout_staleness
from .operator import RosEpisodeBridge, restore_console_logging
from .pinch import KionPinchExecutor
from .reset import DUAL_ARM_HOME_SERVICE, EpisodeResetController


LOGGER = logging.getLogger(__name__)
LEFT_PINCH_KEY = "action.left_pinch"
RIGHT_PINCH_KEY = "action.right_pinch"


class EpisodeState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    TERMINATED = "terminated"
    ABORTED = "aborted"
    RESETTING = "resetting"
    CLOSED = "closed"


class PlainKionEpisodeClient:
    """Episode-driven plain TOPPRA client controlled by terminal and/or ROS services."""

    def __init__(
        self,
        *,
        ros_types: RosTypes,
        topics: KionTopics,
        agent_factory: Callable[[], GR00TAgent],
        log_root: Path,
        control_frequency: float,
        servo_gain: int,
        startup_timeout_s: float,
        max_state_age_s: float,
        max_image_age_s: float,
        task: str,
        inference_mode: str,
        dry_run: bool,
        target_safety: TargetSafetyGuard | None = None,
        episode_duration_s: float = 80.0,
        control_interface: str = "gui",
        ros_namespace: str = "/gr00t_rollout",
        pinch_enabled: bool = True,
        pinch_max_rate_hz: float = 30.0,
        reset_mode: str = "manual",
        reset_service: str = DUAL_ARM_HOME_SERVICE,
        reset_timeout_s: float = 10.0,
    ) -> None:
        self.ros_types = ros_types
        self.rospy = ros_types.rospy
        self.topics = topics
        self.agent_factory = agent_factory
        self.log_root = Path(log_root)
        self.control_frequency = float(control_frequency)
        self.control_period_ns = int(round(1e9 / self.control_frequency))
        self.servo_gain = int(servo_gain)
        self.startup_timeout_s = float(startup_timeout_s)
        self.max_state_age_s = float(max_state_age_s)
        self.max_image_age_s = float(max_image_age_s)
        self.task = task
        self.inference_mode = inference_mode
        self.dry_run = bool(dry_run)
        if not self.dry_run and target_safety is None:
            raise ValueError("Real motion requires a TargetSafetyGuard")
        self.target_safety = target_safety
        self.episode_duration_s = float(episode_duration_s)
        self.control_interface = control_interface
        self.ros_namespace = ros_namespace
        self.pinch_enabled = bool(pinch_enabled)
        self.pinch_max_rate_hz = float(pinch_max_rate_hz)
        self.reset_controller = EpisodeResetController(
            self.rospy,
            mode=reset_mode,
            service_name=reset_service,
            service_timeout_s=reset_timeout_s,
            dry_run=self.dry_run,
        )

        self.observations = RolloutObservationBuffer(ros_types, topics)
        self.agent: GR00TAgent | None = None
        self.servo: KionDualArmServo | None = None
        self.pinch: KionPinchExecutor | None = None
        self.recorder: TrackingRecorder | None = None
        self.state = EpisodeState.IDLE
        self.episode_number = 0
        self._episode_started_s: float | None = None
        self._last_target: dict[str, np.ndarray] | None = None
        self._sample_index = 0
        self._commands: Queue[str] = Queue()
        self._shutdown = threading.Event()
        self._terminal_thread: threading.Thread | None = None
        self._ros_bridge: RosEpisodeBridge | None = None
        self._finalizer: threading.Thread | None = None
        self._finalization_error: str | None = None
        self._last_outcome: str | None = None
        self._last_safe_success: bool | None = None
        self._tracking_log: str | None = None
        self._last_command_error: str | None = None
        self._stale_fields: tuple[str, ...] = ()
        self._critical_stale_fields: tuple[str, ...] = ()
        self._twist_stale = True
        self._hardware_safety_error: str | None = None

    def enqueue_command(self, command: str) -> None:
        self._commands.put(command)

    def run(self) -> None:
        try:
            self.observations.wait_until_ready(self.startup_timeout_s)
            self._start_control_interfaces()
            self._run_control_loop()
        except BaseException:
            if self.state == EpisodeState.RUNNING:
                try:
                    self._finish_episode(
                        success=False,
                        reason="control-loop fault",
                        control_fault=True,
                    )
                except BaseException:
                    LOGGER.exception("Failed to finalize the control-fault episode")
            raise
        finally:
            self.close()

    def _start_control_interfaces(self) -> None:
        if self.control_interface in {"terminal", "both"}:
            self._terminal_thread = threading.Thread(
                target=self._terminal_loop,
                daemon=True,
                name="ToppraTerminal",
            )
            self._terminal_thread.start()
        if self.control_interface in {"gui", "both"}:
            self._ros_bridge = RosEpisodeBridge(
                self.rospy,
                self.enqueue_command,
                self.status,
                namespace=self.ros_namespace,
            )
            self._ros_bridge.publish_status(force=True)
        LOGGER.info(
            "Plain TOPPRA episode client ready: interface=%s namespace=%s",
            self.control_interface,
            self.ros_namespace,
        )

    def _run_control_loop(self) -> None:
        next_tick_ns = time.monotonic_ns()
        last_deadline_warning_ns = 0
        while not self.rospy.is_shutdown() and not self._shutdown.is_set():
            loop_started_ns = time.monotonic_ns()
            snapshot = self.observations.snapshot(
                max_state_age_s=self.max_state_age_s,
                max_image_age_s=self.max_image_age_s,
            )
            critical_stale, twist_stale = rollout_staleness(snapshot.stale_fields)
            self._stale_fields = snapshot.stale_fields
            self._critical_stale_fields = critical_stale
            self._twist_stale = twist_stale
            self._drain_commands()
            if (
                self.state == EpisodeState.RUNNING
                and self._episode_started_s is not None
                and time.monotonic() - self._episode_started_s >= self.episode_duration_s
            ):
                self._finish_episode(success=False, reason="episode timeout")

            target_left: np.ndarray | None = None
            target_right: np.ndarray | None = None
            diagnostics: dict[str, Any] = {
                "execution_state": self.state.value,
                "active_sequence_id": None,
                "active_buffer_size": 0,
            }
            if self.state == EpisodeState.RUNNING:
                if self.pinch is not None:
                    self.pinch.raise_if_failed()
                if critical_stale:
                    target_left, target_right = self._hold_target(snapshot)
                    diagnostics["execution_state"] = "critical_observation_stale_hold"
                else:
                    assert self.agent is not None
                    action = self.agent.act(
                        policy_observation(
                            snapshot.observation,
                            twist_stale=twist_stale,
                        ),
                        self.task,
                    )
                    target_left = np.asarray(action[LEFT_TARGET_KEY], dtype=np.float64)
                    target_right = np.asarray(action[RIGHT_TARGET_KEY], dtype=np.float64)
                    if self.target_safety is not None:
                        try:
                            self.target_safety.validate(
                                target_left,
                                target_right,
                                snapshot.left_pose,
                                snapshot.right_pose,
                            )
                        except TargetSafetyError as exc:
                            self._hardware_safety_error = str(exc)
                            LOGGER.error("Hardware target safety abort: %s", exc)
                            self._abort_episode(reason=f"target safety: {exc}")
                            target_left = None
                            target_right = None
                            diagnostics["execution_state"] = "target_safety_abort"
                            continue
                    self._last_target = {
                        "left": target_left.copy(),
                        "right": target_right.copy(),
                    }
                    if self.pinch is not None:
                        self.pinch.publish(action[LEFT_PINCH_KEY], action[RIGHT_PINCH_KEY])
                    diagnostics = self.agent.diagnostics()
            elif self.state == EpisodeState.TERMINATED and self._last_target is not None:
                target_left = self._last_target["left"]
                target_right = self._last_target["right"]

            if self.servo is not None and target_left is not None and target_right is not None:
                published_monotonic_ns, published_ros_ns = self.servo.publish(
                    target_left,
                    target_right,
                )
                if self.recorder is not None:
                    self.recorder.record(
                        make_tracking_row(
                            sample_index=self._sample_index,
                            loop_started_monotonic_ns=loop_started_ns,
                            published_monotonic_ns=published_monotonic_ns,
                            published_ros_ns=published_ros_ns,
                            loop_duration_ns=time.monotonic_ns() - loop_started_ns,
                            deadline_lateness_ns=max(loop_started_ns - next_tick_ns, 0),
                            active_sequence_id=diagnostics.get("active_sequence_id"),
                            execution_state=str(
                                diagnostics.get("execution_state", self.state.value)
                            ),
                            active_buffer_size=int(diagnostics.get("active_buffer_size") or 0),
                            left_pose_ros_ns=snapshot.left_pose_ros_ns,
                            left_pose_received_monotonic_ns=(
                                snapshot.left_pose_received_monotonic_ns
                            ),
                            right_pose_ros_ns=snapshot.right_pose_ros_ns,
                            right_pose_received_monotonic_ns=(
                                snapshot.right_pose_received_monotonic_ns
                            ),
                            twist_ros_ns=snapshot.twist_ros_ns,
                            twist_received_monotonic_ns=snapshot.twist_received_monotonic_ns,
                            target_left=target_left,
                            measured_left=snapshot.left_pose,
                            measured_left_twist=snapshot.left_twist,
                            target_right=target_right,
                            measured_right=snapshot.right_pose,
                            measured_right_twist=snapshot.right_twist,
                        )
                    )
                    self._sample_index += 1

            if self._ros_bridge is not None:
                self._ros_bridge.publish_status()
            next_tick_ns += self.control_period_ns
            remaining_ns = next_tick_ns - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns * 1e-9)
            else:
                now_ns = time.monotonic_ns()
                if now_ns - last_deadline_warning_ns >= 1_000_000_000:
                    LOGGER.warning(
                        "Control deadline missed by %.3f ms; samples are not caught up",
                        -remaining_ns * 1e-6,
                    )
                    last_deadline_warning_ns = now_ns
                next_tick_ns = now_ns + self.control_period_ns

    def _hold_target(self, snapshot: Any) -> tuple[np.ndarray, np.ndarray]:
        if self._last_target is None:
            self._last_target = {
                "left": snapshot.left_pose.copy(),
                "right": snapshot.right_pose.copy(),
            }
        return self._last_target["left"], self._last_target["right"]

    def _terminal_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                line = input("toppra-rollout> ")
            except EOFError:
                self.enqueue_command("quit")
                return
            self.enqueue_command(line)

    def _drain_commands(self) -> None:
        while True:
            try:
                line = self._commands.get_nowait()
            except Empty:
                return
            try:
                self.handle_command(line)
                self._last_command_error = None
            except Exception as exc:
                self._last_command_error = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("Rollout command failed: %s", line)

    def handle_command(self, line: str) -> dict[str, Any] | None:
        parts = shlex.split(line)
        if not parts:
            return None
        command = parts[0].lower()
        if command == "start":
            self._start_episode()
        elif command == "success":
            self._finish_episode(success=True, reason="operator success")
        elif command == "failure":
            self._finish_episode(success=False, reason="operator failure")
        elif command == "abort":
            self._abort_episode(reason="operator abort")
        elif command == "reset":
            self._begin_reset()
        elif command == "ready":
            self._confirm_reset_ready()
        elif command == "approve":
            raise RuntimeError("approve is only available in Speed-RL calibration")
        elif command == "status":
            LOGGER.info("Rollout status: %s", json.dumps(self.status(), sort_keys=True))
        elif command == "quit":
            if self.state == EpisodeState.RUNNING:
                raise RuntimeError("quit is refused while an episode is RUNNING; abort first")
            self._shutdown.set()
        else:
            raise ValueError(f"Unknown rollout command {command!r}")
        return self.status()

    def _start_episode(self) -> None:
        if self.state != EpisodeState.IDLE:
            raise RuntimeError(
                f"Start requires IDLE state, got {self.state.value}; finish and reset first"
            )
        if self._finalizer is not None and self._finalizer.is_alive():
            raise RuntimeError("The previous episode is still finalizing")
        if self._finalization_error is not None:
            raise RuntimeError(
                f"The previous episode failed to finalize: {self._finalization_error}"
            )
        if self._critical_stale_fields:
            raise RuntimeError(
                "Cannot start while required rollout observations are stale: "
                + ", ".join(self._critical_stale_fields)
            )
        if not self.dry_run:
            require_real_policy_cameras(
                self.rospy,
                (self.topics.left_camera, self.topics.right_camera, self.topics.head_camera),
            )

        self._close_episode_resources()
        if self.agent is not None:
            self.agent.teardown()
            self.agent = None
        agent: GR00TAgent | None = None
        servo: KionDualArmServo | None = None
        pinch: KionPinchExecutor | None = None
        recorder: TrackingRecorder | None = None
        next_episode = self.episode_number + 1
        run_directory = self.log_root / (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_episode_{next_episode:03d}"
        )
        try:
            agent = self.agent_factory()
            servo = KionDualArmServo(
                self.ros_types,
                self.topics,
                control_frequency=self.control_frequency,
                gain=self.servo_gain,
                dry_run=self.dry_run,
            )
            servo.wait_for_connection(self.startup_timeout_s)
            servo.configure(self.startup_timeout_s)
            if self.pinch_enabled:
                pinch = KionPinchExecutor(
                    self.rospy,
                    self.ros_types.hand_namespace,
                    max_rate_hz=self.pinch_max_rate_hz,
                    dry_run=self.dry_run,
                )
            recorder = TrackingRecorder(
                run_directory,
                {
                    "episode": next_episode,
                    "mode": "plain",
                    "inference_mode": self.inference_mode,
                    "task": self.task,
                    "control_frequency": self.control_frequency,
                    "topics": asdict(self.topics),
                },
            )
        except BaseException:
            for resource in (recorder, pinch, servo):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception:
                        LOGGER.exception("Failed to clean up a partial rollout start")
            if agent is not None:
                try:
                    agent.teardown()
                except Exception:
                    LOGGER.exception("Failed to stop the partial rollout agent")
            raise

        self.agent = agent
        self.servo = servo
        self.pinch = pinch
        self.recorder = recorder
        self.episode_number = next_episode
        self._tracking_log = str(recorder.csv_path)
        self._sample_index = 0
        self._last_target = None
        self._last_outcome = None
        self._last_safe_success = None
        self._hardware_safety_error = None
        self._episode_started_s = time.monotonic()
        self.state = EpisodeState.RUNNING
        LOGGER.info(
            "Started plain TOPPRA episode=%d inference_mode=%s",
            self.episode_number,
            self.inference_mode,
        )

    def _begin_reset(self) -> None:
        if self.state not in {
            EpisodeState.TERMINATED,
            EpisodeState.ABORTED,
            EpisodeState.RESETTING,
        }:
            raise RuntimeError("Reset requires a finished or aborted episode")
        if self._finalizer is not None and self._finalizer.is_alive():
            raise RuntimeError("The episode is still finalizing; wait before reset")
        if self._finalization_error is not None:
            raise RuntimeError(f"Episode finalization failed: {self._finalization_error}")
        self._close_motion_resources()
        if self.agent is not None:
            self.agent.teardown()
            self.agent = None
        self._last_target = None
        self._episode_started_s = None
        self.reset_controller.begin()
        self.state = EpisodeState.RESETTING
        LOGGER.warning(
            "Servo control released for reset (mode=%s). Confirm robot and scene reset, then "
            "issue 'ready'.",
            self.reset_controller.mode,
        )

    def _confirm_reset_ready(self) -> None:
        if self.state != EpisodeState.RESETTING:
            raise RuntimeError("Ready requires RESETTING state")
        reset_status = self.reset_controller.status()
        if not reset_status.operator_may_confirm_ready:
            detail = reset_status.error or "dual-arm home request is still in progress"
            raise RuntimeError(f"Reset cannot be confirmed ready: {detail}")
        if self._critical_stale_fields:
            raise RuntimeError(
                "Cannot confirm reset while required observations are stale: "
                + ", ".join(self._critical_stale_fields)
            )
        self.state = EpisodeState.IDLE
        LOGGER.info("Operator confirmed robot and scene reset; the next episode may start")

    def _finish_episode(
        self,
        *,
        success: bool,
        reason: str,
        control_fault: bool = False,
    ) -> None:
        if self.state != EpisodeState.RUNNING:
            raise RuntimeError("No RUNNING episode to finish")
        duration_s = time.monotonic() - float(self._episode_started_s)
        recorder = self.recorder
        self.recorder = None
        self.state = EpisodeState.TERMINATED
        self._last_outcome = "success" if success else "failure"
        self._last_safe_success = bool(success and not control_fault)
        self._schedule_finalization(
            recorder,
            {
                "success": bool(success),
                "abort": False,
                "control_fault": bool(control_fault),
                "reason": reason,
                "duration_s": duration_s,
            },
        )

    def _abort_episode(self, *, reason: str) -> None:
        if self.state != EpisodeState.RUNNING:
            raise RuntimeError("No RUNNING episode to abort")
        duration_s = time.monotonic() - float(self._episode_started_s)
        recorder = self.recorder
        self.recorder = None
        self._close_motion_resources()
        self.state = EpisodeState.ABORTED
        self._last_outcome = "abort"
        self._last_safe_success = False
        self._schedule_finalization(
            recorder,
            {
                "success": False,
                "abort": True,
                "control_fault": False,
                "reason": reason,
                "duration_s": duration_s,
            },
        )

    def _schedule_finalization(
        self,
        recorder: TrackingRecorder | None,
        outcome: dict[str, Any],
    ) -> None:
        if self._finalizer is not None and self._finalizer.is_alive():
            raise RuntimeError("Another episode finalizer is already running")
        self._finalization_error = None

        def finalize() -> None:
            try:
                if recorder is not None:
                    recorder.close()
                    (recorder.directory / "outcome.json").write_text(
                        json.dumps(outcome, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
            except BaseException as exc:
                self._finalization_error = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("Episode finalization failed")

        self._finalizer = threading.Thread(
            target=finalize,
            daemon=True,
            name=f"ToppraEpisodeFinalizer{self.episode_number}",
        )
        self._finalizer.start()

    def status(self) -> dict[str, Any]:
        diagnostics = {} if self.agent is None else self.agent.diagnostics()
        reset_status = self.reset_controller.status()
        return {
            "state": self.state.value,
            "mode": "plain",
            "inference_mode": self.inference_mode,
            "task": self.task,
            "episode": self.episode_number,
            "episode_elapsed_s": (
                None
                if self._episode_started_s is None or self.state != EpisodeState.RUNNING
                else time.monotonic() - self._episode_started_s
            ),
            "active_buffer_size": diagnostics.get("active_buffer_size"),
            "active_sequence_id": diagnostics.get("active_sequence_id"),
            "execution_state": diagnostics.get("execution_state", self.state.value),
            "inference_latency_s": diagnostics.get("latest_latency_s"),
            "planning_latency_s": diagnostics.get("latest_planning_latency_s"),
            "latest_error": diagnostics.get("latest_error"),
            "command_error": self._last_command_error,
            "speed_action": None,
            "speed_scale": None,
            "speed_violation": False,
            "hardware_safety_error": self._hardware_safety_error,
            "twist_stale": self._twist_stale,
            "stale_fields": self._stale_fields,
            "critical_stale_fields": self._critical_stale_fields,
            "last_outcome": self._last_outcome,
            "last_safe_success": self._last_safe_success,
            "tracking_log": self._tracking_log,
            "finalization_in_progress": (
                self._finalizer is not None and self._finalizer.is_alive()
            ),
            "finalization_error": self._finalization_error,
            "reset": reset_status.to_dict(),
        }

    def _close_motion_resources(self) -> None:
        if self.pinch is not None:
            self.pinch.close()
            self.pinch = None
        if self.servo is not None:
            self.servo.close()
            self.servo = None

    def _close_episode_resources(self) -> None:
        self._close_motion_resources()
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None

    def close(self) -> None:
        if self.state == EpisodeState.CLOSED:
            return
        self._shutdown.set()
        if self.state == EpisodeState.RUNNING:
            self._abort_episode(reason="client shutdown")
        if self._ros_bridge is not None:
            self._ros_bridge.close()
            self._ros_bridge = None
        self._close_episode_resources()
        if self._finalizer is not None and self._finalizer.is_alive():
            self._finalizer.join(timeout=15.0)
            if self._finalizer.is_alive():
                raise RuntimeError("Timed out finalizing the last rollout episode")
        if self.agent is not None:
            self.agent.teardown()
            self.agent = None
        self.reset_controller.wait(0.1)
        self.observations.close()
        self.state = EpisodeState.CLOSED


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run episode-driven plain TOPPRA rollout with GUI or terminal labeling."
    )
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=47866)
    parser.add_argument("--timeout-ms", type=int, default=15_000)
    parser.add_argument("--policy-frequency", type=_positive_float, default=30.0)
    parser.add_argument("--control-frequency", type=_positive_float, default=250.0)
    parser.add_argument("--inference-mode", choices=("sync", "async"), default="sync")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--refill-threshold", type=int, default=20)
    parser.add_argument("--max-latency-s", type=float, default=0.35)
    parser.add_argument("--scheduling-margin-s", type=float, default=0.05)
    parser.add_argument("--handoff-margin-s", type=float, default=0.05)
    parser.add_argument("--open-loop-horizon", type=int, default=8)
    parser.add_argument("--tts-samples", type=int, default=1)
    parser.add_argument("--tts-waypoint-count", type=int, default=5)
    parser.add_argument("--max-linear-velocity", type=_positive_float, default=1.0)
    parser.add_argument("--max-angular-velocity", type=_positive_float, default=3.0)
    parser.add_argument("--max-linear-acceleration", type=_positive_float, default=5.0)
    parser.add_argument("--max-angular-acceleration", type=_positive_float, default=15.0)
    parser.add_argument("--safety-margin", type=_positive_float, default=0.9)
    parser.add_argument("--velocity-filter", type=_positive_float, default=1.0)
    parser.add_argument("--servo-gain", type=int, default=800)
    parser.add_argument("--startup-timeout-s", type=_positive_float, default=30.0)
    parser.add_argument("--max-state-age-s", type=_positive_float, default=0.25)
    parser.add_argument("--max-image-age-s", type=_positive_float, default=1.0)
    parser.add_argument("--episode-duration-s", type=_positive_float, default=80.0)
    parser.add_argument("--log-root", type=Path, default=Path("logs/kion_rollout"))
    parser.add_argument("--node-name", default="gr00t_kion_rollout")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--control-interface",
        choices=("gui", "terminal", "both"),
        default="gui",
    )
    parser.add_argument("--ros-namespace", default="/gr00t_rollout")
    parser.add_argument("--disable-pinch", action="store_true")
    parser.add_argument("--pinch-max-rate-hz", type=_positive_float, default=30.0)
    parser.add_argument("--reset-mode", choices=("manual", "go-home"), default="manual")
    parser.add_argument("--reset-service", default=DUAL_ARM_HOME_SERVICE)
    parser.add_argument("--reset-timeout-s", type=_positive_float, default=10.0)
    parser.add_argument("--left-camera-topic", default=KionTopics.left_camera)
    parser.add_argument("--right-camera-topic", default=KionTopics.right_camera)
    parser.add_argument("--head-camera-topic", default=KionTopics.head_camera)
    parser.add_argument("--left-workspace-bounds")
    parser.add_argument("--right-workspace-bounds")
    parser.add_argument("--max-target-position-error-m", type=_positive_float)
    parser.add_argument("--max-target-rotation-error-rad", type=_positive_float)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if not 100 <= args.servo_gain <= 1000:
        raise ValueError("--servo-gain must be in the SDK range [100, 1000]")
    if not 0 < args.safety_margin <= 1:
        raise ValueError("--safety-margin must be in (0, 1]")
    target_guard_args = (
        args.left_workspace_bounds,
        args.right_workspace_bounds,
        args.max_target_position_error_m,
        args.max_target_rotation_error_rad,
    )
    if not args.dry_run and any(value is None for value in target_guard_args):
        raise ValueError(
            "Real motion requires explicit --left-workspace-bounds, "
            "--right-workspace-bounds, --max-target-position-error-m, and "
            "--max-target-rotation-error-rad"
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
    )
    preflight = GR00TPolicyClient(args.server_host, args.server_port, timeout_ms=args.timeout_ms)
    try:
        preflight.ping()
    finally:
        preflight.close()
    LOGGER.info("Policy server preflight passed: %s:%d", args.server_host, args.server_port)
    ros_types = load_ros_types()
    ros_types.rospy.init_node(args.node_name, disable_signals=False)
    restore_console_logging()
    topics = KionTopics(
        left_camera=args.left_camera_topic,
        right_camera=args.right_camera_topic,
        head_camera=args.head_camera_topic,
    )
    limits = CartesianLimits(
        max_linear_velocity=(args.max_linear_velocity,) * 3,
        max_angular_velocity=(args.max_angular_velocity,) * 3,
        max_linear_acceleration=(args.max_linear_acceleration,) * 3,
        max_angular_acceleration=(args.max_angular_acceleration,) * 3,
        safety_margin=args.safety_margin,
    )
    target_safety = None
    if args.left_workspace_bounds is not None or args.right_workspace_bounds is not None:
        if args.left_workspace_bounds is None or args.right_workspace_bounds is None:
            raise ValueError("Both left and right workspace bounds must be supplied together")
        if args.max_target_position_error_m is None or args.max_target_rotation_error_rad is None:
            raise ValueError("Workspace bounds require both target tracking error limits")
        target_safety = TargetSafetyGuard(
            parse_workspace_bounds(args.left_workspace_bounds),
            parse_workspace_bounds(args.right_workspace_bounds),
            max_position_error_m=args.max_target_position_error_m,
            max_rotation_error_rad=args.max_target_rotation_error_rad,
        )

    def agent_factory() -> GR00TAgent:
        return GR00TAgent(
            host=args.server_host,
            port=args.server_port,
            fps=args.policy_frequency,
            refill_threshold=args.refill_threshold,
            max_latency_s=args.max_latency_s,
            task_default=args.task,
            timeout_ms=args.timeout_ms,
            left_limits=limits,
            right_limits=limits,
            inference_mode=args.inference_mode,
            control_frequency=args.control_frequency,
            scheduling_margin_s=args.scheduling_margin_s,
            handoff_margin_s=args.handoff_margin_s,
            open_loop_horizon=args.open_loop_horizon,
            tts_samples=args.tts_samples,
            tts_waypoint_count=args.tts_waypoint_count,
            left_velocity_state_key=LEFT_TWIST_KEY,
            right_velocity_state_key=RIGHT_TWIST_KEY,
            velocity_filter=args.velocity_filter,
            include_velocity_in_policy_observation=False,
        )

    client = PlainKionEpisodeClient(
        ros_types=ros_types,
        topics=topics,
        agent_factory=agent_factory,
        log_root=args.log_root,
        control_frequency=args.control_frequency,
        servo_gain=args.servo_gain,
        startup_timeout_s=args.startup_timeout_s,
        max_state_age_s=args.max_state_age_s,
        max_image_age_s=args.max_image_age_s,
        task=args.task,
        inference_mode=args.inference_mode,
        dry_run=args.dry_run,
        target_safety=target_safety,
        episode_duration_s=args.episode_duration_s,
        control_interface=args.control_interface,
        ros_namespace=args.ros_namespace,
        pinch_enabled=not args.disable_pinch,
        pinch_max_rate_hz=args.pinch_max_rate_hz,
        reset_mode=args.reset_mode,
        reset_service=args.reset_service,
        reset_timeout_s=args.reset_timeout_s,
    )
    client.run()
