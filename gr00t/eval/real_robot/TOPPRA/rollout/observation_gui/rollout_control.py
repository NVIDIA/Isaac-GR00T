from __future__ import annotations

import json
from queue import Empty, Full, Queue
import threading
import time
from typing import Any

from controlloop import Mode
import dearpygui.dearpygui as dpg
from gui.module import GUIModule
import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger


class RolloutControlPanel(GUIModule):
    """ObservationGUILite panel for an external 250 Hz GR00T rollout process."""

    _HIDDEN_STANDARD_ITEMS = (
        "task_input",
        "btn_record",
        "btn_inference",
        "btn_training",
        "btn_stop",
        "btn_save",
        "btn_discard",
        "chk_recording_states",
        "btn_success",
        "status_text",
    )

    def __init__(self, namespace: str = "/gr00t_rollout") -> None:
        super().__init__("GR00T Rollout", placement="top")
        self.namespace = "/" + namespace.strip("/")
        self._status: dict[str, Any] = {"state": "connecting"}
        self._status_lock = threading.Lock()
        self._last_status_s = 0.0
        self._last_result = "Waiting for rollout status"
        self._commands: Queue[str] = Queue(maxsize=8)
        self._shutdown = threading.Event()
        self._control_loop = None
        self._control_state = None
        self._recording_episode: int | None = None
        self._saved_episode: int | None = None
        self._subscriber = rospy.Subscriber(
            f"{self.namespace}/status",
            String,
            self._status_callback,
            queue_size=1,
        )
        self._worker = threading.Thread(
            target=self._command_loop,
            daemon=True,
            name="RolloutGUICommands",
        )
        self._worker.start()

    def bind(self, gui: Any) -> None:
        """Attach the passive ObservationGUILite recorder without granting motion control."""

        self._control_loop = gui.control_loop
        self._control_state = gui.ctrl_state

    def _status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(status, dict):
            return
        with self._status_lock:
            self._status = status
            self._last_status_s = time.monotonic()

    def _enqueue(self, command: str) -> None:
        try:
            self._commands.put_nowait(command)
        except Full:
            self._last_result = "Command queue is busy"

    def _command_loop(self) -> None:
        while not self._shutdown.is_set() and not rospy.is_shutdown():
            try:
                command = self._commands.get(timeout=0.2)
            except Empty:
                continue
            service_name = f"{self.namespace}/{command}"
            try:
                rospy.wait_for_service(service_name, timeout=2.0)
                response = rospy.ServiceProxy(service_name, Trigger)()
                self._last_result = response.message
            except Exception as exc:
                self._last_result = f"{command} failed: {exc}"

    def setup_gui(self) -> None:
        for item in self._HIDDEN_STANDARD_ITEMS:
            if dpg.does_item_exist(item):
                dpg.configure_item(item, show=False)

        dpg.add_spacer(height=8)
        dpg.add_text("External GR00T Rollout", color=(255, 200, 100))
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Start Episode",
                height=60,
                tag="rollout_start",
                callback=lambda: self._enqueue("start"),
            )
            dpg.add_button(
                label="Success",
                height=60,
                tag="rollout_success",
                callback=lambda: self._enqueue("success"),
            )
            dpg.add_button(
                label="Failure",
                height=60,
                tag="rollout_failure",
                callback=lambda: self._enqueue("failure"),
            )
            dpg.add_button(
                label="Abort / Close Servo",
                height=60,
                tag="rollout_abort",
                callback=lambda: self._enqueue("abort"),
            )
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Reset / Release Servo",
                height=50,
                tag="rollout_reset",
                callback=lambda: self._enqueue("reset"),
            )
            dpg.add_button(
                label="Reset Complete / Ready",
                height=50,
                tag="rollout_ready",
                callback=lambda: self._enqueue("ready"),
            )
            dpg.add_button(
                label="Approve Calibrated Speed",
                height=60,
                tag="rollout_approve",
                callback=lambda: self._enqueue("approve"),
            )
        dpg.add_text("State: connecting", tag="rollout_state_text")
        dpg.add_text("Trajectory: ---", tag="rollout_trajectory_text")
        dpg.add_text("Speed-RL: ---", tag="rollout_speed_text")
        dpg.add_text("Safety: ---", tag="rollout_safety_text")
        dpg.add_text("Reset: ---", tag="rollout_reset_text")
        dpg.add_text("Dataset: idle", tag="rollout_dataset_text")
        dpg.add_text(self._last_result, tag="rollout_command_text", wrap=1800)

    @staticmethod
    def _fmt(value: Any, suffix: str = "") -> str:
        if value is None:
            return "---"
        if isinstance(value, float):
            return f"{value:.3f}{suffix}"
        return f"{value}{suffix}"

    def update_gui(self) -> None:
        with self._status_lock:
            status = dict(self._status)
            age_s = time.monotonic() - self._last_status_s if self._last_status_s else None
        state = str(status.get("state", "connecting"))
        self._sync_passive_recording(status)
        running = state == "running"
        finalizing = bool(status.get("finalization_in_progress", False))
        mode = status.get("mode", "---")
        phase = status.get("phase")
        reset = status.get("reset") or {}
        stale_label = "no status" if age_s is None else f"status age {age_s:.1f}s"

        dpg.set_value(
            "rollout_state_text",
            f"State: {state} | mode={mode} | inference={status.get('inference_mode', '---')} "
            f"| episode={status.get('episode', 0)} | task={status.get('task', '---')} "
            f"| {stale_label}",
        )
        dpg.set_value(
            "rollout_trajectory_text",
            "Trajectory: sequence={} | buffer={} | inference={} | planning={} | error={}".format(
                self._fmt(status.get("active_sequence_id")),
                self._fmt(status.get("active_buffer_size")),
                self._fmt(status.get("inference_latency_s"), " s"),
                self._fmt(status.get("planning_latency_s"), " s"),
                status.get("command_error") or status.get("latest_error") or "none",
            ),
        )
        dpg.set_value(
            "rollout_speed_text",
            "Speed-RL: phase={} | action={} | scale={} | policy={} | epsilon={} | "
            "replay={} | per-speed={} | updates={}".format(
                phase or "disabled",
                self._fmt(status.get("speed_action")),
                self._fmt(status.get("speed_scale")),
                self._fmt(status.get("policy_version")),
                self._fmt(status.get("epsilon")),
                self._fmt(status.get("replay_size")),
                status.get("replay_action_counts", "---"),
                self._fmt(status.get("learner_update_count")),
            ),
        )
        dpg.set_value(
            "rollout_safety_text",
            "Safety: violation={} | twist_stale={} | critical_stale={} | mask={} | "
            "target_guard={} | outcome={} | safe_success={}".format(
                status.get("speed_violation", False),
                status.get("twist_stale", "---"),
                status.get("critical_stale_fields", "---"),
                status.get("action_mask", "---"),
                status.get("hardware_safety_error") or "clear",
                status.get("last_outcome", "---"),
                status.get("last_safe_success", "---"),
            ),
        )
        dpg.set_value(
            "rollout_reset_text",
            "Reset: mode={} | home_request={} | operator_ready={} | error={}".format(
                reset.get("mode", "---"),
                (
                    "running"
                    if reset.get("home_request_in_progress")
                    else ("complete" if reset.get("home_request_complete") else "not-started")
                ),
                reset.get("operator_may_confirm_ready", False),
                reset.get("error") or "none",
            ),
        )
        recording_mode = (
            "unbound" if self._control_state is None else self._control_state.get_mode().name
        )
        frame_count = 0 if self._control_state is None else self._control_state.frame_count
        dpg.set_value(
            "rollout_dataset_text",
            f"Dataset: mode={recording_mode} | frames={frame_count} | "
            f"recording_episode={self._recording_episode or '---'} | "
            f"last_saved_episode={self._saved_episode or '---'}",
        )
        dpg.set_value("rollout_command_text", self._last_result)
        start_blocked = bool(status.get("critical_stale_fields")) or bool(
            status.get("finalization_error")
        )
        dpg.configure_item(
            "rollout_start",
            enabled=state == "idle" and not finalizing and not start_blocked,
        )
        dpg.configure_item("rollout_success", enabled=running)
        dpg.configure_item("rollout_failure", enabled=running)
        dpg.configure_item("rollout_abort", enabled=running)
        dpg.configure_item(
            "rollout_reset",
            enabled=state in {"terminated", "aborted", "resetting"} and not finalizing,
        )
        dpg.configure_item(
            "rollout_ready",
            enabled=state == "resetting"
            and bool(reset.get("operator_may_confirm_ready"))
            and not bool(status.get("critical_stale_fields")),
        )
        awaiting_approval = bool((status.get("calibration") or {}).get("awaiting_approval"))
        dpg.configure_item(
            "rollout_approve",
            enabled=(mode in {"speed-rl", "speed-rl-baseline"})
            and awaiting_approval
            and not running,
        )

    def _sync_passive_recording(self, status: dict[str, Any]) -> None:
        if self._control_loop is None or self._control_state is None:
            return
        try:
            episode = int(status.get("episode", 0))
        except (TypeError, ValueError):
            return
        state = str(status.get("state", ""))
        if state == "running" and episode > 0 and episode != self._recording_episode:
            if self._control_loop.frames:
                self._control_loop.discard_episode()
            self._control_state.task = str(status.get("task", ""))
            self._control_state.episode_success = False
            self._control_state.set_mode(Mode.RECORDING)
            self._recording_episode = episode
            self._last_result = f"Recording external rollout episode {episode}"
            return
        if self._recording_episode != episode or state not in {
            "terminated",
            "aborted",
            "resetting",
            "idle",
        }:
            return

        outcome = status.get("last_outcome")
        if outcome not in {"success", "failure", "abort"}:
            return
        self._control_state.set_mode(Mode.REVIEWING)
        if outcome == "abort":
            self._control_loop.discard_episode()
            self._last_result = f"Discarded aborted rollout episode {episode}"
        else:
            frame_index = max(len(self._control_loop.frames) - 2, 0)
            self._control_loop.debug_events.append(
                {
                    "frame_index": frame_index,
                    "events": [
                        {
                            "type": "external_rollout_outcome",
                            "episode": episode,
                            "outcome": outcome,
                            "safe_success": bool(status.get("last_safe_success", False)),
                            "speed_violation": bool(status.get("speed_violation", False)),
                            "tracking_log": status.get("tracking_log"),
                        }
                    ],
                }
            )
            self._control_state.episode_success = bool(
                outcome == "success" and status.get("last_safe_success", True)
            )
            self._control_loop.save_episode()
            self._saved_episode = episode
            self._last_result = f"Saved rollout episode {episode} with label {outcome}"
        self._recording_episode = None
