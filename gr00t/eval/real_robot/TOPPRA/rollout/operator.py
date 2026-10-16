from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import importlib
import json
import logging
import math
import threading
import time
from typing import Any

import numpy as np


CommandSink = Callable[[str], None]
StatusProvider = Callable[[], Mapping[str, Any]]


def restore_console_logging() -> logging.Handler:
    """Restore one stderr handler after ``rospy.init_node`` replaces root handlers."""

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if getattr(handler, "_gr00t_console_handler", False):
            return handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s"))
    setattr(handler, "_gr00t_console_handler", True)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    return handler


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def status_json(status: Mapping[str, Any]) -> str:
    """Serialize one rollout status snapshot for ROS and the observation GUI."""

    return json.dumps(
        _json_ready(status),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class RosEpisodeBridge:
    """Expose the episode command queue through ROS Trigger services.

    Service callbacks only enqueue commands. Robot state changes remain serialized in the
    250 Hz control thread, so a GUI callback can never configure or close Servo concurrently.
    """

    COMMANDS = ("start", "success", "failure", "abort", "reset", "ready", "approve")

    def __init__(
        self,
        rospy: Any,
        command_sink: CommandSink,
        status_provider: StatusProvider,
        *,
        namespace: str = "/gr00t_rollout",
        publish_hz: float = 10.0,
    ) -> None:
        self.rospy = rospy
        self.command_sink = command_sink
        self.status_provider = status_provider
        self.namespace = "/" + namespace.strip("/")
        self.publish_period_s = 1.0 / float(publish_hz)
        self._last_publish_s = 0.0
        self._lock = threading.Lock()

        std_srvs = importlib.import_module("std_srvs.srv")
        std_msgs = importlib.import_module("std_msgs.msg")
        self._trigger_response = getattr(std_srvs, "TriggerResponse")
        self._string_type = getattr(std_msgs, "String")
        trigger_type = getattr(std_srvs, "Trigger")
        self._publisher = rospy.Publisher(
            f"{self.namespace}/status",
            self._string_type,
            queue_size=1,
            latch=True,
        )
        self._services = [
            rospy.Service(
                f"{self.namespace}/{command}",
                trigger_type,
                self._command_callback(command),
            )
            for command in self.COMMANDS
        ]
        self._services.append(
            rospy.Service(
                f"{self.namespace}/status",
                trigger_type,
                self._status_callback,
            )
        )

    def _command_callback(self, command: str) -> Callable[[Any], Any]:
        def callback(_request: Any) -> Any:
            queued_command = command
            if command == "approve":
                queued_command = "approve ObservationGUILite operator approval"
            self.command_sink(queued_command)
            return self._trigger_response(
                success=True,
                message=f"Queued rollout command: {command}",
            )

        return callback

    def _status_callback(self, _request: Any) -> Any:
        try:
            message = status_json(self.status_provider())
        except Exception as exc:
            return self._trigger_response(success=False, message=str(exc))
        return self._trigger_response(success=True, message=message)

    def publish_status(self, *, force: bool = False) -> None:
        now_s = time.monotonic()
        with self._lock:
            if not force and now_s - self._last_publish_s < self.publish_period_s:
                return
            self._last_publish_s = now_s
        message = self._string_type()
        message.data = status_json(self.status_provider())
        self._publisher.publish(message)

    def close(self) -> None:
        for service in self._services:
            shutdown = getattr(service, "shutdown", None)
            if callable(shutdown):
                shutdown("rollout client closed")
        unregister = getattr(self._publisher, "unregister", None)
        if callable(unregister):
            unregister()
