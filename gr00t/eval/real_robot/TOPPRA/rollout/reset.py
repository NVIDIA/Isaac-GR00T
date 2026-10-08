from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import threading
from typing import Any


DUAL_ARM_HOME_SERVICE = "/zj_humanoid/upperlimb/go_home/dual_arm"


@dataclass(frozen=True)
class ResetStatus:
    mode: str
    home_request_in_progress: bool
    home_request_complete: bool
    error: str | None
    response: str | None

    @property
    def operator_may_confirm_ready(self) -> bool:
        return (
            not self.home_request_in_progress and self.home_request_complete and self.error is None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "operator_may_confirm_ready": self.operator_may_confirm_ready,
        }


class EpisodeResetController:
    """Release rollout control and optionally request the SDK's built-in dual-arm home."""

    def __init__(
        self,
        rospy: Any,
        *,
        mode: str = "manual",
        service_name: str = DUAL_ARM_HOME_SERVICE,
        service_timeout_s: float = 10.0,
        dry_run: bool = False,
    ) -> None:
        if mode not in {"manual", "go-home"}:
            raise ValueError("reset mode must be 'manual' or 'go-home'")
        if service_timeout_s <= 0:
            raise ValueError("reset service timeout must be positive")
        self.rospy = rospy
        self.mode = mode
        self.service_name = str(service_name)
        self.service_timeout_s = float(service_timeout_s)
        self.dry_run = bool(dry_run)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._home_request_in_progress = False
        self._home_request_complete = False
        self._error: str | None = None
        self._response: str | None = None

    def begin(self) -> None:
        with self._lock:
            if self._home_request_in_progress:
                raise RuntimeError("The dual-arm home request is already in progress")
            self._error = None
            self._response = None
            self._home_request_complete = self.mode == "manual" or self.dry_run
            self._home_request_in_progress = self.mode == "go-home" and not self.dry_run
        if not self._home_request_in_progress:
            return
        self._thread = threading.Thread(
            target=self._request_home,
            daemon=True,
            name="KionDualArmHomeReset",
        )
        self._thread.start()

    def _request_home(self) -> None:
        error: str | None = None
        response_message: str | None = None
        completed = False
        try:
            trigger_module = importlib.import_module("std_srvs.srv")
            trigger_type = getattr(trigger_module, "Trigger")
            self.rospy.wait_for_service(self.service_name, timeout=self.service_timeout_s)
            response = self.rospy.ServiceProxy(self.service_name, trigger_type)()
            response_message = str(getattr(response, "message", ""))
            if not bool(getattr(response, "success", False)):
                raise RuntimeError(
                    f"dual-arm home service rejected the request: {response_message or 'no message'}"
                )
            completed = True
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._home_request_in_progress = False
            self._home_request_complete = completed
            self._error = error
            self._response = response_message

    def status(self) -> ResetStatus:
        with self._lock:
            return ResetStatus(
                mode=self.mode,
                home_request_in_progress=self._home_request_in_progress,
                home_request_complete=self._home_request_complete,
                error=self._error,
                response=self._response,
            )

    def wait(self, timeout_s: float = 0.0) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(float(timeout_s), 0.0))
