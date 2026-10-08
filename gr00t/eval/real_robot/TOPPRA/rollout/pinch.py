from __future__ import annotations

import importlib
import logging
from queue import Empty, Full, Queue
import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray


LOGGER = logging.getLogger(__name__)
_STOP = object()

LEFT_LOW = np.array((0.2, 0.05, 0.0, 0.0, 0.0, 0.0), dtype=np.float64)
LEFT_HIGH = np.array((0.2, 1.2, 1.0, 1.0, 1.0, 1.0), dtype=np.float64)
RIGHT_LOW = np.array((-0.2, 0.6, 0.0, 0.0, 1.3, 1.3), dtype=np.float64)
RIGHT_HIGH = np.array((0.21, 0.92, 0.93, 1.0, 1.3, 1.3), dtype=np.float64)


def pinch_posture(
    pinchness: NDArray[Any] | float,
    low: NDArray[Any],
    high: NDArray[Any],
) -> NDArray:
    value = np.asarray(pinchness, dtype=np.float64).reshape(-1)
    if value.shape != (1,) or np.any(~np.isfinite(value)):
        raise ValueError("pinch action must be one finite scalar")
    amount = float(np.clip(value[0], 0.0, 1.0))
    return (
        np.asarray(low, dtype=np.float64)
        + (np.asarray(high, dtype=np.float64) - np.asarray(low, dtype=np.float64)) * amount
    )


class _HandWorker:
    def __init__(
        self,
        rospy: Any,
        service_type: type,
        side: str,
        low: NDArray[Any],
        high: NDArray[Any],
        *,
        max_rate_hz: float,
        dry_run: bool,
    ) -> None:
        self.rospy = rospy
        self.side = side
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.period_s = 1.0 / float(max_rate_hz)
        self.dry_run = bool(dry_run)
        self.service_name = f"/zj_humanoid/hand/joint_switch/{side}"
        self._queue: Queue[NDArray[Any] | object] = Queue(maxsize=1)
        self._proxy = None
        self._error_lock = threading.Lock()
        self._error: BaseException | None = None
        if not self.dry_run:
            self._proxy = rospy.ServiceProxy(self.service_name, service_type, persistent=True)
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"Kion{side.title()}Pinch",
        )
        self._thread.start()

    def submit(self, value: NDArray[Any]) -> None:
        try:
            self._queue.get_nowait()
        except Empty:
            pass
        try:
            self._queue.put_nowait(np.asarray(value, dtype=np.float64).copy())
        except Full:
            pass

    def _run(self) -> None:
        last_call_s = 0.0
        while True:
            value = self._queue.get()
            if value is _STOP:
                return
            posture = pinch_posture(value, self.low, self.high)
            if self.dry_run:
                continue
            wait_s = self.period_s - (time.monotonic() - last_call_s)
            if wait_s > 0:
                time.sleep(wait_s)
            try:
                assert self._proxy is not None
                self._proxy(posture)
                last_call_s = time.monotonic()
            except Exception as exc:
                with self._error_lock:
                    if self._error is None:
                        self._error = exc
                LOGGER.exception("%s pinch service call failed", self.side)

    def raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"{self.side} pinch worker failed") from error

    def close(self) -> None:
        try:
            self._queue.get_nowait()
        except Empty:
            pass
        self._queue.put(_STOP)
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError(f"{self.side} pinch worker did not stop")
        close = getattr(self._proxy, "close", None)
        if callable(close):
            close()


class KionPinchExecutor:
    """Publish the latest model pinch without blocking the 250 Hz arm loop."""

    def __init__(
        self,
        rospy: Any,
        hand_namespace: str,
        *,
        max_rate_hz: float = 30.0,
        dry_run: bool = False,
    ) -> None:
        service_module_name = hand_namespace.rsplit(".", 1)[0] + ".srv"
        service_module = importlib.import_module(service_module_name)
        service_type = getattr(service_module, "HandJoint")
        self._last = {"left": None, "right": None}
        self._workers = {
            "left": _HandWorker(
                rospy,
                service_type,
                "left",
                LEFT_LOW,
                LEFT_HIGH,
                max_rate_hz=max_rate_hz,
                dry_run=dry_run,
            ),
            "right": _HandWorker(
                rospy,
                service_type,
                "right",
                RIGHT_LOW,
                RIGHT_HIGH,
                max_rate_hz=max_rate_hz,
                dry_run=dry_run,
            ),
        }

    def publish(self, left: NDArray[Any], right: NDArray[Any]) -> None:
        self.raise_if_failed()
        for side, value in (("left", left), ("right", right)):
            normalized = np.asarray(value, dtype=np.float64).reshape(-1)
            if normalized.shape != (1,) or np.any(~np.isfinite(normalized)):
                raise ValueError(f"{side} pinch action must have finite shape (1,)")
            previous = self._last[side]
            if previous is not None and np.allclose(previous, normalized, atol=1e-3, rtol=0.0):
                continue
            self._last[side] = normalized.copy()
            self._workers[side].submit(normalized)

    def raise_if_failed(self) -> None:
        for worker in self._workers.values():
            worker.raise_if_failed()

    def close(self) -> None:
        errors = []
        for worker in self._workers.values():
            try:
                worker.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("Failed to close one or more pinch workers") from errors[0]
