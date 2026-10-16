from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def parse_twist_thresholds(value: str) -> FloatArray:
    try:
        thresholds = np.asarray([float(part) for part in value.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("twist thresholds must be six comma-separated numbers") from exc
    if thresholds.shape != (6,) or np.any(~np.isfinite(thresholds)) or np.any(thresholds <= 0):
        raise ValueError("twist thresholds must be six finite positive numbers")
    return thresholds


@dataclass(frozen=True)
class SpeedSafetyStatus:
    violation_latched: bool
    consecutive_counts: tuple[tuple[int, ...], tuple[int, ...]]
    twist_stale: bool
    action_mask: tuple[bool, ...]


class SpeedViolationMonitor:
    """Latch after any measured twist component exceeds its limit for 100 valid frames."""

    def __init__(
        self,
        left_thresholds: Sequence[float],
        right_thresholds: Sequence[float],
        *,
        consecutive_frames: int = 100,
        action_count: int = 4,
        stale_safe_action_count: int = 2,
    ) -> None:
        self.thresholds = np.stack(
            (
                self._validated_thresholds(left_thresholds, "left_thresholds"),
                self._validated_thresholds(right_thresholds, "right_thresholds"),
            )
        )
        if consecutive_frames < 1:
            raise ValueError("consecutive_frames must be positive")
        self.consecutive_frames = int(consecutive_frames)
        self.action_count = int(action_count)
        self.stale_safe_action_count = int(stale_safe_action_count)
        if self.action_count < 1:
            raise ValueError("action_count must be positive")
        if not 1 <= self.stale_safe_action_count <= self.action_count:
            raise ValueError("stale_safe_action_count must lie inside the action space")
        self._counts = np.zeros((2, 6), dtype=np.int64)
        self._latched = False
        self._twist_stale = True

    @staticmethod
    def _validated_thresholds(value: Sequence[float], name: str) -> FloatArray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (6,) or np.any(~np.isfinite(array)) or np.any(array <= 0):
            raise ValueError(f"{name} must contain six finite positive values")
        return array.copy()

    def reset_episode(self) -> None:
        self._counts.fill(0)
        self._latched = False
        self._twist_stale = True

    def update(
        self,
        left_twist: NDArray[Any],
        right_twist: NDArray[Any],
        *,
        stale: bool,
    ) -> SpeedSafetyStatus:
        self._twist_stale = bool(stale)
        if stale:
            self._counts.fill(0)
            return self.status()
        twists = np.stack(
            (
                np.asarray(left_twist, dtype=np.float64),
                np.asarray(right_twist, dtype=np.float64),
            )
        )
        if twists.shape != (2, 6) or np.any(~np.isfinite(twists)):
            raise ValueError("measured bimanual twists must have finite shape (2, 6)")
        exceeded = np.abs(twists) > self.thresholds
        self._counts = np.where(exceeded, self._counts + 1, 0)
        if np.any(self._counts >= self.consecutive_frames):
            self._latched = True
        return self.status()

    def status(self) -> SpeedSafetyStatus:
        if self._twist_stale:
            mask = (True,) * self.stale_safe_action_count + (False,) * (
                self.action_count - self.stale_safe_action_count
            )
        else:
            mask = (True,) * self.action_count
        return SpeedSafetyStatus(
            violation_latched=self._latched,
            consecutive_counts=tuple(
                tuple(int(value) for value in arm_counts) for arm_counts in self._counts
            ),  # type: ignore[arg-type]
            twist_stale=self._twist_stale,
            action_mask=mask,
        )
