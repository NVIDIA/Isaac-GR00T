from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from queue import Full, Queue
import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

POSE_NAMES = ("x", "y", "z", "qw", "qx", "qy", "qz")
TWIST_NAMES = ("vx", "vy", "vz", "wx", "wy", "wz")
_STOP = object()


def _columns(prefix: str, names: tuple[str, ...]) -> list[str]:
    return [f"{prefix}_{name}" for name in names]


TRACKING_COLUMNS = [
    "sample_index",
    "loop_started_monotonic_ns",
    "published_monotonic_ns",
    "published_ros_ns",
    "loop_duration_ns",
    "deadline_lateness_ns",
    "active_sequence_id",
    "execution_state",
    "active_buffer_size",
    "left_pose_ros_ns",
    "left_pose_received_monotonic_ns",
    "right_pose_ros_ns",
    "right_pose_received_monotonic_ns",
    "twist_ros_ns",
    "twist_received_monotonic_ns",
    *_columns("target_left", POSE_NAMES),
    *_columns("measured_left", POSE_NAMES),
    *_columns("measured_left", TWIST_NAMES),
    *_columns("target_right", POSE_NAMES),
    *_columns("measured_right", POSE_NAMES),
    *_columns("measured_right", TWIST_NAMES),
    "left_position_error_m",
    "left_rotation_error_rad",
    "right_position_error_m",
    "right_rotation_error_rad",
]


def rotation_error_rad(target_wxyz: FloatArray, measured_wxyz: FloatArray) -> float:
    target = np.asarray(target_wxyz, dtype=np.float64)
    measured = np.asarray(measured_wxyz, dtype=np.float64)
    target_norm = np.linalg.norm(target)
    measured_norm = np.linalg.norm(measured)
    if target_norm <= 1e-12 or measured_norm <= 1e-12:
        return math.nan
    cosine = abs(float(np.dot(target / target_norm, measured / measured_norm)))
    return 2.0 * math.acos(float(np.clip(cosine, -1.0, 1.0)))


def make_tracking_row(
    *,
    sample_index: int,
    loop_started_monotonic_ns: int,
    published_monotonic_ns: int,
    published_ros_ns: int,
    loop_duration_ns: int,
    deadline_lateness_ns: int,
    active_sequence_id: int | None,
    execution_state: str,
    active_buffer_size: int,
    left_pose_ros_ns: int,
    left_pose_received_monotonic_ns: int,
    right_pose_ros_ns: int,
    right_pose_received_monotonic_ns: int,
    twist_ros_ns: int,
    twist_received_monotonic_ns: int,
    target_left: FloatArray,
    measured_left: FloatArray,
    measured_left_twist: FloatArray,
    target_right: FloatArray,
    measured_right: FloatArray,
    measured_right_twist: FloatArray,
) -> tuple[Any, ...]:
    target_left = np.asarray(target_left, dtype=np.float64)
    measured_left = np.asarray(measured_left, dtype=np.float64)
    measured_left_twist = np.asarray(measured_left_twist, dtype=np.float64)
    target_right = np.asarray(target_right, dtype=np.float64)
    measured_right = np.asarray(measured_right, dtype=np.float64)
    measured_right_twist = np.asarray(measured_right_twist, dtype=np.float64)
    for name, value, shape in (
        ("target_left", target_left, (7,)),
        ("measured_left", measured_left, (7,)),
        ("measured_left_twist", measured_left_twist, (6,)),
        ("target_right", target_right, (7,)),
        ("measured_right", measured_right, (7,)),
        ("measured_right_twist", measured_right_twist, (6,)),
    ):
        if value.shape != shape or np.any(~np.isfinite(value)):
            raise ValueError(f"{name} must have finite shape {shape}, got {value.shape}")

    left_position_error = float(np.linalg.norm(target_left[:3] - measured_left[:3]))
    right_position_error = float(np.linalg.norm(target_right[:3] - measured_right[:3]))
    return (
        int(sample_index),
        int(loop_started_monotonic_ns),
        int(published_monotonic_ns),
        int(published_ros_ns),
        int(loop_duration_ns),
        int(deadline_lateness_ns),
        -1 if active_sequence_id is None else int(active_sequence_id),
        str(execution_state),
        int(active_buffer_size),
        int(left_pose_ros_ns),
        int(left_pose_received_monotonic_ns),
        int(right_pose_ros_ns),
        int(right_pose_received_monotonic_ns),
        int(twist_ros_ns),
        int(twist_received_monotonic_ns),
        *target_left.tolist(),
        *measured_left.tolist(),
        *measured_left_twist.tolist(),
        *target_right.tolist(),
        *measured_right.tolist(),
        *measured_right_twist.tolist(),
        left_position_error,
        rotation_error_rad(target_left[3:], measured_left[3:]),
        right_position_error,
        rotation_error_rad(target_right[3:], measured_right[3:]),
    )


class TrackingRecorder:
    """Write control-loop tracking samples without blocking the servo loop on disk I/O."""

    def __init__(
        self,
        directory: str | Path,
        metadata: dict[str, Any],
        *,
        queue_capacity: int = 50_000,
        flush_rows: int = 250,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.csv_path = self.directory / "tcp_tracking.csv"
        self.metadata_path = self.directory / "metadata.json"
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )
        self._queue: Queue[tuple[Any, ...] | object] = Queue(maxsize=queue_capacity)
        self._flush_rows = max(1, int(flush_rows))
        self._dropped_rows = 0
        self._closed = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="KionTrackingWriter",
        )
        self._thread.start()

    @property
    def dropped_rows(self) -> int:
        return self._dropped_rows

    def record(self, row: tuple[Any, ...]) -> None:
        if self._closed:
            raise RuntimeError("TrackingRecorder is closed")
        if len(row) != len(TRACKING_COLUMNS):
            raise ValueError(
                f"Tracking row has {len(row)} fields, expected {len(TRACKING_COLUMNS)}"
            )
        if self._error is not None:
            raise RuntimeError("Tracking writer failed") from self._error
        try:
            self._queue.put_nowait(row)
        except Full:
            self._dropped_rows += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STOP)
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("Tracking writer did not stop")
        if self._error is not None:
            raise RuntimeError("Tracking writer failed") from self._error

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        metadata["tracking_dropped_rows"] = self._dropped_rows
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )

    def _writer_loop(self) -> None:
        try:
            with self.csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(TRACKING_COLUMNS)
                rows_since_flush = 0
                while True:
                    item = self._queue.get()
                    if item is _STOP:
                        file.flush()
                        return
                    writer.writerow(item)
                    rows_since_flush += 1
                    if rows_since_flush >= self._flush_rows:
                        file.flush()
                        rows_since_flush = 0
        except BaseException as exc:
            self._error = exc


@dataclass(frozen=True)
class DelayEstimate:
    arm: str
    delay_ms: float
    zero_lag_position_rmse_m: float
    best_lag_position_rmse_m: float
    best_lag_rotation_rmse_rad: float
    control_rate_hz: float
    pose_feedback_rate_hz: float
    moving_samples: int


def _load_numeric_columns(path: str | Path, columns: tuple[str, ...]) -> dict[str, FloatArray]:
    result = {column: [] for column in columns}
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = set(columns).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Tracking CSV is missing columns: {sorted(missing)}")
        for row in reader:
            for column in columns:
                result[column].append(float(row[column]))
    if not result[columns[0]]:
        raise ValueError("Tracking CSV contains no samples")
    return {key: np.asarray(value, dtype=np.float64) for key, value in result.items()}


def _median_rate_hz(times_s: FloatArray) -> float:
    if len(times_s) < 2:
        return math.nan
    intervals = np.diff(times_s)
    intervals = intervals[intervals > 0]
    if not len(intervals):
        return math.nan
    return float(1.0 / np.median(intervals))


def _deduplicate_feedback(
    times_ns: FloatArray,
    positions: FloatArray,
    quaternions: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    valid = times_ns > 0
    times_ns = times_ns[valid]
    positions = positions[valid]
    quaternions = quaternions[valid]
    if not len(times_ns):
        raise ValueError("Tracking CSV contains no timestamped pose feedback")
    keep = np.concatenate(([True], np.diff(times_ns) > 0))
    return times_ns[keep] * 1e-9, positions[keep], quaternions[keep]


def _interpolate_positions(
    target_times_s: FloatArray,
    target_positions: FloatArray,
    query_times_s: FloatArray,
) -> FloatArray:
    return np.column_stack(
        [np.interp(query_times_s, target_times_s, target_positions[:, axis]) for axis in range(3)]
    )


def _interpolate_quaternions_wxyz(
    target_times_s: FloatArray,
    target_quaternions: FloatArray,
    query_times_s: FloatArray,
) -> FloatArray:
    indices = np.searchsorted(target_times_s, query_times_s, side="right") - 1
    indices = np.clip(indices, 0, len(target_times_s) - 2)
    start_time = target_times_s[indices]
    end_time = target_times_s[indices + 1]
    blend = (query_times_s - start_time) / np.maximum(end_time - start_time, 1e-12)
    start = target_quaternions[indices]
    end = target_quaternions[indices + 1].copy()
    end[np.sum(start * end, axis=1) < 0] *= -1
    result = start + blend[:, np.newaxis] * (end - start)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    return result


def estimate_tracking_delay(
    csv_path: str | Path,
    arm: str,
    *,
    max_lag_ms: float = 500.0,
    lag_step_ms: float = 1.0,
    min_target_speed_m_s: float = 0.005,
) -> DelayEstimate:
    if arm not in {"left", "right"}:
        raise ValueError("arm must be 'left' or 'right'")
    if max_lag_ms < 0 or lag_step_ms <= 0:
        raise ValueError("max_lag_ms must be non-negative and lag_step_ms must be positive")

    target_columns = tuple(_columns(f"target_{arm}", POSE_NAMES))
    measured_columns = tuple(_columns(f"measured_{arm}", POSE_NAMES))
    feedback_time_column = f"{arm}_pose_received_monotonic_ns"
    columns = (
        "published_monotonic_ns",
        feedback_time_column,
        *target_columns,
        *measured_columns,
    )
    table = _load_numeric_columns(csv_path, columns)
    target_times_s = table["published_monotonic_ns"] * 1e-9
    target_positions = np.column_stack([table[name] for name in target_columns[:3]])
    target_quaternions = np.column_stack([table[name] for name in target_columns[3:]])
    measured_positions_all = np.column_stack([table[name] for name in measured_columns[:3]])
    measured_quaternions_all = np.column_stack([table[name] for name in measured_columns[3:]])
    feedback_times_s, measured_positions, measured_quaternions = _deduplicate_feedback(
        table[feedback_time_column],
        measured_positions_all,
        measured_quaternions_all,
    )

    unique_target = np.concatenate(([True], np.diff(target_times_s) > 0))
    target_times_s = target_times_s[unique_target]
    target_positions = target_positions[unique_target]
    target_quaternions = target_quaternions[unique_target]
    if len(target_times_s) < 3 or len(feedback_times_s) < 3:
        raise ValueError("At least three target and feedback samples are required")

    target_speed = np.linalg.norm(
        np.gradient(target_positions, target_times_s, axis=0),
        axis=1,
    )
    lags_s = np.arange(0.0, max_lag_ms * 1e-3 + lag_step_ms * 5e-4, lag_step_ms * 1e-3)
    scores = np.full(len(lags_s), np.inf, dtype=np.float64)
    sample_counts = np.zeros(len(lags_s), dtype=np.int64)
    for index, lag_s in enumerate(lags_s):
        query = feedback_times_s - lag_s
        valid = (query >= target_times_s[0]) & (query <= target_times_s[-1])
        if not np.any(valid):
            continue
        moving = np.interp(query[valid], target_times_s, target_speed) >= min_target_speed_m_s
        valid_indices = np.flatnonzero(valid)
        if np.count_nonzero(moving) >= 10:
            valid_indices = valid_indices[moving]
        if len(valid_indices) < 3:
            continue
        predicted = _interpolate_positions(
            target_times_s,
            target_positions,
            query[valid_indices],
        )
        error = predicted - measured_positions[valid_indices]
        scores[index] = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
        sample_counts[index] = len(valid_indices)

    if not np.any(np.isfinite(scores)):
        raise ValueError("No overlapping target and measured motion was found")
    best_index = int(np.nanargmin(scores))
    best_lag_s = float(lags_s[best_index])

    query = feedback_times_s - best_lag_s
    valid = (query >= target_times_s[0]) & (query <= target_times_s[-1])
    predicted_quaternions = _interpolate_quaternions_wxyz(
        target_times_s,
        target_quaternions,
        query[valid],
    )
    cosine = np.abs(np.sum(predicted_quaternions * measured_quaternions[valid], axis=1))
    rotation_errors = 2.0 * np.arccos(np.clip(cosine, -1.0, 1.0))

    return DelayEstimate(
        arm=arm,
        delay_ms=best_lag_s * 1000.0,
        zero_lag_position_rmse_m=float(scores[0]),
        best_lag_position_rmse_m=float(scores[best_index]),
        best_lag_rotation_rmse_rad=float(np.sqrt(np.mean(rotation_errors**2))),
        control_rate_hz=_median_rate_hz(target_times_s),
        pose_feedback_rate_hz=_median_rate_hz(feedback_times_s),
        moving_samples=int(sample_counts[best_index]),
    )


def delay_estimate_as_dict(estimate: DelayEstimate) -> dict[str, Any]:
    return asdict(estimate)
