from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
MOCK_CAMERA_NODE_MARKER = "gr00t_pipeline_mock_inputs"


def parse_workspace_bounds(value: str) -> "WorkspaceBounds":
    """Parse xmin,xmax,ymin,ymax,zmin,zmax without inventing hardware defaults."""

    try:
        values = np.asarray([float(part) for part in value.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("workspace bounds must be six comma-separated numbers") from exc
    if values.shape != (6,) or np.any(~np.isfinite(values)):
        raise ValueError("workspace bounds must be six finite numbers")
    lower = values[[0, 2, 4]]
    upper = values[[1, 3, 5]]
    if np.any(lower >= upper):
        raise ValueError("each workspace minimum must be strictly below its maximum")
    return WorkspaceBounds(lower=lower, upper=upper)


@dataclass(frozen=True)
class WorkspaceBounds:
    lower: FloatArray
    upper: FloatArray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("workspace lower and upper bounds must each have shape [3]")
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            raise ValueError("workspace bounds must be finite")
        if np.any(lower >= upper):
            raise ValueError("workspace lower bounds must be below upper bounds")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def contains(self, position: Sequence[float]) -> bool:
        point = np.asarray(position, dtype=np.float64)
        return bool(
            point.shape == (3,) and np.all(point >= self.lower) and np.all(point <= self.upper)
        )

    def to_list(self) -> list[float]:
        return [
            float(self.lower[0]),
            float(self.upper[0]),
            float(self.lower[1]),
            float(self.upper[1]),
            float(self.lower[2]),
            float(self.upper[2]),
        ]


class TargetSafetyError(RuntimeError):
    pass


class TargetSafetyGuard:
    """Fail closed before Servo publication when a Cartesian target is unsafe."""

    def __init__(
        self,
        left_workspace: WorkspaceBounds,
        right_workspace: WorkspaceBounds,
        *,
        max_position_error_m: float,
        max_rotation_error_rad: float,
    ) -> None:
        if not math.isfinite(max_position_error_m) or max_position_error_m <= 0:
            raise ValueError("max_position_error_m must be finite and positive")
        if not math.isfinite(max_rotation_error_rad) or not 0 < max_rotation_error_rad <= math.pi:
            raise ValueError("max_rotation_error_rad must be in (0, pi]")
        self.workspaces = {"left": left_workspace, "right": right_workspace}
        self.max_position_error_m = float(max_position_error_m)
        self.max_rotation_error_rad = float(max_rotation_error_rad)

    @staticmethod
    def _pose(value: Sequence[float], name: str) -> FloatArray:
        pose = np.asarray(value, dtype=np.float64).reshape(-1)
        if pose.shape != (7,) or np.any(~np.isfinite(pose)):
            raise TargetSafetyError(f"{name} must be a finite [x,y,z,qw,qx,qy,qz] pose")
        norm = float(np.linalg.norm(pose[3:]))
        if norm <= 1e-12:
            raise TargetSafetyError(f"{name} has a zero quaternion")
        normalized = pose.copy()
        normalized[3:] /= norm
        return normalized

    def validate(
        self,
        target_left: Sequence[float],
        target_right: Sequence[float],
        measured_left: Sequence[float],
        measured_right: Sequence[float],
    ) -> None:
        for side, target_value, measured_value in (
            ("left", target_left, measured_left),
            ("right", target_right, measured_right),
        ):
            target = self._pose(target_value, f"{side} target")
            measured = self._pose(measured_value, f"{side} measured pose")
            if not self.workspaces[side].contains(target[:3]):
                bounds = self.workspaces[side].to_list()
                raise TargetSafetyError(
                    f"{side} target position {target[:3].tolist()} is outside certified "
                    f"workspace {bounds}"
                )
            position_error = float(np.linalg.norm(target[:3] - measured[:3]))
            if position_error > self.max_position_error_m:
                raise TargetSafetyError(
                    f"{side} target tracking error {position_error:.6f} m exceeds "
                    f"{self.max_position_error_m:.6f} m"
                )
            quaternion_dot = float(np.clip(abs(np.dot(target[3:], measured[3:])), 0.0, 1.0))
            rotation_error = 2.0 * math.acos(quaternion_dot)
            if rotation_error > self.max_rotation_error_rad:
                raise TargetSafetyError(
                    f"{side} target rotation error {rotation_error:.6f} rad exceeds "
                    f"{self.max_rotation_error_rad:.6f} rad"
                )


def mock_camera_publishers(rospy: Any, camera_topics: Sequence[str]) -> tuple[str, ...]:
    """Return mock relay nodes publishing any policy camera topic."""

    code, message, state = rospy.get_master().getSystemState()
    if code != 1:
        raise RuntimeError(f"Could not query ROS system state: {message}")
    requested = {rospy.resolve_name(topic) for topic in camera_topics}
    publishers = state[0]
    collisions = {
        node
        for topic, nodes in publishers
        if topic in requested
        for node in nodes
        if MOCK_CAMERA_NODE_MARKER in node
    }
    return tuple(sorted(collisions))


def require_real_policy_cameras(rospy: Any, camera_topics: Sequence[str]) -> None:
    collisions = mock_camera_publishers(rospy, camera_topics)
    if collisions:
        raise RuntimeError(
            "Real motion is refused while test-only wrist-camera relay nodes are active: "
            + ", ".join(collisions)
        )
