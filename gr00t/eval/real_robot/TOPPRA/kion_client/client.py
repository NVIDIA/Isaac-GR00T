from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import importlib
import logging
import math
from pathlib import Path
import threading
import time
from types import ModuleType
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from ..eval_toppra_bimanual import DEFAULT_TASK, CartesianLimits, GR00TAgent
from .tracking import TrackingRecorder, make_tracking_row


LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]

LEFT_POSE_KEY = "observation.state.left_tcp"
RIGHT_POSE_KEY = "observation.state.right_tcp"
LEFT_TWIST_KEY = "observation.state.left_tcp_twist"
RIGHT_TWIST_KEY = "observation.state.right_tcp_twist"
LEFT_TARGET_KEY = "action.left_tcp"
RIGHT_TARGET_KEY = "action.right_tcp"


@dataclass(frozen=True)
class KionTopics:
    left_camera: str = "/zj_humanoid/sensor/left_wrist/image_raw/compressed"
    right_camera: str = "/zj_humanoid/sensor/right_wrist/image_raw/compressed"
    head_camera: str = "/zj_humanoid/sensor/realsense_head/color/image_raw/compressed"
    left_pose: str = "/zj_humanoid/upperlimb/tcp_pose/left_arm"
    right_pose: str = "/zj_humanoid/upperlimb/tcp_pose/right_arm"
    dual_twist: str = "/zj_humanoid/upperlimb/tcp_speed/dual_arm"
    left_pressure: str = "/zj_humanoid/hand/finger_pressures/left"
    right_pressure: str = "/zj_humanoid/hand/finger_pressures/right"
    left_wrist_force: str = "/wrist_force_control/left_arm_compensated_force"
    right_wrist_force: str = "/wrist_force_control/right_arm_compensated_force"
    dual_servo: str = "/zj_humanoid/upperlimb/servol/dual_arm"
    set_servo_params: str = "/zj_humanoid/upperlimb/set_servo_params"
    clear_servo_params: str = "/zj_humanoid/upperlimb/clear_servo_params"


@dataclass(frozen=True)
class RosTypes:
    rospy: ModuleType
    DualPose: type
    Pose: type
    TcpSpeed: type
    Servo: type
    ServoRequest: type
    PressureSensor: type
    CompressedImage: type
    WrenchStamped: type
    upperlimb_namespace: str
    hand_namespace: str


@dataclass(frozen=True)
class RobotSnapshot:
    observation: dict[str, NDArray[Any]]
    left_pose: FloatArray
    right_pose: FloatArray
    left_twist: FloatArray
    right_twist: FloatArray
    left_pose_ros_ns: int
    left_pose_received_monotonic_ns: int
    right_pose_ros_ns: int
    right_pose_received_monotonic_ns: int
    twist_ros_ns: int
    twist_received_monotonic_ns: int
    stale_fields: tuple[str, ...]


def _import_first(names: tuple[str, ...]) -> tuple[ModuleType, str]:
    errors: list[str] = []
    for name in names:
        try:
            return importlib.import_module(name), name
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
    raise ImportError("Could not import any supported ROS SDK namespace:\n" + "\n".join(errors))


def load_ros_types() -> RosTypes:
    rospy = importlib.import_module("rospy")
    compressed_image_module = importlib.import_module("sensor_msgs.msg")
    wrench_module = importlib.import_module("geometry_msgs.msg")
    upperlimb_msg, upperlimb_namespace = _import_first(
        ("zj_humanoid.upperlimb.msg", "upperlimb.msg")
    )
    upperlimb_srv, upperlimb_srv_namespace = _import_first(
        ("zj_humanoid.upperlimb.srv", "upperlimb.srv")
    )
    hand_msg, hand_namespace = _import_first(("zj_humanoid.hand.msg", "hand.msg"))
    if upperlimb_namespace.rsplit(".", 1)[0] != upperlimb_srv_namespace.rsplit(".", 1)[0]:
        raise ImportError(
            "Upperlimb message and service namespaces do not match: "
            f"{upperlimb_namespace}, {upperlimb_srv_namespace}"
        )
    return RosTypes(
        rospy=rospy,
        DualPose=getattr(upperlimb_msg, "DualPose"),
        Pose=getattr(upperlimb_msg, "Pose"),
        TcpSpeed=getattr(upperlimb_msg, "TcpSpeed"),
        Servo=getattr(upperlimb_srv, "Servo"),
        ServoRequest=getattr(upperlimb_srv, "ServoRequest"),
        PressureSensor=getattr(hand_msg, "PressureSensor"),
        CompressedImage=getattr(compressed_image_module, "CompressedImage"),
        WrenchStamped=getattr(wrench_module, "WrenchStamped"),
        upperlimb_namespace=upperlimb_namespace,
        hand_namespace=hand_namespace,
    )


def _ros_stamp_ns(message: Any) -> int:
    try:
        return int(message.header.stamp.to_nsec())
    except (AttributeError, TypeError):
        return 0


def _normalized_pose(value: NDArray[Any], name: str) -> FloatArray:
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or np.any(~np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite [x,y,z,qw,qx,qy,qz] pose")
    quaternion_norm = float(np.linalg.norm(pose[3:]))
    if quaternion_norm <= 1e-12:
        raise ValueError(f"{name} contains a zero quaternion")
    result = pose.copy()
    result[3:] /= quaternion_norm
    return result


def fill_geometry_pose(message: Any, wxyz_pose: NDArray[Any]) -> None:
    pose = _normalized_pose(wxyz_pose, "TCP target")
    message.position.x = float(pose[0])
    message.position.y = float(pose[1])
    message.position.z = float(pose[2])
    message.orientation.x = float(pose[4])
    message.orientation.y = float(pose[5])
    message.orientation.z = float(pose[6])
    message.orientation.w = float(pose[3])


class KionObservationBuffer:
    REQUIRED_KEYS = (
        "observation.images.left",
        "observation.images.right",
        "observation.images.head",
        LEFT_POSE_KEY,
        RIGHT_POSE_KEY,
        LEFT_TWIST_KEY,
        RIGHT_TWIST_KEY,
        "observation.state.left_finger_pressure",
        "observation.state.right_finger_pressure",
        "observation.state.left_wrist_force",
        "observation.state.right_wrist_force",
    )

    def __init__(
        self,
        ros_types: RosTypes,
        topics: KionTopics,
        *,
        image_size: int = 224,
        wrist_force_deadzone: float = 10.0,
    ) -> None:
        self.ros_types = ros_types
        self.rospy = ros_types.rospy
        self.topics = topics
        self.image_size = int(image_size)
        self.wrist_force_deadzone = float(wrist_force_deadzone)
        self._lock = threading.Lock()
        self._values: dict[str, NDArray[Any]] = {}
        self._received_ns: dict[str, int] = {}
        self._ros_ns: dict[str, int] = {}
        self._previous_quaternion: dict[str, FloatArray | None] = {
            "left": None,
            "right": None,
        }
        self._filtered_force: dict[str, FloatArray | None] = {
            "left": None,
            "right": None,
        }
        self._subscribers = [
            self.rospy.Subscriber(
                topics.left_camera,
                ros_types.CompressedImage,
                self._camera_callback,
                callback_args="left",
                queue_size=1,
                tcp_nodelay=True,
            ),
            self.rospy.Subscriber(
                topics.right_camera,
                ros_types.CompressedImage,
                self._camera_callback,
                callback_args="right",
                queue_size=1,
                tcp_nodelay=True,
            ),
            self.rospy.Subscriber(
                topics.head_camera,
                ros_types.CompressedImage,
                self._camera_callback,
                callback_args="head",
                queue_size=1,
                tcp_nodelay=True,
            ),
            self.rospy.Subscriber(
                topics.left_pose,
                ros_types.Pose,
                self._pose_callback,
                callback_args="left",
                queue_size=1,
                tcp_nodelay=True,
            ),
            self.rospy.Subscriber(
                topics.right_pose,
                ros_types.Pose,
                self._pose_callback,
                callback_args="right",
                queue_size=1,
                tcp_nodelay=True,
            ),
            self.rospy.Subscriber(
                topics.dual_twist,
                ros_types.TcpSpeed,
                self._twist_callback,
                queue_size=1,
                tcp_nodelay=True,
            ),
            self.rospy.Subscriber(
                topics.left_pressure,
                ros_types.PressureSensor,
                self._pressure_callback,
                callback_args="left",
                queue_size=1,
                tcp_nodelay=True,
            ),
            self.rospy.Subscriber(
                topics.right_pressure,
                ros_types.PressureSensor,
                self._pressure_callback,
                callback_args="right",
                queue_size=1,
                tcp_nodelay=True,
            ),
            self.rospy.Subscriber(
                topics.left_wrist_force,
                ros_types.WrenchStamped,
                self._force_callback,
                callback_args="left",
                queue_size=1,
                tcp_nodelay=True,
            ),
            self.rospy.Subscriber(
                topics.right_wrist_force,
                ros_types.WrenchStamped,
                self._force_callback,
                callback_args="right",
                queue_size=1,
                tcp_nodelay=True,
            ),
        ]

    def _store(self, key: str, value: NDArray[Any], message: Any) -> None:
        received_ns = time.monotonic_ns()
        ros_ns = _ros_stamp_ns(message)
        with self._lock:
            self._values[key] = value
            self._received_ns[key] = received_ns
            self._ros_ns[key] = ros_ns

    def _camera_callback(self, message: Any, side: str) -> None:
        encoded = np.frombuffer(message.data, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            LOGGER.error("Failed to decode %s camera frame", side)
            return
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        chw = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.uint8)
        self._store(f"observation.images.{side}", chw, message)

    def _pose_callback(self, message: Any, arm: str) -> None:
        quaternion = np.array(
            [
                message.quaternion.w,
                message.quaternion.x,
                message.quaternion.y,
                message.quaternion.z,
            ],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12 or np.any(~np.isfinite(quaternion)):
            LOGGER.error("Rejected invalid %s TCP quaternion", arm)
            return
        quaternion /= norm
        with self._lock:
            previous = self._previous_quaternion[arm]
            if previous is not None and float(np.dot(previous, quaternion)) < 0.0:
                quaternion = -quaternion
            self._previous_quaternion[arm] = quaternion.copy()
        pose = np.array(
            [
                message.position.x,
                message.position.y,
                message.position.z,
                *quaternion,
            ],
            dtype=np.float32,
        )
        self._store(f"observation.state.{arm}_tcp", pose, message)

    def _twist_callback(self, message: Any) -> None:
        received_ns = time.monotonic_ns()
        ros_ns = _ros_stamp_ns(message)
        left = np.asarray(message.left_arm, dtype=np.float32).reshape(-1)
        right = np.asarray(message.right_arm, dtype=np.float32).reshape(-1)
        if left.shape != (6,) or right.shape != (6,):
            LOGGER.error("Rejected TcpSpeed with shapes left=%s right=%s", left.shape, right.shape)
            return
        if np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
            LOGGER.error("Rejected non-finite TcpSpeed")
            return
        with self._lock:
            for key, value in ((LEFT_TWIST_KEY, left), (RIGHT_TWIST_KEY, right)):
                self._values[key] = value
                self._received_ns[key] = received_ns
                self._ros_ns[key] = ros_ns

    def _pressure_callback(self, message: Any, arm: str) -> None:
        pressure = np.asarray(message.pressure, dtype=np.float32).reshape(-1)
        if pressure.shape != (6,) or np.any(~np.isfinite(pressure)):
            LOGGER.error("Rejected %s finger pressure with shape %s", arm, pressure.shape)
            return
        self._store(f"observation.state.{arm}_finger_pressure", pressure, message)

    def _force_callback(self, message: Any, arm: str) -> None:
        force = np.array(
            [message.wrench.force.x, message.wrench.force.y, message.wrench.force.z],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(force)):
            LOGGER.error("Rejected non-finite %s wrist force", arm)
            return
        with self._lock:
            previous = self._filtered_force[arm]
            filtered = force if previous is None else 0.95 * previous + 0.05 * force
            self._filtered_force[arm] = filtered
        deadzone = self.wrist_force_deadzone
        filtered = (np.abs(filtered) > deadzone) * (filtered - np.sign(filtered) * deadzone)
        self._store(
            f"observation.state.{arm}_wrist_force",
            filtered.astype(np.float32),
            message,
        )

    def wait_until_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        last_missing: tuple[str, ...] | None = None
        while not self.rospy.is_shutdown():
            with self._lock:
                missing = tuple(key for key in self.REQUIRED_KEYS if key not in self._values)
            if not missing:
                return
            if missing != last_missing:
                LOGGER.info("Waiting for ROS observations: %s", ", ".join(missing))
                last_missing = missing
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for ROS observations: " + ", ".join(missing))
            time.sleep(0.05)
        raise RuntimeError("ROS shutdown while waiting for observations")

    def snapshot(
        self,
        *,
        max_state_age_s: float,
        max_image_age_s: float,
    ) -> RobotSnapshot:
        now_ns = time.monotonic_ns()
        with self._lock:
            missing = [key for key in self.REQUIRED_KEYS if key not in self._values]
            if missing:
                raise RuntimeError("Missing ROS observations: " + ", ".join(missing))
            values = dict(self._values)
            received_ns = dict(self._received_ns)
            ros_ns = dict(self._ros_ns)

        stale: list[str] = []
        for key in self.REQUIRED_KEYS:
            max_age_s = max_image_age_s if key.startswith("observation.images") else max_state_age_s
            if (now_ns - received_ns[key]) * 1e-9 > max_age_s:
                stale.append(key)
        return RobotSnapshot(
            observation=values,
            left_pose=np.asarray(values[LEFT_POSE_KEY], dtype=np.float64),
            right_pose=np.asarray(values[RIGHT_POSE_KEY], dtype=np.float64),
            left_twist=np.asarray(values[LEFT_TWIST_KEY], dtype=np.float64),
            right_twist=np.asarray(values[RIGHT_TWIST_KEY], dtype=np.float64),
            left_pose_ros_ns=ros_ns[LEFT_POSE_KEY],
            left_pose_received_monotonic_ns=received_ns[LEFT_POSE_KEY],
            right_pose_ros_ns=ros_ns[RIGHT_POSE_KEY],
            right_pose_received_monotonic_ns=received_ns[RIGHT_POSE_KEY],
            twist_ros_ns=ros_ns[LEFT_TWIST_KEY],
            twist_received_monotonic_ns=received_ns[LEFT_TWIST_KEY],
            stale_fields=tuple(stale),
        )

    def close(self) -> None:
        for subscriber in self._subscribers:
            subscriber.unregister()


class KionDualArmServo:
    def __init__(
        self,
        ros_types: RosTypes,
        topics: KionTopics,
        *,
        control_frequency: float,
        gain: int,
        dry_run: bool,
    ) -> None:
        self.ros_types = ros_types
        self.rospy = ros_types.rospy
        self.topics = topics
        self.period_s = 1.0 / float(control_frequency)
        self.gain = int(gain)
        self.dry_run = bool(dry_run)
        self._configured = False
        self._publisher = None
        if not self.dry_run:
            self._publisher = self.rospy.Publisher(
                topics.dual_servo,
                ros_types.DualPose,
                queue_size=1,
                tcp_nodelay=True,
            )

    def wait_for_connection(self, timeout_s: float) -> None:
        if self.dry_run:
            return
        assert self._publisher is not None
        deadline = time.monotonic() + timeout_s
        while not self.rospy.is_shutdown():
            if self._publisher.get_num_connections() > 0:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "No subscriber connected to "
                    f"{self.topics.dual_servo}; verify its ROS type is "
                    f"{self.ros_types.DualPose._type}"
                )
            time.sleep(0.05)
        raise RuntimeError("ROS shutdown while waiting for dual-arm servo")

    def configure(self, timeout_s: float) -> None:
        if self.dry_run:
            return
        self.rospy.wait_for_service(self.topics.set_servo_params, timeout=timeout_s)
        request = self.ros_types.ServoRequest()
        request.v = 0.0
        request.acc = 0.0
        request.time = self.period_s
        request.lookahead_time = 0.2
        request.gain = self.gain
        request.arm_type = 0
        response = self.rospy.ServiceProxy(
            self.topics.set_servo_params,
            self.ros_types.Servo,
        )(request)
        if not response.success:
            raise RuntimeError(f"set_servo_params failed: {response.message}")
        self._configured = True
        LOGGER.info(
            "Configured dual-arm servo: period=%.6f s gain=%d response=%s",
            self.period_s,
            self.gain,
            response.message,
        )

    def publish(self, left_pose: NDArray[Any], right_pose: NDArray[Any]) -> tuple[int, int]:
        left_pose = _normalized_pose(left_pose, "left TCP target")
        right_pose = _normalized_pose(right_pose, "right TCP target")
        published_monotonic_ns = time.monotonic_ns()
        published_ros_ns = int(self.rospy.Time.now().to_nsec())
        if not self.dry_run:
            assert self._publisher is not None
            message = self.ros_types.DualPose()
            fill_geometry_pose(message.left_arm_pose, left_pose)
            fill_geometry_pose(message.right_arm_pose, right_pose)
            self._publisher.publish(message)
        return published_monotonic_ns, published_ros_ns

    def close(self) -> None:
        if self.dry_run or not self._configured:
            return
        try:
            self.rospy.wait_for_service(self.topics.clear_servo_params, timeout=2.0)
            response = self.rospy.ServiceProxy(
                self.topics.clear_servo_params,
                self.ros_types.Servo,
            )(self.ros_types.ServoRequest())
            if not response.success:
                LOGGER.error("clear_servo_params failed: %s", response.message)
        except Exception:
            LOGGER.exception("Failed to clear servo parameters")
        finally:
            self._configured = False


class KionToppraRosClient:
    def __init__(
        self,
        *,
        ros_types: RosTypes,
        topics: KionTopics,
        agent: GR00TAgent,
        recorder: TrackingRecorder,
        control_frequency: float,
        servo_gain: int,
        startup_timeout_s: float,
        max_state_age_s: float,
        max_image_age_s: float,
        task: str,
        dry_run: bool,
    ) -> None:
        self.rospy = ros_types.rospy
        self.agent = agent
        self.recorder = recorder
        self.task = task
        self.control_frequency = float(control_frequency)
        self.control_period_ns = int(round(1e9 / self.control_frequency))
        self.startup_timeout_s = float(startup_timeout_s)
        self.max_state_age_s = float(max_state_age_s)
        self.max_image_age_s = float(max_image_age_s)
        self.observations = KionObservationBuffer(ros_types, topics)
        self.servo = KionDualArmServo(
            ros_types,
            topics,
            control_frequency=control_frequency,
            gain=servo_gain,
            dry_run=dry_run,
        )
        self._last_target: dict[str, FloatArray] | None = None
        self._closed = False

    def run(self) -> None:
        self.observations.wait_until_ready(self.startup_timeout_s)
        self.servo.wait_for_connection(self.startup_timeout_s)
        self.servo.configure(self.startup_timeout_s)
        LOGGER.info("Starting Kion TOPPRA control loop at %.3f Hz", self.control_frequency)
        next_tick_ns = time.monotonic_ns()
        sample_index = 0
        last_stale_fields: tuple[str, ...] = ()
        last_deadline_warning_ns = 0

        while not self.rospy.is_shutdown():
            loop_started_ns = time.monotonic_ns()
            lateness_ns = max(loop_started_ns - next_tick_ns, 0)
            snapshot = self.observations.snapshot(
                max_state_age_s=self.max_state_age_s,
                max_image_age_s=self.max_image_age_s,
            )

            if snapshot.stale_fields:
                if snapshot.stale_fields != last_stale_fields:
                    LOGGER.error(
                        "Holding target because ROS observations are stale: %s",
                        ", ".join(snapshot.stale_fields),
                    )
                if self._last_target is None:
                    target_left = snapshot.left_pose.copy()
                    target_right = snapshot.right_pose.copy()
                else:
                    target_left = self._last_target["left"]
                    target_right = self._last_target["right"]
                diagnostics: dict[str, Any] = {
                    "active_sequence_id": None,
                    "execution_state": "state_stale_hold",
                    "active_buffer_size": 0,
                }
            else:
                if last_stale_fields:
                    LOGGER.info("ROS observations recovered; resuming trajectory consumption")
                action = self.agent.act(snapshot.observation, self.task)
                target_left = _normalized_pose(action[LEFT_TARGET_KEY], LEFT_TARGET_KEY)
                target_right = _normalized_pose(action[RIGHT_TARGET_KEY], RIGHT_TARGET_KEY)
                self._last_target = {
                    "left": target_left.copy(),
                    "right": target_right.copy(),
                }
                diagnostics = self.agent.diagnostics()
            last_stale_fields = snapshot.stale_fields

            published_monotonic_ns, published_ros_ns = self.servo.publish(
                target_left,
                target_right,
            )
            loop_duration_ns = time.monotonic_ns() - loop_started_ns
            self.recorder.record(
                make_tracking_row(
                    sample_index=sample_index,
                    loop_started_monotonic_ns=loop_started_ns,
                    published_monotonic_ns=published_monotonic_ns,
                    published_ros_ns=published_ros_ns,
                    loop_duration_ns=loop_duration_ns,
                    deadline_lateness_ns=lateness_ns,
                    active_sequence_id=diagnostics.get("active_sequence_id"),
                    execution_state=str(diagnostics.get("execution_state", "unknown")),
                    active_buffer_size=int(diagnostics.get("active_buffer_size") or 0),
                    left_pose_ros_ns=snapshot.left_pose_ros_ns,
                    left_pose_received_monotonic_ns=snapshot.left_pose_received_monotonic_ns,
                    right_pose_ros_ns=snapshot.right_pose_ros_ns,
                    right_pose_received_monotonic_ns=snapshot.right_pose_received_monotonic_ns,
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
            sample_index += 1

            next_tick_ns += self.control_period_ns
            now_ns = time.monotonic_ns()
            remaining_ns = next_tick_ns - now_ns
            if remaining_ns > 0:
                time.sleep(remaining_ns * 1e-9)
                continue

            if now_ns - last_deadline_warning_ns >= 1_000_000_000:
                LOGGER.warning(
                    "Control deadline missed by %.3f ms; trajectory samples are not caught up",
                    -remaining_ns * 1e-6,
                )
                last_deadline_warning_ns = now_ns
            next_tick_ns = now_ns + self.control_period_ns

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for close in (
            self.servo.close,
            self.agent.teardown,
            self.observations.close,
            self.recorder.close,
        ):
            try:
                close()
            except BaseException as exc:
                errors.append(exc)
                LOGGER.exception("Kion client cleanup failed")
        if errors:
            raise RuntimeError(
                f"Kion client cleanup failed with {len(errors)} error(s)"
            ) from errors[0]


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bimanual GR00T TOPPRA agent against the Kion ROS SDK."
    )
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--timeout-ms", type=int, default=15_000)
    parser.add_argument("--policy-frequency", type=_positive_float, default=30.0)
    parser.add_argument("--control-frequency", type=_positive_float, default=250.0)
    parser.add_argument("--inference-mode", choices=("sync", "async"), default="async")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--refill-threshold", type=int, default=20)
    parser.add_argument("--max-latency-s", type=float, default=0.12)
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
    parser.add_argument("--log-root", type=Path, default=Path("logs/kion_toppra"))
    parser.add_argument("--node-name", default="gr00t_kion_toppra_client")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--left-camera-topic", default=KionTopics.left_camera)
    parser.add_argument("--right-camera-topic", default=KionTopics.right_camera)
    parser.add_argument("--head-camera-topic", default=KionTopics.head_camera)
    return parser.parse_args()


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
    )
    ros_types = load_ros_types()
    ros_types.rospy.init_node(args.node_name, disable_signals=False)
    if not 100 <= args.servo_gain <= 1000:
        raise ValueError("--servo-gain must be in the SDK range [100, 1000]")
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
    run_directory = args.log_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    metadata = {
        "arguments": _jsonable_args(args),
        "topics": asdict(topics),
        "upperlimb_python_namespace": ros_types.upperlimb_namespace,
        "upperlimb_dual_pose_ros_type": ros_types.DualPose._type,
        "upperlimb_tcp_speed_ros_type": ros_types.TcpSpeed._type,
        "hand_python_namespace": ros_types.hand_namespace,
        "pose_semantics": "[x,y,z,qw,qx,qy,qz] in robot base frame",
        "twist_semantics": "[vx,vy,vz,wx,wy,wz], expected in robot base frame",
    }
    agent: GR00TAgent | None = None
    recorder: TrackingRecorder | None = None
    client: KionToppraRosClient | None = None
    try:
        agent = GR00TAgent(
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
        recorder = TrackingRecorder(run_directory, metadata)
        client = KionToppraRosClient(
            ros_types=ros_types,
            topics=topics,
            agent=agent,
            recorder=recorder,
            control_frequency=args.control_frequency,
            servo_gain=args.servo_gain,
            startup_timeout_s=args.startup_timeout_s,
            max_state_age_s=args.max_state_age_s,
            max_image_age_s=args.max_image_age_s,
            task=args.task,
            dry_run=args.dry_run,
        )
        LOGGER.info(
            "ROS SDK namespace=%s DualPose=%s TcpSpeed=%s tracking_log=%s",
            ros_types.upperlimb_namespace,
            ros_types.DualPose._type,
            ros_types.TcpSpeed._type,
            recorder.csv_path,
        )
        client.run()
    finally:
        if client is not None:
            client.close()
        else:
            if agent is not None:
                agent.teardown()
            if recorder is not None:
                recorder.close()


if __name__ == "__main__":
    main()
