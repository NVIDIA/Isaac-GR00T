from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import cv2
from gr00t.eval.real_robot.TOPPRA.kion_client.client import (
    KionDualArmServo,
    KionObservationBuffer,
    KionTopics,
    RosTypes,
    fill_geometry_pose,
)
from gr00t.eval.real_robot.TOPPRA.kion_client.tracking import (
    TrackingRecorder,
    estimate_tracking_delay,
    make_tracking_row,
)
import numpy as np


def _identity_pose(position: np.ndarray) -> np.ndarray:
    return np.concatenate((np.asarray(position, dtype=np.float64), [1.0, 0.0, 0.0, 0.0]))


def _tracking_row(
    *,
    index: int,
    publish_ns: int,
    feedback_ns: int,
    target_pose: np.ndarray,
    measured_pose: np.ndarray,
) -> tuple:
    return make_tracking_row(
        sample_index=index,
        loop_started_monotonic_ns=publish_ns,
        published_monotonic_ns=publish_ns,
        published_ros_ns=publish_ns,
        loop_duration_ns=100_000,
        deadline_lateness_ns=0,
        active_sequence_id=1,
        execution_state="executing",
        active_buffer_size=100,
        left_pose_ros_ns=feedback_ns,
        left_pose_received_monotonic_ns=feedback_ns,
        right_pose_ros_ns=feedback_ns,
        right_pose_received_monotonic_ns=feedback_ns,
        twist_ros_ns=feedback_ns,
        twist_received_monotonic_ns=feedback_ns,
        target_left=target_pose,
        measured_left=measured_pose,
        measured_left_twist=np.zeros(6),
        target_right=target_pose,
        measured_right=measured_pose,
        measured_right_twist=np.zeros(6),
    )


def test_fill_geometry_pose_converts_wxyz_to_ros_xyzw() -> None:
    message = SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
    )
    fill_geometry_pose(
        message,
        np.array([0.4, -0.2, 0.5, 2.0, 0.2, 0.4, 0.6]),
    )
    assert (message.position.x, message.position.y, message.position.z) == (0.4, -0.2, 0.5)
    expected = np.array([2.0, 0.2, 0.4, 0.6])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(
        [
            message.orientation.w,
            message.orientation.x,
            message.orientation.y,
            message.orientation.z,
        ],
        expected,
    )


def test_tracking_recorder_and_delay_estimator_recover_known_delay(tmp_path) -> None:
    recorder = TrackingRecorder(
        tmp_path / "rollout",
        {"control_frequency": 250.0},
        flush_rows=20,
    )
    control_dt = 1.0 / 250.0
    feedback_dt = 1.0 / 100.0
    expected_delay_s = 0.08
    base_ns = 10_000_000_000
    latest_feedback_s = 0.0
    latest_measured = _identity_pose(np.zeros(3))

    def position_at(time_s: float) -> np.ndarray:
        return np.array(
            [
                0.10 * np.sin(2.0 * np.pi * 0.7 * time_s),
                0.06 * np.sin(2.0 * np.pi * 1.1 * time_s + 0.3),
                0.04 * np.sin(2.0 * np.pi * 0.4 * time_s + 0.8),
            ]
        )

    duration_s = 4.0
    for index, target_time_s in enumerate(np.arange(0.0, duration_s, control_dt)):
        feedback_time_s = math_floor(target_time_s / feedback_dt) * feedback_dt
        if feedback_time_s > latest_feedback_s or index == 0:
            latest_feedback_s = feedback_time_s
            latest_measured = _identity_pose(
                position_at(max(feedback_time_s - expected_delay_s, 0.0))
            )
        target = _identity_pose(position_at(target_time_s))
        recorder.record(
            _tracking_row(
                index=index,
                publish_ns=base_ns + round(target_time_s * 1e9),
                feedback_ns=base_ns + round(latest_feedback_s * 1e9),
                target_pose=target,
                measured_pose=latest_measured,
            )
        )
    recorder.close()

    metadata = json.loads(recorder.metadata_path.read_text(encoding="utf-8"))
    assert metadata["tracking_dropped_rows"] == 0
    with recorder.csv_path.open(newline="", encoding="utf-8") as file:
        assert sum(1 for _row in csv.reader(file)) == round(duration_s / control_dt) + 1

    estimate = estimate_tracking_delay(
        recorder.csv_path,
        "left",
        max_lag_ms=150.0,
        lag_step_ms=1.0,
    )
    assert abs(estimate.delay_ms - expected_delay_s * 1000.0) <= 2.0
    assert abs(estimate.control_rate_hz - 250.0) < 0.1
    assert abs(estimate.pose_feedback_rate_hz - 100.0) < 0.1
    assert estimate.best_lag_position_rmse_m < estimate.zero_lag_position_rmse_m


def math_floor(value: float) -> int:
    return int(np.floor(value + 1e-12))


class _FakeStamp:
    def __init__(self, nanoseconds: int) -> None:
        self.nanoseconds = nanoseconds

    def to_nsec(self) -> int:
        return self.nanoseconds


def _header(nanoseconds: int = 123) -> SimpleNamespace:
    return SimpleNamespace(stamp=_FakeStamp(nanoseconds))


class _FakeGeometryPose:
    def __init__(self) -> None:
        self.position = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)


class _FakeDualPose:
    _type = "upperlimb/DualPose"

    def __init__(self) -> None:
        self.left_arm_pose = _FakeGeometryPose()
        self.right_arm_pose = _FakeGeometryPose()


class _FakeTcpSpeed:
    _type = "upperlimb/TcpSpeed"


class _FakeServoRequest:
    def __init__(self) -> None:
        self.v = 0.0
        self.acc = 0.0
        self.time = 0.0
        self.lookahead_time = 0.0
        self.gain = 0
        self.arm_type = 0


class _FakePublisher:
    def __init__(self) -> None:
        self.messages: list[_FakeDualPose] = []

    def get_num_connections(self) -> int:
        return 1

    def publish(self, message: _FakeDualPose) -> None:
        self.messages.append(message)


class _FakeRospy:
    class Time:
        @staticmethod
        def now() -> _FakeStamp:
            return _FakeStamp(456)

    def __init__(self) -> None:
        self.publisher = _FakePublisher()
        self.service_requests: dict[str, _FakeServoRequest] = {}

    def Subscriber(
        self,
        _name,
        _data_class,
        _callback=None,
        callback_args=None,
        queue_size=None,
        buff_size=65536,
        tcp_nodelay=False,
    ) -> SimpleNamespace:
        del callback_args, queue_size, buff_size, tcp_nodelay
        return SimpleNamespace(unregister=lambda: None)

    def Publisher(self, *_args, **_kwargs) -> _FakePublisher:
        return self.publisher

    def ServiceProxy(self, name, _service_type):
        def call(request):
            self.service_requests[name] = request
            return SimpleNamespace(success=True, message="ok")

        return call

    def wait_for_service(self, _name, timeout) -> None:
        assert timeout > 0

    def is_shutdown(self) -> bool:
        return False


def _fake_ros_types() -> RosTypes:
    rospy = _FakeRospy()
    return RosTypes(
        rospy=rospy,  # type: ignore[arg-type]
        DualPose=_FakeDualPose,
        Pose=object,
        TcpSpeed=_FakeTcpSpeed,
        Servo=object,
        ServoRequest=_FakeServoRequest,
        PressureSensor=object,
        CompressedImage=object,
        WrenchStamped=object,
        upperlimb_namespace="upperlimb.msg",
        hand_namespace="hand.msg",
    )


def test_ros_observation_snapshot_and_dual_pose_publish() -> None:
    ros_types = _fake_ros_types()
    topics = KionTopics()
    observations = KionObservationBuffer(ros_types, topics)
    encoded_ok, encoded = cv2.imencode(".jpg", np.full((16, 16, 3), 127, dtype=np.uint8))
    assert encoded_ok
    camera = SimpleNamespace(data=encoded.tobytes(), header=_header())
    for side in ("left", "right", "head"):
        observations._camera_callback(camera, side)

    def pose_message(y: float) -> SimpleNamespace:
        return SimpleNamespace(
            header=_header(),
            position=SimpleNamespace(x=0.4, y=y, z=0.5),
            quaternion=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
        )

    observations._pose_callback(pose_message(0.2), "left")
    observations._pose_callback(pose_message(-0.2), "right")
    observations._twist_callback(
        SimpleNamespace(
            header=_header(),
            left_arm=[0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
            right_arm=[0.0, -0.01, 0.0, 0.0, 0.0, 0.0],
        )
    )
    for arm in ("left", "right"):
        observations._pressure_callback(
            SimpleNamespace(header=_header(), pressure=np.arange(6)),
            arm,
        )
        observations._force_callback(
            SimpleNamespace(
                header=_header(),
                wrench=SimpleNamespace(force=SimpleNamespace(x=11.0, y=0.0, z=-12.0)),
            ),
            arm,
        )

    snapshot = observations.snapshot(max_state_age_s=1.0, max_image_age_s=1.0)
    assert not snapshot.stale_fields
    assert set(snapshot.observation) == set(KionObservationBuffer.REQUIRED_KEYS)
    np.testing.assert_allclose(snapshot.left_pose, [0.4, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(snapshot.left_twist, [0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        snapshot.observation["observation.state.left_wrist_force"],
        [1.0, 0.0, -2.0],
    )

    servo = KionDualArmServo(
        ros_types,
        topics,
        control_frequency=250.0,
        gain=800,
        dry_run=False,
    )
    servo.wait_for_connection(1.0)
    servo.configure(1.0)
    servo.publish(snapshot.left_pose, snapshot.right_pose)
    message = ros_types.rospy.publisher.messages[-1]
    assert message.left_arm_pose.orientation.w == 1.0
    assert np.isclose(message.right_arm_pose.position.y, -0.2)
    request = ros_types.rospy.service_requests[topics.set_servo_params]
    assert request.time == 0.004
    assert request.gain == 800
    servo.close()
    observations.close()
