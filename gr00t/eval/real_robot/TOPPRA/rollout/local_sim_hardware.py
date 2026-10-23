"""Localhost-only ROS hardware simulator for end-to-end rollout checks.

The simulator publishes every observation required by the parcel rollout, provides
the two Servo configuration services, and subscribes to the dual-arm Servo target.
Received targets are fed back as the next measured TCP poses.  It refuses to run
unless ROS_MASTER_URI points at loopback, so it cannot impersonate hardware on the
robot ROS graph.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import ipaddress
import os
import threading
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from ..kion_client.client import KionTopics, load_ros_types
from .reset import DUAL_ARM_HOME_SERVICE


@dataclass(frozen=True)
class SimulationConfig:
    image_frequency: float = 30.0
    state_frequency: float = 250.0
    print_every: int = 25


def require_loopback_master(uri: str) -> None:
    """Reject any ROS graph that could be connected to physical hardware."""

    parsed = urlparse(uri)
    host = parsed.hostname
    if parsed.scheme != "http" or host is None:
        raise RuntimeError(f"ROS_MASTER_URI is invalid: {uri!r}")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == "localhost"
    if not loopback:
        raise RuntimeError(
            "The local rollout simulator requires a loopback ROS master; "
            f"refusing ROS_MASTER_URI={uri!r}"
        )


def _jpeg_image() -> bytes:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[:, :112] = (40, 80, 160)
    image[:, 112:] = (160, 80, 40)
    cv2.putText(
        image,
        "GR00T LOCAL SIM",
        (18, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Could not encode the simulated camera image")
    return encoded.tobytes()


def _copy_geometry_pose(message: Any) -> np.ndarray:
    return np.array(
        [
            message.position.x,
            message.position.y,
            message.position.z,
            message.orientation.w,
            message.orientation.x,
            message.orientation.y,
            message.orientation.z,
        ],
        dtype=np.float64,
    )


class LocalRolloutHardwareSimulator:
    def __init__(self, rospy: Any, config: SimulationConfig) -> None:
        from geometry_msgs.msg import WrenchStamped
        from sensor_msgs.msg import CompressedImage
        from std_srvs.srv import Trigger

        self.rospy = rospy
        self.config = config
        self.ros_types = load_ros_types()
        self.topics = KionTopics()
        self._lock = threading.Lock()
        self._servo_targets = 0
        self._left_pose = np.array(
            [0.7303, 0.3359, 0.8453, 0.7419, 0.1857, -0.6276, -0.1457],
            dtype=np.float64,
        )
        self._right_pose = np.array(
            [0.7316, -0.3139, 0.9298, 0.5486, -0.1116, -0.8227, 0.0990],
            dtype=np.float64,
        )
        self._jpeg = _jpeg_image()

        self._camera_publishers = [
            rospy.Publisher(topic, CompressedImage, queue_size=1)
            for topic in (
                self.topics.head_camera,
                self.topics.left_camera,
                self.topics.right_camera,
            )
        ]
        self._left_pose_publisher = rospy.Publisher(
            self.topics.left_pose,
            self.ros_types.Pose,
            queue_size=1,
        )
        self._right_pose_publisher = rospy.Publisher(
            self.topics.right_pose,
            self.ros_types.Pose,
            queue_size=1,
        )
        self._twist_publisher = rospy.Publisher(
            self.topics.dual_twist,
            self.ros_types.TcpSpeed,
            queue_size=1,
        )
        self._left_force_publisher = rospy.Publisher(
            self.topics.left_wrist_force,
            WrenchStamped,
            queue_size=1,
        )
        self._right_force_publisher = rospy.Publisher(
            self.topics.right_wrist_force,
            WrenchStamped,
            queue_size=1,
        )
        self._servo_subscriber = rospy.Subscriber(
            self.topics.dual_servo,
            self.ros_types.DualPose,
            self._servo_target_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self._services = [
            rospy.Service(
                self.topics.set_servo_params,
                self.ros_types.Servo,
                self._servo_service,
            ),
            rospy.Service(
                self.topics.clear_servo_params,
                self.ros_types.Servo,
                self._servo_service,
            ),
            rospy.Service(DUAL_ARM_HOME_SERVICE, Trigger, self._home_service),
        ]
        self._image_timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / config.image_frequency),
            self._publish_images,
        )
        self._state_timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / config.state_frequency),
            self._publish_state,
        )

    @property
    def servo_target_count(self) -> int:
        with self._lock:
            return self._servo_targets

    def _servo_service(self, _request: Any) -> Any:
        return self.ros_types.Servo._response_class(
            success=True,
            message="local simulation accepted",
        )

    @staticmethod
    def _home_service(_request: Any) -> Any:
        from std_srvs.srv import TriggerResponse

        return TriggerResponse(success=True, message="local simulation accepted home request")

    def _servo_target_callback(self, message: Any) -> None:
        left = _copy_geometry_pose(message.left_arm_pose)
        right = _copy_geometry_pose(message.right_arm_pose)
        with self._lock:
            self._left_pose = left
            self._right_pose = right
            self._servo_targets += 1
            count = self._servo_targets
        if count == 1 or count % self.config.print_every == 0:
            self.rospy.loginfo(
                "SERVO ACTION #%d left=%s right=%s",
                count,
                np.array2string(left, precision=5, separator=","),
                np.array2string(right, precision=5, separator=","),
            )

    def _publish_images(self, _event: Any) -> None:
        from sensor_msgs.msg import CompressedImage

        stamp = self.rospy.Time.now()
        message = CompressedImage()
        message.header.stamp = stamp
        message.format = "jpeg"
        message.data = self._jpeg
        for publisher in self._camera_publishers:
            publisher.publish(message)

    @staticmethod
    def _fill_pose(message: Any, pose: np.ndarray, stamp: Any) -> None:
        message.header.stamp = stamp
        message.position.x, message.position.y, message.position.z = pose[:3]
        message.quaternion.w = pose[3]
        message.quaternion.x = pose[4]
        message.quaternion.y = pose[5]
        message.quaternion.z = pose[6]

    def _publish_state(self, _event: Any) -> None:
        from geometry_msgs.msg import WrenchStamped

        stamp = self.rospy.Time.now()
        with self._lock:
            left = self._left_pose.copy()
            right = self._right_pose.copy()

        left_pose = self.ros_types.Pose()
        right_pose = self.ros_types.Pose()
        self._fill_pose(left_pose, left, stamp)
        self._fill_pose(right_pose, right, stamp)
        self._left_pose_publisher.publish(left_pose)
        self._right_pose_publisher.publish(right_pose)

        twist = self.ros_types.TcpSpeed()
        twist.header.stamp = stamp
        twist.left_arm = [0.0] * 6
        twist.right_arm = [0.0] * 6
        self._twist_publisher.publish(twist)

        force = WrenchStamped()
        force.header.stamp = stamp
        self._left_force_publisher.publish(force)
        self._right_force_publisher.publish(force)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-frequency", type=float, default=30.0)
    parser.add_argument("--state-frequency", type=float, default=250.0)
    parser.add_argument("--print-every", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.image_frequency <= 0 or args.state_frequency <= 0 or args.print_every <= 0:
        raise ValueError("simulation frequencies and print interval must be positive")
    master_uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
    require_loopback_master(master_uri)

    import rospy

    rospy.init_node("gr00t_local_sim_hardware", anonymous=False)
    simulator = LocalRolloutHardwareSimulator(
        rospy,
        SimulationConfig(
            image_frequency=args.image_frequency,
            state_frequency=args.state_frequency,
            print_every=args.print_every,
        ),
    )
    rospy.logwarn(
        "LOCAL SIMULATION ONLY: publishing synthetic observations on %s; Servo targets will be "
        "printed and fed back as measured TCP poses",
        master_uri,
    )
    rospy.spin()
    rospy.loginfo("Local simulation received %d Servo targets", simulator.servo_target_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
