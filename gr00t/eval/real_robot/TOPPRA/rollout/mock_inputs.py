"""Test-only ROS relay for missing wrist cameras.

This module exists solely to verify the rollout data path. It never publishes robot
commands. The head image is copied to both wrist-camera topics; force observations
continue to come from the robot's real compensation node.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import time
from typing import Any


@dataclass(frozen=True)
class MockInputTopics:
    head_camera: str = "/zj_humanoid/sensor/realsense_head/color/image_raw/compressed"
    left_wrist_camera: str = "/zj_humanoid/sensor/left_wrist/image_raw/compressed"
    right_wrist_camera: str = "/zj_humanoid/sensor/right_wrist/image_raw/compressed"


class PipelineMockInputRelay:
    """Forward the real head image to otherwise missing wrist-camera inputs."""

    def __init__(
        self,
        rospy: Any,
        compressed_image_type: type[Any],
        topics: MockInputTopics,
    ) -> None:
        self.rospy = rospy
        self.topics = topics
        self._seen_sources: set[str] = set()
        self._closed = False

        targets = (
            topics.left_wrist_camera,
            topics.right_wrist_camera,
        )
        published = {
            self.rospy.resolve_name(name)
            for name, _message_type in self.rospy.get_published_topics("/")
        }
        collisions = [name for name in targets if self.rospy.resolve_name(name) in published]
        if collisions:
            raise RuntimeError(
                "Refusing to mask existing ROS publishers on mock target topics: "
                + ", ".join(collisions)
            )

        self._left_camera_publisher = self.rospy.Publisher(
            topics.left_wrist_camera,
            compressed_image_type,
            queue_size=1,
        )
        self._right_camera_publisher = self.rospy.Publisher(
            topics.right_wrist_camera,
            compressed_image_type,
            queue_size=1,
        )
        self._subscribers = (
            self.rospy.Subscriber(
                topics.head_camera,
                compressed_image_type,
                self._forward_head_image,
                queue_size=1,
            ),
        )

    @property
    def ready(self) -> bool:
        return self._seen_sources == {"head_camera"}

    @property
    def missing_sources(self) -> tuple[str, ...]:
        expected = {"head_camera"}
        return tuple(sorted(expected - self._seen_sources))

    def _mark_seen(self, source: str) -> None:
        if source not in self._seen_sources:
            self._seen_sources.add(source)
            self.rospy.loginfo("Mock input source is live: %s", source)

    def _forward_head_image(self, message: Any) -> None:
        self._mark_seen("head_camera")
        self._left_camera_publisher.publish(message)
        self._right_camera_publisher.publish(message)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for subscriber in self._subscribers:
            subscriber.unregister()
        for publisher in (
            self._left_camera_publisher,
            self._right_camera_publisher,
        ):
            publisher.unregister()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = MockInputTopics()
    parser = argparse.ArgumentParser(
        description="Publish test-only substitutes for missing rollout ROS inputs."
    )
    parser.add_argument(
        "--head-camera-topic",
        default=os.environ.get("HEAD_CAMERA_TOPIC", defaults.head_camera),
    )
    parser.add_argument(
        "--left-wrist-camera-topic",
        default=os.environ.get("LEFT_CAMERA_TOPIC", defaults.left_wrist_camera),
    )
    parser.add_argument(
        "--right-wrist-camera-topic",
        default=os.environ.get("RIGHT_CAMERA_TOPIC", defaults.right_wrist_camera),
    )
    parser.add_argument("--startup-timeout-s", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.startup_timeout_s <= 0:
        raise ValueError("--startup-timeout-s must be positive")

    import rospy
    from sensor_msgs.msg import CompressedImage

    # An anonymous name lets the publisher-collision check reject a second relay without
    # ROS first shutting down the already healthy relay for reusing its node name.
    rospy.init_node("gr00t_pipeline_mock_inputs", anonymous=True)
    rospy.logwarn(
        "TEST-ONLY MOCK INPUTS: head images will impersonate both wrist cameras. Real force "
        "topics are not touched. Do not use the mock images to evaluate policy quality."
    )
    topics = MockInputTopics(
        head_camera=args.head_camera_topic,
        left_wrist_camera=args.left_wrist_camera_topic,
        right_wrist_camera=args.right_wrist_camera_topic,
    )
    relay = PipelineMockInputRelay(rospy, CompressedImage, topics)
    rospy.on_shutdown(relay.close)

    deadline = time.monotonic() + args.startup_timeout_s
    rate = rospy.Rate(20)
    while not rospy.is_shutdown() and not relay.ready and time.monotonic() < deadline:
        rate.sleep()
    if not relay.ready:
        relay.close()
        missing = ", ".join(relay.missing_sources)
        rospy.logerr("Mock input startup failed; no messages received from: %s", missing)
        return 2

    rospy.logwarn(
        "TEST-ONLY WRIST CAMERA MOCK ACTIVE. The head stream is live; no force, Servo, or hand "
        "topic is published by this node."
    )
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
