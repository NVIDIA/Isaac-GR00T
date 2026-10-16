# ruff: noqa: E402
import importlib
import sys
import types
import warnings


warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="Gym has been unmaintained")


def _install_ros_sdk_namespace_compatibility() -> None:
    """Expose the locally installed flat ROS packages under the SDK's nested namespace."""

    try:
        importlib.import_module("zj_humanoid.upperlimb.msg")
        importlib.import_module("zj_humanoid.hand.msg")
        return
    except ModuleNotFoundError:
        pass

    root = sys.modules.get("zj_humanoid")
    if root is None:
        root = types.ModuleType("zj_humanoid")
        root.__path__ = []
        sys.modules["zj_humanoid"] = root
    for package_name in ("upperlimb", "hand"):
        package = importlib.import_module(package_name)
        setattr(root, package_name, package)
        sys.modules[f"zj_humanoid.{package_name}"] = package
        for child_name in ("msg", "srv"):
            child = importlib.import_module(f"{package_name}.{child_name}")
            sys.modules[f"zj_humanoid.{package_name}.{child_name}"] = child


_install_ros_sdk_namespace_compatibility()

from agents.lerobuffer import LeroBuffer
from agents.rollout_passive import PassiveRolloutAgent
from gui import ObservationGUI
from gui.rollout_control import RolloutControlPanel
from observations import (
    LeftFingerPressure,
    LeftTCP,
    LeftWristForce,
    ObservationController,
    ParcelCamera,
    RightFingerPressure,
    RightTCP,
    RightWristForce,
)
import rospy


def main() -> None:
    rospy.init_node("learning_framework_parcel_rollout_view")
    controller = ObservationController(
        [
            ParcelCamera(
                "head",
                "/zj_humanoid/sensor/realsense_head/color/image_raw/compressed",
            ),
            ParcelCamera(
                "left",
                "/zj_humanoid/sensor/left_wrist/image_raw/compressed",
            ),
            ParcelCamera(
                "right",
                "/zj_humanoid/sensor/right_wrist/image_raw/compressed",
            ),
            LeftTCP(),
            RightTCP(),
            LeftFingerPressure(),
            RightFingerPressure(),
            LeftWristForce(deadzone=10),
            RightWristForce(deadzone=10),
        ]
    )
    buffer = LeroBuffer(
        "parcel_rollout_view",
        "datasets/parcel_rollout_view",
        controller.generate_spec(),
        fps=30,
    )
    agent = PassiveRolloutAgent()
    rollout_panel = RolloutControlPanel("/gr00t_rollout")
    gui = ObservationGUI(
        controller,
        buffer,
        agent,
        fps=30,
        default_task="move parcel onto conveyor belt one by one",
        modules=[rollout_panel],
    )
    rollout_panel.bind(gui)
    gui.run()


if __name__ == "__main__":
    main()
