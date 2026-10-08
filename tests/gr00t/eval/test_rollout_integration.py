from __future__ import annotations

from enum import Enum, auto
import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from gr00t.eval.real_robot.TOPPRA.kion_client.client import KionTopics
from gr00t.eval.real_robot.TOPPRA.rollout import plain_client as plain_module
from gr00t.eval.real_robot.TOPPRA.rollout.hardware_safety import (
    TargetSafetyError,
    TargetSafetyGuard,
    mock_camera_publishers,
    parse_workspace_bounds,
)
from gr00t.eval.real_robot.TOPPRA.rollout.local_sim_hardware import require_loopback_master
from gr00t.eval.real_robot.TOPPRA.rollout.mock_inputs import MockInputTopics, PipelineMockInputRelay
from gr00t.eval.real_robot.TOPPRA.rollout.observation import (
    CRITICAL_ROLLOUT_KEYS,
    LOCAL_TWIST_KEYS,
    RolloutObservationBuffer,
    policy_observation,
    rollout_staleness,
)
from gr00t.eval.real_robot.TOPPRA.rollout.operator import (
    RosEpisodeBridge,
    restore_console_logging,
    status_json,
)
from gr00t.eval.real_robot.TOPPRA.rollout.pinch import (
    LEFT_HIGH,
    LEFT_LOW,
    RIGHT_HIGH,
    RIGHT_LOW,
    pinch_posture,
)
from gr00t.eval.real_robot.TOPPRA.rollout.plain_client import _parse_args as parse_plain_args
from gr00t.eval.real_robot.TOPPRA.rollout.reset import EpisodeResetController
from gr00t.eval.real_robot.TOPPRA.speed_rl.client import _parse_args as parse_speed_args
import numpy as np
import pytest


class _TriggerResponse:
    def __init__(self, *, success: bool, message: str) -> None:
        self.success = success
        self.message = message


class _String:
    def __init__(self) -> None:
        self.data = ""


class _Publisher:
    def __init__(self) -> None:
        self.messages: list[Any] = []
        self.closed = False

    def publish(self, message: Any) -> None:
        self.messages.append(message)

    def unregister(self) -> None:
        self.closed = True


class _Service:
    def __init__(self, name: str, callback: Any) -> None:
        self.name = name
        self.callback = callback
        self.closed = False

    def shutdown(self, _reason: str) -> None:
        self.closed = True


class _Rospy:
    def __init__(self) -> None:
        self.publisher = _Publisher()
        self.services: dict[str, _Service] = {}

    def Publisher(self, *_args: Any, **_kwargs: Any) -> _Publisher:  # noqa: N802
        return self.publisher

    def Service(self, name: str, _service_type: Any, callback: Any) -> _Service:  # noqa: N802
        service = _Service(name, callback)
        self.services[name] = service
        return service


class _RelayEndpoint:
    def __init__(self, topic: str, callback: Any | None = None) -> None:
        self.topic = topic
        self.callback = callback
        self.messages: list[Any] = []
        self.closed = False

    def publish(self, message: Any) -> None:
        self.messages.append(message)

    def unregister(self) -> None:
        self.closed = True


class _RelayRospy:
    def __init__(self, published_topics: tuple[str, ...] = ()) -> None:
        self.initial_published_topics = published_topics
        self.publishers: dict[str, _RelayEndpoint] = {}
        self.subscribers: dict[str, _RelayEndpoint] = {}
        self.logs: list[str] = []

    def resolve_name(self, name: str) -> str:
        return name if name.startswith("/") else f"/{name}"

    def get_published_topics(self, _namespace: str) -> list[tuple[str, str]]:
        return [(topic, "test/Message") for topic in self.initial_published_topics]

    def Publisher(self, topic: str, *_args: Any, **_kwargs: Any) -> _RelayEndpoint:  # noqa: N802
        endpoint = _RelayEndpoint(topic)
        self.publishers[topic] = endpoint
        return endpoint

    def Subscriber(  # noqa: N802
        self,
        topic: str,
        _message_type: Any,
        callback: Any,
        **_kwargs: Any,
    ) -> _RelayEndpoint:
        endpoint = _RelayEndpoint(topic, callback)
        self.subscribers[topic] = endpoint
        return endpoint

    def loginfo(self, message: str, source: str) -> None:
        self.logs.append(message % source)


def test_rollout_status_json_normalizes_numpy_and_nonfinite_values() -> None:
    encoded = status_json(
        {
            "mask": np.array([True, False]),
            "action": np.int64(2),
            "latency": float("nan"),
            "nested": (np.float32(1.25),),
        }
    )
    assert json.loads(encoded) == {
        "action": 2,
        "latency": None,
        "mask": [True, False],
        "nested": [1.25],
    }


def test_restore_console_logging_is_idempotent() -> None:
    root_logger = logging.getLogger()
    original_level = root_logger.level
    existing = [
        handler
        for handler in root_logger.handlers
        if getattr(handler, "_gr00t_console_handler", False)
    ]
    for handler in existing:
        root_logger.removeHandler(handler)
    try:
        first = restore_console_logging()
        second = restore_console_logging()
        assert first is second
        assert (
            sum(
                bool(getattr(handler, "_gr00t_console_handler", False))
                for handler in root_logger.handlers
            )
            == 1
        )
    finally:
        for handler in tuple(root_logger.handlers):
            if getattr(handler, "_gr00t_console_handler", False):
                root_logger.removeHandler(handler)
        for handler in existing:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)


def test_pipeline_mock_relay_forwards_inputs_and_never_creates_command_publishers() -> None:
    fake_rospy = _RelayRospy()
    topics = MockInputTopics()
    relay = PipelineMockInputRelay(fake_rospy, object, topics)

    head_message = object()
    fake_rospy.subscribers[topics.head_camera].callback(head_message)

    assert relay.ready
    assert fake_rospy.publishers[topics.left_wrist_camera].messages == [head_message]
    assert fake_rospy.publishers[topics.right_wrist_camera].messages == [head_message]
    assert not any("servo" in topic or "hand" in topic for topic in fake_rospy.publishers)

    relay.close()
    assert all(endpoint.closed for endpoint in fake_rospy.publishers.values())
    assert all(endpoint.closed for endpoint in fake_rospy.subscribers.values())


def test_pipeline_mock_relay_refuses_to_mask_real_publishers() -> None:
    topics = MockInputTopics()
    fake_rospy = _RelayRospy((topics.left_wrist_camera,))

    with pytest.raises(RuntimeError, match="Refusing to mask existing ROS publishers"):
        PipelineMockInputRelay(fake_rospy, object, topics)


def test_local_hardware_simulator_requires_loopback_ros_master() -> None:
    require_loopback_master("http://127.0.0.1:11312")
    require_loopback_master("http://localhost:11311")

    with pytest.raises(RuntimeError, match="loopback ROS master"):
        require_loopback_master("http://192.168.217.1:11311")


def test_hardware_target_guard_checks_workspace_and_tracking_error() -> None:
    workspace = parse_workspace_bounds("0.2,1.0,-0.8,0.8,0.1,1.5")
    guard = TargetSafetyGuard(
        workspace,
        workspace,
        max_position_error_m=0.1,
        max_rotation_error_rad=0.2,
    )
    measured = np.array([0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])
    guard.validate(measured, measured, measured, measured)

    outside = measured.copy()
    outside[0] = 1.01
    with pytest.raises(TargetSafetyError, match="outside certified workspace"):
        guard.validate(outside, measured, measured, measured)

    far = measured.copy()
    far[1] = 0.11
    with pytest.raises(TargetSafetyError, match="tracking error"):
        guard.validate(far, measured, measured, measured)

    rotated = measured.copy()
    rotated[3:] = [np.cos(0.11), np.sin(0.11), 0.0, 0.0]
    with pytest.raises(TargetSafetyError, match="rotation error"):
        guard.validate(rotated, measured, measured, measured)


def test_hardware_target_guard_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="strictly below"):
        parse_workspace_bounds("1,0,-1,1,0,1")
    with pytest.raises(ValueError, match="six"):
        parse_workspace_bounds("0,1,0,1")


def test_real_motion_detects_test_only_camera_relay() -> None:
    class Master:
        def getSystemState(self) -> tuple[int, str, list[Any]]:  # noqa: N802
            return (
                1,
                "ok",
                [
                    [
                        (
                            "/left/image",
                            ["/gr00t_rollout/gr00t_pipeline_mock_inputs_123"],
                        ),
                        ("/head/image", ["/real_camera"]),
                    ],
                    [],
                    [],
                ],
            )

    class Rospy:
        @staticmethod
        def get_master() -> Master:
            return Master()

        @staticmethod
        def resolve_name(topic: str) -> str:
            return topic

    assert mock_camera_publishers(Rospy(), ("/left/image", "/head/image")) == (
        "/gr00t_rollout/gr00t_pipeline_mock_inputs_123",
    )


def test_ros_episode_bridge_queues_commands_and_publishes_status(monkeypatch) -> None:
    fake_rospy = _Rospy()
    commands: list[str] = []

    def fake_import(name: str) -> Any:
        if name == "std_srvs.srv":
            return SimpleNamespace(Trigger=object, TriggerResponse=_TriggerResponse)
        if name == "std_msgs.msg":
            return SimpleNamespace(String=_String)
        raise ImportError(name)

    monkeypatch.setattr(
        "gr00t.eval.real_robot.TOPPRA.rollout.operator.importlib.import_module",
        fake_import,
    )
    bridge = RosEpisodeBridge(
        fake_rospy,
        commands.append,
        lambda: {"state": "running", "speed_scale": np.float32(1.3)},
        namespace="test_rollout",
    )
    response = fake_rospy.services["/test_rollout/start"].callback(None)
    assert response.success
    assert commands == ["start"]
    fake_rospy.services["/test_rollout/approve"].callback(None)
    assert commands[-1].startswith("approve ObservationGUILite")
    fake_rospy.services["/test_rollout/reset"].callback(None)
    fake_rospy.services["/test_rollout/ready"].callback(None)
    assert commands[-2:] == ["reset", "ready"]

    status_response = fake_rospy.services["/test_rollout/status"].callback(None)
    assert json.loads(status_response.message)["state"] == "running"
    bridge.publish_status(force=True)
    assert json.loads(fake_rospy.publisher.messages[-1].data)["speed_scale"] == np.float32(1.3)
    bridge.close()
    assert fake_rospy.publisher.closed
    assert all(service.closed for service in fake_rospy.services.values())


def test_pinch_mapping_matches_observation_gui_calibration() -> None:
    np.testing.assert_allclose(pinch_posture(np.array([0.0]), LEFT_LOW, LEFT_HIGH), LEFT_LOW)
    np.testing.assert_allclose(pinch_posture(np.array([1.0]), LEFT_LOW, LEFT_HIGH), LEFT_HIGH)
    np.testing.assert_allclose(pinch_posture(np.array([-1.0]), RIGHT_LOW, RIGHT_HIGH), RIGHT_LOW)
    np.testing.assert_allclose(pinch_posture(np.array([2.0]), RIGHT_LOW, RIGHT_HIGH), RIGHT_HIGH)


def test_unified_clients_default_to_sync_single_candidate_gui_control() -> None:
    plain = parse_plain_args([])
    speed = parse_speed_args(
        [
            "--left-twist-thresholds",
            "1,1,1,1,1,1",
            "--right-twist-thresholds",
            "1,1,1,1,1,1",
        ]
    )
    for args in (plain, speed):
        assert args.inference_mode == "sync"
        assert args.tts_samples == 1
        assert args.control_interface == "gui"
        assert args.ros_namespace == "/gr00t_rollout"
        assert args.server_port == 47866
        assert args.reset_mode == "manual"


def test_go_home_reset_uses_sdk_trigger_and_requires_operator_confirmation() -> None:
    class Rospy:
        def __init__(self) -> None:
            self.waited_for: tuple[str, float] | None = None
            self.called_service: str | None = None

        def wait_for_service(self, name: str, timeout: float) -> None:
            self.waited_for = (name, timeout)

        def ServiceProxy(self, name: str, _service_type: Any) -> Any:  # noqa: N802
            self.called_service = name
            return lambda: SimpleNamespace(success=True, message="home accepted")

    rospy = Rospy()
    controller = EpisodeResetController(
        rospy,
        mode="go-home",
        service_name="/test/go_home",
        service_timeout_s=3.0,
    )
    controller.begin()
    controller.wait(1.0)
    status = controller.status()

    assert rospy.waited_for == ("/test/go_home", 3.0)
    assert rospy.called_service == "/test/go_home"
    assert status.home_request_complete
    assert status.operator_may_confirm_ready
    assert status.response == "home accepted"


def test_rollout_observation_contract_ignores_pressure_and_omits_stale_twist() -> None:
    assert all("finger_pressure" not in key for key in RolloutObservationBuffer.REQUIRED_KEYS)
    critical_key = next(iter(CRITICAL_ROLLOUT_KEYS))
    critical, twist_stale = rollout_staleness((critical_key, LOCAL_TWIST_KEYS[0]))
    assert critical == (critical_key,)
    assert twist_stale

    observation = {
        key: np.zeros(1, dtype=np.float32) for key in RolloutObservationBuffer.REQUIRED_KEYS
    }
    observation["observation.state.left_finger_pressure"] = np.ones(6, dtype=np.float32)
    selected = policy_observation(observation, twist_stale=True)
    assert not any(key in selected for key in LOCAL_TWIST_KEYS)
    assert "observation.state.left_finger_pressure" not in selected


def test_observation_gui_passively_saves_labels_and_discards_abort(monkeypatch) -> None:
    class Mode(Enum):
        IDLE = auto()
        RECORDING = auto()
        REVIEWING = auto()

    dearpygui_package = ModuleType("dearpygui")
    dearpygui_module = ModuleType("dearpygui.dearpygui")
    dearpygui_package.dearpygui = dearpygui_module
    gui_package = ModuleType("gui")
    gui_module = ModuleType("gui.module")
    gui_module.GUIModule = object
    gui_package.module = gui_module
    controlloop = ModuleType("controlloop")
    controlloop.Mode = Mode
    std_msgs = ModuleType("std_msgs.msg")
    std_msgs.String = object
    std_srvs = ModuleType("std_srvs.srv")
    std_srvs.Trigger = object
    for name, module in {
        "dearpygui": dearpygui_package,
        "dearpygui.dearpygui": dearpygui_module,
        "gui": gui_package,
        "gui.module": gui_module,
        "controlloop": controlloop,
        "rospy": ModuleType("rospy"),
        "std_msgs.msg": std_msgs,
        "std_srvs.srv": std_srvs,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    source = (
        Path(__file__).parents[3]
        / "gr00t/eval/real_robot/TOPPRA/rollout/observation_gui/rollout_control.py"
    )
    spec = importlib.util.spec_from_file_location("test_rollout_control_overlay", source)
    assert spec is not None and spec.loader is not None
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)

    class FakeState:
        def __init__(self) -> None:
            self.mode = Mode.IDLE
            self.task = ""
            self.episode_success = False

        def set_mode(self, mode: Mode) -> None:
            self.mode = mode

    class FakeLoop:
        def __init__(self) -> None:
            self.frames: list[dict[str, Any]] = []
            self.debug_events: list[dict[str, Any]] = []
            self.saved: list[tuple[bool, list[dict[str, Any]]]] = []
            self.discarded = 0

        def save_episode(self) -> None:
            self.saved.append((state.episode_success, list(self.debug_events)))
            self.frames.clear()
            self.debug_events.clear()

        def discard_episode(self) -> None:
            self.discarded += 1
            self.frames.clear()

    state = FakeState()
    loop = FakeLoop()
    panel = overlay.RolloutControlPanel.__new__(overlay.RolloutControlPanel)
    panel._control_loop = loop
    panel._control_state = state
    panel._recording_episode = None
    panel._saved_episode = None
    panel._last_result = ""

    panel._sync_passive_recording({"state": "running", "episode": 3, "task": "parcel"})
    assert state.mode is Mode.RECORDING
    assert state.task == "parcel"
    loop.frames.extend([{}, {}, {}])
    panel._sync_passive_recording(
        {
            "state": "resetting",
            "episode": 3,
            "last_outcome": "success",
            "last_safe_success": True,
            "speed_violation": False,
            "tracking_log": "tracking.csv",
        }
    )
    assert panel._saved_episode == 3
    assert loop.saved[0][0] is True
    assert loop.saved[0][1][0]["events"][0]["outcome"] == "success"

    panel._sync_passive_recording({"state": "running", "episode": 4, "task": "parcel"})
    panel._sync_passive_recording({"state": "aborted", "episode": 4, "last_outcome": "abort"})
    assert loop.discarded == 1


def test_plain_episode_lifecycle_recreates_agent_and_persists_operator_outcome(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeObservations:
        def __init__(self, *_args, **_kwargs) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeServo:
        instances = []

        def __init__(self, *_args, **_kwargs) -> None:
            self.closed = False
            self.instances.append(self)

        def wait_for_connection(self, _timeout: float) -> None:
            return None

        def configure(self, _timeout: float) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class FakePinch(FakeServo):
        pass

    class FakeRecorder:
        def __init__(self, directory, _metadata) -> None:
            self.directory = directory
            self.directory.mkdir(parents=True)
            self.csv_path = self.directory / "tcp_tracking.csv"
            self.csv_path.touch()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeAgent:
        def __init__(self) -> None:
            self.closed = False

        def diagnostics(self) -> dict[str, Any]:
            return {"execution_state": "hold"}

        def teardown(self) -> None:
            self.closed = True

    monkeypatch.setattr(plain_module, "RolloutObservationBuffer", FakeObservations)
    monkeypatch.setattr(plain_module, "KionDualArmServo", FakeServo)
    monkeypatch.setattr(plain_module, "KionPinchExecutor", FakePinch)
    monkeypatch.setattr(plain_module, "TrackingRecorder", FakeRecorder)

    agents: list[FakeAgent] = []

    def agent_factory() -> FakeAgent:
        agent = FakeAgent()
        agents.append(agent)
        return agent

    ros_types = SimpleNamespace(rospy=object(), hand_namespace="test.hand.msg")
    client = plain_module.PlainKionEpisodeClient(
        ros_types=ros_types,
        topics=KionTopics(),
        agent_factory=agent_factory,
        log_root=tmp_path,
        control_frequency=250,
        servo_gain=800,
        startup_timeout_s=1,
        max_state_age_s=1,
        max_image_age_s=1,
        task="parcel",
        inference_mode="sync",
        dry_run=True,
    )
    client._start_episode()
    assert client.state is plain_module.EpisodeState.RUNNING
    first_directory = client.recorder.directory
    client._finish_episode(success=True, reason="operator success")
    client._finalizer.join(timeout=2)
    assert client.state is plain_module.EpisodeState.TERMINATED
    assert json.loads((first_directory / "outcome.json").read_text())["success"] is True

    with pytest.raises(RuntimeError, match="finish and reset first"):
        client._start_episode()
    client._begin_reset()
    assert client.state is plain_module.EpisodeState.RESETTING
    assert agents[0].closed
    client._confirm_reset_ready()
    assert client.state is plain_module.EpisodeState.IDLE

    client._start_episode()
    assert len(agents) == 2
    assert agents[0].closed
    client._abort_episode(reason="operator abort")
    client._finalizer.join(timeout=2)
    assert client.state is plain_module.EpisodeState.ABORTED
    assert all(item.closed for item in FakeServo.instances)
    client.close()
    assert client.state is plain_module.EpisodeState.CLOSED
