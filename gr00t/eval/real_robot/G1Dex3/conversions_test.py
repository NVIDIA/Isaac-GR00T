# SPDX-License-Identifier: Apache-2.0
"""Round-trip tests for the G1-Dex3 <-> GR00T conversions.

Requires the co-installed env (gr00t + policy_interfaces + pydrake); run with::

    pytest gr00t/eval/real_robot/G1Dex3/conversions_test.py
"""

import numpy as np
import pytest


pydrake = pytest.importorskip("pydrake")
pytest.importorskip("policy_interfaces")

from gr00t.eval.real_robot.G1Dex3.conversions import (  # noqa: E402
    G1DexConversionConfig,
    _eef9d_to_rigid_transform,
    _rigid_transform_to_eef9d,
    gr00t_action_to_poses_and_grippers,
    multiarm_obs_to_gr00t,
)
from policy_interfaces.robot_gym.multiarm_spaces import (  # noqa: E402
    CameraImageSet,
    CameraRgbImage,
    MultiarmObservation,
    PosesAndGrippers,
    PosesAndGrippersActualAndDesired,
)
from pydrake.math import RigidTransform, RotationMatrix  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


CFG = G1DexConversionConfig()
H, W = 32, 48


def _rand_transform(seed: int) -> RigidTransform:
    rng = np.random.default_rng(seed)
    rot = Rotation.random(random_state=seed).as_matrix()
    return RigidTransform(R=RotationMatrix(rot), p=rng.uniform(-1, 1, size=3))


def _fake_camera() -> CameraImageSet:
    rgb = CameraRgbImage(
        array=np.zeros((H, W, 3), dtype=np.uint8),
        K=np.eye(3),
        X_TC=RigidTransform(),
    )
    return CameraImageSet(rgb=rgb)


def _fake_obs() -> MultiarmObservation:
    grippers = {}
    for joint_names in CFG.hand_joint_map.values():
        for i, name in enumerate(joint_names):
            grippers[name] = float(i) / 10.0
    actual = PosesAndGrippers(
        poses={
            "/current_left_hand_ee_link": _rand_transform(1),
            "/current_right_hand_ee_link": _rand_transform(2),
        },
        grippers=grippers,
    )
    robot = PosesAndGrippersActualAndDesired(
        actual=actual, desired=PosesAndGrippers(poses={}, grippers={})
    )
    visuo = {view: _fake_camera() for view in CFG.camera_map}
    return MultiarmObservation(
        robot=robot, visuo=visuo, language_instruction="place the cube in the hand"
    )


def test_eef9d_round_trip():
    original = _rand_transform(7)
    eef9d = _rigid_transform_to_eef9d(original)
    assert eef9d.shape == (9,)
    restored = _eef9d_to_rigid_transform(eef9d)
    np.testing.assert_allclose(restored.translation(), original.translation(), atol=1e-6)
    np.testing.assert_allclose(
        restored.rotation().matrix(), original.rotation().matrix(), atol=1e-6
    )


def test_obs_to_gr00t_shapes():
    obs = _fake_obs()
    gr00t_obs = multiarm_obs_to_gr00t(obs, CFG)

    assert set(gr00t_obs["video"]) == {"ego_view", "left_wrist", "right_wrist"}
    for view in gr00t_obs["video"].values():
        assert view.shape == (1, 1, H, W, 3)
        assert view.dtype == np.uint8

    assert gr00t_obs["state"]["left_arm_eef"].shape == (1, 1, 9)
    assert gr00t_obs["state"]["right_arm_eef"].shape == (1, 1, 9)
    assert gr00t_obs["state"]["left_hand"].shape == (1, 1, 7)
    assert gr00t_obs["state"]["right_hand"].shape == (1, 1, 7)

    assert gr00t_obs["language"][CFG.language_key] == [["place the cube in the hand"]]


def test_state_matches_observation():
    obs = _fake_obs()
    gr00t_obs = multiarm_obs_to_gr00t(obs, CFG)
    # The 9D state must reconstruct the original EEF pose (relative-EEF decode
    # depends on this being the true current pose).
    restored = _eef9d_to_rigid_transform(gr00t_obs["state"]["left_arm_eef"][0, 0])
    original = obs.robot.actual.poses["/current_left_hand_ee_link"]
    np.testing.assert_allclose(
        restored.rotation().matrix(), original.rotation().matrix(), atol=1e-6
    )


def test_action_to_poses_and_grippers():
    # Fake absolute action chunk: (B=1, T=4, D) per key.
    horizon = 4
    action = {
        "left_arm_eef": np.zeros((1, horizon, 9), dtype=np.float32),
        "right_arm_eef": np.zeros((1, horizon, 9), dtype=np.float32),
        "left_hand": np.zeros((1, horizon, 7), dtype=np.float32),
        "right_hand": np.zeros((1, horizon, 7), dtype=np.float32),
    }
    # Fill step 2 with valid poses for both arms (an all-zero 9D is not a valid
    # rot6d and would make the rotation decode singular).
    action["left_arm_eef"][0, 2] = _rigid_transform_to_eef9d(_rand_transform(3))
    action["right_arm_eef"][0, 2] = _rigid_transform_to_eef9d(_rand_transform(4))
    action["left_hand"][0, 2] = np.arange(7, dtype=np.float32)

    result = gr00t_action_to_poses_and_grippers(action, step=2, cfg=CFG)

    assert set(result.poses) == {"/ee_target_left", "/ee_target_right"}
    assert isinstance(result.poses["/ee_target_left"], RigidTransform)
    expected_hand_names = set(CFG.hand_joint_map["left_hand"]) | set(CFG.hand_joint_map["right_hand"])
    assert len(result.grippers) == 14  # 7 Dex3 joints x 2 hands, named scalars
    assert set(result.grippers) == expected_hand_names
    # Values are written in HAND_JOINT_ORDER.
    for i, name in enumerate(CFG.hand_joint_map["left_hand"]):
        assert result.grippers[name] == float(i)
