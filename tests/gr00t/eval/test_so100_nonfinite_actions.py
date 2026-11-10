# SPDX-License-Identifier: Apache-2.0
"""Reject non-finite SO100 policy actions before they can be returned."""

import numpy as np
import pytest

eval_so100 = pytest.importorskip("gr00t.eval.real_robot.SO100.eval_so100")

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

NONFINITE_VALUES = [
    float("nan"),
    float("inf"),
    float("-inf"),
]


@pytest.mark.parametrize("joint_idx", range(len(JOINTS)), ids=JOINTS)
@pytest.mark.parametrize(
    "value",
    NONFINITE_VALUES,
    ids=["nan", "positive_inf", "negative_inf"],
)
def test_decode_action_chunk_rejects_nonfinite_values(joint_idx, value):
    adapter = eval_so100.So100Adapter(policy_client=None)

    single_arm = np.zeros((1, 1, 5), dtype=np.float64)
    gripper = np.zeros((1, 1, 1), dtype=np.float64)

    if joint_idx < 5:
        single_arm[0, 0, joint_idx] = value
    else:
        gripper[0, 0, 0] = value

    chunk = {"single_arm": single_arm, "gripper": gripper}

    with pytest.raises(ValueError, match="non-finite action"):
        adapter.decode_action_chunk(chunk, 0)


def test_decode_action_chunk_preserves_finite_values():
    adapter = eval_so100.So100Adapter(policy_client=None)
    values = np.array([1.0, -2.0, 3.5, 0.0, 7.0, 0.25])

    chunk = {
        "single_arm": values[:5].reshape(1, 1, 5),
        "gripper": values[5:].reshape(1, 1, 1),
    }

    action = adapter.decode_action_chunk(chunk, 0)

    assert list(action) == JOINTS
    assert list(action.values()) == values.tolist()
