from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from ..kion_client.client import (
    LEFT_POSE_KEY,
    LEFT_TWIST_KEY,
    RIGHT_POSE_KEY,
    RIGHT_TWIST_KEY,
    KionObservationBuffer,
)


POLICY_IMAGE_KEYS = (
    "observation.images.head",
    "observation.images.left",
    "observation.images.right",
)
POLICY_STATE_KEYS = (
    LEFT_POSE_KEY,
    "observation.state.left_wrist_force",
    RIGHT_POSE_KEY,
    "observation.state.right_wrist_force",
)
LOCAL_TWIST_KEYS = (LEFT_TWIST_KEY, RIGHT_TWIST_KEY)
REQUIRED_ROLLOUT_KEYS = (*POLICY_IMAGE_KEYS, *POLICY_STATE_KEYS, *LOCAL_TWIST_KEYS)
CRITICAL_ROLLOUT_KEYS = frozenset((*POLICY_IMAGE_KEYS, *POLICY_STATE_KEYS))


class RolloutObservationBuffer(KionObservationBuffer):
    """Kion buffer restricted to the modalities used by the parcel checkpoint.

    Finger pressure remains subscribed by the frozen base implementation for monitoring, but it
    is not a VLA input and cannot block rollout readiness or trajectory consumption.
    """

    REQUIRED_KEYS = REQUIRED_ROLLOUT_KEYS


def rollout_staleness(stale_fields: tuple[str, ...]) -> tuple[tuple[str, ...], bool]:
    stale = tuple(str(key) for key in stale_fields)
    critical = tuple(key for key in stale if key in CRITICAL_ROLLOUT_KEYS)
    twist_stale = any(key in LOCAL_TWIST_KEYS for key in stale)
    return critical, twist_stale


def policy_observation(
    observation: dict[str, NDArray[Any]],
    *,
    twist_stale: bool,
) -> dict[str, NDArray[Any]]:
    """Return only checkpoint inputs plus fresh local-only twists.

    Stale twists are omitted so TOPPRA falls back to its commanded twist estimate. They are never
    sent to the policy because the rollout agents set include_velocity_in_policy_observation=False.
    """

    selected = (*POLICY_IMAGE_KEYS, *POLICY_STATE_KEYS)
    if not twist_stale:
        selected = (*selected, *LOCAL_TWIST_KEYS)
    missing = [key for key in selected if key not in observation]
    if missing:
        raise KeyError(f"Rollout observation is missing configured inputs: {missing}")
    return {key: observation[key] for key in selected}
