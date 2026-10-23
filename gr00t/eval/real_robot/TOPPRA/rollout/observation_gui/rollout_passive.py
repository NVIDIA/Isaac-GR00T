from __future__ import annotations

from typing import Any

from agents import Agent
from numpy.typing import NDArray


class PassiveRolloutAgent(Agent):
    """Keep ObservationGUILite read-only while the 250 Hz rollout owns motion."""

    def act(
        self,
        obs: dict[str, NDArray],
        task: str,
    ) -> dict[str, NDArray]:
        del obs, task
        raise RuntimeError(
            "ObservationGUILite is in external-rollout mode; use the GR00T Rollout panel"
        )

    def teardown(self) -> None:
        return None

    def drain_debug_events(self) -> list[dict[str, Any]]:
        return []
