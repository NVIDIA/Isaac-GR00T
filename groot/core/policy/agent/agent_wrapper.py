# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""System-2 agent policy wrapper backed by a remote agent server.

Same seam as :class:`groot.core.policy.cot.cot_wrapper.CoTSyncPolicyWrapper`: a
:class:`~groot.core.policy.base.PolicyWrapper` that translates a high-level goal
+ current camera frames into a next-step subtask, writes it into the observation
under ``instruction_key``, then delegates to the wrapped (System-1) policy.

System-2 runs as a *full agent loop* (nanobot ``EmbodiedSession`` + a VLM) hosted
behind a port; this wrapper is a thin client and imports no nanobot. System-1
still owns the control loop. At a re-inference boundary the wrapper ships the
current frames to the server and blocks for the agent's next subtask (one
``robot_step`` turn); between boundaries it coasts on the cached subtask. This
mirrors ``CoTSyncPolicyWrapper``'s ``max_drift`` cadence and its blocking step-0
``prepare_new_episode``.

Robot-specific concerns (camera-frame extraction, the instruction key, the
system prompt / robot description sent to the server) are injected, so any
embodiment can reuse this wrapper. The server owns all VLM configuration
(provider, model, generation params).
"""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any
import uuid

from loguru import logger
from PIL import Image

from groot.core.policy import base
from groot.core.policy.agent.agent_client import AgentClient

# Maps an observation dict to ``{camera_name: [frame, ...]}`` in the order the
# VLM should see them; the first entry is the primary camera.
FrameExtractor = Callable[[dict[str, Any]], dict[str, list[Image.Image]]]

_DEFAULT_INSTRUCTION_KEY = "annotation.task"
# Key under which a fresh post-setup observation may be passed in reset options
# to trigger step-0 inference (matches the CoT wrapper convention).
RESET_OBSERVATION_KEY = "observation"


class AgentSyncPolicyWrapper(base.PolicyWrapper):
    """Synchronous System-2 agent wrapper (one ``robot_step`` per re-inference)."""

    def __init__(
        self,
        policy: base.BasePolicy,
        *,
        frame_extractor: FrameExtractor,
        system_prompt: str,
        robot_description: str,
        server_address: str,
        instruction_key: str = _DEFAULT_INSTRUCTION_KEY,
        max_drift: float = 2.0,
        rpc_timeout_ms: int = 300_000,
        fallback_response: str = "wait and observe",
        subtask_candidates: list[str] | None = None,
        strict: bool = True,
        client: AgentClient | None = None,
    ) -> None:
        super().__init__(policy, strict=strict)
        self._frame_extractor = frame_extractor
        self._instruction_key = instruction_key
        self._system_prompt = system_prompt
        self._robot_description = robot_description
        self._max_drift = max_drift
        self._fallback = fallback_response
        # Optional closed vocabulary: when set, every episode's ``robot_step`` is
        # enum-constrained to these candidates (fixed at reset). ``None`` → the
        # server generates free-form subtasks.
        self._subtask_candidates = subtask_candidates
        # The only link to System-2: a thin RPC client to the agent server.
        self._client = (
            client if client is not None else AgentClient(server_address, timeout_ms=rpc_timeout_ms)
        )

        self._session_id: str | None = None
        self._session_finished = False
        self.last_agent_input = ""
        self.last_agent_output = ""
        self.last_inference_time = 0.0
        self.start_time = time.time()

    # --- BasePolicy abstract contract (forward to the wrapped policy) ----
    def check_observation(self, observation: dict[str, Any]) -> None:
        self.policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        self.policy.check_action(action)

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.policy._get_action(observation, options)  # noqa: SLF001

    def get_modality_config(self):
        return self.policy.get_modality_config()

    # --- helpers (mirrors _CoTPolicyWrapperBase) ------------------------
    @staticmethod
    def _normalize_goal(value: Any) -> str:
        while isinstance(value, list | tuple):
            if not value:
                return ""
            value = value[0]
        return value

    def _read_goal(self, observation: dict[str, Any]) -> str:
        return self._normalize_goal(observation[self._instruction_key])

    @staticmethod
    def _split_reset_options(
        options: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not options or RESET_OBSERVATION_KEY not in options:
            return options, None
        inner = {k: v for k, v in options.items() if k != RESET_OBSERVATION_KEY}
        return (inner or None), options[RESET_OBSERVATION_KEY]

    def _forward_get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None, kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.policy.get_action(observation, options, **kwargs)

    def _ordered_frames(self, observation: dict[str, Any]) -> list[Image.Image]:
        frames_by_camera = self._frame_extractor(observation)
        ordered = [img for imgs in frames_by_camera.values() for img in imgs]
        if not ordered:
            raise ValueError("frame_extractor returned no camera frames")
        return ordered

    # --- remote exchange ------------------------------------------------
    def _mark_session_ended(self) -> None:
        self._session_finished = True
        logger.warning(
            "System-2 agent session ended (server-side); coasting on last subtask {!r}",
            self.last_agent_output,
        )

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        goal = self._read_goal(observation)
        now = time.time()
        boundary = (goal != self.last_agent_input) or (
            now - self.last_inference_time >= self._max_drift
        )
        if boundary and not self._session_finished and self._session_id is not None:
            result = self._client.step(
                session_id=self._session_id, images=self._ordered_frames(observation)
            )
            if result.session_ended:
                self._mark_session_ended()
            elif result.text is not None:
                self.last_agent_output = result.text
                logger.info("System-2 subtask: {!r}", self.last_agent_output)
            self.last_agent_input = goal
            self.last_inference_time = now

        if self.last_agent_output:
            observation[self._instruction_key] = self.last_agent_output
        return self._forward_get_action(observation, options, kwargs)

    def prepare_new_episode(self, observation: dict[str, Any]) -> None:
        """Start a fresh remote agent session and run the blocking step-0 inference.

        Opens a new server-side session (fresh ``session_id``), posts the fresh
        post-setup observation, and blocks for the first subtask so System-1 gets
        a real subtask from step 0.
        """
        self._teardown()
        self.start_time = time.time()
        goal = self._read_goal(observation)
        self._session_finished = False
        self._session_id = uuid.uuid4().hex
        result = self._client.reset(
            session_id=self._session_id,
            goal=goal,
            system_prompt=self._system_prompt,
            robot_description=self._robot_description,
            images=self._ordered_frames(observation),
            candidates=self._subtask_candidates,
        )
        if result.session_ended:
            self._mark_session_ended()
            self.last_agent_output = self._fallback
        else:
            self.last_agent_output = result.text if result.text is not None else self._fallback
        self.last_agent_input = goal
        self.last_inference_time = time.time()
        logger.info(
            "System-2 agent step-0 on fresh scene: {!r} -> {!r}", goal, self.last_agent_output
        )

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        inner_options, step0_obs = self._split_reset_options(options)
        self._teardown()
        self.last_agent_input = ""
        self.last_agent_output = ""
        self.last_inference_time = 0.0
        self._session_finished = False
        self.start_time = time.time()
        info = self.policy.reset(inner_options)
        if step0_obs is not None:
            self.prepare_new_episode(step0_obs)
        return info

    def _teardown(self) -> None:
        if self._session_id is not None:
            self._client.close(session_id=self._session_id)
            self._session_id = None

    def shutdown(self) -> None:
        self._teardown()


__all__ = ["RESET_OBSERVATION_KEY", "AgentSyncPolicyWrapper", "FrameExtractor"]
