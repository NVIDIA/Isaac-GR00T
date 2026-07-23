# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AgentSyncPolicyWrapper cadence / coasting / teardown, with a fake agent client.

The wrapper is a thin client to a remote agent server; we inject a fake client
that returns numbered subtasks (and optionally ends the session) so we can
exercise the wrapper's System-1-facing behaviour (re-inference cadence, coasting
on the cached subtask, coasting after the session ends, session teardown) without
any network, nanobot, or robot.
"""

from __future__ import annotations

from typing import Any

from PIL import Image as PILImage
import pytest

from groot.core.policy import base
from groot.core.policy.agent import agent_wrapper
from groot.core.policy.agent.agent_client import Subtask


class _FakeAgentClient:
    """Stands in for AgentClient: numbered subtasks, optional session end.

    ``reset`` emits ``subtask_1``; each ``step`` advances. If ``max_subtasks`` is
    set, the exchange that would exceed it reports ``session_ended`` instead. If
    ``reset_ends`` is True, ``reset`` itself reports session-ended (the "agent
    called task_done on turn 0" case).
    """

    def __init__(self, *, max_subtasks: int | None = None, reset_ends: bool = False) -> None:
        self.max_subtasks = max_subtasks
        self.reset_ends = reset_ends
        self.i = 0
        self.reset_sessions: list[str] = []
        self.reset_candidates: list[list[str] | None] = []
        self.closed_sessions: list[str] = []

    def reset(
        self, *, session_id, goal, system_prompt, robot_description, images, candidates=None
    ) -> Subtask:
        self.reset_sessions.append(session_id)
        self.reset_candidates.append(candidates)
        if self.reset_ends:
            return Subtask(text=None, session_ended=True)
        self.i = 1
        return Subtask(text="subtask_1", session_ended=False)

    def step(self, *, session_id, images) -> Subtask:
        self.i += 1
        if self.max_subtasks is not None and self.i > self.max_subtasks:
            return Subtask(text=None, session_ended=True)
        return Subtask(text=f"subtask_{self.i}", session_ended=False)

    def close(self, *, session_id) -> None:
        self.closed_sessions.append(session_id)


class _RecordingPolicy(base.BasePolicy):
    """Inner System-1 stand-in: records the instruction it is handed each step."""

    def __init__(self) -> None:
        super().__init__(strict=False)
        self.seen_tasks: list[str] = []
        self.reset_calls = 0

    def check_observation(self, observation: dict[str, Any]) -> None:
        pass

    def check_action(self, action: dict[str, Any]) -> None:
        pass

    def _get_action(self, observation, options=None):
        return {"a": 0.0}, {}

    def get_action(self, observation, options=None, **kwargs):
        self.seen_tasks.append(observation["annotation.task"])
        return {"a": 0.0}, {}

    def reset(self, options=None):
        self.reset_calls += 1
        return {}

    def get_modality_config(self):
        return {}


def _frame_extractor(_obs: dict[str, Any]) -> dict[str, list]:
    return {"top": [PILImage.new("RGB", (2, 2))]}


def _make_wrapper(inner, client, *, max_drift, subtask_candidates=None):
    return agent_wrapper.AgentSyncPolicyWrapper(
        inner,
        frame_extractor=_frame_extractor,
        system_prompt="sys",
        robot_description="desc",
        server_address="tcp://unused",
        max_drift=max_drift,
        subtask_candidates=subtask_candidates,
        client=client,
    )


def _obs(task: str = "do the task") -> dict[str, Any]:
    return {"annotation.task": task, "top_camera_image": PILImage.new("RGB", (2, 2))}


def test_forwards_subtask_candidates_on_reset():
    """Closed vocab, when set, is forwarded to the server on reset (fixed per episode)."""
    vocab = ["put the candy in the white box", "close the white box", "other"]
    client = _FakeAgentClient()
    wrapper = _make_wrapper(_RecordingPolicy(), client, max_drift=0.0, subtask_candidates=vocab)
    wrapper.prepare_new_episode(_obs())
    wrapper.shutdown()
    assert client.reset_candidates == [vocab]


def test_no_candidates_forwards_none_by_default():
    """Free-form (default) sends no candidates, preserving the pre-closed-vocab path."""
    client = _FakeAgentClient()
    wrapper = _make_wrapper(_RecordingPolicy(), client, max_drift=0.0)
    wrapper.prepare_new_episode(_obs())
    wrapper.shutdown()
    assert client.reset_candidates == [None]


def test_reinfers_every_boundary():
    """max_drift=0 -> every get_action is a boundary -> subtask advances each step."""
    client = _FakeAgentClient()
    inner = _RecordingPolicy()
    wrapper = _make_wrapper(inner, client, max_drift=0.0)
    wrapper.prepare_new_episode(_obs())  # step-0 -> subtask_1
    for _ in range(3):
        wrapper.get_action(_obs())
    wrapper.shutdown()

    # step-0 primed subtask_1; each get_action re-infers the next subtask.
    assert inner.seen_tasks == ["subtask_2", "subtask_3", "subtask_4"]


def test_coasts_between_boundaries():
    """Large max_drift -> no re-inference -> System-1 coasts on the step-0 subtask."""
    client = _FakeAgentClient()
    inner = _RecordingPolicy()
    wrapper = _make_wrapper(inner, client, max_drift=100.0)
    wrapper.prepare_new_episode(_obs())  # step-0 -> subtask_1
    for _ in range(3):
        wrapper.get_action(_obs())
    wrapper.shutdown()

    assert inner.seen_tasks == ["subtask_1", "subtask_1", "subtask_1"]


def test_coasts_after_session_ends():
    """When the agent session ends, the wrapper coasts on the last subtask, no hang."""
    client = _FakeAgentClient(max_subtasks=2)
    inner = _RecordingPolicy()
    wrapper = _make_wrapper(inner, client, max_drift=0.0)
    wrapper.prepare_new_episode(_obs())  # subtask_1
    for _ in range(5):
        wrapper.get_action(_obs())
    wrapper.shutdown()

    assert wrapper._session_finished
    # subtask_2 was the last live subtask; every step after the end coasts on it.
    assert inner.seen_tasks[-3:] == ["subtask_2", "subtask_2", "subtask_2"]


def test_reset_ends_session_coasts_on_fallback():
    """Agent calls task_done on turn 0 -> wrapper coasts on the fallback response."""
    client = _FakeAgentClient(reset_ends=True)
    inner = _RecordingPolicy()
    wrapper = _make_wrapper(inner, client, max_drift=0.0)
    wrapper.prepare_new_episode(_obs())
    wrapper.get_action(_obs())
    wrapper.shutdown()

    assert wrapper._session_finished
    assert wrapper.last_agent_output == wrapper._fallback
    assert inner.seen_tasks == [wrapper._fallback]


def test_shutdown_closes_remote_session():
    """shutdown() closes the server-side session and clears local state."""
    client = _FakeAgentClient()
    inner = _RecordingPolicy()
    wrapper = _make_wrapper(inner, client, max_drift=0.0)
    wrapper.prepare_new_episode(_obs())
    session_id = wrapper._session_id
    wrapper.get_action(_obs())
    wrapper.shutdown()

    assert session_id is not None
    assert client.closed_sessions == [session_id]
    assert wrapper._session_id is None


def test_new_episode_starts_fresh_session():
    """prepare_new_episode closes the prior session and opens a new session_id."""
    client = _FakeAgentClient()
    inner = _RecordingPolicy()
    wrapper = _make_wrapper(inner, client, max_drift=0.0)
    wrapper.prepare_new_episode(_obs())
    first = wrapper._session_id
    wrapper.prepare_new_episode(_obs())
    second = wrapper._session_id
    wrapper.shutdown()

    assert first != second
    assert client.reset_sessions == [first, second]
    assert first in client.closed_sessions  # prior session torn down


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
