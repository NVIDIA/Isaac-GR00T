# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentClient request framing (no network): the reset payload's optional candidates.

We stub ``_rpc`` to capture the request dict so we can assert the wire framing
without a live ZMQ server.
"""

from __future__ import annotations

from typing import Any

from PIL import Image as PILImage

from groot.core.policy.agent.agent_client import AgentClient


def _client_capturing() -> tuple[AgentClient, list[dict[str, Any]]]:
    client = AgentClient("tcp://unused")
    sent: list[dict[str, Any]] = []

    def _fake_rpc(payload: dict[str, Any]) -> dict[str, Any]:
        sent.append(payload)
        return {"ok": True, "subtask": "x", "session_ended": False}

    client._rpc = _fake_rpc  # type: ignore[assignment]
    return client, sent


def _reset(client: AgentClient, **kwargs):
    return client.reset(
        session_id="s0",
        goal="g",
        system_prompt="sys",
        robot_description="desc",
        images=[PILImage.new("RGB", (2, 2))],
        **kwargs,
    )


def test_reset_includes_candidates_when_set():
    client, sent = _client_capturing()
    vocab = ["put the candy in the white box", "other"]
    _reset(client, candidates=vocab)
    assert sent[0]["candidates"] == vocab


def test_reset_omits_candidates_when_absent():
    """Backward compat: no candidates key so older servers see the legacy request."""
    client, sent = _client_capturing()
    _reset(client)
    assert "candidates" not in sent[0]


def test_reset_omits_candidates_when_empty():
    client, sent = _client_capturing()
    _reset(client, candidates=[])
    assert "candidates" not in sent[0]
