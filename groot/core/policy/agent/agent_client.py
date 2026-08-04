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
"""Thin ZMQ/msgpack client for the System-2 agent server.

This is the *only* thing the System-1 side (a robot control loop) needs to talk
to System-2. It knows nothing about nanobot: it ships camera frames + opaque
framing strings over a port and gets back a next-step subtask string. All the
agent machinery (nanobot ``EmbodiedSession``, the VLM provider/model) lives
behind the port on the server.

Protocol mirrors ``nanobot.embodied.backends.agent_sync.server``: session-stateful,
keyed by ``session_id``; ``reset``/``step`` block until the server's agent turn
(the VLM round-trip) completes, so the blocking RPC IS the synchronous S1<->S2
handoff.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
from typing import Any

from PIL.Image import Image


@dataclass(slots=True)
class Subtask:
    """Server reply to a reset/step: the next subtask, or session-ended."""

    text: str | None
    session_ended: bool


@dataclass(slots=True)
class Prediction:
    """Server reply to a stateless predict: one action, and whether it fell back."""

    action: str | None
    used_fallback: bool


class AgentClient:
    """ZMQ REQ client for the agent server (one REQ socket per request)."""

    def __init__(self, address: str, *, timeout_ms: int = 300_000, jpeg_quality: int = 92) -> None:
        # Accept "host:port" or "tcp://host:port".
        self.endpoint = address if "://" in address else f"tcp://{address}"
        self.timeout_ms = int(timeout_ms)
        self.jpeg_quality = int(jpeg_quality)
        import zmq  # local import: keep module import light for non-agent code paths

        self._zmq = zmq
        self._context = zmq.Context.instance()

    def _rpc(self, payload: dict[str, Any]) -> dict[str, Any]:
        import msgpack

        # A fresh REQ socket per request: robust to timeouts (a timed-out REQ
        # socket is otherwise stuck in the wrong state) and thread-affinity.
        socket = self._context.socket(self._zmq.REQ)
        socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(self._zmq.LINGER, 0)
        socket.connect(self.endpoint)
        try:
            socket.send(msgpack.packb(payload, use_bin_type=True))
            reply = dict(msgpack.unpackb(socket.recv(), raw=False))
        except self._zmq.Again as exc:
            raise TimeoutError(f"agent server {self.endpoint} timed out") from exc
        finally:
            socket.close(0)
        if not reply.get("ok"):
            raise RuntimeError(f"agent server error: {reply.get('error')}")
        return reply

    def _jpeg(self, image: Image) -> bytes:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=self.jpeg_quality)
        return buffer.getvalue()

    def ping(self) -> dict[str, Any]:
        return self._rpc({"cmd": "ping"})

    def reset(
        self,
        *,
        session_id: str,
        goal: str,
        system_prompt: str,
        robot_description: str,
        images: list[Image],
        candidates: list[str] | None = None,
    ) -> Subtask:
        request: dict[str, object] = {
            "cmd": "reset",
            "session_id": session_id,
            "goal": goal,
            "system_prompt": system_prompt,
            "robot_description": robot_description,
            "images": [self._jpeg(img) for img in images],
        }
        # Optional closed-vocabulary: only sent when non-empty so older servers
        # (which ignore unknown keys) stay backward-compatible and the free-form
        # path is byte-identical to before.
        if candidates:
            request["candidates"] = list(candidates)
        reply = self._rpc(request)
        return Subtask(text=reply.get("subtask"), session_ended=bool(reply.get("session_ended")))

    def step(self, *, session_id: str, images: list[Image]) -> Subtask:
        reply = self._rpc(
            {
                "cmd": "step",
                "session_id": session_id,
                "images": [self._jpeg(img) for img in images],
            }
        )
        return Subtask(text=reply.get("subtask"), session_ended=bool(reply.get("session_ended")))

    def predict(
        self,
        *,
        goal: str,
        system_prompt: str,
        robot_description: str,
        images: list[Image],
        candidates: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Prediction:
        """One stateless action, no session.

        The agent server exposes this alongside reset/step for callers that need
        a single action rather than a closed-loop subtask stream (e.g. the G1
        navigation waypoint bridge, which asks for one image point per goal).
        """
        request: dict[str, object] = {
            "cmd": "predict",
            "goal": goal,
            "system_prompt": system_prompt,
            "robot_description": robot_description,
            "images": [self._jpeg(img) for img in images],
        }
        # Omit optional keys when unset so the server keeps its own defaults.
        if candidates:
            request["candidates"] = list(candidates)
        if max_tokens is not None:
            request["max_tokens"] = int(max_tokens)
        if temperature is not None:
            request["temperature"] = float(temperature)
        reply = self._rpc(request)
        return Prediction(
            action=reply.get("action"),
            used_fallback=bool(reply.get("used_fallback")),
        )

    def close(self, *, session_id: str) -> None:
        try:
            self._rpc({"cmd": "close", "session_id": session_id})
        except (TimeoutError, RuntimeError):
            # Best-effort teardown: a dead/unreachable server must not raise out
            # of shutdown paths.
            pass


__all__ = ["AgentClient", "Prediction", "Subtask"]
