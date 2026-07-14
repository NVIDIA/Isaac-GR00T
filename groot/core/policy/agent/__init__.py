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
"""Robot-agnostic System-2 agent policy wrapper (thin client to an agent server).

A VLM runs as a full agent loop (nanobot ``EmbodiedSession``) hosted behind a
port; :class:`AgentSyncPolicyWrapper` is a thin client that ships camera frames
and gets back next-step subtasks, which are written into the observation before
the wrapped System-1 policy computes actions. System-1 owns the control loop; the
agent and loop exchange control synchronously (each ``step`` RPC blocks for one
agent turn). This package imports no nanobot — the agent server (which does)
lives in ``nanobot.embodied.backends.agent_sync``.
"""

from __future__ import annotations

from groot.core.policy.agent.agent_client import AgentClient, Subtask
from groot.core.policy.agent.agent_wrapper import AgentSyncPolicyWrapper

__all__ = ["AgentClient", "AgentSyncPolicyWrapper", "Subtask"]
