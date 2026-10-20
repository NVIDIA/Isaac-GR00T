# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
