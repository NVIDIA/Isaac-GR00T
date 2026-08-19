# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Select the coding agent used for AutoResearch candidate edits."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from skills.autoresearch.claude_editor import ClaudeEditor, ClaudeEditorConfig
from skills.autoresearch.codex_editor import CodexEditor, CodexEditorConfig, EditResult


class AgenticEditor(Protocol):
    """Interface implemented by every AutoResearch coding-agent host."""

    def edit(self, *, worktree: Path, prompt: str, artifact_dir: Path) -> EditResult:
        """Edit one candidate worktree and return a structured result."""


EditorFactory = Callable[[Mapping[str, Any]], AgenticEditor]


def _codex(agent: Mapping[str, Any]) -> AgenticEditor:
    sandbox = agent.get("sandbox", "workspace-write")
    if not isinstance(sandbox, str) or not sandbox:
        raise ValueError("agent.sandbox must be a non-empty string")
    return CodexEditor(
        CodexEditorConfig(
            executable=str(agent.get("executable") or "codex"),
            model=str(agent["model"]) if agent.get("model") else None,
            timeout_s=int(agent.get("timeout_s", 3600)),
            sandbox=sandbox,
        )
    )


def _claude(agent: Mapping[str, Any]) -> AgenticEditor:
    if "sandbox" in agent:
        raise ValueError("agent.sandbox is supported only by agent.host=codex")
    return ClaudeEditor(
        ClaudeEditorConfig(
            executable=str(agent.get("executable") or "claude"),
            model=str(agent["model"]) if agent.get("model") else None,
            timeout_s=int(agent.get("timeout_s", 3600)),
        )
    )


# Adding another host requires one factory and one entry here.
_EDITOR_FACTORIES: dict[str, EditorFactory] = {
    "claude": _claude,
    "codex": _codex,
}


def agent_host(agent: Mapping[str, Any]) -> str:
    """Return and validate the configured host, defaulting to Codex."""
    host = agent.get("host", "codex")
    if not isinstance(host, str) or host not in _EDITOR_FACTORIES:
        supported = ", ".join(sorted(_EDITOR_FACTORIES))
        raise ValueError(f"agent.host must be one of: {supported}")
    return host


def create_editor(agent: Mapping[str, Any]) -> AgenticEditor:
    """Build the configured editor without invoking its CLI."""
    return _EDITOR_FACTORIES[agent_host(agent)](agent)


def agent_executable(agent: Mapping[str, Any]) -> str:
    """Return the selected CLI executable for preflight validation."""
    host = agent_host(agent)
    return str(agent.get("executable") or host)
