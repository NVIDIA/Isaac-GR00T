# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
from skills.autoresearch.agent_editor import agent_executable, create_editor
from skills.autoresearch.claude_editor import ClaudeEditor, ClaudeEditorConfig
from skills.autoresearch.codex_editor import CodexEditor


def test_editor_factory_defaults_to_codex() -> None:
    assert isinstance(create_editor({}), CodexEditor)
    assert agent_executable({}) == "codex"


def test_editor_factory_selects_claude() -> None:
    editor = create_editor({"host": "claude", "model": "sonnet", "timeout_s": 42})

    assert isinstance(editor, ClaudeEditor)
    assert editor.config.model == "sonnet"
    assert editor.config.timeout_s == 42
    assert agent_executable({"host": "claude"}) == "claude"


def test_editor_factory_rejects_unknown_host() -> None:
    with pytest.raises(ValueError, match="agent.host"):
        create_editor({"host": "unknown"})


def test_claude_rejects_codex_sandbox() -> None:
    with pytest.raises(ValueError, match="agent.sandbox"):
        create_editor({"host": "claude", "sandbox": "workspace-write"})


def test_claude_editor_uses_structured_bounded_cli(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write(
            json.dumps(
                {
                    "is_error": False,
                    "structured_output": {
                        "hypothesis": "Reduce dropout.",
                        "validation": ["Read the changed config."],
                    },
                }
            )
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    editor = ClaudeEditor(ClaudeEditorConfig(model="sonnet", timeout_s=30))

    result = editor.edit(
        worktree=worktree,
        prompt="Make one bounded change.",
        artifact_dir=tmp_path / "artifacts",
    )

    command, kwargs = calls[0]
    assert command[:2] == ["claude", "-p"]
    assert "--safe-mode" in command
    assert command[command.index("--permission-mode") + 1] == "acceptEdits"
    assert command[command.index("--tools") + 1] == "Read,Edit,Write,Glob,Grep"
    assert "Bash" not in command[command.index("--tools") + 1]
    assert command[command.index("--model") + 1] == "sonnet"
    assert command[command.index("--output-format") + 1] == "json"
    schema = json.loads(command[command.index("--json-schema") + 1])
    assert schema["required"] == ["hypothesis", "validation"]
    assert kwargs["cwd"] == worktree
    assert kwargs["input"] == "Make one bounded change."
    assert result.hypothesis == "Reduce dropout."
    assert result.validation == ("Read the changed config.",)
    metadata = json.loads((result.result_path.parent / "attempt.json").read_text())
    assert metadata["host"] == "claude"
    assert metadata["status"] == "succeeded"


def test_claude_editor_rejects_error_output(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command, **kwargs):
        kwargs["stdout"].write(json.dumps({"is_error": True, "result": "not authenticated"}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="not authenticated"):
        ClaudeEditor().edit(
            worktree=tmp_path,
            prompt="Make one change.",
            artifact_dir=tmp_path / "artifacts",
        )


def test_claude_editor_records_failed_execution(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 7),
    )

    with pytest.raises(RuntimeError, match="status 7"):
        ClaudeEditor().edit(
            worktree=tmp_path,
            prompt="Make one change.",
            artifact_dir=tmp_path / "artifacts",
        )

    metadata = json.loads((tmp_path / "artifacts/attempt-0001/attempt.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["returncode"] == 7


def test_claude_editor_uses_numbered_retry_attempts(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command, **kwargs):
        kwargs["stdout"].write(
            json.dumps(
                {
                    "is_error": False,
                    "structured_output": {
                        "hypothesis": "Try one change.",
                        "validation": [],
                    },
                }
            )
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    editor = ClaudeEditor()
    artifact_dir = tmp_path / "artifacts"

    first = editor.edit(worktree=tmp_path, prompt="First.", artifact_dir=artifact_dir)
    second = editor.edit(worktree=tmp_path, prompt="Second.", artifact_dir=artifact_dir)

    assert first.result_path.parent.name == "attempt-0001"
    assert second.result_path.parent.name == "attempt-0002"


def test_claude_editor_config_requires_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        ClaudeEditorConfig(timeout_s=0)
