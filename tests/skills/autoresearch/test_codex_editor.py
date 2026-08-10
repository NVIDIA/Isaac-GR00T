# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ARG001

from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess

import pytest
from skills.autoresearch.codex_editor import CodexEditor, CodexEditorConfig


def _result_payload() -> dict[str, object]:
    return {
        "hypothesis": "Reduce the learning rate.",
        "validation": ["pytest focused_test.py"],
    }


def test_codex_editor_uses_bounded_noninteractive_cli(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        runtime_home = Path(kwargs["env"]["HOME"])
        runtime_schema = Path(command[command.index("--output-schema") + 1])
        assert runtime_home.is_dir()
        assert runtime_schema.is_file()
        result_path = Path(command[command.index("--output-last-message") + 1])
        result_path.write_text(json.dumps(_result_payload()))
        kwargs["stdout"].write('{"type":"turn.completed"}\n')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("HOME", "/safe/home")
    monkeypatch.setenv("PATH", "/safe/bin")
    monkeypatch.setenv("CODEX_HOME", "/safe/codex")
    monkeypatch.setenv("GITLAB_TOKEN", "must-not-leak")
    monkeypatch.setenv("AWS_PROFILE", "must-not-leak")
    editor = CodexEditor(CodexEditorConfig(model="gpt-test", timeout_s=30))
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    artifact_dir = tmp_path / "artifacts"

    result = editor.edit(
        worktree=worktree,
        prompt="Make one change.",
        artifact_dir=artifact_dir,
    )

    command, kwargs = calls[0]
    runtime_home = Path(kwargs["env"]["HOME"])
    runtime_schema = Path(command[command.index("--output-schema") + 1])
    assert command[:2] == ["codex", "exec"]
    assert "--ephemeral" in command
    assert "--ignore-rules" in command
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert "sandbox_workspace_write.network_access=false" in command
    assert 'shell_environment_policy.inherit="none"' in command
    include_policy = next(
        item for item in command if item.startswith("shell_environment_policy.include_only=")
    )
    assert "CODEX_HOME" not in include_policy
    assert '"HOME"' in include_policy
    assert '"PATH"' in include_policy
    shell_set = next(item for item in command if item.startswith("shell_environment_policy.set="))
    assert "CODEX_HOME" not in shell_set
    assert command[command.index("--model") + 1] == "gpt-test"
    assert kwargs["input"] == "Make one change."
    assert kwargs["timeout"] == 30
    assert runtime_home.name == "home"
    assert artifact_dir not in runtime_home.parents
    assert not runtime_home.exists()
    assert kwargs["env"]["HOME"] != "/safe/home"
    assert kwargs["env"]["PATH"] == "/safe/bin"
    assert kwargs["env"]["CODEX_HOME"] == "/safe/codex"
    assert "GITLAB_TOKEN" not in kwargs["env"]
    assert "AWS_PROFILE" not in kwargs["env"]
    assert result.hypothesis == "Reduce the learning rate."
    assert result.validation == ("pytest focused_test.py",)
    assert result.result_path.parent.name == "attempt-0001"

    attempt_dir = result.result_path.parent
    assert command[command.index("--add-dir") + 1] == str(runtime_home)
    assert stat.S_IMODE(artifact_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(attempt_dir.stat().st_mode) == 0o700
    for name in (
        "attempt.json",
        "codex.jsonl",
        "codex.stderr.log",
        "prompt.txt",
        "result.json",
    ):
        assert stat.S_IMODE((attempt_dir / name).stat().st_mode) == 0o600
    metadata = json.loads((attempt_dir / "attempt.json").read_text())
    assert not (attempt_dir / "home").exists()
    assert not (attempt_dir / "result.schema.json").exists()
    assert not runtime_schema.exists()
    assert metadata["attempt"] == 1
    assert metadata["status"] == "succeeded"
    assert metadata["returncode"] == 0
    assert metadata["sandbox"] == "workspace-write"
    assert metadata["started_at"]
    assert metadata["completed_at"]


def test_codex_editor_rejects_failed_execution(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(subprocess, "run", fake_run)
    editor = CodexEditor()

    with pytest.raises(RuntimeError, match="status 7"):
        editor.edit(
            worktree=tmp_path,
            prompt="Make one change.",
            artifact_dir=tmp_path / "artifacts",
        )

    metadata_path = tmp_path / "artifacts" / "attempt-0001" / "attempt.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "failed"
    assert metadata["returncode"] == 7
    assert "status 7" in metadata["error"]


def test_codex_editor_uses_new_attempt_and_never_reads_stale_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    should_write_result = True

    def fake_run(command, **kwargs):
        if should_write_result:
            result_path = Path(command[command.index("--output-last-message") + 1])
            result_path.write_text(json.dumps(_result_payload()))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    editor = CodexEditor()
    artifact_dir = tmp_path / "artifacts"

    first = editor.edit(worktree=tmp_path, prompt="First.", artifact_dir=artifact_dir)
    should_write_result = False
    with pytest.raises(RuntimeError, match="did not produce a valid structured result"):
        editor.edit(worktree=tmp_path, prompt="Second.", artifact_dir=artifact_dir)

    assert first.result_path.parent.name == "attempt-0001"
    assert first.result_path.is_file()
    assert (artifact_dir / "attempt-0002" / "attempt.json").is_file()
    assert not (artifact_dir / "attempt-0002" / "result.json").exists()


@pytest.mark.parametrize(
    "payload,error",
    [
        ({**_result_payload(), "extra": "field"}, "unexpected fields"),
        ({**_result_payload(), "hypothesis": 7}, "non-empty string"),
        ({**_result_payload(), "validation": [7]}, "list of non-empty strings"),
    ],
)
def test_codex_editor_strictly_validates_structured_result(
    tmp_path: Path,
    monkeypatch,
    payload: dict[str, object],
    error: str,
) -> None:
    def fake_run(command, **kwargs):
        result_path = Path(command[command.index("--output-last-message") + 1])
        result_path.write_text(json.dumps(payload))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=error):
        CodexEditor().edit(
            worktree=tmp_path,
            prompt="Make one change.",
            artifact_dir=tmp_path / "artifacts",
        )


def test_codex_editor_config_requires_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        CodexEditorConfig(timeout_s=0)


def test_codex_editor_allows_explicit_no_bubblewrap_fallback(tmp_path: Path) -> None:
    editor = CodexEditor(CodexEditorConfig(sandbox="danger-full-access"))

    command = editor.command(
        worktree=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        runtime_dir=tmp_path / "runtime",
    )

    assert command[command.index("--sandbox") + 1] == "danger-full-access"
    assert "sandbox_workspace_write.network_access=false" not in command


def test_codex_editor_rejects_unknown_sandbox() -> None:
    with pytest.raises(ValueError, match="sandbox"):
        CodexEditorConfig(sandbox="unknown")


def test_codex_editor_derives_auth_home_before_replacing_process_home(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs["env"])
        result_path = Path(command[command.index("--output-last-message") + 1])
        result_path.write_text(json.dumps(_result_payload()))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("HOME", "/original/home")
    monkeypatch.delenv("CODEX_HOME", raising=False)

    CodexEditor().edit(
        worktree=tmp_path,
        prompt="Make one change.",
        artifact_dir=tmp_path / "artifacts",
    )

    assert captured["CODEX_HOME"] == "/original/home/.codex"
    runtime_home = Path(captured["HOME"])
    assert runtime_home.name == "home"
    assert not runtime_home.exists()
