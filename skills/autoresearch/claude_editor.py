# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Run Claude Code against one isolated AutoResearch worktree."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any

from skills.autoresearch.codex_editor import (
    _RESULT_SCHEMA,
    EditResult,
    _load_edit_result,
    _make_private_directory,
    _new_attempt_directory,
    _open_private_text,
    _utc_now,
    _write_private_json,
    _write_private_text,
)

_TOOLS = "Read,Edit,Write,Glob,Grep"


@dataclass(frozen=True)
class ClaudeEditorConfig:
    """Public Claude Code settings for one candidate edit."""

    executable: str = "claude"
    model: str | None = None
    timeout_s: int = 3600

    def __post_init__(self) -> None:
        if not self.executable.strip():
            raise ValueError("Claude executable must not be empty")
        if self.timeout_s < 1:
            raise ValueError("Claude timeout_s must be >= 1")


class ClaudeEditor:
    """Invoke Claude Code noninteractively with bounded editing tools."""

    def __init__(self, config: ClaudeEditorConfig | None = None) -> None:
        self.config = config or ClaudeEditorConfig()

    def command(self) -> list[str]:
        """Build the noninteractive Claude Code command."""
        command = [
            self.config.executable,
            "-p",
            "--safe-mode",
            "--no-session-persistence",
            "--no-chrome",
            "--disable-slash-commands",
            "--permission-mode",
            "acceptEdits",
            "--tools",
            _TOOLS,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(_RESULT_SCHEMA, separators=(",", ":")),
        ]
        if self.config.model:
            command.extend(["--model", self.config.model])
        return command

    def edit(self, *, worktree: Path, prompt: str, artifact_dir: Path) -> EditResult:
        """Run one Claude attempt and parse its structured result."""
        _make_private_directory(artifact_dir)
        attempt_dir, attempt_number = _new_attempt_directory(artifact_dir)
        prompt_path = attempt_dir / "prompt.txt"
        stdout_path = attempt_dir / "claude.stdout.json"
        stderr_path = attempt_dir / "claude.stderr.log"
        result_path = attempt_dir / "result.json"
        metadata_path = attempt_dir / "attempt.json"

        _write_private_text(prompt_path, prompt)
        command = self.command()
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "attempt": attempt_number,
            "host": "claude",
            "status": "running",
            "started_at": _utc_now(),
            "worktree": str(worktree.resolve()),
            "command": command,
            "model": self.config.model,
            "timeout_s": self.config.timeout_s,
            "returncode": None,
            "error": None,
        }
        _write_private_json(metadata_path, metadata)

        try:
            with (
                _open_private_text(stdout_path) as stdout_file,
                _open_private_text(stderr_path) as stderr_file,
            ):
                completed = subprocess.run(
                    command,
                    cwd=worktree,
                    input=prompt,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    check=False,
                    timeout=self.config.timeout_s,
                )
            metadata["returncode"] = completed.returncode
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Claude exited with status {completed.returncode}; see {stderr_path}"
                )
            _extract_result(stdout_path, result_path)
            result = _load_edit_result(
                result_path=result_path,
                jsonl_path=stdout_path,
                stderr_path=stderr_path,
                source="Claude",
            )
        except Exception as exc:
            metadata["status"] = "failed"
            metadata["error"] = f"{type(exc).__name__}: {exc}"
            raise
        else:
            metadata["status"] = "succeeded"
            return result
        finally:
            if result_path.exists():
                result_path.chmod(0o600)
            metadata["completed_at"] = _utc_now()
            _write_private_json(metadata_path, metadata)


def _extract_result(stdout_path: Path, result_path: Path) -> None:
    """Extract the schema-validated payload from Claude Code output."""
    try:
        envelope = json.loads(stdout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Claude did not produce valid JSON: {stdout_path}") from exc
    if not isinstance(envelope, dict):
        raise TypeError(f"Claude output must be a JSON object: {stdout_path}")
    if envelope.get("is_error"):
        raise RuntimeError(f"Claude reported an error: {envelope.get('result', 'unknown error')}")
    payload = envelope.get("structured_output")
    if not isinstance(payload, dict):
        raise TypeError(f"Claude output is missing structured_output: {stdout_path}")
    _write_private_json(result_path, payload)
