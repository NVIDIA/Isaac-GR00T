# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Run Codex against one isolated AutoResearch worktree."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, TextIO

# Codex output is schema-constrained so controller state stays structured.
_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string", "minLength": 1},
        "validation": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
    "required": [
        "hypothesis",
        "validation",
    ],
    "additionalProperties": False,
}

_SAFE_ENVIRONMENT_VARIABLES = (
    "CODEX_HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TMP",
    "TMPDIR",
)


# Editor dataclasses describe the CLI call and its parsed result.
@dataclass(frozen=True)
class CodexEditorConfig:
    """Public Codex CLI settings for one candidate edit."""

    executable: str = "codex"
    model: str | None = None
    timeout_s: int = 3600
    sandbox: str = "workspace-write"

    def __post_init__(self) -> None:
        """Validate editor config before it reaches subprocess construction."""
        if not self.executable.strip():
            raise ValueError("Codex executable must not be empty")
        if self.timeout_s < 1:
            raise ValueError("Codex timeout_s must be >= 1")
        if self.sandbox not in {"workspace-write", "danger-full-access"}:
            raise ValueError("Codex sandbox must be 'workspace-write' or 'danger-full-access'")


# Hold the validated structured result from Codex.
@dataclass(frozen=True)
class EditResult:
    """Structured result emitted by Codex after editing a worktree."""

    hypothesis: str
    validation: tuple[str, ...]
    jsonl_path: Path
    stderr_path: Path
    result_path: Path


# CodexEditor is the only production path that invokes the Codex CLI.
class CodexEditor:
    """Invoke ``codex exec`` with bounded permissions and durable traces."""

    def __init__(self, config: CodexEditorConfig | None = None) -> None:
        """Store constructor dependencies without performing work."""
        self.config = config or CodexEditorConfig()

    def command(
        self,
        *,
        worktree: Path,
        artifact_dir: Path,
        runtime_dir: Path,
    ) -> list[str]:
        """Assemble the Codex CLI command for one attempt."""
        # Only the structured result is durable; schema and HOME are temporary.
        schema_path = runtime_dir / "result.schema.json"
        result_path = artifact_dir / "result.json"
        agent_home = runtime_dir / "home"
        # Codex runs with a private home and explicit structured output contract.
        command = [
            self.config.executable,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--json",
            "--sandbox",
            self.config.sandbox,
            "-c",
            'approval_policy="never"',
            "-c",
            'shell_environment_policy.inherit="none"',
            "-c",
            ('shell_environment_policy.include_only=["HOME","PATH","LANG","LC_ALL","LC_CTYPE"]'),
            "-c",
            _shell_environment_set(agent_home),
            "-C",
            str(worktree),
            "--add-dir",
            str(agent_home),
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(result_path),
        ]
        # Workspace-write mode disables network access for research edits.
        if self.config.sandbox == "workspace-write":
            command[command.index("-C") : command.index("-C")] = [
                "-c",
                "sandbox_workspace_write.network_access=false",
            ]
        if self.config.model:
            command.extend(["--model", self.config.model])
        command.append("-")
        return command

    def edit(self, *, worktree: Path, prompt: str, artifact_dir: Path) -> EditResult:
        """Run one editor attempt and parse the structured result."""
        # Each editor attempt gets its own private directory for restartability.
        _make_private_directory(artifact_dir)
        attempt_dir, attempt_number = _new_attempt_directory(artifact_dir)
        runtime = tempfile.TemporaryDirectory(prefix="groot-autoresearch-codex-")
        runtime_dir = Path(runtime.name)
        schema_path = runtime_dir / "result.schema.json"
        prompt_path = attempt_dir / "prompt.txt"
        jsonl_path = attempt_dir / "codex.jsonl"
        stderr_path = attempt_dir / "codex.stderr.log"
        result_path = attempt_dir / "result.json"
        metadata_path = attempt_dir / "attempt.json"
        agent_home = runtime_dir / "home"
        # Keep only the prompt; HOME and schema are removed after this attempt.
        _make_private_directory(agent_home)
        _write_private_json(schema_path, _RESULT_SCHEMA)
        _write_private_text(prompt_path, prompt)
        # Unlinking makes the stale-result invariant explicit for restarted attempts.
        result_path.unlink(missing_ok=True)

        command = self.command(
            worktree=worktree,
            artifact_dir=attempt_dir,
            runtime_dir=runtime_dir,
        )
        # Attempt metadata is updated in finally so failures remain inspectable.
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "attempt": attempt_number,
            "status": "running",
            "started_at": _utc_now(),
            "worktree": str(worktree.resolve()),
            "command": command,
            "model": self.config.model,
            "sandbox": self.config.sandbox,
            "timeout_s": self.config.timeout_s,
            "returncode": None,
            "error": None,
        }
        _write_private_json(metadata_path, metadata)

        try:
            # Stream stdout and stderr directly to private files.
            with (
                _open_private_text(jsonl_path) as stdout_file,
                _open_private_text(stderr_path) as stderr_file,
            ):
                completed = subprocess.run(
                    command,
                    input=prompt,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    check=False,
                    timeout=self.config.timeout_s,
                    env=_safe_environment(agent_home),
                )
            metadata["returncode"] = completed.returncode
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Codex exited with status {completed.returncode}; see {stderr_path}"
                )
            result = _load_edit_result(
                result_path=result_path,
                jsonl_path=jsonl_path,
                stderr_path=stderr_path,
            )
        except Exception as exc:
            metadata["status"] = "failed"
            metadata["error"] = f"{type(exc).__name__}: {exc}"
            raise
        else:
            metadata["status"] = "succeeded"
            return result
        finally:
            runtime.cleanup()
            if result_path.exists():
                result_path.chmod(0o600)
            metadata["completed_at"] = _utc_now()
            _write_private_json(metadata_path, metadata)


def _load_edit_result(
    *,
    result_path: Path,
    jsonl_path: Path,
    stderr_path: Path,
) -> EditResult:
    """Result parsing validates Codex output before git changes are accepted."""
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Codex did not produce a valid structured result: {result_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(  # noqa: TRY004
            f"Codex structured result must be an object: {result_path}"
        )

    # Enforce the schema again locally before accepting git changes.
    required = {"hypothesis", "validation"}
    missing = sorted(required - payload.keys())
    extra = sorted(payload.keys() - required)
    if missing:
        raise RuntimeError(f"Codex result is missing required fields: {', '.join(missing)}")
    if extra:
        raise RuntimeError(f"Codex result has unexpected fields: {', '.join(extra)}")

    # Strip user-facing text fields while preserving their required meaning.
    hypothesis = payload["hypothesis"]
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise RuntimeError("Codex result field hypothesis must be a non-empty string")

    validation = payload["validation"]
    if not isinstance(validation, list) or any(
        not isinstance(item, str) or not item.strip() for item in validation
    ):
        raise RuntimeError("Codex result field 'validation' must be a list of non-empty strings")
    return EditResult(
        hypothesis=hypothesis.strip(),
        validation=tuple(item.strip() for item in validation),
        jsonl_path=jsonl_path,
        stderr_path=stderr_path,
        result_path=result_path,
    )


def _safe_environment(agent_home: Path) -> dict[str, str]:
    """Private file and environment helpers reduce leakage between attempts."""
    environment = {
        name: os.environ[name] for name in _SAFE_ENVIRONMENT_VARIABLES if name in os.environ
    }
    if "CODEX_HOME" not in environment and os.environ.get("HOME"):
        environment["CODEX_HOME"] = str(Path(os.environ["HOME"]) / ".codex")
    environment["HOME"] = str(agent_home)
    environment.setdefault("PATH", os.defpath)
    return environment


def _shell_environment_set(agent_home: Path) -> str:
    """Format shell environment overrides for Codex."""
    values = {
        "HOME": str(agent_home),
        "PATH": os.environ.get("PATH", os.defpath),
    }
    for name in ("LANG", "LC_ALL", "LC_CTYPE"):
        if name in os.environ:
            values[name] = os.environ[name]
    assignments = ",".join(f"{name}={json.dumps(value)}" for name, value in values.items())
    return f"shell_environment_policy.set={{{assignments}}}"


def _make_private_directory(path: Path) -> None:
    """Create private attempt files with conservative permissions."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def _new_attempt_directory(artifact_dir: Path) -> tuple[Path, int]:
    """Create private attempt files with conservative permissions."""
    attempt = 1
    while True:
        attempt_dir = artifact_dir / f"attempt-{attempt:04d}"
        try:
            attempt_dir.mkdir(mode=0o700)
        except FileExistsError:
            attempt += 1
            continue
        attempt_dir.chmod(0o700)
        return attempt_dir, attempt


def _open_private_text(path: Path) -> TextIO:
    """Create private attempt files with conservative permissions."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8")


def _write_private_text(path: Path, text: str) -> None:
    """Create private attempt files with conservative permissions."""
    with _open_private_text(path) as stream:
        stream.write(text)


def _write_private_json(path: Path, payload: Any) -> None:
    """Create private attempt files with conservative permissions."""
    _write_private_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _utc_now() -> str:
    """Return timestamp text for attempt metadata."""
    return datetime.now(timezone.utc).isoformat()
