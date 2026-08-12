# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Git worktrees for AutoResearch candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

_HELPER_PATHS = (".codex-autorun-prompt.txt", ".venv")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")


@dataclass(frozen=True)
class CandidateWorkspace:
    """One trial branch checked out as a Git worktree."""

    path: Path
    branch: str
    base_commit: str


class CompositionConflictError(RuntimeError):
    """A queued change conflicts with the current best revision."""

    def __init__(
        self,
        *,
        commit: str,
        conflict_files: tuple[str, ...],
    ) -> None:
        """Record the conflicting commit and files."""
        self.commit = commit
        self.conflict_files = conflict_files
        super().__init__(
            f"Could not compose queued change {commit}; conflicts: " + ", ".join(conflict_files)
        )


class GitWorkspaceManager:
    """Create, commit, publish, and clean up candidate worktrees."""

    def __init__(self, repository: Path, output_dir: Path) -> None:
        self.repository = repository.resolve()
        self.output_dir = output_dir.resolve()

    def create(
        self,
        *,
        trial_id: str,
        base_commit: str,
    ) -> CandidateWorkspace:
        """Create a trial branch and worktree from one commit."""
        path = self._candidate_path(trial_id)
        branch = self.branch_name(trial_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._run(
            ["git", "worktree", "add", "-b", branch, str(path), base_commit],
            cwd=self.repository,
        )
        return CandidateWorkspace(path=path, branch=branch, base_commit=base_commit)

    @staticmethod
    def branch_name(trial_id: str) -> str:
        """Return the deterministic branch name for a trial."""
        return f"codex/autoresearch-{_slug(trial_id)}"

    def reset_incomplete(self, *, trial_id: str) -> None:
        """Remove work left by an interrupted candidate edit."""
        path = self._candidate_path(trial_id)
        branch = self.branch_name(trial_id)
        self._run(
            ["git", "worktree", "remove", "--force", str(path)],
            cwd=self.repository,
            check=False,
        )
        self._run(["git", "worktree", "prune"], cwd=self.repository)
        self._run(
            ["git", "branch", "-D", branch],
            cwd=self.repository,
            check=False,
        )

    def reset_to_commit(self, workspace: CandidateWorkspace, *, commit: str) -> None:
        """Discard one failed edit attempt and restore a clean worktree."""
        self._run(["git", "reset", "--hard", commit], cwd=workspace.path)
        self._run(["git", "clean", "-fd"], cwd=workspace.path)

    def changed_files(self, workspace: CandidateWorkspace) -> list[str]:
        """List tracked and untracked candidate changes."""
        output = self._run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=workspace.path,
        ).stdout
        entries = [entry for entry in output.split("\0") if entry]
        changed: list[str] = []
        index = 0
        while index < len(entries):
            entry = entries[index]
            if len(entry) >= 4:
                status = entry[:2]
                path = entry[3:]
                if not _is_helper_path(path):
                    changed.append(path)
                if "R" in status or "C" in status:
                    index += 1
                    if index < len(entries) and not _is_helper_path(entries[index]):
                        changed.append(entries[index])
            index += 1
        return sorted(set(changed))

    def diff(self, workspace: CandidateWorkspace) -> str:
        """Return a binary diff, including untracked files."""
        output = self._run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=workspace.path,
        ).stdout
        untracked = [path for path in output.split("\0") if path]
        if untracked:
            self._run(["git", "add", "-N", "--", *untracked], cwd=workspace.path)
        try:
            return self._run(
                ["git", "diff", "--binary", "HEAD"],
                cwd=workspace.path,
            ).stdout
        finally:
            if untracked:
                self._run(["git", "reset", "--quiet"], cwd=workspace.path)

    @staticmethod
    def validate_changed_files(
        changed_files: list[str],
        *,
        editable_paths: tuple[str, ...],
        frozen_paths: tuple[str, ...],
    ) -> list[str]:
        """Return changes outside the configured research surface."""
        violations = []
        for path in changed_files:
            if _matches_any(path, frozen_paths):
                violations.append(f"{path}: frozen")
            elif not _matches_any(path, editable_paths):
                violations.append(f"{path}: outside editable paths")
        return violations

    def commit(
        self,
        workspace: CandidateWorkspace,
        *,
        message: str,
        changed_files: list[str] | tuple[str, ...],
    ) -> str:
        """Commit the files already approved by the controller."""
        files = sorted(set(changed_files))
        if not files:
            raise ValueError("Codex made no candidate changes")
        self._run(["git", "add", "--", *files], cwd=workspace.path)
        self._run(
            ["git", "-c", "core.hooksPath=/dev/null", "commit", "-m", message],
            cwd=workspace.path,
        )
        return self.head(workspace)

    def commit_diff(self, commit: str) -> str:
        """Return the patch stored in a candidate commit."""
        return self._run(
            ["git", "show", "--binary", "--format=", commit],
            cwd=self.repository,
        ).stdout

    def apply_commit(self, workspace: CandidateWorkspace, *, commit: str) -> None:
        """Apply one good change without committing the composition."""
        try:
            self._run(
                ["git", "-c", "core.hooksPath=/dev/null", "cherry-pick", "--no-commit", commit],
                cwd=workspace.path,
            )
        except RuntimeError as exc:
            conflict_files = tuple(self._conflict_files(workspace))
            if not conflict_files:
                raise
            self._run(["git", "reset", "--merge", "HEAD"], cwd=workspace.path)
            raise CompositionConflictError(
                commit=commit,
                conflict_files=conflict_files,
            ) from exc

    def head(self, workspace: CandidateWorkspace) -> str:
        """Return the worktree HEAD commit."""
        return self._run(["git", "rev-parse", "HEAD"], cwd=workspace.path).stdout.strip()

    def publish_branch(self, *, branch: str, commit: str, remote: str = "origin") -> None:
        """Push the candidate commit to its trial branch."""
        self._run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "push",
                remote,
                f"{commit}:refs/heads/{branch}",
            ],
            cwd=self.repository,
        )

    def remove(self, workspace: CandidateWorkspace) -> None:
        """Remove a completed candidate worktree and keep its branch."""
        if workspace.path.exists():
            self._remove_worktree(workspace.path)

    def discard(self, workspace: CandidateWorkspace) -> None:
        """Remove a rejected worktree and its local branch."""
        self.remove(workspace)
        self._run(
            ["git", "branch", "-D", workspace.branch],
            cwd=self.repository,
            check=False,
        )

    def _candidate_path(self, trial_id: str) -> Path:
        """Return the worktree path for a safe trial identifier."""
        if not _SAFE_ID.fullmatch(trial_id):
            raise ValueError(f"Unsafe trial_id: {trial_id!r}")
        return self.output_dir / trial_id / "worktree"

    def _remove_worktree(self, path: Path) -> None:
        """Force-remove one candidate worktree."""
        self._run(
            ["git", "worktree", "remove", "--force", str(path)],
            cwd=self.repository,
        )

    def _conflict_files(self, workspace: CandidateWorkspace) -> list[str]:
        """List files with unresolved cherry-pick conflicts."""
        output = self._run(
            ["git", "diff", "--name-only", "--diff-filter=U", "-z"],
            cwd=workspace.path,
        ).stdout
        return sorted(path for path in output.split("\0") if path)

    @staticmethod
    def _run(
        command: list[str],
        *,
        cwd: Path,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run Git and surface its stdout and stderr on failure."""
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit code {result.returncode}: {' '.join(command)}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    """Return whether a path matches a configured file or directory."""
    normalized = path.strip("/")
    for pattern in patterns:
        clean = pattern.strip("/")
        if normalized == clean:
            return True
        if pattern.endswith("/") and normalized.startswith(f"{clean}/"):
            return True
    return False


def _is_helper_path(path: str) -> bool:
    """Return whether a path is a local Codex helper artifact."""
    normalized = path.rstrip("/")
    return any(
        normalized == helper or normalized.startswith(f"{helper}/") for helper in _HELPER_PATHS
    )


def _slug(value: str) -> str:
    """Convert an identifier into a Git branch component."""
    return re.sub(r"[^a-z0-9.-]+", "-", value.lower()).strip(".-") or "trial"
