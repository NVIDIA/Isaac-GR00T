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

from pathlib import Path
import subprocess

import pytest
from skills.autoresearch.git_workspace import (
    CandidateWorkspace,
    CompositionConflictError,
    GitWorkspaceManager,
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "AutoResearch Test")
    _git(repository, "config", "user.email", "autoresearch@example.com")
    (repository / "groot/core/training/models").mkdir(parents=True)
    (repository / "groot/core/training/models/model.py").write_text("BASE = 1\n")
    (repository / "groot/infra").mkdir(parents=True)
    (repository / "groot/infra/eval.py").write_text("METRIC = 1\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "baseline")
    return repository


def _create_conflicting_composition(
    repository: Path,
    tmp_path: Path,
) -> tuple[
    GitWorkspaceManager,
    CandidateWorkspace,
    str,
    str,
    CompositionConflictError,
]:
    manager = GitWorkspaceManager(repository, tmp_path / "runs")
    base = _git(repository, "rev-parse", "HEAD")
    source = manager.create(trial_id="source-conflict", base_commit=base)
    (source.path / "groot/core/training/models/model.py").write_text("BASE = 2\n")
    source_commit = manager.commit(
        source,
        message="Source value",
        changed_files=manager.changed_files(source),
    )
    manager.remove(source)

    parent = manager.create(trial_id="parent-conflict", base_commit=base)
    (parent.path / "groot/core/training/models/model.py").write_text("BASE = 3\n")
    parent_commit = manager.commit(
        parent,
        message="Parent value",
        changed_files=manager.changed_files(parent),
    )
    manager.remove(parent)

    composition = manager.create(
        trial_id="composition-conflict",
        base_commit=parent_commit,
    )
    with pytest.raises(CompositionConflictError) as exc_info:
        manager.apply_commit(composition, commit=source_commit)
    return manager, composition, source_commit, parent_commit, exc_info.value


def test_workspace_commits_editable_change(repository: Path, tmp_path: Path) -> None:
    manager = GitWorkspaceManager(repository, tmp_path / "runs")
    workspace = manager.create(
        trial_id="trial-0001",
        base_commit=_git(repository, "rev-parse", "HEAD"),
    )
    assert workspace.path == tmp_path / "runs" / "trial-0001" / "worktree"
    model = workspace.path / "groot/core/training/models/model.py"
    model.write_text("BASE = 2\n")

    changed = manager.changed_files(workspace)
    violations = manager.validate_changed_files(
        changed,
        editable_paths=("groot/core/training/models/",),
        frozen_paths=("groot/infra/",),
    )

    assert changed == ["groot/core/training/models/model.py"]
    assert violations == []
    commit = manager.commit(
        workspace,
        message="AutoResearch trial 1",
        changed_files=changed,
    )
    assert commit == _git(workspace.path, "rev-parse", "HEAD")
    manager.remove(workspace)


def test_reset_to_commit_discards_tracked_and_untracked_changes(
    repository: Path,
    tmp_path: Path,
) -> None:
    manager = GitWorkspaceManager(repository, tmp_path / "runs")
    base_commit = _git(repository, "rev-parse", "HEAD")
    workspace = manager.create(trial_id="trial-retry", base_commit=base_commit)
    model = workspace.path / "groot/core/training/models/model.py"
    untracked = workspace.path / "groot/core/training/models/partial.py"
    model.write_text("BASE = 99\n")
    untracked.write_text("PARTIAL = True\n")

    manager.reset_to_commit(workspace, commit=base_commit)

    assert manager.head(workspace) == base_commit
    assert model.read_text() == "BASE = 1\n"
    assert not untracked.exists()
    assert manager.changed_files(workspace) == []
    manager.discard(workspace)


def test_workspace_rejects_frozen_and_out_of_scope_changes(
    repository: Path,
    tmp_path: Path,
) -> None:
    manager = GitWorkspaceManager(repository, tmp_path / "runs")
    workspace = manager.create(
        trial_id="trial-0002",
        base_commit=_git(repository, "rev-parse", "HEAD"),
    )
    (workspace.path / "groot/infra/eval.py").write_text("METRIC = 2\n")
    (workspace.path / "README.md").write_text("out of scope\n")

    violations = manager.validate_changed_files(
        manager.changed_files(workspace),
        editable_paths=("groot/core/training/models/",),
        frozen_paths=("groot/infra/",),
    )

    assert violations == ["README.md: outside editable paths", "groot/infra/eval.py: frozen"]
    manager.remove(workspace)


def test_workspace_commits_only_approved_files(repository: Path, tmp_path: Path) -> None:
    manager = GitWorkspaceManager(repository, tmp_path / "runs")
    workspace = manager.create(
        trial_id="trial-0003",
        base_commit=_git(repository, "rev-parse", "HEAD"),
    )
    (workspace.path / "groot/core/training/models/model.py").write_text("BASE = 2\n")
    validated = manager.changed_files(workspace)
    (workspace.path / "README.md").write_text("late change\n")

    commit = manager.commit(
        workspace,
        message="AutoResearch trial 3",
        changed_files=validated,
    )

    assert _git(workspace.path, "show", "--pretty=", "--name-only", commit) == validated[0]
    assert manager.changed_files(workspace) == ["README.md"]
    manager.remove(workspace)


def test_workspace_replays_candidate_change_without_moving_head(
    repository: Path,
    tmp_path: Path,
) -> None:
    manager = GitWorkspaceManager(repository, tmp_path / "runs")
    base = _git(repository, "rev-parse", "HEAD")
    source = manager.create(trial_id="source", base_commit=base)
    source_file = source.path / "groot/core/training/models/source.py"
    source_file.write_text("ENABLED = True\n")
    source_commit = manager.commit(
        source,
        message="Source change",
        changed_files=manager.changed_files(source),
    )
    manager.remove(source)

    parent = manager.create(trial_id="parent", base_commit=base)
    model = parent.path / "groot/core/training/models/model.py"
    model.write_text("BASE = 2\n")
    parent_commit = manager.commit(
        parent,
        message="Parent recipe",
        changed_files=manager.changed_files(parent),
    )
    manager.remove(parent)

    composition = manager.create(
        trial_id="composition",
        base_commit=parent_commit,
    )
    manager.apply_commit(composition, commit=source_commit)

    assert manager.head(composition) == parent_commit
    assert manager.changed_files(composition) == ["groot/core/training/models/source.py"]
    manager.discard(composition)


def test_workspace_reports_conflicting_composition(repository: Path, tmp_path: Path) -> None:
    manager, composition, source_commit, parent_commit, conflict = _create_conflicting_composition(
        repository, tmp_path
    )

    assert conflict.commit == source_commit
    assert conflict.conflict_files == ("groot/core/training/models/model.py",)
    assert manager.head(composition) == parent_commit
    assert manager.changed_files(composition) == []
    assert _git(composition.path, "status", "--porcelain") == ""
    manager.discard(composition)


def test_publish_branch_is_idempotent_after_worktree_removal(
    repository: Path,
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(repository, "remote", "add", "origin", str(remote))
    manager = GitWorkspaceManager(repository, tmp_path / "runs")
    workspace = manager.create(
        trial_id="trial-0004",
        base_commit=_git(repository, "rev-parse", "HEAD"),
    )
    (workspace.path / "groot/core/training/models/model.py").write_text("BASE = 4\n")
    changed = manager.changed_files(workspace)
    commit = manager.commit(
        workspace,
        message="AutoResearch trial 4",
        changed_files=changed,
    )
    branch = workspace.branch
    manager.remove(workspace)

    manager.publish_branch(branch=branch, commit=commit)
    manager.publish_branch(branch=branch, commit=commit)

    assert (
        _git(repository, "ls-remote", "--heads", "origin", f"refs/heads/{branch}").split()[0]
        == commit
    )


def test_workspace_rejects_unsafe_trial_id(repository: Path, tmp_path: Path) -> None:
    manager = GitWorkspaceManager(repository, tmp_path / "runs")

    with pytest.raises(ValueError, match="Unsafe trial_id"):
        manager.reset_incomplete(trial_id="../not-owned")


def test_reset_incomplete_removes_only_stale_worktree(
    repository: Path,
    tmp_path: Path,
) -> None:
    manager = GitWorkspaceManager(repository, tmp_path / "runs")
    workspace = manager.create(
        trial_id="trial-0005",
        base_commit=_git(repository, "rev-parse", "HEAD"),
    )
    artifact = workspace.path.parent / "codex" / "result.txt"
    artifact.parent.mkdir()
    artifact.write_text("keep me\n")

    manager.reset_incomplete(trial_id="trial-0005")

    assert not workspace.path.exists()
    assert artifact.read_text() == "keep me\n"
    result = subprocess.run(
        ["git", "show-ref", "--verify", f"refs/heads/{workspace.branch}"],
        cwd=repository,
        check=False,
    )
    assert result.returncode != 0


def test_branch_name_does_not_repeat_study_id(
    repository: Path,
    tmp_path: Path,
) -> None:
    manager = GitWorkspaceManager(repository, tmp_path / "runs")
    trial_id = "ar-yam-smoke-0005"

    workspace = manager.create(
        trial_id=trial_id,
        base_commit=_git(repository, "rev-parse", "HEAD"),
    )

    assert workspace.branch == "codex/autoresearch-ar-yam-smoke-0005"
    manager.discard(workspace)


def test_discard_removes_only_registered_candidate_branch(
    repository: Path,
    tmp_path: Path,
) -> None:
    manager = GitWorkspaceManager(repository, tmp_path / "runs")
    workspace = manager.create(
        trial_id="trial-0006",
        base_commit=_git(repository, "rev-parse", "HEAD"),
    )

    manager.discard(workspace)

    assert not workspace.path.exists()
    result = subprocess.run(
        ["git", "show-ref", "--verify", f"refs/heads/{workspace.branch}"],
        cwd=repository,
        check=False,
    )
    assert result.returncode != 0
