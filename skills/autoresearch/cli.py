# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Command-line interface for standalone AutoResearch."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlsplit

from skills.autoresearch.agent_editor import agent_executable, create_editor
from skills.autoresearch.config import StudyConfig
from skills.autoresearch.git_workspace import GitWorkspaceManager
from skills.autoresearch.script_runner import ScriptTrialRunner
from skills.autoresearch.study_state import StudyState, Trial
from skills.autoresearch.wave_controller import WaveController


def main(argv: list[str] | None = None) -> int:
    """Run or resume the study described by one config file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="AutoResearch YAML path")
    args = parser.parse_args(argv)

    configure_logging()
    config = StudyConfig.from_yaml(args.config)
    controller = _controller(config)

    _validate(config)
    controller.check_baseline_train_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # The state file determines whether this invocation initializes or resumes.
    if controller.state_path.exists():
        state = controller.load_state()
        phase = "resuming"
    else:
        state = controller.initialize()
        phase = "initialized"
    _print_progress(_study_summary(config, state, phase=phase))

    return _drive_controller(controller)


def configure_logging() -> None:
    """Show AutoResearch info logs without raising third-party log levels."""
    logging.basicConfig(format="%(message)s")
    logging.getLogger("skills.autoresearch").setLevel(logging.INFO)


def _controller(config: StudyConfig) -> WaveController:
    """The controller factory wires the script runner, editor, and workspace implementation."""
    editor = create_editor(config.agent)
    return WaveController(
        config=config,
        workspace_manager=GitWorkspaceManager(config.workspace, config.output_dir),
        editor=editor,
        runner=ScriptTrialRunner(config, repository=config.workspace),
        progress=_print_progress,
    )


def _drive_controller(controller: WaveController) -> int:
    """Loop saved one-step transitions until the controller stops or pauses."""
    # Each loop performs at most one controller transition.
    while True:
        result = controller.run_once()
        _print_progress(f"[AutoResearch] transition={result.action} | {result.message}")
        if result.done:
            return 0


def _print_progress(message: str) -> None:
    """Print progress immediately so long-running jobs remain observable."""
    print(message, flush=True)  # noqa: T201


def _study_summary(config: StudyConfig, state: StudyState, *, phase: str) -> str:
    """Build one self-contained initialization or resume summary."""
    active_wave = state.active_wave_index if state.active_wave_index is not None else "none"
    best_trial = state.best_trial_id or "baseline"
    return (
        f"[AutoResearch] {phase} | study={config.study_id} | wave={active_wave} | "
        f"trials={len(state.trials)}/{config.max_trials} | best={best_trial} | "
        f"best_score={state.best_score} | output_dir={config.output_dir}"
    )


def _validate(config: StudyConfig) -> None:
    """Validate local inputs without running user scripts."""
    if not config.workspace.is_dir():
        raise ValueError(f"Workspace does not exist: {config.workspace}")
    create_editor(config.agent)
    executable = agent_executable(config.agent)
    if shutil.which(executable) is None:
        raise ValueError(f"Agent CLI was not found on PATH: {executable}")

    # Script validation checks the wrapper paths and supplied baseline result.
    try:
        config.output_dir.relative_to(config.workspace)
    except ValueError:
        pass
    else:
        raise ValueError(f"output_dir must be outside workspace: {config.output_dir}")
    _validate_git_remote(
        config.workspace,
        config.git_remote,
        config.base_branch,
        config.base_commit,
    )
    trial = Trial(
        trial_id="ar-validation-0001",
        index=1,
        parent_trial_id=None,
        idea_type=config.idea_types[0],
        hypothesis="validation placeholder",
        code_branch="codex/autoresearch-validation",
        code_revision=config.base_commit,
    )
    runner = ScriptTrialRunner(config, repository=config.workspace)
    runner.validate(trial)
    runner.validate_baseline()


def _validate_git_remote(
    workspace: Path,
    remote: str,
    base_branch: str | None,
    base_commit: str,
) -> None:
    """Verify the configured Git remote and immutable base commit."""
    result = subprocess.run(
        ["git", "remote", "get-url", remote],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"Git remote {remote!r} does not exist")
    parsed = urlsplit(result.stdout.strip())
    if parsed.scheme in {"http", "https"} and (
        parsed.username is not None or parsed.password is not None
    ):
        raise ValueError(
            f"Git remote {remote!r} contains embedded credentials; use a credential "
            "helper or askpass configuration instead"
        )
    commit = subprocess.run(
        ["git", "cat-file", "-e", f"{base_commit}^{{commit}}"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0:
        raise ValueError(f"Invalid base commit {base_commit!r}: {commit.stderr.strip()}")

    if base_branch is None:
        return
    branch_format = subprocess.run(
        ["git", "check-ref-format", "--branch", base_branch],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if branch_format.returncode != 0:
        raise ValueError(f"Invalid base branch {base_branch!r}")
    branch_commit = None
    for ref in (
        base_branch,
        f"refs/heads/{base_branch}",
        f"refs/remotes/{remote}/{base_branch}",
    ):
        branch = subprocess.run(
            ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        if branch.returncode == 0:
            branch_commit = branch.stdout.strip()
            break
    if branch_commit is None:
        raise ValueError(
            f"Base branch {base_branch!r} does not exist locally or under remote {remote!r}"
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base_commit, branch_commit],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError(
            f"Base commit {base_commit!r} is not contained in base branch {base_branch!r}"
        )
