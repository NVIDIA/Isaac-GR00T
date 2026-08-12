# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""User-script train/evaluation runner for standalone AutoResearch."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess

from skills.autoresearch.config import TRIAL_EVAL_RESULT, StudyConfig
from skills.autoresearch.script_artifacts import (
    collect_script_evaluation_artifacts,
    collect_script_training_artifacts,
)
from skills.autoresearch.study_state import StageState, Trial


# ScriptTrialRunner executes user scripts and waits for generic artifacts.
class ScriptTrialRunner:
    """Run user-provided train/eval scripts and collect their generic artifacts."""

    def __init__(self, config: StudyConfig, *, repository: Path | None = None) -> None:
        """Store constructor dependencies without performing work."""
        self.config = config
        self.repository = repository.resolve() if repository is not None else None

    def validate(self, trial: Trial) -> None:
        """Validation checks script paths and per-trial output boundaries before launch."""
        _require_branch(trial)
        self._require_scripts()
        self._trial_root(trial)

    def validate_baseline(self) -> None:
        """Check the supplied baseline result and configured score."""
        self._require_scripts()
        evaluation = collect_script_evaluation_artifacts(
            result_file=self.config.eval_result,
            result_path=self.config.objective.result_path,
        )
        if evaluation.score != self.config.objective.baseline:
            raise RuntimeError(
                f"Configured baseline_result {self.config.objective.baseline} does not match "
                f"{self.config.objective.result_path}={evaluation.score} in "
                f"{evaluation.result_path}"
            )

    def run_train(self, trial: Trial) -> StageState:
        """Run the training script and return its checkpoint."""
        _require_branch(trial)
        completed = self._run_script(trial, previous=trial.train, stage="train")
        if completed.status != "succeeded":
            return completed
        try:
            result = self._collect_training(self._trial_root(trial))
        except RuntimeError as exc:
            return replace(completed, status="failed", error=str(exc))
        return replace(completed, checkpoint=str(result.checkpoint))

    def run_eval(self, trial: Trial) -> tuple[StageState, float | None]:
        """Run the evaluation script and return its score."""
        _require_branch(trial)
        if not trial.train.checkpoint:
            return (
                replace(
                    trial.evaluation,
                    status="failed",
                    error="training stage must include checkpoint",
                ),
                None,
            )
        completed = self._run_script(trial, previous=trial.evaluation, stage="eval")
        if completed.status != "succeeded":
            return completed, None
        try:
            result = collect_script_evaluation_artifacts(
                result_file=self._result_file(self._trial_root(trial)),
                result_path=self.config.objective.result_path,
            )
        except RuntimeError as exc:
            return replace(completed, status="failed", error=str(exc)), None
        return completed, result.score

    def _require_scripts(self) -> None:
        """Enforce one runner invariant with a focused error."""
        for label, script in (
            ("train", self.config.train_script),
            ("eval", self.config.eval_script),
        ):
            if not script.is_file():
                raise RuntimeError(f"{label} script does not exist: {script}")

    def _run_script(
        self,
        trial: Trial,
        *,
        previous: StageState,
        stage: str,
    ) -> StageState:
        """Run one user script and record its exit status."""
        if previous.status != "running":
            raise ValueError("script attempt must be running before execution")
        run_root = self._trial_root(trial)
        run_root.mkdir(parents=True, exist_ok=True)
        result_file = self._result_file(run_root)
        if stage == "eval":
            # A retry must prove that this attempt produced a new result.
            result_file.unlink(missing_ok=True)
        attempt = previous.attempts
        stdout_path = run_root / f"{stage}.attempt-{attempt:04d}.stdout.log"
        stderr_path = run_root / f"{stage}.attempt-{attempt:04d}.stderr.log"
        script = self.config.train_script if stage == "train" else self.config.eval_script
        command = _script_command(script)
        source_root = self._ensure_source_worktree(trial, run_root)
        env = self._stage_env(
            trial, previous, stage=stage, run_root=run_root, source_root=source_root
        )
        cwd = source_root or script.parent
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=self.config.script_timeout_s,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            return replace(
                previous,
                status="failed",
                error=f"{stage} script timed out after {exc.timeout} seconds",
            )
        if completed.returncode != 0:
            return replace(
                previous,
                status="failed",
                error=f"{stage} script failed with exit code {completed.returncode}: {stderr_path}",
            )
        return replace(previous, status="succeeded", error=None)

    def _stage_env(
        self,
        trial: Trial,
        stage_state: StageState,
        *,
        stage: str,
        run_root: Path,
        source_root: Path | None,
    ) -> dict[str, str]:
        """Build the environment passed to a user train or evaluation script."""
        # Resolve the paths that AutoResearch promises to every user script.
        result_file = self._result_file(run_root)
        checkpoint = trial.train.checkpoint
        if not checkpoint:
            checkpoint = str(
                self._checkpoint_override(run_root) or (run_root / self.config.output_dir_name)
            )
        # Start from the caller environment, then layer configured runner values.
        values = os.environ.copy()
        values.update(self.config.env or {})
        values.update(
            {
                "AUTORESEARCH_RUN_ROOT": str(run_root),
                "AUTORESEARCH_TRIAL_ID": trial.trial_id,
                "AUTORESEARCH_STAGE": stage,
                "AUTORESEARCH_ATTEMPT": str(stage_state.attempts),
                "RUN_ROOT": str(run_root),
                "GROOT_BRANCH": trial.code_branch,
                "GROOT_REVISION": trial.code_revision,
                "TRAINING_OUTPUT_DIR": str(run_root / self.config.output_dir_name),
                "OUTPUT_DIR": str(run_root / self.config.output_dir_name),
                "CHECKPOINT_PATH": checkpoint,
                "AUTORESEARCH_CHECKPOINT_PATH": checkpoint,
                "EVAL_OUTPUT_DIR": str(result_file.parent),
                "RESULT_PATH": str(result_file),
                "AUTORESEARCH_RESULT_PATH": str(result_file),
            }
        )
        # Expose the detached candidate source root when one was created.
        if source_root is not None:
            values.update(
                {
                    "AUTORESEARCH_REPO_ROOT": str(source_root),
                    "REPO_ROOT": str(source_root),
                }
            )
        # Stage-specific environment values have final precedence.
        values.update((self.config.train_env if stage == "train" else self.config.eval_env) or {})
        return values

    def _collect_training(self, run_root: Path):
        """Reuse script artifact collection with this runner config."""
        return collect_script_training_artifacts(
            run_root=run_root,
            checkpoint_path=self._checkpoint_override(run_root),
            output_dir_name=self.config.output_dir_name,
        )

    def _ensure_source_worktree(self, trial: Trial, run_root: Path) -> Path | None:
        """Local scripts get a detached source worktree at the exact candidate commit."""
        if self.repository is None:
            return None
        source_root = run_root / "source"
        if source_root.exists():
            existing = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=source_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if existing.returncode == 0 and existing.stdout.strip() == trial.code_revision:
                return source_root
            raise RuntimeError(
                f"Source worktree already exists at a different revision: {source_root}"
            )
        result = subprocess.run(
            ["git", "worktree", "add", "--detach", str(source_root), trial.code_revision],
            cwd=self.repository,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stdout + result.stderr).strip()
            raise RuntimeError(f"Could not create script source worktree: {detail}")
        return source_root

    def cleanup(self, trial: Trial) -> None:
        """Remove the reconstructible source worktree after a terminal trial."""
        if self.repository is None:
            return
        source_root = self._trial_root(trial) / "source"
        if not source_root.exists():
            return
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(source_root)],
            cwd=self.repository,
            capture_output=True,
            text=True,
            check=False,
        )
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=self.repository,
            capture_output=True,
            text=True,
            check=False,
        )

    def _trial_root(self, trial: Trial) -> Path:
        """Resolve trial identity, paths, or ordering helpers."""
        root = (self.config.output_dir / trial.trial_id).resolve()
        _require_child(root, self.config.output_dir.resolve(), "script run root")
        return root

    def _checkpoint_override(self, run_root: Path) -> Path | None:
        """Resolve an optional configured checkpoint path under the run root."""
        if self.config.checkpoint_path is None:
            return None
        return _resolve_run_path(run_root, self.config.checkpoint_path)

    def _result_file(self, run_root: Path) -> Path:
        """Resolve the configured eval result file under the run root."""
        result_file = _resolve_run_path(run_root, TRIAL_EVAL_RESULT)
        _require_child(result_file, run_root.resolve(), "evaluation result")
        return result_file


def _require_branch(trial: Trial) -> None:
    """Enforce one runner invariant with a focused error."""
    if not trial.code_branch:
        raise ValueError(f"trial {trial.trial_id} has no published code branch")
    if not trial.code_revision:
        raise ValueError(f"trial {trial.trial_id} has no immutable code revision")


def _require_child(path: Path, parent: Path, label: str) -> None:
    """Enforce one runner invariant with a focused error."""
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise RuntimeError(f"{label} {path} is outside {parent}") from exc
    if path == parent:
        raise RuntimeError(f"{label} must be below, not equal to, {parent}")


def _script_command(script: Path) -> list[str]:
    """Build one script command as an argv list."""
    if script.suffix == ".sh":
        return ["bash", str(script)]
    return [str(script)]


def _resolve_run_path(run_root: Path, value: str) -> Path:
    """Resolve a configured run path into canonical form."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = run_root / path
    return path.resolve()
