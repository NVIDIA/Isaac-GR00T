# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Per-trial AutoResearch lifecycle controller."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Literal

import yaml

from skills.autoresearch.agent_editor import AgenticEditor
from skills.autoresearch.config import StudyConfig, expand_environment_variables
from skills.autoresearch.git_workspace import (
    CandidateWorkspace,
    CompositionConflictError,
    GitWorkspaceManager,
)
from skills.autoresearch.study_state import StageState, StudyState, Trial
from skills.autoresearch.train_config_lock import MISSING
from skills.autoresearch.trial_prompt import build_prompt

StageName = Literal["train", "eval"]


# Each controller step returns one visible action for logs and debuggers.
@dataclass(frozen=True)
class StepResult:
    action: str
    message: str
    done: bool = False


# TrialController owns one trial's lifecycle; WaveController only schedules trials.
class TrialController:
    """Advance individual trials through code, train, eval, and scoring."""

    def __init__(
        self,
        *,
        config: StudyConfig,
        workspace_manager: GitWorkspaceManager,
        editor: AgenticEditor,
        runner: Any,
        save_state: Callable[[StudyState], None],
        progress: Callable[[str], None],
    ) -> None:
        """Store trial dependencies and the wave-owned state callback."""
        self.config = config
        self.workspace_manager = workspace_manager
        self.editor = editor
        self.runner = runner
        self._save_state = save_state
        self._progress = progress

    def advance(
        self,
        state: StudyState,
        trials: list[Trial],
    ) -> StepResult | None:
        """Advance unfinished trials, or return None when the wave can settle."""
        work: list[tuple[Trial, StageName]] = []
        for trial in sorted(trials, key=lambda item: item.index):
            if self._is_terminal(trial):
                self.runner.cleanup(trial)
                continue
            if not trial.code_revision:
                return self._edit_trial(state, trial)
            if not trial.code_published:
                return self._publish_trial(state, trial)
            if trial.train.status != "succeeded":
                work.append((trial, "train"))
                continue
            if not trial.train.checkpoint:
                raise RuntimeError(f"Trial {trial.trial_id} succeeded without a checkpoint")
            if trial.evaluation.status != "succeeded":
                work.append((trial, "eval"))
                continue
            raise RuntimeError(f"Trial {trial.trial_id} succeeded without a score")

        return self._run_stages(state, work) if work else None

    def _is_terminal(self, trial: Trial) -> bool:
        """Return whether a trial needs no more work in its current wave."""
        return (
            trial.score is not None
            or trial.train.status == "skipped"
            or trial.evaluation.status == "skipped"
            or (trial.train.status == "failed" and trial.train.attempts > self.config.max_retry)
            or (
                trial.evaluation.status == "failed"
                and trial.evaluation.attempts > self.config.max_retry
            )
        )

    def _edit_trial(self, state: StudyState, trial: Trial) -> StepResult:
        """Candidate editing owns worktree lifecycle from prompt through validated commit."""
        # Set up the editor artifact location and exact parent revision first.
        artifact_dir = self.config.output_dir / trial.trial_id / "codex"
        base_revision = trial.parent_revision
        self._report(
            trial,
            "prepare_candidate",
            f"creating an isolated worktree from {base_revision}",
        )

        # Remove only controller-owned incomplete work from a prior interrupted edit.
        self.workspace_manager.reset_incomplete(trial_id=trial.trial_id)
        workspace = self.workspace_manager.create(
            trial_id=trial.trial_id,
            base_commit=base_revision,
        )
        initial_head = self.workspace_manager.head(workspace)
        candidate = trial
        changed: list[str] = []
        keep_branch = False
        try:
            if trial.composition_source_trial_id:
                # Composition trials replay a previously useful isolated source change.
                self._report(
                    trial,
                    "compose_candidate",
                    f"applying queued change from {trial.composition_source_trial_id}",
                )
                source = state.get_trial(trial.composition_source_trial_id)
                candidate = replace(
                    trial,
                    hypothesis=(
                        f"Compose queued change from {source.trial_id}: {source.hypothesis}"
                    ),
                )
                try:
                    self.workspace_manager.apply_commit(
                        workspace,
                        commit=source.code_revision,
                    )
                except CompositionConflictError as conflict:
                    reason = "Composition conflict: " + ", ".join(conflict.conflict_files)
                    return self._reject_candidate(state, candidate, reason)
            else:
                prompt = build_prompt(
                    config=self.config,
                    state=state,
                    trial_id=trial.trial_id,
                    trial_index=trial.index,
                    parent_revision=base_revision,
                    parent_score=trial.parent_score,
                )
                for retry in range(self.config.max_retry + 1):
                    attempt = retry + 1
                    if retry:
                        self._report(
                            trial,
                            "reset_candidate",
                            f"restoring a clean worktree before Codex attempt {attempt}",
                        )
                        self.workspace_manager.reset_to_commit(
                            workspace,
                            commit=initial_head,
                        )
                    self._report(
                        trial,
                        "edit_candidate",
                        f"running Codex attempt {attempt}/{self.config.max_retry + 1}",
                    )
                    try:
                        result = self.editor.edit(
                            worktree=workspace.path,
                            prompt=prompt,
                            artifact_dir=artifact_dir,
                        )
                        break
                    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                        if retry >= self.config.max_retry:
                            self._report(
                                trial,
                                "edit_failed",
                                _error_detail(exc),
                            )
                            raise
                        delay = _retry_delay_s(retry)
                        self._report(
                            trial,
                            "edit_retry",
                            f"{_error_detail(exc)}; retrying in {delay}s",
                        )
                        time.sleep(delay)
                candidate = replace(trial, hypothesis=result.hypothesis)
            if self.workspace_manager.head(workspace) != initial_head:
                raise RuntimeError("Candidate preparation changed git HEAD")
            # Validate the diff before committing anything to a candidate branch.
            self._report(candidate, "validate_candidate", "checking the candidate diff")
            changed = self.workspace_manager.changed_files(workspace)
            candidate = replace(
                candidate,
                changed_files=tuple(changed),
            )
            violations = self.workspace_manager.validate_changed_files(
                changed,
                editable_paths=self.config.editable_paths,
                frozen_paths=self.config.frozen_paths,
            )
            violations = [*violations, *self._train_config_violations(workspace)]
            self._save_diff(artifact_dir, self.workspace_manager.diff(workspace))
            # Empty diffs and frozen-path edits are terminal skips, not runner failures.
            if not changed or violations:
                if not changed and trial.composition_source_trial_id:
                    reason = "Queued change is already present or produces an empty composition"
                else:
                    reason = "Codex made no changes" if not changed else "; ".join(violations)
                return self._reject_candidate(state, candidate, reason)

            # Commit only the validated file list and persist the immutable revision.
            commit = self.workspace_manager.commit(
                workspace,
                message=f"AutoResearch {self.config.study_id} {trial.trial_id}",
                changed_files=changed,
            )
            candidate = replace(
                candidate,
                code_branch=workspace.branch,
                code_revision=commit,
                code_published=False,
            )
            self._save_diff(artifact_dir, self.workspace_manager.commit_diff(commit))
            self._save_trial(state, candidate)
            keep_branch = True
            self._report(
                candidate,
                "candidate_committed",
                f"revision={commit} branch={workspace.branch}",
            )
            return StepResult("candidate_committed", f"Committed {workspace.branch} at {commit}")
        except Exception as exc:  # noqa: BLE001
            # Preserve a diagnostic diff when the editor failed after touching files.
            if not changed:
                try:
                    changed = self.workspace_manager.changed_files(workspace)
                    candidate = replace(candidate, changed_files=tuple(changed))
                    if changed:
                        self._save_diff(artifact_dir, self.workspace_manager.diff(workspace))
                except Exception:  # noqa: BLE001, S110
                    # Preserve the primary editor or validation failure even if diagnostics fail.
                    pass
            failed = replace(
                candidate,
                train=StageState(status="skipped", error=str(exc)),
                evaluation=StageState(status="skipped", error=str(exc)),
            )
            self._save_trial(state, failed)
            self._report(failed, "candidate_failed", _error_detail(exc))
            return StepResult("candidate_failed", str(exc))
        finally:
            # Published candidate branches keep their ref; failed work is discarded.
            if keep_branch:
                self.workspace_manager.remove(workspace)
            else:
                self.workspace_manager.discard(workspace)

    def _publish_trial(self, state: StudyState, trial: Trial) -> StepResult:
        """Make the immutable candidate revision visible to user scripts."""
        for retry in range(self.config.max_retry + 1):
            attempt = retry + 1
            self._report(
                trial,
                "publish_candidate",
                f"pushing {trial.code_branch} attempt {attempt}/{self.config.max_retry + 1}",
            )
            try:
                self.workspace_manager.publish_branch(
                    branch=trial.code_branch,
                    commit=trial.code_revision,
                    remote=self.config.git_remote,
                )
                break
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                if retry >= self.config.max_retry:
                    reason = (
                        f"Candidate publish failed after {retry + 1} attempts: {_error_detail(exc)}"
                    )
                    failed = replace(
                        trial,
                        train=StageState(status="skipped", error=reason),
                        evaluation=StageState(status="skipped", error=reason),
                    )
                    self._save_trial(state, failed)
                    self._report(failed, "publish_failed", reason)
                    return StepResult("candidate_publish_failed", reason)
                delay = _retry_delay_s(retry)
                self._report(
                    trial,
                    "publish_retry",
                    f"{_error_detail(exc)}; retrying in {delay}s",
                )
                time.sleep(delay)
        published = replace(trial, code_published=True)
        self._save_trial(state, published)
        self._report(
            published,
            "candidate_published",
            f"revision={trial.code_revision} branch={trial.code_branch}",
        )
        return StepResult(
            "candidate_published",
            f"Published {trial.code_branch} at {trial.code_revision}",
        )

    def _run_stages(
        self,
        state: StudyState,
        work: list[tuple[Trial, StageName]],
    ) -> StepResult:
        """Run each trial's requested train or eval stage concurrently."""
        if not work:
            raise RuntimeError("No trial stages are ready")

        # Mark every selected stage as running before invoking user scripts.
        prepared_state = state
        runnable: list[tuple[Trial, StageName]] = []
        for source_trial, stage in work:
            current = _stage_state(source_trial, stage)
            # A new attempt increments the counter; a recovered running attempt
            # keeps its counter so restarting does not consume an extra retry.
            if current.status in {"pending", "failed"}:
                prepared = replace(
                    current,
                    status="running",
                    attempts=current.attempts + 1,
                    error=None,
                )
            elif current.status == "running":
                prepared = replace(current, error=None)
            else:
                raise RuntimeError(f"Unexpected {stage} status: {current.status}")
            prepared_trial = _replace_stage(source_trial, stage, prepared)
            prepared_state = prepared_state.with_updated_trial(prepared_trial)
            runnable.append((prepared_trial, stage))

        # A restart can now rerun the same trial/stage/attempt safely.
        self._save_state(prepared_state)
        for trial, stage in runnable:
            stage_state = _stage_state(trial, stage)
            trial_dir = self.config.output_dir / trial.trial_id
            stdout_path = trial_dir / f"{stage}.attempt-{stage_state.attempts:04d}.stdout.log"
            stderr_path = trial_dir / f"{stage}.attempt-{stage_state.attempts:04d}.stderr.log"
            self._report(
                trial,
                f"{stage}_started",
                f"running user {stage} script; stdout={stdout_path}; stderr={stderr_path}",
            )

        completed_state = prepared_state
        with ThreadPoolExecutor(max_workers=len(runnable)) as executor:
            # Submit train and eval through the same executor so a wave can run
            # independent trial stages in parallel.
            futures = {}
            for trial, stage in runnable:
                run = self.runner.run_train if stage == "train" else self.runner.run_eval
                futures[executor.submit(run, trial)] = (trial, stage)

            # Apply results as scripts finish instead of waiting in submit order.
            for future in as_completed(futures):
                trial, stage = futures[future]
                try:
                    result = future.result()
                    # Evaluation additionally produces the objective score
                    # used for parent comparison.
                    if stage == "train":
                        if not isinstance(result, StageState):
                            raise TypeError("run_train must return StageState")
                        completed = replace(trial, train=result)
                    else:
                        evaluation, score = result
                        if not isinstance(evaluation, StageState):
                            raise TypeError("run_eval must return StageState and score")
                        completed = replace(trial, evaluation=evaluation)
                        if evaluation.status == "succeeded":
                            if score is None:
                                raise RuntimeError("successful evaluation returned no score")
                            completed = self._score_trial(state, completed, score)
                except Exception as exc:  # noqa: BLE001
                    # Isolate a script or result-parsing failure to its stage so
                    # the remaining parallel trials can still finish.
                    failed = replace(
                        _stage_state(trial, stage),
                        status="failed",
                        error=f"{stage} execution failed: {exc}",
                    )
                    completed = _replace_stage(trial, stage, failed)
                completed_state = completed_state.with_updated_trial(completed)
                # Persist each completed stage immediately. A slow sibling trial
                # must not keep already-finished results only in process memory.
                self._save_state(completed_state)
                stage_state = _stage_state(completed, stage)
                self._report(
                    completed,
                    f"{stage}_{stage_state.status}",
                    _stage_result_detail(completed, stage),
                )

        # Build one concise progress message for the train/eval stages that ran.
        summaries = []
        for stage in ("train", "eval"):
            finished = [
                _stage_state(completed_state.get_trial(trial.trial_id), stage)
                for trial, item_stage in work
                if item_stage == stage
            ]
            if finished:
                succeeded = sum(item.status == "succeeded" for item in finished)
                failed = sum(item.status == "failed" for item in finished)
                summaries.append(f"{stage}: {succeeded} succeeded, {failed} failed")
        return StepResult(
            "trials_executed",
            f"Executed {len(work)} trial stages (" + "; ".join(summaries) + ")",
        )

    def _score_trial(self, state: StudyState, trial: Trial, score: float) -> Trial:
        """Record whether a score improves on the trial's frozen parent."""
        parent_score = (
            trial.parent_score if trial.parent_score is not None else state.baseline_score
        )
        return replace(
            trial,
            score=score,
            improved_over_parent=self.config.objective.is_better(score, parent_score),
        )

    def check_baseline_train_config(self) -> None:
        """Fail before any agent run when the locked baseline config is unusable."""
        lock = self.config.train_config_lock
        if lock is None:
            return
        baseline = self._baseline_train_config()
        violations = lock.violations(
            baseline=baseline,
            candidate=baseline,
            extra_eval_dataset_paths=self._extra_eval_dataset_paths(),
        )
        if violations:
            raise ValueError("Baseline training config is invalid: " + "; ".join(violations))

    def _train_config_violations(self, workspace: CandidateWorkspace) -> list[str]:
        """Compare the candidate training config with the pinned baseline, not the parent."""
        lock = self.config.train_config_lock
        if lock is None:
            return []
        candidate_path = workspace.path / lock.path
        candidate = (
            self._render_train_config(candidate_path.read_text(encoding="utf-8"), candidate_path)
            if candidate_path.is_file()
            else MISSING
        )
        return lock.violations(
            baseline=self._baseline_train_config(),
            candidate=candidate,
            extra_eval_dataset_paths=self._extra_eval_dataset_paths(),
        )

    def _baseline_train_config(self) -> Any:
        """Render the locked training config stored at the pinned baseline commit."""
        lock = self.config.train_config_lock
        text = self.workspace_manager.read_file(commit=self.config.base_commit, path=lock.path)
        return self._render_train_config(text, Path(f"{self.config.base_commit}:{lock.path}"))

    def _render_train_config(self, text: str, source: Path) -> Any:
        """Expand ${NAME} with the same layered environment the train script receives."""
        environ = {**os.environ, **self.config.env, **self.config.train_env}
        return expand_environment_variables(yaml.safe_load(text), source, environ)

    def _extra_eval_dataset_paths(self) -> tuple[str, ...]:
        """Treat the study-level eval dataset as held out alongside the lock's list."""
        return (self.config.eval_dataset,) if self.config.eval_dataset else ()

    def _reject_candidate(
        self,
        state: StudyState,
        trial: Trial,
        reason: str,
    ) -> StepResult:
        """Mark a candidate as skipped after editing or composition fails validation."""
        skipped = replace(
            trial,
            train=StageState(status="skipped", error=reason),
            evaluation=StageState(status="skipped", error=reason),
        )
        self._save_trial(state, skipped)
        self._report(skipped, "candidate_rejected", reason)
        return StepResult("candidate_rejected", reason)

    @staticmethod
    def _save_diff(artifact_dir: Path, content: str) -> None:
        """Save the candidate diff with private permissions."""
        artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = artifact_dir / "codex.diff"
        path.write_text(content, encoding="utf-8")
        path.chmod(0o600)

    def _save_trial(self, state: StudyState, trial: Trial) -> None:
        """Persist a trial update."""
        self._save_state(state.with_updated_trial(trial))

    def _report(self, trial: Trial, action: str, detail: str) -> None:
        """Report one self-contained trial progress line to the CLI."""
        total_attempts = self.config.max_retry + 1
        self._progress(
            f"[Wave {trial.wave_index}]"
            f"[Trial {trial.index}/{self.config.max_trials} {trial.trial_id}] "
            f"action={action} | train={_stage_progress(trial.train, total_attempts)} | "
            f"eval={_stage_progress(trial.evaluation, total_attempts)} | {detail}"
        )


def _retry_delay_s(retry: int) -> int:
    """Return bounded exponential backoff after a failed external command."""
    return min(2**retry, 30)


def _stage_progress(stage: StageState, total_attempts: int) -> str:
    """Format one stage so status and retry budget are visible together."""
    return f"{stage.status}(attempt={stage.attempts}/{total_attempts})"


def _stage_result_detail(trial: Trial, stage: StageName) -> str:
    """Summarize the useful output of one completed train or eval stage."""
    stage_state = _stage_state(trial, stage)
    if stage_state.status != "succeeded":
        return _error_detail(stage_state.error or f"{stage} did not succeed")
    if stage == "train":
        checkpoint = stage_state.checkpoint or "missing"
        return f"training succeeded; checkpoint={checkpoint}"
    return f"evaluation succeeded; score={trial.score}"


def _error_detail(error: object, max_length: int = 240) -> str:
    """Keep failures readable as one bounded progress line."""
    detail = " ".join(str(error).split())
    if len(detail) <= max_length:
        return detail
    return detail[: max_length - 3] + "..."


def _stage_state(trial: Trial, stage: StageName) -> StageState:
    """Return one stage from a trial."""
    return trial.train if stage == "train" else trial.evaluation


def _replace_stage(trial: Trial, stage: StageName, value: StageState) -> Trial:
    """Return a trial with one stage replaced."""
    if stage == "train":
        return replace(trial, train=value)
    return replace(trial, evaluation=value)
