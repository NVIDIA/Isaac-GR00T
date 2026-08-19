# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# ruff: noqa: ARG002, SLF001

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import threading

import pytest
from skills.autoresearch.codex_editor import EditResult
from skills.autoresearch.config import StudyConfig
from skills.autoresearch.git_workspace import (
    CandidateWorkspace,
    CompositionConflictError,
)
from skills.autoresearch.study_state import (
    Objective,
    StageState,
    StudyState,
    Trial,
    make_trial_id,
)
from skills.autoresearch.wave_controller import WaveController


def initialize_state(path: Path, state: StudyState) -> None:
    """Write test state using the controller's plain JSON format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict()), encoding="utf-8")


def load_state(path: Path) -> StudyState:
    """Read test state using the controller's plain JSON format."""
    return StudyState.from_dict(json.loads(path.read_text(encoding="utf-8")))


class FakeWorkspaceManager:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.pushed = []
        self.created = []
        self.applied = []
        self.reset_commits = []

    def reset_incomplete(self, **kwargs):
        return None

    def create(self, *, trial_id, base_commit):
        path = self.tmp_path / trial_id
        path.mkdir(parents=True, exist_ok=True)
        self.created.append((trial_id, base_commit))
        return CandidateWorkspace(path, f"codex/{trial_id}", base_commit)

    def head(self, workspace):
        return workspace.base_commit

    def reset_to_commit(self, workspace, *, commit):
        self.reset_commits.append((workspace.path.name, commit))

    def changed_files(self, workspace):
        return ["groot/core/training/models/n2/model.py"]

    def validate_changed_files(self, changed, **kwargs):
        return []

    def diff(self, workspace):
        return "diff --git a/model.py b/model.py\n"

    def commit(self, workspace, *, message, changed_files):
        return f"{workspace.path.name}-sha"

    def commit_diff(self, commit):
        return "diff --git a/model.py b/model.py\n"

    def apply_commit(self, workspace, *, commit):
        self.applied.append((workspace.path.name, commit))

    def publish_branch(self, *, branch, commit, remote):
        self.pushed.append((branch, commit, remote))

    def remove(self, workspace):
        return None

    def discard(self, workspace):
        return None


class ConflictWorkspaceManager(FakeWorkspaceManager):
    def __init__(
        self,
        tmp_path: Path,
        *,
        conflict_files=("groot/core/training/models/n2/model.py",),
    ) -> None:
        super().__init__(tmp_path)
        self.conflict_files = tuple(conflict_files)

    def apply_commit(self, workspace, *, commit):
        self.applied.append((workspace.path.name, commit))
        raise CompositionConflictError(
            commit=commit,
            conflict_files=self.conflict_files,
        )


class BrokenCompositionWorkspaceManager(FakeWorkspaceManager):
    def apply_commit(self, workspace, *, commit):
        self.applied.append((workspace.path.name, commit))
        raise RuntimeError("bad source revision")


class FakeEditor:
    def __init__(self):
        self.calls = []

    def edit(self, *, worktree, prompt, artifact_dir):
        self.calls.append((worktree, prompt))
        artifact_dir.mkdir(parents=True, exist_ok=True)
        result = artifact_dir / "result.json"
        result.write_text("{}")
        return EditResult(
            hypothesis="Reduce the learning rate.",
            validation=("pytest focused.py",),
            jsonl_path=artifact_dir / "codex.jsonl",
            stderr_path=artifact_dir / "codex.stderr.log",
            result_path=result,
        )


class FakeRunnerConfig:
    git_remote = "origin"


class FakeRunner:
    config = FakeRunnerConfig()

    def __init__(self, scores=None):
        self.scores = scores or {}
        self.train_runs = []
        self.eval_runs = []
        self.cleaned = []

    def cleanup(self, trial):
        self.cleaned.append(trial.trial_id)

    def run_train(self, trial):
        self.train_runs.append(trial.trial_id)
        return replace(
            trial.train,
            status="succeeded",
            checkpoint=f"/shared/{trial.trial_id}/checkpoint-10",
        )

    def run_eval(self, trial):
        self.eval_runs.append(trial.trial_id)
        return (
            replace(trial.evaluation, status="succeeded"),
            self.scores.get(trial.trial_id, 0.025),
        )


def _config(tmp_path: Path) -> StudyConfig:
    return StudyConfig(
        study_id="yam-smoke",
        workspace=tmp_path,
        base_commit="5587e83bece8d8588991a23ace81c095784e8d48",
        train_dataset="/dataset",
        eval_dataset="/dataset",
        objective=Objective(
            result_path="results.local.average.mse",
            direction="minimize",
            baseline=0.0268,
        ),
        max_trials=1,
        editable_paths=("groot/core/training/models/",),
        frozen_paths=("groot/infra/",),
        idea_types=("recipe_tuning",),
        agent={},
        eval_result=tmp_path / "baseline/eval/results.json",
        output_dir=tmp_path,
        train_script=tmp_path / "train.sh",
        eval_script=tmp_path / "eval.sh",
    )


def _study_state(config: StudyConfig, **values) -> StudyState:
    """Build a state that satisfies the current persisted study contract."""
    return StudyState(
        study_id=config.study_id,
        baseline_score=config.objective.baseline,
        base_commit=config.base_commit,
        eval_metric=config.objective.result_path,
        direction=config.objective.direction,
        min_delta=config.objective.min_delta,
        **values,
    )


def _trial_id(index: int, wave_index: int = 1) -> str:
    """Return a deterministic trial ID for controller tests."""
    return make_trial_id("yam-smoke", wave_index, index)


def _save_score(
    controller: WaveController,
    state: StudyState,
    trial: Trial,
    score: float,
) -> None:
    scored = controller.trial_controller._score_trial(state, trial, score)
    controller.trial_controller._save_trial(state, scored)


def _queued_composition_state(config: StudyConfig) -> StudyState:
    winner = Trial(
        trial_id="yam-smoke_wave1_trial1",
        index=1,
        wave_index=1,
        parent_trial_id=None,
        parent_revision=config.base_commit,
        idea_type="recipe_tuning",
        hypothesis="Change A",
        parent_score=config.objective.baseline,
        code_revision="winner-sha",
        score=0.02,
        improved_over_parent=True,
        is_new_best=True,
    )
    source = replace(
        winner,
        trial_id="yam-smoke_wave1_trial2",
        index=2,
        hypothesis="Change B",
        code_revision="source-sha",
        score=0.021,
        is_new_best=False,
    )
    return _study_state(
        config,
        best_score=winner.score,
        best_trial_id=winner.trial_id,
        trials=(winner, source),
        pending_change_trial_ids=(source.trial_id,),
    )


def test_controller_runs_one_resumable_trial_to_new_best(tmp_path: Path) -> None:
    output_dir = tmp_path / "state"
    config = replace(_config(tmp_path), output_dir=output_dir)
    state_path = output_dir / "state.json"
    workspace_manager = FakeWorkspaceManager(tmp_path / "worktrees")
    runner = FakeRunner()
    editor = FakeEditor()
    progress: list[str] = []
    controller = WaveController(
        config=config,
        workspace_manager=workspace_manager,
        editor=editor,
        runner=runner,
        progress=progress.append,
    )
    controller.initialize()
    assert load_state(state_path).base_commit == config.base_commit

    actions = [controller.run_once().action for _ in range(6)]

    assert actions == [
        "wave_created",
        "candidate_committed",
        "candidate_published",
        "trials_executed",
        "trials_executed",
        "wave_settled",
    ]
    state = load_state(state_path)
    assert state.best_score == 0.025
    assert state.best_trial_id == "yam-smoke_wave1_trial1"
    assert state.trials[0].improved_over_parent is True
    assert state.trials[0].is_new_best is True
    assert runner.cleaned == ["yam-smoke_wave1_trial1"]
    assert (output_dir / state.trials[0].trial_id / "codex" / "result.json").is_file()
    assert not (output_dir / "insights").exists()
    assert "Prior Insights:" in editor.calls[0][1]
    assert "No relevant published Insights were available" in editor.calls[0][1]
    assert not (output_dir / "trials").exists()
    output = "\n".join(progress)
    trial_prefix = "[Wave 1][Trial 1/1 yam-smoke_wave1_trial1]"
    assert "[Wave 1] started | trials=1-1/1" in output
    assert f"{trial_prefix} action=edit_candidate" in output
    assert "train=pending(attempt=0/1) | eval=pending(attempt=0/1)" in output
    assert f"{trial_prefix} action=train_started" in output
    assert "train=running(attempt=1/1)" in output
    assert "train.attempt-0001.stdout.log" in output
    assert f"{trial_prefix} action=train_succeeded" in output
    assert "checkpoint=/shared/yam-smoke_wave1_trial1/checkpoint-10" in output
    assert f"{trial_prefix} action=eval_started" in output
    assert f"{trial_prefix} action=eval_succeeded" in output
    assert "evaluation succeeded; score=0.025" in output
    assert "[Wave 1] settled | winner=yam-smoke_wave1_trial1" in output


def test_controller_reuses_prior_insights_interface_per_wave(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "state"
    config = replace(
        _config(tmp_path),
        output_dir=output_dir,
        max_trials=2,
        max_parallel_trials=2,
    )
    waves: list[int] = []

    def fake_provider(*, config, state, wave_index):
        del config, state
        waves.append(wave_index)
        return "- Reserved prior knowledge."

    monkeypatch.setattr(
        "skills.autoresearch.trial_controller.prior_insights_for_wave",
        fake_provider,
    )
    editor = FakeEditor()
    controller = WaveController(
        config=config,
        workspace_manager=FakeWorkspaceManager(tmp_path / "worktrees"),
        editor=editor,
        runner=FakeRunner(),
    )
    controller.initialize()

    assert [controller.run_once().action for _ in range(4)] == [
        "wave_created",
        "candidate_committed",
        "candidate_published",
        "candidate_committed",
    ]
    assert waves == [1]
    assert len(editor.calls) == 2
    assert all("- Reserved prior knowledge." in prompt for _, prompt in editor.calls)
    assert not (output_dir / "insights").exists()


def test_controller_requires_explicit_initialization(tmp_path: Path) -> None:
    config = _config(tmp_path)
    controller = WaveController(
        config=config,
        workspace_manager=FakeWorkspaceManager(tmp_path / "worktrees"),
        editor=FakeEditor(),
        runner=FakeRunner(),
    )

    try:
        controller.run_once()
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("run_once must not silently initialize missing state")


def test_controller_resumes_running_stage_with_same_attempt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    trial = Trial(
        trial_id="yam-smoke_wave1_trial1",
        index=1,
        wave_index=1,
        parent_trial_id=None,
        parent_revision=config.base_commit,
        idea_type="recipe_tuning",
        hypothesis="Tune learning rate.",
        code_branch="codex/trial",
        code_revision="candidate-sha",
        code_published=True,
        parent_score=config.objective.baseline,
        train=StageState(status="running", attempts=1),
    )
    state = _study_state(
        config,
        best_score=config.objective.baseline,
        trials=(trial,),
        active_wave_index=1,
    )
    state_path = tmp_path / "state.json"
    initialize_state(state_path, state)
    runner = FakeRunner()
    controller = WaveController(
        config=config,
        workspace_manager=FakeWorkspaceManager(tmp_path / "worktrees"),
        editor=FakeEditor(),
        runner=runner,
    )

    assert controller.run_once().action == "trials_executed"
    resumed = load_state(state_path).trials[0].train
    assert resumed.status == "succeeded"
    assert resumed.attempts == 1


def test_controller_preserves_candidate_context_when_commit_fails(tmp_path: Path) -> None:
    class FailingWorkspaceManager(FakeWorkspaceManager):
        def commit(self, workspace, *, message, changed_files):
            raise RuntimeError("commit failed")

    config = _config(tmp_path)
    state_path = tmp_path / "state.json"
    controller = WaveController(
        config=config,
        workspace_manager=FailingWorkspaceManager(tmp_path / "worktrees"),
        editor=FakeEditor(),
        runner=FakeRunner(),
    )
    controller.initialize()
    controller.run_once()

    result = controller.run_once()
    trial = load_state(state_path).trials[0]

    assert result.action == "candidate_failed"
    assert trial.hypothesis == "Reduce the learning rate."
    assert trial.changed_files == ("groot/core/training/models/n2/model.py",)
    assert trial.train.error == "commit failed"


def test_codex_edit_retries_with_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FlakyEditor(FakeEditor):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def edit(self, **kwargs):
            self.attempts += 1
            if self.attempts < 3:
                raise RuntimeError("temporary Codex failure")
            return super().edit(**kwargs)

    delays = []
    monkeypatch.setattr("skills.autoresearch.trial_controller.time.sleep", delays.append)
    config = replace(_config(tmp_path), max_retry=2)
    editor = FlakyEditor()
    workspace_manager = FakeWorkspaceManager(tmp_path / "worktrees")
    controller = WaveController(
        config=config,
        workspace_manager=workspace_manager,
        editor=editor,
        runner=FakeRunner(),
    )
    controller.initialize()
    controller.run_once()

    assert controller.run_once().action == "candidate_committed"
    assert editor.attempts == 3
    assert workspace_manager.reset_commits == [
        ("yam-smoke_wave1_trial1", config.base_commit),
        ("yam-smoke_wave1_trial1", config.base_commit),
    ]
    assert delays == [1, 2]


def test_git_publish_retries_with_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FlakyWorkspaceManager(FakeWorkspaceManager):
        def __init__(self, path):
            super().__init__(path)
            self.publish_attempts = 0

        def publish_branch(self, **kwargs):
            self.publish_attempts += 1
            if self.publish_attempts < 3:
                raise RuntimeError("temporary Git failure")
            return super().publish_branch(**kwargs)

    delays = []
    monkeypatch.setattr("skills.autoresearch.trial_controller.time.sleep", delays.append)
    config = replace(_config(tmp_path), max_retry=2)
    workspace_manager = FlakyWorkspaceManager(tmp_path / "worktrees")
    controller = WaveController(
        config=config,
        workspace_manager=workspace_manager,
        editor=FakeEditor(),
        runner=FakeRunner(),
    )
    controller.initialize()
    controller.run_once()
    controller.run_once()

    assert controller.run_once().action == "candidate_published"
    assert workspace_manager.publish_attempts == 3
    assert delays == [1, 2]


def test_exhausted_publish_failure_skips_trial_and_continues_wave(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AlwaysFailingWorkspaceManager(FakeWorkspaceManager):
        def __init__(self, path):
            super().__init__(path)
            self.publish_attempts = 0

        def publish_branch(self, **kwargs):
            self.publish_attempts += 1
            raise RuntimeError("Git remote unavailable")

    delays = []
    monkeypatch.setattr("skills.autoresearch.trial_controller.time.sleep", delays.append)
    config = replace(
        _config(tmp_path),
        max_trials=2,
        max_parallel_trials=2,
        max_retry=1,
        output_dir=tmp_path / "state",
    )
    workspace_manager = AlwaysFailingWorkspaceManager(tmp_path / "worktrees")
    controller = WaveController(
        config=config,
        workspace_manager=workspace_manager,
        editor=FakeEditor(),
        runner=FakeRunner(),
    )
    controller.initialize()
    assert controller.run_once().action == "wave_created"
    assert controller.run_once().action == "candidate_committed"

    result = controller.run_once()

    assert result.action == "candidate_publish_failed"
    assert workspace_manager.publish_attempts == 2
    assert delays == [1]
    failed = controller.load_state().trials[0]
    assert failed.train.status == "skipped"
    assert failed.evaluation.status == "skipped"
    assert "Candidate publish failed after 2 attempts" in (failed.train.error or "")
    assert controller.run_once().action == "candidate_committed"
    assert controller.load_state().trials[1].code_revision


def test_parallel_wave_runs_all_training_scripts_together(tmp_path: Path) -> None:
    class ConcurrentRunner(FakeRunner):
        def __init__(self):
            super().__init__()
            self.barrier = threading.Barrier(3)

        def run_train(self, trial):
            self.barrier.wait(timeout=5)
            return super().run_train(trial)

    config = replace(
        _config(tmp_path),
        max_trials=3,
        max_parallel_trials=3,
        output_dir=tmp_path / "state",
    )
    state_path = config.output_dir / "state.json"
    runner = ConcurrentRunner()
    controller = WaveController(
        config=config,
        workspace_manager=FakeWorkspaceManager(tmp_path / "worktrees"),
        editor=FakeEditor(),
        runner=runner,
    )
    controller.initialize()

    actions = [controller.run_once().action for _ in range(8)]
    state = load_state(state_path)

    assert actions[:7] == [
        "wave_created",
        "candidate_committed",
        "candidate_published",
        "candidate_committed",
        "candidate_published",
        "candidate_committed",
        "candidate_published",
    ]
    assert actions[7] == "trials_executed"
    assert set(runner.train_runs) == {trial.trial_id for trial in state.trials}
    assert {trial.train.status for trial in state.trials} == {"succeeded"}
    assert {trial.wave_index for trial in state.trials} == {1}
    assert {trial.parent_trial_id for trial in state.trials} == {None}
    assert {trial.parent_revision for trial in state.trials} == {config.base_commit}


def test_wave_runs_train_and_eval_selected_by_each_trial(tmp_path: Path) -> None:
    class MixedRunner(FakeRunner):
        def __init__(self):
            super().__init__()
            self.barrier = threading.Barrier(2)

        def run_train(self, trial):
            self.barrier.wait(timeout=5)
            return super().run_train(trial)

        def run_eval(self, trial):
            self.barrier.wait(timeout=5)
            return super().run_eval(trial)

    config = replace(_config(tmp_path), max_trials=2, max_parallel_trials=2)
    common = {
        "wave_index": 1,
        "parent_trial_id": None,
        "parent_revision": config.base_commit,
        "idea_type": "recipe_tuning",
        "code_published": True,
        "parent_score": config.objective.baseline,
    }
    train_trial = Trial(
        trial_id="yam-smoke_wave1_trial1",
        index=1,
        hypothesis="Train next.",
        code_branch="codex/train",
        code_revision="train-sha",
        **common,
    )
    eval_trial = Trial(
        trial_id="yam-smoke_wave1_trial2",
        index=2,
        hypothesis="Evaluate next.",
        code_branch="codex/eval",
        code_revision="eval-sha",
        train=StageState(
            status="succeeded",
            checkpoint="/shared/checkpoint",
        ),
        **common,
    )
    state = _study_state(
        config,
        best_score=config.objective.baseline,
        trials=(train_trial, eval_trial),
        active_wave_index=1,
    )
    initialize_state(tmp_path / "state.json", state)
    runner = MixedRunner()
    controller = WaveController(
        config=config,
        workspace_manager=FakeWorkspaceManager(tmp_path / "worktrees"),
        editor=FakeEditor(),
        runner=runner,
    )

    result = controller.run_once()

    assert result.action == "trials_executed"
    assert "train: 1 succeeded" in result.message
    assert "eval: 1 succeeded" in result.message
    assert runner.train_runs == [train_trial.trial_id]
    assert runner.eval_runs == [eval_trial.trial_id]


def _settle_in_order(tmp_path: Path, order: tuple[int, ...]) -> StudyState:
    config = replace(_config(tmp_path), max_trials=3, max_parallel_trials=3)
    trials = tuple(
        Trial(
            trial_id=_trial_id(index),
            index=index,
            wave_index=1,
            parent_trial_id=None,
            parent_revision=config.base_commit,
            idea_type="recipe_tuning",
            hypothesis=f"Candidate {index}",
            code_revision=f"candidate-{index}-sha",
            parent_score=config.objective.baseline,
        )
        for index in range(1, 4)
    )
    state = _study_state(
        config,
        best_score=config.objective.baseline,
        trials=trials,
        active_wave_index=1,
    )
    state_path = tmp_path / "state.json"
    initialize_state(state_path, state)
    controller = WaveController(
        config=config,
        workspace_manager=FakeWorkspaceManager(tmp_path / "worktrees"),
        editor=FakeEditor(),
        runner=FakeRunner(),
    )
    scores = {1: 0.025, 2: 0.024, 3: 0.027}
    for position, index in enumerate(order):
        current = load_state(state_path)
        trial = next(trial for trial in current.trials if trial.index == index)
        _save_score(controller, current, trial, scores[index])
        if position < len(order) - 1:
            partial = load_state(state_path)
            assert partial.best_score == config.objective.baseline
            assert not any(trial.is_new_best for trial in partial.trials)
            assert partial.pending_change_trial_ids == ()

    assert controller.run_once().action == "wave_settled"
    return load_state(state_path)


def test_wave_settlement_is_completion_order_invariant(tmp_path: Path) -> None:
    first = _settle_in_order(tmp_path / "first", (1, 3, 2))
    second = _settle_in_order(tmp_path / "second", (2, 1, 3))

    def projection(state: StudyState):
        return (
            state.best_score,
            state.best_trial_id,
            state.pending_change_trial_ids,
            tuple(
                (trial.index, trial.improved_over_parent, trial.is_new_best)
                for trial in state.trials
            ),
        )

    assert projection(first) == projection(second)
    assert first.best_trial_id == "yam-smoke_wave1_trial2"
    assert first.pending_change_trial_ids == ("yam-smoke_wave1_trial1",)
    assert projection(first)[3] == (
        (1, True, False),
        (2, True, True),
        (3, False, False),
    )
    feedback = {trial.index: trial for trial in first.trials}
    assert feedback[1].result_summary == ("Improved the metric from parent score 0.0268 to 0.025.")
    assert feedback[1].next_guidance.startswith("Preserve this parent improvement")
    assert feedback[2].next_guidance.startswith("Build on this new global-best result")
    assert feedback[3].result_summary == (
        "Did not improve the metric over parent score 0.0268; observed 0.027."
    )


def test_restart_resumes_partial_wave_without_duplicate_trials(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), max_trials=2, max_parallel_trials=2)
    trials = tuple(
        Trial(
            trial_id=_trial_id(index),
            index=index,
            wave_index=1,
            parent_trial_id=None,
            parent_revision=config.base_commit,
            idea_type="recipe_tuning",
            hypothesis=f"Candidate {index}",
            code_branch=f"codex/candidate-{index}",
            code_revision=f"candidate-{index}-sha",
            code_published=True,
            parent_score=config.objective.baseline,
            train=StageState(
                status="succeeded",
                checkpoint=f"/checkpoint-{index}",
            ),
            evaluation=StageState(
                status="running",
            ),
        )
        for index in range(1, 3)
    )
    state = _study_state(
        config,
        best_score=config.objective.baseline,
        trials=trials,
        active_wave_index=1,
    )
    state_path = tmp_path / "state.json"
    initialize_state(state_path, state)
    first = WaveController(
        config=config,
        workspace_manager=FakeWorkspaceManager(tmp_path / "worktrees-1"),
        editor=FakeEditor(),
        runner=FakeRunner(),
    )
    current = load_state(state_path)
    _save_score(first, current, current.trials[0], 0.024)

    runner = FakeRunner(scores={trials[1].trial_id: 0.025})
    resumed = WaveController(
        config=config,
        workspace_manager=FakeWorkspaceManager(tmp_path / "worktrees-2"),
        editor=FakeEditor(),
        runner=runner,
    )

    assert resumed.run_once().action == "trials_executed"
    assert len(load_state(state_path).trials) == 2
    assert resumed.run_once().action == "wave_settled"
    final = load_state(state_path)
    assert final.best_trial_id == trials[0].trial_id
    assert len(final.trials) == 2
    assert runner.train_runs == []
    assert runner.eval_runs == [trials[1].trial_id]


def test_edit_uses_recorded_parent_revision_not_current_best(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), max_trials=3)
    parent_a = Trial(
        trial_id="yam-smoke_wave1_trial1",
        index=1,
        wave_index=1,
        parent_trial_id=None,
        parent_revision=config.base_commit,
        idea_type="recipe_tuning",
        hypothesis="Parent A",
        parent_score=config.objective.baseline,
        code_revision="parent-a-sha",
        score=0.02,
        improved_over_parent=True,
        is_new_best=True,
    )
    parent_b = Trial(
        trial_id="yam-smoke_wave2_trial2",
        index=2,
        wave_index=2,
        parent_trial_id=parent_a.trial_id,
        parent_revision=parent_a.code_revision,
        idea_type="recipe_tuning",
        hypothesis="Parent B",
        parent_score=parent_a.score,
        code_revision="parent-b-sha",
        score=0.019,
        improved_over_parent=True,
        is_new_best=True,
    )
    candidate = Trial(
        trial_id="yam-smoke_wave3_trial3",
        index=3,
        wave_index=3,
        parent_trial_id=parent_a.trial_id,
        parent_revision=parent_a.code_revision,
        idea_type="free_roam",
        hypothesis="",
        parent_score=parent_a.score,
    )
    state = _study_state(
        config,
        best_score=parent_b.score,
        best_trial_id=parent_b.trial_id,
        trials=(parent_a, parent_b, candidate),
        active_wave_index=3,
    )
    state_path = tmp_path / "state.json"
    initialize_state(state_path, state)
    workspace = FakeWorkspaceManager(tmp_path / "worktrees")
    controller = WaveController(
        config=config,
        workspace_manager=workspace,
        editor=FakeEditor(),
        runner=FakeRunner(),
    )

    assert controller.run_once().action == "candidate_committed"
    assert workspace.created == [(candidate.trial_id, parent_a.code_revision)]


def test_good_losing_composition_requeues_its_atomic_source(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), max_trials=6, max_parallel_trials=2)
    a = Trial(
        trial_id="yam-smoke_wave1_trial1",
        index=1,
        wave_index=1,
        parent_trial_id=None,
        parent_revision=config.base_commit,
        idea_type="recipe_tuning",
        hypothesis="Change A",
        parent_score=config.objective.baseline,
        code_revision="a-sha",
        score=0.017,
        improved_over_parent=True,
        is_new_best=True,
    )
    b = replace(
        a,
        trial_id="yam-smoke_wave1_trial2",
        index=2,
        hypothesis="Change B",
        code_revision="b-sha",
        score=0.018,
        is_new_best=False,
    )
    c = replace(
        a,
        trial_id="yam-smoke_wave1_trial3",
        index=3,
        hypothesis="Change C",
        code_revision="c-sha",
        score=0.019,
        is_new_best=False,
    )
    state = _study_state(
        config,
        best_score=a.score,
        best_trial_id=a.trial_id,
        trials=(a, b, c),
        pending_change_trial_ids=(b.trial_id, c.trial_id),
    )
    state_path = tmp_path / "state.json"
    initialize_state(state_path, state)
    workspace = FakeWorkspaceManager(tmp_path / "worktrees")
    editor = FakeEditor()
    controller = WaveController(
        config=config,
        workspace_manager=workspace,
        editor=editor,
        runner=FakeRunner(),
    )

    assert controller.run_once().action == "wave_created"
    created = load_state(state_path)
    compositions = created.trials[-2:]
    assert [trial.parent_trial_id for trial in compositions] == [a.trial_id, a.trial_id]
    assert [trial.composition_source_trial_id for trial in compositions] == [
        b.trial_id,
        c.trial_id,
    ]
    assert created.pending_change_trial_ids == ()

    assert controller.run_once().action == "candidate_committed"
    assert controller.run_once().action == "candidate_published"
    assert controller.run_once().action == "candidate_committed"
    assert editor.calls == []
    assert workspace.applied == [
        (compositions[0].trial_id, b.code_revision),
        (compositions[1].trial_id, c.code_revision),
    ]

    for score in (0.016, 0.0165):
        current = load_state(state_path)
        trial = next(trial for trial in current.trials if trial.score is None)
        _save_score(controller, current, trial, score)
    assert controller.run_once().action == "wave_settled"
    settled = load_state(state_path)
    assert settled.best_trial_id == compositions[0].trial_id
    assert settled.pending_change_trial_ids == (c.trial_id,)

    assert controller.run_once().action == "wave_created"
    next_trial = load_state(state_path).trials[-1]
    assert next_trial.parent_trial_id == compositions[0].trial_id
    assert next_trial.composition_source_trial_id == c.trial_id


def test_composition_conflict_defaults_to_audited_skip(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), max_trials=3)
    state_path = tmp_path / "state.json"
    initialize_state(state_path, _queued_composition_state(config))
    workspace = ConflictWorkspaceManager(tmp_path / "worktrees")
    editor = FakeEditor()
    controller = WaveController(
        config=config,
        workspace_manager=workspace,
        editor=editor,
        runner=FakeRunner(),
    )

    assert controller.run_once().action == "wave_created"
    result = controller.run_once()

    assert result.action == "candidate_rejected"
    assert "Composition conflict:" in result.message
    assert editor.calls == []
    trial = load_state(state_path).trials[-1]
    assert "groot/core/training/models/n2/model.py" in (trial.train.error or "")
    assert trial.train.status == "skipped"
    assert trial.evaluation.status == "skipped"


def test_non_conflict_composition_failure_is_reported(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), max_trials=3)
    state_path = tmp_path / "state.json"
    initialize_state(state_path, _queued_composition_state(config))
    workspace = BrokenCompositionWorkspaceManager(tmp_path / "worktrees")
    editor = FakeEditor()
    controller = WaveController(
        config=config,
        workspace_manager=workspace,
        editor=editor,
        runner=FakeRunner(),
    )

    assert controller.run_once().action == "wave_created"
    result = controller.run_once()

    assert result.action == "candidate_failed"
    assert result.message == "bad source revision"
    assert editor.calls == []


def test_wave_retains_non_winners_and_superseded_best(
    tmp_path: Path,
) -> None:
    config = replace(_config(tmp_path), max_trials=3, max_parallel_trials=2)
    old_best = Trial(
        trial_id="yam-smoke_wave1_trial1",
        index=1,
        wave_index=1,
        parent_trial_id=None,
        parent_revision=config.base_commit,
        idea_type="recipe_tuning",
        hypothesis="Old best",
        parent_score=config.objective.baseline,
        code_revision="old-sha",
        train=StageState(
            status="succeeded",
            checkpoint="/experiments/old/output",
        ),
        score=0.02,
        improved_over_parent=True,
        is_new_best=True,
    )
    winner = Trial(
        trial_id="yam-smoke_wave2_trial2",
        index=2,
        wave_index=2,
        parent_trial_id=old_best.trial_id,
        parent_revision=old_best.code_revision,
        idea_type="recipe_tuning",
        hypothesis="Winner",
        code_revision="winner-sha",
        train=StageState(
            status="succeeded",
            checkpoint="/experiments/winner/output",
        ),
        parent_score=old_best.score,
        score=0.019,
        improved_over_parent=True,
    )
    loser = replace(
        winner,
        trial_id="yam-smoke_wave2_trial3",
        index=3,
        hypothesis="Loser",
        code_revision="loser-sha",
        train=StageState(
            status="succeeded",
            checkpoint="/experiments/loser/output",
        ),
        score=0.021,
        improved_over_parent=False,
    )
    state = _study_state(
        config,
        best_score=old_best.score,
        best_trial_id=old_best.trial_id,
        trials=(old_best, winner, loser),
        active_wave_index=2,
    )
    state_path = tmp_path / "state.json"
    initialize_state(state_path, state)
    runner = FakeRunner()
    controller = WaveController(
        config=config,
        workspace_manager=FakeWorkspaceManager(tmp_path / "worktrees"),
        editor=FakeEditor(),
        runner=runner,
    )

    assert controller.run_once().action == "wave_settled"
    settled = load_state(state_path)
    assert settled.best_trial_id == winner.trial_id
    assert [trial.train.checkpoint for trial in settled.trials] == [
        "/experiments/old/output",
        "/experiments/winner/output",
        "/experiments/loser/output",
    ]
    assert controller.run_once().action == "stopped"


def test_failed_train_keeps_checkpoint_returned_by_script_runner(tmp_path: Path) -> None:
    class ArtifactFailureRunner(FakeRunner):
        def run_train(self, trial):
            self.train_runs.append(trial.trial_id)
            return replace(
                trial.train,
                status="failed",
                checkpoint="/experiments/failed-attempt/output",
                error="checkpoint is incomplete",
            )

    config = _config(tmp_path)
    trial = Trial(
        trial_id="yam-smoke_wave1_trial1",
        index=1,
        wave_index=1,
        parent_trial_id=None,
        parent_revision=config.base_commit,
        idea_type="recipe_tuning",
        hypothesis="Candidate",
        code_branch="codex/candidate",
        code_revision="candidate-sha",
        code_published=True,
        parent_score=config.objective.baseline,
    )
    state = _study_state(
        config,
        best_score=config.objective.baseline,
        trials=(trial,),
        active_wave_index=1,
    )
    state_path = tmp_path / "state.json"
    initialize_state(state_path, state)
    controller = WaveController(
        config=config,
        workspace_manager=FakeWorkspaceManager(tmp_path / "worktrees"),
        editor=FakeEditor(),
        runner=ArtifactFailureRunner(),
    )

    assert controller.run_once().action == "trials_executed"
    failed = load_state(state_path).trials[0].train
    assert failed.status == "failed"
    assert failed.attempts == 1
    assert failed.checkpoint == "/experiments/failed-attempt/output"
    assert controller.run_once().action == "wave_settled"
    settled = load_state(state_path).trials[0]
    assert settled.result_summary == "The experiment produced no valid metric result."
    assert settled.next_guidance.startswith("Do not draw a conclusion")


@pytest.mark.parametrize("stage_name", ["train", "eval"])
def test_stage_failure_retries_exactly_max_retry_times(
    tmp_path: Path,
    stage_name: str,
) -> None:
    class AlwaysFailingRunner(FakeRunner):
        def run_train(self, trial):
            self.train_runs.append(trial.trial_id)
            return replace(trial.train, status="failed", error="transient train failure")

        def run_eval(self, trial):
            self.eval_runs.append(trial.trial_id)
            return (
                replace(
                    trial.evaluation,
                    status="failed",
                    error="transient eval failure",
                ),
                None,
            )

    config = replace(_config(tmp_path), max_retry=2)
    train = (
        StageState(status="failed", attempts=1)
        if stage_name == "train"
        else StageState(status="succeeded", attempts=1, checkpoint="/checkpoint")
    )
    evaluation = StageState() if stage_name == "train" else StageState(status="failed", attempts=1)
    trial = Trial(
        trial_id="yam-smoke_wave1_trial1",
        index=1,
        wave_index=1,
        parent_trial_id=None,
        parent_revision=config.base_commit,
        idea_type="recipe_tuning",
        hypothesis="Candidate",
        code_branch="codex/candidate",
        code_revision="candidate-sha",
        code_published=True,
        parent_score=config.objective.baseline,
        train=train,
        evaluation=evaluation,
    )
    state = _study_state(
        config,
        best_score=config.objective.baseline,
        trials=(trial,),
        active_wave_index=1,
    )
    state_path = tmp_path / "state.json"
    initialize_state(state_path, state)
    runner = AlwaysFailingRunner()
    controller = WaveController(
        config=config,
        workspace_manager=FakeWorkspaceManager(tmp_path / "worktrees"),
        editor=FakeEditor(),
        runner=runner,
    )

    assert controller.run_once().action == "trials_executed"
    assert controller.run_once().action == "trials_executed"
    assert controller.run_once().action == "wave_settled"

    finished = load_state(state_path).trials[0]
    stage = finished.train if stage_name == "train" else finished.evaluation
    runs = runner.train_runs if stage_name == "train" else runner.eval_runs
    assert stage.status == "failed"
    assert stage.attempts == 3
    assert len(runs) == 2


def test_parallel_stage_results_are_saved_as_each_future_finishes(tmp_path: Path) -> None:
    class RecordingWaveController(WaveController):
        def __init__(self, **kwargs):
            self.saved_states = []
            super().__init__(**kwargs)

        def _save_state(self, state):
            self.saved_states.append(state)
            super()._save_state(state)

    config = replace(_config(tmp_path), max_trials=2, max_parallel_trials=2)
    common = {
        "wave_index": 1,
        "parent_trial_id": None,
        "parent_revision": config.base_commit,
        "idea_type": "recipe_tuning",
        "code_published": True,
        "parent_score": config.objective.baseline,
    }
    trials = tuple(
        Trial(
            trial_id=_trial_id(index),
            index=index,
            hypothesis=f"Trial {index}",
            code_branch=f"codex/trial-{index}",
            code_revision=f"revision-{index}",
            **common,
        )
        for index in (1, 2)
    )
    state = _study_state(
        config,
        best_score=config.objective.baseline,
        trials=trials,
        active_wave_index=1,
    )
    initialize_state(tmp_path / "state.json", state)
    controller = RecordingWaveController(
        config=config,
        workspace_manager=FakeWorkspaceManager(tmp_path / "worktrees"),
        editor=FakeEditor(),
        runner=FakeRunner(),
    )

    assert controller.run_once().action == "trials_executed"

    train_statuses = [
        [trial.train.status for trial in saved.trials] for saved in controller.saved_states
    ]
    assert ["succeeded", "running"] in train_statuses or [
        "running",
        "succeeded",
    ] in train_statuses


@pytest.mark.parametrize(
    ("field", "modified_config"),
    [
        (
            "base_commit",
            lambda config: replace(config, base_commit="b" * 40),
        ),
        (
            "eval_metric",
            lambda config: replace(
                config,
                objective=replace(config.objective, result_path="different.metric"),
            ),
        ),
        (
            "direction",
            lambda config: replace(
                config,
                objective=replace(config.objective, direction="maximize"),
            ),
        ),
        (
            "min_delta",
            lambda config: replace(
                config,
                objective=replace(config.objective, min_delta=0.1),
            ),
        ),
        (
            "baseline_score",
            lambda config: replace(
                config,
                objective=replace(config.objective, baseline=1.0),
            ),
        ),
    ],
)
def test_resume_rejects_changed_study_contract(
    tmp_path: Path,
    field: str,
    modified_config,
) -> None:
    config = _config(tmp_path)
    original = WaveController(
        config=config,
        workspace_manager=FakeWorkspaceManager(tmp_path / "worktrees"),
        editor=FakeEditor(),
        runner=FakeRunner(),
    )
    original.initialize()
    resumed = WaveController(
        config=modified_config(config),
        workspace_manager=FakeWorkspaceManager(tmp_path / "worktrees"),
        editor=FakeEditor(),
        runner=FakeRunner(),
    )

    with pytest.raises(ValueError, match=field):
        resumed.load_state()


def test_state_write_uses_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    config = _config(tmp_path)
    controller = WaveController(
        config=config,
        workspace_manager=FakeWorkspaceManager(tmp_path / "worktrees"),
        editor=FakeEditor(),
        runner=FakeRunner(),
    )
    real_replace = os.replace
    replacements = []

    def recording_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(
        "skills.autoresearch.wave_controller.os.replace",
        recording_replace,
    )

    controller.initialize()

    assert replacements
    temporary_path, destination = replacements[-1]
    assert destination == controller.state_path
    assert not temporary_path.exists()
    assert load_state(controller.state_path).study_id == config.study_id
