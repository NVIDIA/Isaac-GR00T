# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for infrastructure-independent AutoResearch models."""

from __future__ import annotations

from dataclasses import replace

import pytest
from skills.autoresearch.study_state import (
    Objective,
    StageState,
    StudyState,
    Trial,
    make_trial_id,
)


def _study_state(**values) -> StudyState:
    """Build a state with the required current-schema identity."""
    return StudyState(
        base_commit="a" * 40,
        eval_metric="results.score",
        direction="minimize",
        min_delta=0.0,
        **values,
    )


@pytest.mark.parametrize(
    ("direction", "candidate", "reference", "expected"),
    [
        ("minimize", 0.2, 0.3, True),
        ("minimize", 0.3, 0.2, False),
        ("maximize", 0.3, 0.2, True),
        ("maximize", 0.2, 0.3, False),
    ],
)
def test_objective_comparison(
    direction: str,
    candidate: float,
    reference: float,
    expected: bool,
) -> None:
    objective = Objective(
        result_path="results.score",
        direction=direction,  # type: ignore[arg-type]
        baseline=0.25,
    )

    assert objective.is_better(candidate, reference) is expected
    assert objective.is_better(reference, reference) is False


def test_objective_min_delta() -> None:
    objective = Objective(
        result_path="results.local.average.mse",
        direction="minimize",
        baseline=0.0268,
        min_delta=0.001,
    )

    assert objective.is_better(0.0258, 0.0268)
    assert not objective.is_better(0.026, 0.0268)


def test_objective_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="direction"):
        Objective("score", "sideways", 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_delta"):
        Objective("score", "maximize", 0.0, min_delta=-0.1)
    with pytest.raises(ValueError, match="finite"):
        Objective("score", "maximize", float("nan"))


@pytest.mark.parametrize("status", ["running", "failed"])
def test_stage_state_preserves_status_and_checkpoint(status: str) -> None:
    stage = StageState(
        status=status,  # type: ignore[arg-type]
        checkpoint="/results/checkpoint-10",
        attempts=2,
        error="preempted" if status == "failed" else None,
    )

    assert StageState.from_dict(stage.to_dict()) == stage


def test_stage_state_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="unknown stage status"):
        StageState(status="unknown")  # type: ignore[arg-type]


def test_stable_trial_ids_and_next_id() -> None:
    state = StudyState.create(
        study_id="YAM bowls",
        baseline_score=0.0268,
        base_commit="a" * 40,
        eval_metric="results.score",
        direction="minimize",
        min_delta=0.0,
    )
    trial = Trial(
        trial_id=state.next_trial_id,
        index=state.next_trial_index,
        parent_trial_id=None,
        idea_type="recipe_tuning",
        hypothesis="Lower the learning rate.",
        wave_index=1,
        parent_revision=state.base_commit,
        parent_score=state.baseline_score,
    )
    state = state.with_trial(trial)

    assert trial.trial_id == "yam-bowls_wave1_trial1"
    assert state.next_trial_id == "yam-bowls_wave2_trial2"
    assert make_trial_id("YAM bowls", 2, 2) == state.next_trial_id


def test_study_state_persists_pinned_base_commit() -> None:
    state = StudyState.create(
        study_id="yam",
        baseline_score=0.0268,
        base_commit="a" * 40,
        eval_metric="results.local.average.mse",
        direction="minimize",
        min_delta=0.0001,
    )

    assert StudyState.from_dict(state.to_dict()) == state


def test_parent_improvement_is_distinct_from_global_best() -> None:
    state = StudyState.create(
        study_id="yam",
        baseline_score=0.0268,
        base_commit="a" * 40,
        eval_metric="results.score",
        direction="minimize",
        min_delta=0.0,
    )
    best = Trial(
        trial_id=make_trial_id("yam", 1, 1),
        index=1,
        parent_trial_id=None,
        parent_revision=state.base_commit,
        idea_type="recipe_tuning",
        hypothesis="First candidate",
        wave_index=1,
        result_summary="Improved the metric.",
        next_guidance="Build on this result.",
        code_branch="autoresearch/yam/0001",
        code_revision="abc123",
        parent_score=0.0268,
        train=StageState(status="succeeded", attempts=1),
        evaluation=StageState(status="succeeded", attempts=1),
        score=0.025,
        improved_over_parent=True,
        is_new_best=True,
    )
    state = replace(
        state.with_trial(best),
        best_score=best.score,
        best_trial_id=best.trial_id,
    )
    improved_but_not_best = Trial(
        trial_id=make_trial_id("yam", 1, 2),
        index=2,
        parent_trial_id=None,
        parent_revision=state.base_commit,
        idea_type="recipe_tuning",
        hypothesis="Parallel candidate",
        wave_index=1,
        parent_score=0.0268,
        score=0.0255,
        improved_over_parent=True,
        is_new_best=False,
    )
    state = state.with_trial(improved_but_not_best)

    assert state.trials[1].improved_over_parent is True
    assert state.trials[1].is_new_best is False
    assert state.best_trial_id == best.trial_id
    assert StudyState.from_dict(state.to_dict()) == state


def test_trial_flags_require_a_score() -> None:
    with pytest.raises(ValueError, match="flags require a score"):
        Trial(
            trial_id="ar-yam-0001",
            index=1,
            parent_trial_id=None,
            idea_type="recipe_tuning",
            hypothesis="Candidate",
            improved_over_parent=True,
        )


def test_parallel_state_round_trips_wave_queue_and_composition_provenance() -> None:
    source = Trial(
        trial_id="yam_wave1_trial1",
        index=1,
        wave_index=1,
        parent_trial_id=None,
        parent_revision="base-sha",
        idea_type="recipe_tuning",
        hypothesis="Lower the learning rate.",
        parent_score=0.0268,
        code_revision="source-sha",
        score=0.02,
        improved_over_parent=True,
        is_new_best=False,
    )
    queued_source = replace(
        source,
        trial_id="yam_wave1_trial2",
        index=2,
        hypothesis="Tune dropout.",
        code_revision="queued-source-sha",
    )
    composition = Trial(
        trial_id="yam_wave2_trial3",
        index=3,
        wave_index=2,
        parent_trial_id=None,
        parent_revision="base-sha",
        composition_source_trial_id=source.trial_id,
        idea_type="composition",
        hypothesis="Replay the queued change.",
        parent_score=0.0268,
    )
    state = _study_state(
        study_id="yam",
        baseline_score=0.0268,
        best_score=0.0268,
        trials=(source, queued_source, composition),
        active_wave_index=2,
        pending_change_trial_ids=(queued_source.trial_id,),
    )
    assert StudyState.from_dict(state.to_dict()) == state


def test_parallel_state_rejects_unknown_composition_and_queue_references() -> None:
    trial = Trial(
        trial_id="yam_wave0_trial1",
        index=1,
        parent_trial_id=None,
        idea_type="composition",
        hypothesis="Replay a missing change.",
        composition_source_trial_id="yam_wave0_trial9999",
    )
    with pytest.raises(ValueError, match="unknown composition source"):
        _study_state(
            study_id="yam",
            baseline_score=0.0268,
            best_score=0.0268,
            trials=(trial,),
        )

    plain = replace(
        trial,
        idea_type="recipe_tuning",
        composition_source_trial_id=None,
    )
    with pytest.raises(ValueError, match="pending change queue references unknown"):
        _study_state(
            study_id="yam",
            baseline_score=0.0268,
            best_score=0.0268,
            trials=(plain,),
            pending_change_trial_ids=("yam_wave0_trial9999",),
        )


def test_state_rejects_best_in_pending_change_queue() -> None:
    best = Trial(
        trial_id="yam_wave1_trial1",
        index=1,
        wave_index=1,
        parent_trial_id=None,
        parent_revision="base-sha",
        idea_type="recipe_tuning",
        hypothesis="Best",
        parent_score=0.0268,
        code_revision="best-sha",
        score=0.02,
        improved_over_parent=True,
        is_new_best=True,
    )
    common = {
        "study_id": "yam",
        "baseline_score": 0.0268,
        "best_score": best.score,
        "best_trial_id": best.trial_id,
        "trials": (best,),
    }

    with pytest.raises(ValueError, match="already in the best"):
        _study_state(**common, pending_change_trial_ids=(best.trial_id,))


def test_integer_fields_reject_booleans_and_floats() -> None:
    with pytest.raises(TypeError, match="attempts"):
        StageState(attempts=True)
    with pytest.raises(TypeError, match="index"):
        Trial(
            trial_id="ar-yam-0001",
            index=1.5,  # type: ignore[arg-type]
            parent_trial_id=None,
            idea_type="recipe_tuning",
            hypothesis="Candidate",
        )


def test_parallel_state_rejects_mismatched_frozen_parent_score() -> None:
    parent = Trial(
        trial_id="yam_wave1_trial1",
        index=1,
        wave_index=1,
        parent_trial_id=None,
        parent_revision="base-sha",
        parent_score=0.0268,
        idea_type="recipe_tuning",
        hypothesis="Parent",
        code_revision="parent-sha",
        score=0.02,
        improved_over_parent=True,
        is_new_best=True,
    )
    child = Trial(
        trial_id="yam_wave2_trial2",
        index=2,
        wave_index=2,
        parent_trial_id=parent.trial_id,
        parent_revision=parent.code_revision,
        parent_score=0.9,
        idea_type="recipe_tuning",
        hypothesis="Child",
    )

    with pytest.raises(ValueError, match="parent score does not match"):
        _study_state(
            study_id="yam",
            baseline_score=0.0268,
            best_score=parent.score,
            best_trial_id=parent.trial_id,
            trials=(parent, child),
            active_wave_index=2,
        )


def test_parallel_state_rejects_best_without_code_revision() -> None:
    best = Trial(
        trial_id="yam_wave0_trial1",
        index=1,
        parent_trial_id=None,
        idea_type="recipe_tuning",
        hypothesis="Best",
        score=0.02,
        improved_over_parent=True,
        is_new_best=True,
    )

    with pytest.raises(ValueError, match="best trial must have a code revision"):
        _study_state(
            study_id="yam",
            baseline_score=0.0268,
            best_score=best.score,
            best_trial_id=best.trial_id,
            trials=(best,),
        )
