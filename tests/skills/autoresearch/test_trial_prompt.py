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

from skills.autoresearch.config import StudyConfig
from skills.autoresearch.study_state import (
    Objective,
    StageState,
    StudyState,
    Trial,
    make_trial_id,
)
from skills.autoresearch.trial_prompt import build_prompt, idea_type_for_index


def _config(
    tmp_path: Path,
    *,
    max_steps: int = 120,
    eval_steps: int = 17,
) -> StudyConfig:
    del max_steps, eval_steps
    script_dir = tmp_path / "scripts"
    script_dir.mkdir(exist_ok=True)
    train_script = script_dir / "train.sh"
    eval_script = script_dir / "eval.sh"
    train_script.write_text("#!/usr/bin/env bash\n")
    eval_script.write_text("#!/usr/bin/env bash\n")
    return StudyConfig(
        study_id="yam-smoke",
        workspace=tmp_path,
        base_commit="5587e83bece8d8588991a23ace81c095784e8d48",
        train_dataset="/dataset/train",
        eval_dataset="/dataset/eval",
        objective=Objective(
            result_path="results.local.average.mse",
            direction="minimize",
            baseline=0.0268,
        ),
        max_trials=4,
        editable_paths=("groot/core/training/models/",),
        frozen_paths=("groot/infra/",),
        idea_types=("recipe_tuning", "free_roam"),
        agent={},
        eval_result=tmp_path / "baseline/eval/results.json",
        output_dir=tmp_path / "runs",
        train_script=train_script,
        eval_script=eval_script,
        train_env={"AUTH": "must-not-appear"},
    )


def test_prompt_states_direction_guards_and_previous_result(tmp_path: Path) -> None:
    config = _config(tmp_path)
    best = Trial(
        trial_id="yam-smoke_wave1_trial1",
        index=1,
        wave_index=1,
        parent_trial_id=None,
        parent_revision=config.base_commit,
        idea_type="recipe_tuning",
        hypothesis="Reduce the learning rate.",
        result_summary="Improved the metric from 0.0268 to 0.025.",
        next_guidance="Build on this global-best result.",
        code_revision="best-candidate-sha",
        parent_score=config.objective.baseline,
        score=0.025,
        improved_over_parent=True,
        is_new_best=True,
        changed_files=("groot/core/training/configs/training/optimizer.yaml",),
    )
    current = Trial(
        trial_id="yam-smoke_wave2_trial2",
        index=2,
        wave_index=2,
        parent_trial_id=best.trial_id,
        parent_revision=best.code_revision,
        idea_type="free_roam",
        hypothesis="",
        parent_score=0.025,
    )
    pending_sibling = Trial(
        trial_id="yam-smoke_wave2_trial3",
        index=3,
        wave_index=2,
        parent_trial_id=best.trial_id,
        parent_revision=best.code_revision,
        idea_type="recipe_tuning",
        hypothesis="",
        parent_score=0.025,
    )
    state = StudyState(
        study_id=config.study_id,
        baseline_score=config.objective.baseline,
        base_commit=config.base_commit,
        eval_metric=config.objective.result_path,
        direction=config.objective.direction,
        min_delta=config.objective.min_delta,
        best_score=0.025,
        best_trial_id=best.trial_id,
        trials=(best, current, pending_sibling),
    )

    prompt = build_prompt(
        config=config,
        state=state,
        trial_id="yam-smoke_wave2_trial2",
        trial_index=2,
        parent_revision="best-candidate-sha",
        parent_score=0.025,
    )

    assert "lower is better" in prompt
    assert "`free_roam`" in prompt
    assert "`groot/infra/`" in prompt
    assert "Reduce the learning rate" in prompt
    assert "Do not commit, push" in prompt
    assert "pinned pretrained/base model" in prompt
    assert "Candidate parent: `yam-smoke_wave1_trial1`" in prompt
    assert "revision `best-candidate-sha`" in prompt
    assert "user-provided train/eval scripts" in prompt
    assert "train_script" in prompt
    assert "eval_result" in prompt
    assert "training and evaluation compute budget is fixed" in prompt
    assert "maximum training steps or epochs" in prompt
    assert "training-side search surface" in prompt
    assert "implementation correctness" in prompt
    assert "changed_files" not in prompt
    assert "result_summary Improved the metric from 0.0268 to 0.025" in prompt
    assert "next_guidance Build on this global-best result" in prompt
    assert "avoid broad rewrite" not in prompt.lower()
    assert '"train_dataset": "/dataset/train"' in prompt
    assert "10-step/10-point" not in prompt
    assert "must-not-appear" not in prompt
    # Active-wave trials have no completed result and stay out of feedback.
    assert prompt.count("`yam-smoke_wave2_trial2`") == 1
    assert "yam-smoke_wave2_trial3" not in prompt


def test_prompt_includes_all_completed_results(tmp_path: Path) -> None:
    config = _config(tmp_path)
    trials = [
        Trial(
            trial_id="yam-smoke_wave1_trial1",
            index=1,
            wave_index=1,
            parent_trial_id=None,
            parent_revision=config.base_commit,
            parent_score=config.objective.baseline,
            idea_type="recipe_tuning",
            hypothesis="The older winning idea.",
            code_revision="older-best-sha",
            result_summary="Improved the metric to 0.02.",
            next_guidance="Build on this global-best result.",
            score=0.02,
            improved_over_parent=True,
            is_new_best=True,
        )
    ]
    for index in range(2, 11):
        reason = "Rejected: changed frozen evaluation code" if index == 10 else "No improvement"
        trials.append(
            Trial(
                trial_id=make_trial_id(config.study_id, index, index),
                index=index,
                wave_index=index,
                parent_trial_id=None,
                parent_revision=config.base_commit,
                parent_score=config.objective.baseline,
                idea_type="free_roam",
                hypothesis=f"Attempt {index}.",
                result_summary="The experiment produced no valid metric result.",
                next_guidance="Inspect the recorded stage error.",
                train=StageState(status="skipped", error=reason),
                evaluation=StageState(status="skipped", error=reason),
            )
        )
    state = StudyState(
        study_id=config.study_id,
        baseline_score=config.objective.baseline,
        base_commit=config.base_commit,
        eval_metric=config.objective.result_path,
        direction=config.objective.direction,
        min_delta=config.objective.min_delta,
        best_score=0.02,
        best_trial_id="yam-smoke_wave1_trial1",
        trials=tuple(trials),
    )

    prompt = build_prompt(
        config=config,
        state=state,
        trial_id="yam-smoke_wave11_trial11",
        trial_index=11,
    )

    assert "Completed experiment feedback:" in prompt
    assert "The older winning idea" in prompt
    assert "outcome current global best" in prompt
    assert "error Rejected: changed frozen evaluation code" in prompt
    for index in range(1, 11):
        trial_id = make_trial_id(config.study_id, index, index)
        assert f"trial `{trial_id}`" in prompt
    assert prompt.count("result_summary") == 10
    assert "changed_files" not in prompt


def test_idea_types_cycle(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert idea_type_for_index(config, 1) == "recipe_tuning"
    assert idea_type_for_index(config, 2) == "free_roam"
    assert idea_type_for_index(config, 3) == "recipe_tuning"
