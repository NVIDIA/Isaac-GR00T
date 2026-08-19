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

import pytest
from skills.autoresearch.config import StudyConfig
from skills.autoresearch.prior_insights import _render_handoff, prior_insights_for_wave
from skills.autoresearch.study_state import Objective, StudyState, Trial
from skills.autoresearch.trial_prompt import build_prompt


def _config(tmp_path: Path) -> StudyConfig:
    return StudyConfig(
        study_id="yam-smoke",
        workspace=tmp_path / "workspace",
        base_commit="5587e83bece8d8588991a23ace81c095784e8d48",
        train_dataset="/dataset/yam/train",
        eval_dataset="/dataset/yam/eval",
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
        train_script=tmp_path / "train.sh",
        eval_script=tmp_path / "eval.sh",
        user_guidance="Change only learning rate.",
    )


def _state(config: StudyConfig) -> StudyState:
    previous = Trial(
        trial_id="yam-smoke_wave1_trial1",
        index=1,
        wave_index=1,
        parent_trial_id=None,
        parent_revision=config.base_commit,
        idea_type="recipe_tuning",
        hypothesis="Reduce learning rate.",
        result_summary="Improved validation MSE.",
        next_guidance="Try a smaller follow-up adjustment.",
        code_revision="candidate-sha",
        parent_score=config.objective.baseline,
        score=0.025,
        improved_over_parent=True,
        is_new_best=True,
    )
    current = Trial(
        trial_id="yam-smoke_wave2_trial2",
        index=2,
        wave_index=2,
        parent_trial_id=previous.trial_id,
        parent_revision=previous.code_revision,
        idea_type="free_roam",
        hypothesis="",
        parent_score=previous.score,
    )
    return StudyState(
        study_id=config.study_id,
        baseline_score=config.objective.baseline,
        best_score=previous.score,
        base_commit=config.base_commit,
        eval_metric=config.objective.result_path,
        direction=config.objective.direction,
        min_delta=config.objective.min_delta,
        best_trial_id=previous.trial_id,
        trials=(previous, current),
        active_wave_index=2,
    )


def test_prior_insights_provider_returns_empty_without_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)

    def unexpected_search(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Insights retrieval must remain disabled")

    monkeypatch.setattr(
        "skills.autoresearch.prior_insights.search_insights",
        unexpected_search,
    )

    assert (
        prior_insights_for_wave(
            config=config,
            state=_state(config),
            wave_index=2,
        )
        == ""
    )
    assert not config.output_dir.exists()


def test_prior_insights_renderer_remains_available() -> None:
    handoff = _render_handoff(
        [
            {
                "purpose": "Ideas to build on",
                "result": {
                    "available": True,
                    "projection": {
                        "generated_at": "2026-08-17T00:00:00Z",
                        "sha256": "projection-sha",
                    },
                    "results": [
                        {
                            "key": "doc:1",
                            "finding": {
                                "id": "finding:doc:1",
                                "title": "Finding",
                                "text": "Complete finding body.",
                            },
                            "chunk": {
                                "id": "doc:1",
                                "title": "Source",
                                "text": "Complete supporting evidence.",
                            },
                        }
                    ],
                },
            }
        ]
    )

    assert "### Retrieved for Ideas to build on" in handoff
    assert "Complete finding body." in handoff
    assert "Complete supporting evidence." in handoff
    assert "sha256=projection-sha" in handoff


def test_candidate_prompt_marks_insights_as_directional_prior(tmp_path: Path) -> None:
    config = _config(tmp_path)

    prompt = build_prompt(
        config=config,
        state=_state(config),
        trial_id="yam-smoke_wave2_trial2",
        trial_index=2,
        prior_insights="- Finding `finding:doc:lr`: A smaller LR helped a related run.",
    )

    assert "Prior Insights:" in prompt
    assert "finding:doc:lr" in prompt
    assert "historical guidance, not evidence" in prompt
    assert "User guidance and current-study train/eval results take precedence" in prompt
