# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# ruff: noqa: S108
"""Tests for the AutoResearch HTML visualization skill."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).parents[3]
RENDERER_PATH = REPO_ROOT / "skills" / "visualize-autoresearch" / "scripts" / "render.py"


def _load_renderer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("autoresearch_visualizer", RENDERER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


renderer = _load_renderer()


def _stage(
    status: str = "succeeded",
    *,
    attempts: int = 1,
    checkpoint: str | None = None,
) -> dict:
    stage = {"status": status, "attempts": attempts}
    if status == "failed":
        stage["error"] = "stage failed"
    if checkpoint is not None:
        stage["checkpoint"] = checkpoint
    return stage


def _trial(
    index: int,
    *,
    parent: str | None = None,
    wave: int = 1,
    parent_score: float = 0.5,
    score: float | None = None,
    improved: bool | None = None,
    train: str = "succeeded",
    evaluation: str = "succeeded",
    composition_source: str | None = None,
) -> dict:
    trial_id = f"ar-visual-test-{index:04d}"
    return {
        "trial_id": trial_id,
        "index": index,
        "wave_index": wave,
        "parent_trial_id": parent,
        "parent_revision": "a" * 40,
        "composition_source_trial_id": composition_source,
        "idea_type": "recipe_tuning",
        "hypothesis": f"Try recipe change {index}.",
        "result_summary": f"Observed result {index}.",
        "next_guidance": f"Follow up on result {index}.",
        "code_branch": f"codex/autoresearch-{index}",
        "code_revision": str(index) * 40,
        "code_published": True,
        "parent_score": parent_score,
        "train": _stage(train, checkpoint=f"/checkpoints/{trial_id}"),
        "evaluation": _stage(evaluation),
        "score": score,
        "improved_over_parent": improved,
        "is_new_best": False if score is not None else None,
        "changed_files": ["groot/config.yaml"],
    }


def _state(*, direction: str = "maximize", trials: list[dict] | None = None) -> dict:
    return {
        "study_id": "visual-test",
        "baseline_score": 0.5,
        "best_score": 0.5,
        "base_commit": "a" * 40,
        "eval_metric": "results.success_rate",
        "direction": direction,
        "min_delta": 0.0,
        "best_trial_id": None,
        "active_wave_index": None,
        "pending_change_trial_ids": [],
        "trials": trials or [],
    }


def test_maximize_branching_layout_and_result_semantics() -> None:
    first = _trial(1, score=0.6, improved=True)
    regression = _trial(2, score=0.4, improved=False)
    winner = _trial(
        3,
        parent=first["trial_id"],
        wave=2,
        parent_score=0.6,
        score=0.75,
        improved=True,
    )
    running = _trial(
        4,
        parent=first["trial_id"],
        wave=2,
        parent_score=0.6,
        score=None,
        improved=None,
        train="running",
        evaluation="pending",
    )
    state = _state(trials=[first, regression, winner, running])
    state["best_score"] = 0.75
    state["best_trial_id"] = winner["trial_id"]
    state["active_wave_index"] = 2

    report = renderer.build_report(state)
    by_id = {trial["trial_id"]: trial for trial in report["trials"]}

    assert by_id[first["trial_id"]]["status"] == "improved"
    assert by_id[regression["trial_id"]]["status"] == "regressed"
    assert by_id[winner["trial_id"]]["depth"] == 2
    assert by_id[running["trial_id"]]["status"] == "running"
    assert by_id[winner["trial_id"]]["intensity"] > by_id[first["trial_id"]]["intensity"]
    assert (
        by_id[first["trial_id"]]["position"]["x"] == by_id[regression["trial_id"]]["position"]["x"]
    )
    assert by_id[winner["trial_id"]]["position"]["x"] == by_id[running["trial_id"]]["position"]["x"]
    assert by_id[winner["trial_id"]]["position"]["x"] > by_id[first["trial_id"]]["position"]["x"]
    assert report["study_status"] == "Running"

    output = renderer.render_html(state, Path("/tmp/state.json"))
    assert output.count('marker-end="url(#arrow-parent)"') == 4
    assert 'data-result="improved"' in output
    assert 'data-result="regressed"' in output
    assert 'data-result="running"' in output
    assert "BEST" in output
    assert "WAVE 1" in output
    assert "WAVE 2" in output
    assert "ACTIVE · 2 TRIALS" in output
    assert "/checkpoints/ar-visual-test-0001" in output


def test_parallel_wave_layout_stays_compact_for_current_state_shape() -> None:
    trials = [
        _trial(index, wave=((index - 1) // 4) + 1, score=0.0, improved=False)
        for index in range(1, 17)
    ]
    for trial in trials[12:]:
        trial["score"] = None
        trial["improved_over_parent"] = None
        trial["is_new_best"] = None
        trial["train"] = _stage("running")
        trial["evaluation"] = _stage("pending")
    state = _state(trials=trials)
    state["active_wave_index"] = 4

    report = renderer.build_report(state)
    output = renderer.render_html(state, Path("/tmp/state.json"))

    assert report["graph_height"] < 1100
    assert len({trial["position"]["x"] for trial in report["trials"]}) == 4
    assert output.count('class="wave-title') == 4
    assert "ACTIVE · 4 TRIALS" in output


def test_minimize_scores_map_lower_to_green_and_higher_to_red() -> None:
    better = _trial(1, parent_score=0.5, score=0.35, improved=True)
    worse = _trial(2, parent_score=0.5, score=0.65, improved=False)
    report = renderer.build_report(_state(direction="minimize", trials=[better, worse]))

    assert report["trials"][0]["improvement"] == pytest.approx(0.15)
    assert report["trials"][1]["improvement"] == pytest.approx(-0.15)
    assert report["trials"][0]["fill"].startswith("hsl(145")
    assert report["trials"][1]["fill"].startswith("hsl(5")


def test_composition_adds_a_distinct_dashed_source_arrow() -> None:
    source = _trial(1, score=0.6, improved=True)
    parent = _trial(2, score=0.65, improved=True)
    composition = _trial(
        3,
        parent=parent["trial_id"],
        wave=2,
        parent_score=0.65,
        score=0.7,
        improved=True,
        composition_source=source["trial_id"],
    )
    output = renderer.render_html(
        _state(trials=[source, parent, composition]),
        Path("/tmp/state.json"),
    )

    assert output.count('marker-end="url(#arrow-parent)"') == 3
    assert output.count('marker-end="url(#arrow-composition)"') == 1
    assert "Source trial: ar-visual-test-0001" in output


def test_failed_unscored_trial_is_not_mislabeled_as_regression() -> None:
    failed = _trial(
        1,
        score=None,
        improved=None,
        train="succeeded",
        evaluation="failed",
    )
    report = renderer.build_report(_state(trials=[failed]))
    output = renderer.render_html(_state(trials=[failed]), Path("/tmp/state.json"))

    assert report["trials"][0]["status"] == "failed"
    assert report["trials"][0]["improvement"] is None
    assert report["study_status"] == "Complete with failures"
    assert 'data-result="failed"' in output
    assert 'data-result="regressed"' not in output
    assert "TRAIN SUCCEEDED · EVAL FAILED" in output
    assert "stage failed" in output


def test_empty_study_renders_without_graph() -> None:
    empty = _state()
    output = renderer.render_html(empty, Path("/tmp/state.json"))

    assert "No trials have been created yet." in output
    assert "<svg" not in output


def test_user_text_is_escaped_in_nodes_and_detail_templates() -> None:
    unsafe = _trial(1, score=0.6, improved=True)
    unsafe["result_summary"] = "</template><script>alert('x')</script>"
    unsafe["hypothesis"] = "<img src=x onerror=alert(1)>"
    state = _state(trials=[unsafe])
    state["study_id"] = "<script>study</script>"

    output = renderer.render_html(state, Path("/tmp/<state>.json"))

    assert "<script>study</script>" not in output
    assert "</template><script>alert('x')</script>" not in output
    assert "<img src=x onerror=alert(1)>" not in output
    assert "&lt;script&gt;study&lt;/script&gt;" in output
    assert "&lt;img src=x onerror=alert(1)&gt;" in output


def test_unknown_parent_and_invalid_score_raise_clear_errors() -> None:
    unknown_parent = _trial(1, parent="missing", score=0.6, improved=True)
    with pytest.raises(ValueError, match="unknown parent"):
        renderer.build_report(_state(trials=[unknown_parent]))

    invalid_score = _trial(1, score=None)
    invalid_score["score"] = "not-a-number"
    with pytest.raises(ValueError, match="must be a number"):
        renderer.build_report(_state(trials=[invalid_score]))


def test_cli_writes_a_portable_html_report(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    output_path = tmp_path / "report.html"
    state_path.write_text(json.dumps(_state()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(RENDERER_PATH),
            str(state_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rendered = output_path.read_text(encoding="utf-8")
    assert completed.stdout.strip() == str(output_path.resolve())
    assert rendered.startswith("<!doctype html>")
    assert 'class="drawer"' in rendered
    assert "https://" not in rendered


def test_load_state_rejects_malformed_or_incomplete_json(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        renderer.load_state(malformed)

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"study_id": "missing-scores"}), encoding="utf-8")
