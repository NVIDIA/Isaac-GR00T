# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: SLF001

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess

import pytest
from skills.autoresearch.config import StudyConfig
from skills.autoresearch.script_runner import ScriptTrialRunner
from skills.autoresearch.study_state import Objective, StageState, Trial


def _trial() -> Trial:
    return Trial(
        trial_id="ar-yam-smoke-0001",
        index=1,
        parent_trial_id=None,
        idea_type="recipe_tuning",
        hypothesis="Tune learning rate.",
        code_branch="codex/autoresearch-yam-0001",
        code_revision="a" * 40,
        parent_score=0.0268,
    )


def _script_runner_layout(tmp_path: Path) -> StudyConfig:
    baseline = tmp_path / "baseline"
    evaluation = baseline / "eval"
    evaluation.mkdir(parents=True)
    (evaluation / "results.json").write_text('{"success_rate": 0.25}\n')
    train_script = tmp_path / "train.sh"
    train_script.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${RUN_ROOT}/training"
printf '{}\n' > "${RUN_ROOT}/training/config.json"
printf 'weights\n' > "${RUN_ROOT}/training/model.safetensors"
ref_path="${RUN_ROOT}/training/ref.txt"
printf '%s %s %s\n' "${GROOT_BRANCH}" "${GROOT_REVISION}" "${AUTORESEARCH_ATTEMPT}" > "${ref_path}"
"""
    )
    eval_script = tmp_path / "eval.sh"
    eval_script.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${EVAL_OUTPUT_DIR}"
cat > "${RESULT_PATH}" <<EOF
{"success_rate": 0.75, "checkpoint_path": "${CHECKPOINT_PATH}"}
EOF
"""
    )
    train_script.chmod(0o755)
    eval_script.chmod(0o755)
    return StudyConfig(
        study_id="runner-test",
        workspace=tmp_path,
        base_branch="baseline",
        base_commit="a" * 40,
        train_dataset="",
        eval_dataset="",
        objective=Objective(
            result_path="success_rate",
            direction="maximize",
            baseline=0.25,
        ),
        max_trials=1,
        editable_paths=("groot/",),
        frozen_paths=("eval/",),
        idea_types=("recipe_tuning",),
        agent={},
        eval_result=evaluation / "results.json",
        output_dir=tmp_path / "runs",
        train_script=train_script,
        eval_script=eval_script,
        script_timeout_s=5,
        git_remote="origin",
    )


def test_script_runner_executes_scripts_and_collects_metric(tmp_path: Path) -> None:
    config = _script_runner_layout(tmp_path)
    runner = ScriptTrialRunner(config)
    trial = replace(_trial(), train=StageState(status="running", attempts=1))

    runner.validate_baseline()
    train = runner.run_train(trial)
    assert train.status == "succeeded"
    assert train.checkpoint is not None
    assert Path(train.checkpoint).is_dir()
    trial_dir = config.output_dir / trial.trial_id
    assert (trial_dir / "train.attempt-0001.stdout.log").is_file()
    assert (trial_dir / "train.attempt-0001.stderr.log").is_file()
    assert (Path(train.checkpoint) / "ref.txt").read_text().strip() == (
        f"{trial.code_branch} {trial.code_revision} 1"
    )
    assert not (config.output_dir / trial.trial_id / "train").exists()

    trial = replace(
        trial,
        train=train,
        evaluation=StageState(status="running", attempts=1),
    )
    evaluation, score = runner.run_eval(trial)

    assert evaluation.status == "succeeded"
    assert score == 0.75
    assert (trial_dir / "eval" / "results.json").is_file()
    assert not (config.output_dir / trial.trial_id / "eval" / "artifacts").exists()


def test_validate_baseline_rejects_mismatched_configured_score(tmp_path: Path) -> None:
    config = _script_runner_layout(tmp_path)
    config = replace(
        config,
        objective=replace(config.objective, baseline=0.5),
    )

    with pytest.raises(RuntimeError, match="does not match"):
        ScriptTrialRunner(config).validate_baseline()


def test_script_runner_does_not_reuse_stale_eval_result(tmp_path: Path) -> None:
    config = _script_runner_layout(tmp_path)
    config.eval_script.write_text("#!/usr/bin/env bash\nexit 0\n")
    runner = ScriptTrialRunner(config)
    base_trial = _trial()
    stale_result = config.output_dir / base_trial.trial_id / "eval" / "results.json"
    stale_result.parent.mkdir(parents=True)
    stale_result.write_text('{"success_rate": 0.99}\n')
    trial = replace(
        base_trial,
        train=StageState(
            status="succeeded",
            checkpoint=str(tmp_path / "checkpoint"),
        ),
        evaluation=StageState(status="running", attempts=1),
    )

    failed, score = runner.run_eval(trial)

    assert failed.status == "failed"
    assert score is None
    assert "Invalid evaluation result JSON" in (failed.error or "")
    assert not stale_result.exists()


def test_script_runner_marks_nonzero_exit_failed(tmp_path: Path) -> None:
    config = _script_runner_layout(tmp_path)
    config.train_script.write_text("#!/usr/bin/env bash\nexit 17\n")
    runner = ScriptTrialRunner(config)
    trial = replace(_trial(), train=StageState(status="running", attempts=1))

    failed = runner.run_train(trial)

    assert failed.status == "failed"
    assert failed.attempts == 1
    assert "exit code 17" in (failed.error or "")


def test_cleanup_removes_source_worktree_but_keeps_trial_outputs(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
    (repository / "README.md").write_text("test\n")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=repository, check=True, capture_output=True
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    config = _script_runner_layout(tmp_path / "layout")
    runner = ScriptTrialRunner(config, repository=repository)
    trial = replace(_trial(), code_revision=revision)
    run_root = config.output_dir / trial.trial_id
    run_root.mkdir(parents=True)
    output_marker = run_root / "training" / "keep.txt"
    output_marker.parent.mkdir()
    output_marker.write_text("keep\n")
    source_root = runner._ensure_source_worktree(trial, run_root)
    assert source_root is not None and source_root.is_dir()

    runner.cleanup(trial)

    assert not source_root.exists()
    assert output_marker.read_text() == "keep\n"
