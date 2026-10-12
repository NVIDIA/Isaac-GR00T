# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for AutoResearch study configuration."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
from skills.autoresearch.config import StudyConfig, expand_environment_variables
import yaml


def _config_data() -> dict[str, object]:
    return {
        "study_id": "yam-stack-bowls-smoke",
        "workspace": "./repo",
        "train_dataset": "/datasets/yam/train",
        "eval_dataset": "/datasets/yam/eval",
        "max_trials": 4,
        "max_retry": 2,
        "user_guidance": "Change only learning rate, weight decay, or dropout.",
        "output_dir": "study-runs",
        "editable_paths": ["groot/core/training/models/"],
        "frozen_paths": ["groot/infra/eval/", "/datasets/yam/"],
        "idea_types": ["recipe_tuning", "free_roam"],
        "baseline": {
            "script_timeout_s": 86400,
            "train_script": "train.sh",
            "eval_script": "eval.sh",
            "eval_result": "eval/results.json",
            "eval_metric": "results.local.average.mse",
            "direction": "minimize",
            "baseline_result": 0.026848560199141502,
            "min_delta": 0.0001,
        },
        "git": {
            "git_remote": "origin",
            "base_branch": "baseline",
            "base_commit": "5587e83bece8d8588991a23ace81c095784e8d48",
        },
        "agent": {"command": ["codex", "exec"], "model": "gpt-5.6"},
    }


def _write_config(tmp_path: Path, data: dict[str, object]) -> Path:
    baseline = data.get("baseline")
    if isinstance(baseline, dict):
        for key in ("train_script", "eval_script"):
            script = baseline.get(key)
            if not isinstance(script, str):
                continue
            script_path = Path(script)
            if not script_path.is_absolute():
                script_path = tmp_path / script_path
            script_path = script_path.resolve()
            baseline[key] = str(script_path)
            script_path.parent.mkdir(parents=True, exist_ok=True)
            if not script_path.exists():
                script_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        eval_result = baseline.get("eval_result")
        if isinstance(eval_result, str):
            result_path = Path(eval_result)
            if not result_path.is_absolute():
                result_path = tmp_path / "baseline" / result_path
            baseline["eval_result"] = str(result_path.resolve())
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_load_study_config(tmp_path: Path) -> None:
    config = StudyConfig.from_yaml(_write_config(tmp_path, _config_data()))

    assert config.study_id == "yam-stack-bowls-smoke"
    assert config.workspace == (tmp_path / "repo").resolve()
    assert config.base_branch == "baseline"
    assert config.base_commit == "5587e83bece8d8588991a23ace81c095784e8d48"
    assert config.objective.direction == "minimize"
    assert config.objective.result_path == "results.local.average.mse"
    assert config.output_dir == (tmp_path / "study-runs").resolve()
    assert config.output_dir_name == "training"
    assert config.train_script == (tmp_path / "train.sh").resolve()
    assert config.eval_result == (tmp_path / "baseline/eval/results.json").resolve()
    assert config.agent["model"] == "gpt-5.6"
    assert config.max_parallel_trials == 1
    assert config.max_retry == 2
    assert config.user_guidance == "Change only learning rate, weight decay, or dropout."


def test_user_guidance_is_optional_trimmed_and_type_checked(tmp_path: Path) -> None:
    data = _config_data()
    data.pop("user_guidance")
    assert StudyConfig.from_yaml(_write_config(tmp_path, data)).user_guidance == ""

    data = _config_data()
    data["user_guidance"] = "  Change only the learning rate.  "
    assert (
        StudyConfig.from_yaml(_write_config(tmp_path, data)).user_guidance
        == "Change only the learning rate."
    )

    data["user_guidance"] = ["change learning rate"]
    with pytest.raises(TypeError, match="user_guidance must be a string"):
        StudyConfig.from_yaml(_write_config(tmp_path, data))


def test_load_study_config_expands_environment_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "autoresearch"
    monkeypatch.setenv("AUTORESEARCH_ROOT", str(root))
    data = _config_data()
    data["output_dir"] = "${AUTORESEARCH_ROOT}/study"

    config = StudyConfig.from_yaml(_write_config(tmp_path, data))

    assert config.output_dir == (root / "study").resolve()


def test_load_study_config_rejects_missing_environment_paths(tmp_path: Path) -> None:
    data = _config_data()
    data["output_dir"] = "${AUTORESEARCH_MISSING_ROOT}/study"

    with pytest.raises(ValueError, match="AUTORESEARCH_MISSING_ROOT is not set"):
        StudyConfig.from_yaml(_write_config(tmp_path, data))


def test_load_study_config_rejects_placeholder_environment_values(tmp_path: Path) -> None:
    data = _config_data()
    data["train"] = {"env": {"POOL": "your-osmo-pool"}}

    with pytest.raises(ValueError, match="train environment POOL must be set"):
        StudyConfig.from_yaml(_write_config(tmp_path, data))


@pytest.mark.parametrize(
    ("package_dir", "editable_paths", "frozen_paths", "logs"),
    [
        (
            "oss/skills/autoresearch",
            ("oss/groot/core/training/models/", "groot/core/training/models/", "oss/vendor/"),
            ("oss/groot/infra/",),
            [],
        ),
        # An exported checkout is the oss/ package itself: mirrored entries fold together,
        # while an oss/ path that exists in the checkout is kept as written.
        (
            "skills/autoresearch",
            ("groot/core/training/models/", "oss/vendor/"),
            ("groot/infra/",),
            [
                "[AutoResearch] exported layout detected; dropped oss/ prefix from 1 editable_paths",
                "[AutoResearch] exported layout detected; dropped oss/ prefix from 1 frozen_paths",
            ],
        ),
    ],
)
def test_config_maps_oss_paths_to_workspace_layout(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    package_dir: str,
    editable_paths: tuple[str, ...],
    frozen_paths: tuple[str, ...],
    logs: list[str],
) -> None:
    (tmp_path / "repo" / package_dir).mkdir(parents=True)
    (tmp_path / "repo/oss/vendor").mkdir(parents=True)
    data = _config_data()
    data["editable_paths"] = [
        "oss/groot/core/training/models/",
        "groot/core/training/models/",
        "oss/vendor/",
    ]
    data["frozen_paths"] = ["oss/groot/infra/"]

    with caplog.at_level("INFO", logger="skills.autoresearch.config"):
        config = StudyConfig.from_yaml(_write_config(tmp_path, data))

    assert config.editable_paths == editable_paths
    assert config.frozen_paths == frozen_paths
    assert caplog.messages == logs


def test_config_resolves_missing_base_commit_from_branch_head(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=workspace, check=True)
    (workspace / "README.md").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "branch", "-M", "baseline"], cwd=workspace, check=True)
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    data = _config_data()
    git = data["git"]
    assert isinstance(git, dict)
    git.pop("base_commit")

    config = StudyConfig.from_yaml(_write_config(tmp_path, data))

    assert config.base_commit == expected


def test_config_requires_base_branch(tmp_path: Path) -> None:
    data = _config_data()
    git = data["git"]
    assert isinstance(git, dict)
    git.pop("base_branch")

    with pytest.raises(KeyError, match="base_branch"):
        StudyConfig.from_yaml(_write_config(tmp_path, data))


def test_max_parallel_trials_loads_and_validates(tmp_path: Path) -> None:
    data = _config_data()
    data["max_parallel_trials"] = 3
    config = StudyConfig.from_yaml(_write_config(tmp_path, data))

    assert config.max_parallel_trials == 3
    data["max_parallel_trials"] = 0
    with pytest.raises(ValueError, match="max_parallel_trials"):
        StudyConfig.from_yaml(_write_config(tmp_path, data))


def test_max_retry_loads_and_validates(tmp_path: Path) -> None:
    config = StudyConfig.from_yaml(_write_config(tmp_path, _config_data()))

    assert config.max_retry == 2

    for value in (-1, True):
        data = _config_data()
        data["max_retry"] = value
        with pytest.raises((TypeError, ValueError), match="max_retry"):
            StudyConfig.from_yaml(_write_config(tmp_path, data))


def test_relative_paths_are_resolved_against_config_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data = _config_data()
    data["train_dataset"] = "datasets/train"
    data["eval_dataset"] = "datasets/eval"
    config_path = _write_config(config_dir, data)
    monkeypatch.chdir(tmp_path)

    config = StudyConfig.from_yaml(config_path)

    assert config.train_dataset == str((config_dir / "datasets/train").resolve())
    assert config.eval_dataset == str((config_dir / "datasets/eval").resolve())
    assert config.eval_result == (config_dir / "baseline/eval/results.json").resolve()
    assert config.train_script == (config_dir / "train.sh").resolve()


@pytest.mark.parametrize(
    ("field", "relative_value"),
    [
        ("train_script", "train.sh"),
        ("eval_script", "eval.sh"),
        ("eval_result", "eval/results.json"),
    ],
)
def test_config_requires_absolute_baseline_execution_paths(
    tmp_path: Path,
    field: str,
    relative_value: str,
) -> None:
    data = _config_data()
    config_path = _write_config(tmp_path, data)
    baseline = data["baseline"]
    assert isinstance(baseline, dict)
    baseline[field] = relative_value
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"baseline\.{field} must be an absolute path"):
        StudyConfig.from_yaml(config_path)


def test_eval_result_may_be_outside_output_dir(tmp_path: Path) -> None:
    data = _config_data()
    config_path = _write_config(tmp_path, data)
    baseline = data["baseline"]
    assert isinstance(baseline, dict)
    baseline["eval_result"] = str((tmp_path / "outside/results.json").resolve())
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    config = StudyConfig.from_yaml(config_path)

    assert config.eval_result == (tmp_path / "outside/results.json").resolve()


def test_agent_defaults_to_empty_mapping(tmp_path: Path) -> None:
    data = _config_data()
    data.pop("agent")

    config = StudyConfig.from_yaml(_write_config(tmp_path, data))

    assert config.agent == {}


def test_loader_rejects_missing_and_invalid_fields(tmp_path: Path) -> None:
    missing_baseline = _config_data()
    missing_baseline.pop("baseline")
    with pytest.raises(KeyError, match="baseline is required"):
        StudyConfig.from_yaml(_write_config(tmp_path, missing_baseline))

    missing_git = _config_data()
    missing_git.pop("git")
    with pytest.raises(KeyError, match="git is required"):
        StudyConfig.from_yaml(_write_config(tmp_path, missing_git))

    bad_trials = _config_data()
    bad_trials["max_trials"] = 0
    with pytest.raises(ValueError, match="max_trials"):
        StudyConfig.from_yaml(_write_config(tmp_path, bad_trials))


def test_config_rejects_retired_runs_root(tmp_path: Path) -> None:
    data = _config_data()
    data["runs_root"] = "runs"

    with pytest.raises(ValueError, match="runs_root was renamed to output_dir"):
        StudyConfig.from_yaml(_write_config(tmp_path, data))


def test_config_requires_output_dir(tmp_path: Path) -> None:
    data = _config_data()
    data.pop("output_dir")

    with pytest.raises(KeyError, match="output_dir"):
        StudyConfig.from_yaml(_write_config(tmp_path, data))


def test_config_rejects_retired_baseline_score(tmp_path: Path) -> None:
    data = _config_data()
    baseline = data["baseline"]
    assert isinstance(baseline, dict)
    baseline["baseline_score"] = baseline.pop("baseline_result")

    with pytest.raises(ValueError, match="baseline_score was renamed to baseline_result"):
        StudyConfig.from_yaml(_write_config(tmp_path, data))


@pytest.mark.parametrize("base_commit", ["main", "5587e83", "g" * 40])
def test_config_rejects_non_full_commit_sha(tmp_path: Path, base_commit: str) -> None:
    data = _config_data()
    git = data["git"]
    assert isinstance(git, dict)
    git["base_commit"] = base_commit

    with pytest.raises(ValueError, match="full 40-character"):
        StudyConfig.from_yaml(_write_config(tmp_path, data))


@pytest.mark.parametrize(
    "study_id",
    [
        "../escape",
        "nested/study",
        "nested\\study",
        "has space",
        "UpperCase",
        "a" * 65,
        "a..b",
    ],
)
def test_config_rejects_path_unsafe_study_id(tmp_path: Path, study_id: str) -> None:
    data = _config_data()
    data["study_id"] = study_id

    with pytest.raises(ValueError, match="study_id"):
        StudyConfig.from_yaml(_write_config(tmp_path, data))


def test_config_can_omit_datasets(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    train_script = tmp_path / "train.sh"
    eval_script = tmp_path / "eval.sh"
    train_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    eval_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    data = {
        "study_id": "script-smoke",
        "workspace": "./repo",
        "max_trials": 1,
        "editable_paths": ["groot/core/training/models/"],
        "frozen_paths": ["examples/"],
        "idea_types": ["recipe_tuning"],
        "output_dir": "runs",
        "baseline": {
            "script_timeout_s": 86400,
            "train_script": "train.sh",
            "eval_script": "eval.sh",
            "eval_result": "eval/results.json",
            "eval_metric": "success_rate",
            "direction": "maximize",
            "baseline_result": 0.0,
            "min_delta": 0.0,
        },
        "git": {
            "git_remote": "origin",
            "base_branch": "baseline",
            "base_commit": "5587e83bece8d8588991a23ace81c095784e8d48",
        },
    }

    config = StudyConfig.from_yaml(_write_config(tmp_path, data))

    assert config.train_dataset == ""
    assert config.eval_result == (baseline / "eval/results.json").resolve()
    assert config.train_script == train_script.resolve()


@pytest.mark.parametrize("field", ["branch", "working_dir"])
def test_config_rejects_retired_runner_fields(tmp_path: Path, field: str) -> None:
    data = _config_data()
    data[field] = "obsolete"

    with pytest.raises(ValueError, match="no longer supported"):
        StudyConfig.from_yaml(_write_config(tmp_path, data))


def test_genie_sim_example_reads_agent_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "skills/autoresearch/examples/genie_sim_osmo/autoresearch.yaml"
    )
    monkeypatch.setenv("AUTORESEARCH_ROOT", str(tmp_path))
    monkeypatch.setenv("OSMO_POOL", "test-pool")
    monkeypatch.setenv("AUTORESEARCH_AGENT_HOST", "claude")
    monkeypatch.setenv("AUTORESEARCH_AGENT_EXECUTABLE", "/opt/claude")
    monkeypatch.setenv("AUTORESEARCH_AGENT_MODEL", "sonnet")

    payload = expand_environment_variables(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), config_path
    )

    assert payload["agent"] == {
        "host": "claude",
        "executable": "/opt/claude",
        "model": "sonnet",
        "timeout_s": 3600,
    }
