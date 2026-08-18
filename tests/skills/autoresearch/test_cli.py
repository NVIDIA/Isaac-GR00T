# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# ruff: noqa: ARG005, SLF001

from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from skills.autoresearch import cli
import yaml


def _write_config(tmp_path: Path) -> Path:
    baseline = tmp_path / "baseline"
    training = baseline / "training"
    evaluation = baseline / "eval"
    training.mkdir(parents=True)
    evaluation.mkdir()
    (training / "config.json").write_text("{}")
    (training / "model.safetensors").write_bytes(b"weights")
    (evaluation / "results.json").write_text(
        '{"results": {"local": {"average": {"mse": 0.0268}}}}\n'
    )
    train_script = tmp_path / "train.sh"
    eval_script = tmp_path / "eval.sh"
    train_script.write_text("#!/usr/bin/env bash\n")
    eval_script.write_text("#!/usr/bin/env bash\n")
    config = {
        "study_id": "yam-smoke",
        "workspace": str(tmp_path),
        "max_trials": 1,
        "editable_paths": ["groot/core/training/models/"],
        "frozen_paths": ["groot/infra/"],
        "idea_types": ["recipe_tuning"],
        "output_dir": str(tmp_path / "runs"),
        "baseline": {
            "script_timeout_s": 86400,
            "train_script": str(train_script),
            "eval_script": str(eval_script),
            "eval_result": str(evaluation / "results.json"),
            "eval_metric": "results.local.average.mse",
            "direction": "minimize",
            "baseline_result": 0.0268,
            "min_delta": 0.0,
        },
        "git": {
            "git_remote": "origin",
            "base_branch": "baseline",
            "base_commit": "5587e83bece8d8588991a23ace81c095784e8d48",
        },
    }
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_cli_resumes_existing_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path)
    events: list[str] = []

    state_path = tmp_path / "state.json"
    state_path.write_text("{}")
    state = SimpleNamespace(
        active_wave_index=2,
        trials=(object(),),
        best_score=0.025,
        best_trial_id="yam-smoke_wave1_trial1",
    )
    controller = SimpleNamespace(
        state_path=state_path,
        load_state=lambda: events.append("load_state") or state,
        initialize=lambda: events.append("initialize"),
        run_once=lambda: events.append("run_once")
        or SimpleNamespace(action="stopped", message="done", done=True),
    )
    monkeypatch.setattr(cli, "_controller", lambda config: controller)
    monkeypatch.setattr(
        cli,
        "_validate",
        lambda config: events.append("validate"),
    )

    assert cli.main([str(config_path)]) == 0
    assert events == ["validate", "load_state", "run_once"]
    output = capsys.readouterr().out
    assert "[AutoResearch] resuming" in output
    assert "wave=2" in output
    assert "trials=1/1" in output
    assert "best=yam-smoke_wave1_trial1" in output
    assert "transition=stopped" in output


def test_cli_initializes_when_state_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path)
    events: list[str] = []

    state = SimpleNamespace(
        active_wave_index=None,
        trials=(),
        best_score=0.0268,
        best_trial_id=None,
    )
    controller = SimpleNamespace(
        state_path=tmp_path / "state.json",
        initialize=lambda: events.append("initialize") or state,
        run_once=lambda: events.append("run_once")
        or SimpleNamespace(action="stopped", message="done", done=True),
    )
    monkeypatch.setattr(cli, "_controller", lambda config: controller)
    monkeypatch.setattr(
        cli,
        "_validate",
        lambda config: events.append("validate"),
    )

    assert cli.main([str(config_path)]) == 0
    assert events == ["validate", "initialize", "run_once"]
    output = capsys.readouterr().out
    assert "[AutoResearch] initialized" in output
    assert "wave=none" in output
    assert "trials=0/1" in output
    assert "best=baseline" in output


def test_config_resolves_study_output_root(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text())
    payload["output_dir"] = str(tmp_path / "study-output/runs")
    config_path.write_text(yaml.safe_dump(payload))

    config = cli.StudyConfig.from_yaml(config_path)

    assert config.output_dir == (tmp_path / "study-output/runs").resolve()


def test_controller_forwards_explicit_codex_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text())
    payload["agent"] = {"sandbox": "danger-full-access"}
    config_path.write_text(yaml.safe_dump(payload))
    captured = []
    monkeypatch.setattr(
        cli,
        "CodexEditor",
        lambda config: captured.append(config) or SimpleNamespace(),
    )

    cli._controller(cli.StudyConfig.from_yaml(config_path))

    assert captured[0].sandbox == "danger-full-access"


@pytest.mark.parametrize("value", [False, ""])
def test_controller_rejects_invalid_codex_sandbox(tmp_path: Path, value: object) -> None:
    config_path = _write_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text())
    payload["agent"] = {"sandbox": value}
    config_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="agent.sandbox"):
        cli._controller(cli.StudyConfig.from_yaml(config_path))


def test_validate_rejects_credential_bearing_remote(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    credential_remote = "https://oauth2" + ":secret@" + "example.test/project.git"
    subprocess.run(
        [
            "git",
            "remote",
            "add",
            "origin",
            credential_remote,
        ],
        cwd=tmp_path,
        check=True,
    )

    with pytest.raises(ValueError, match="embedded credentials"):
        cli._validate_git_remote(tmp_path, "origin", None, "0" * 40)


def test_validate_rejects_invalid_base_commit(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.test/project.git"],
        cwd=tmp_path,
        check=True,
    )

    with pytest.raises(ValueError, match="Invalid base commit"):
        cli._validate_git_remote(tmp_path, "origin", None, "0" * 40)


def test_validate_rejects_output_dir_inside_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = cli.StudyConfig.from_yaml(_write_config(tmp_path))
    monkeypatch.setattr(cli.shutil, "which", lambda executable: f"/usr/bin/{executable}")

    with pytest.raises(ValueError, match="output_dir must be outside workspace"):
        cli._validate(config)


def test_validate_rejects_commit_outside_base_branch(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "value.txt").write_text("base\n")
    subprocess.run(["git", "add", "value.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "branch", "baseline"], cwd=tmp_path, check=True)
    (tmp_path / "value.txt").write_text("other\n")
    subprocess.run(["git", "commit", "-am", "other"], cwd=tmp_path, check=True, capture_output=True)
    other_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.test/project.git"],
        cwd=tmp_path,
        check=True,
    )

    with pytest.raises(ValueError, match="Invalid base branch"):
        cli._validate_git_remote(tmp_path, "origin", "bad..branch", other_commit)
    with pytest.raises(ValueError, match="not contained in base branch"):
        cli._validate_git_remote(tmp_path, "origin", "baseline", other_commit)
