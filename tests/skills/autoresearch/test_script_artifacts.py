# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from skills.autoresearch.script_artifacts import (
    collect_script_evaluation_artifacts,
    collect_script_training_artifacts,
    lookup_result_path,
)


def test_collect_script_training_artifacts_accepts_checkpoint_root(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    training = run_root / "training"
    training.mkdir(parents=True)
    (training / "config.json").write_text("{}")
    (training / "model.safetensors").write_bytes(b"weights")

    result = collect_script_training_artifacts(run_root=run_root)

    assert result.output_dir == training.resolve()
    assert result.checkpoint == training.resolve()


def test_collect_script_training_artifacts_uses_latest_numbered_checkpoint(
    tmp_path: Path,
) -> None:
    training = tmp_path / "run" / "training"
    for step in (2, 11, 7):
        checkpoint = training / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}")
        (checkpoint / "model.safetensors").write_bytes(b"weights")

    result = collect_script_training_artifacts(run_root=tmp_path / "run")

    assert result.checkpoint.name == "checkpoint-11"


def test_collect_script_training_artifacts_rejects_escaped_checkpoint(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    training = run_root / "training"
    training.mkdir(parents=True)
    outside = tmp_path / "outside-checkpoint"
    outside.mkdir()
    (outside / "config.json").write_text("{}")
    (outside / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(RuntimeError, match="outside expected root"):
        collect_script_training_artifacts(
            run_root=run_root,
            checkpoint_path=outside,
        )


def test_collect_script_evaluation_artifacts_reads_success_rate(tmp_path: Path) -> None:
    result_file = tmp_path / "results.json"
    result_file.write_text(json.dumps({"success_rate": 0.75}))

    result = collect_script_evaluation_artifacts(
        result_file=result_file,
        result_path="success_rate",
    )

    assert result.score == 0.75


def test_collect_script_evaluation_artifacts_rejects_invalid_metric(tmp_path: Path) -> None:
    result_file = tmp_path / "results.json"
    result_file.write_text(json.dumps({"success_rate": True}))

    with pytest.raises(RuntimeError, match="not numeric"):
        collect_script_evaluation_artifacts(
            result_file=result_file,
            result_path="success_rate",
        )


def test_lookup_result_path_reads_nested_metrics() -> None:
    payload = {"results": {"local": {"average": {"mse": 0.0268}}}}

    assert lookup_result_path(payload, "results.local.average.mse") == 0.0268
