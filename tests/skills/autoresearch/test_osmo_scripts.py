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

import os
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).parents[3]
EXAMPLE = REPOSITORY / "skills/autoresearch/examples/genie_sim_osmo"


def _mock_osmo(tmp_path: Path) -> tuple[Path, Path]:
    submit_marker = tmp_path / "submitted"
    executable = tmp_path / "osmo"
    executable.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1 $2" == "workflow query" ]]; then
  if [[ "${MOCK_STAGE}" == "train" ]]; then
    training="${RUN_ROOT}/training"
    mkdir -p "${training}"
    printf '{}\n' > "${training}/config.json"
    printf 'final weights\n' > "${training}/model.safetensors"
  else
    mkdir -p "$(dirname "${RESULT_PATH}")"
    printf '{"success_rate": 1.0}\n' > "${RESULT_PATH}"
  fi
  printf '{"name":"%s","status":"COMPLETED"}\n' "$3"
  exit 0
fi
if [[ "$1 $2" == "workflow submit" ]]; then
  touch "${SUBMIT_MARKER}"
  printf '{"name":"unexpected-submit"}\n'
  exit 0
fi
exit 2
"""
    )
    executable.chmod(0o755)
    return executable, submit_marker


def _environment(tmp_path: Path, *, stage: str) -> tuple[dict[str, str], Path]:
    osmo, submit_marker = _mock_osmo(tmp_path)
    run_root = tmp_path / "run"
    environment = {
        **os.environ,
        "AUTORESEARCH_ATTEMPT": "1",
        "AUTORESEARCH_STAGE": stage,
        "AUTORESEARCH_TRIAL_ID": "script-test_wave1_trial1",
        "RUN_ROOT": str(run_root),
        "GROOT_BRANCH": "test-branch",
        "GROOT_REVISION": "a" * 40,
        "MOCK_STAGE": stage,
        "OSMO": str(osmo),
        "OSMO_POLL_INTERVAL_S": "0",
        "REPO_ROOT": str(REPOSITORY),
        "SUBMIT_MARKER": str(submit_marker),
    }
    return environment, submit_marker


def test_osmo_train_script_reattaches_without_duplicate_submit(tmp_path: Path) -> None:
    environment, submit_marker = _environment(tmp_path, stage="train")

    completed = subprocess.run(
        ["bash", str(EXAMPLE / "train_baseline.sh")],
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "reattached workflow: script-test_wave1_trial1-train-a1" in completed.stdout
    training = Path(environment["RUN_ROOT"]) / "training"
    assert (training / "model.safetensors").is_file()
    assert (training / "config.json").is_file()
    assert not list(training.glob("checkpoint-*"))
    assert not submit_marker.exists()


def test_osmo_eval_script_reattaches_without_duplicate_submit(tmp_path: Path) -> None:
    environment, submit_marker = _environment(tmp_path, stage="eval")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    result_path = tmp_path / "run/eval/results.json"
    environment.update(
        {
            "CHECKPOINT_PATH": str(checkpoint),
            "EVAL_CONFIG_PATH": str(EXAMPLE / "eval_config.yaml"),
            "PYTHON": sys.executable,
            "RESULT_PATH": str(result_path),
        }
    )

    completed = subprocess.run(
        ["bash", str(EXAMPLE / "eval_baseline.sh")],
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "reattached workflow: script-test_wave1_trial1-eval-a1" in completed.stdout
    assert result_path.is_file()
    assert not submit_marker.exists()
