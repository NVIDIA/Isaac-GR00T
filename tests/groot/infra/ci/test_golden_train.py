# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the public golden-run training submit helper."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from groot.infra.ci.golden_run_config import load_variants
from groot.infra.ci.main import golden_train


@pytest.fixture(autouse=True)
def _clear_internal_golden_run_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(golden_train.CONFIG_PATH_ENV_VAR, raising=False)
    monkeypatch.delenv(golden_train.CHECKPOINT_PATH_TEMPLATE_ENV_VAR, raising=False)


def _experiment_config(tmp_path: Path) -> Path:
    config = tmp_path / "golden.yaml"
    config.write_text(
        """
variants:
  sample:
    label: "Sample Experiment"
    run_type: "sample"
    experiment: "n2_yam_simulation_stack_bowls"
    eval: {}
    platforms:
      h100:
        num_gpus: 96
        num_nodes: 13
        batch_size: 4
"""
    )
    return config


def test_print_env_uses_generic_variable_names(capsys: pytest.CaptureFixture[str]):
    golden_train._print_env("n2-minwam-sim", "h100")

    output = capsys.readouterr().out
    assert "GOLDEN_RUN_VARIANT_LABEL='N2 Min-WAM Sim OSS Training'" in output
    assert "GOLDEN_RUN_LAUNCHER=" in output
    assert "NUM_GPUS=96" in output
    assert "NUM_NODES=13" in output
    assert "BATCH_SIZE=4" in output
    assert "GRADIENT_ACCUMULATION_STEPS=1" in output
    assert "GOLDEN_RUN_BASE_MODEL_REPO_ID=nvidia/GR00T-N2-YAM-Pretrain-Bidir-200K" in output
    assert "GOLDEN_RUN_BASE_MODEL_REVISION=checkpoint-200000" in output
    assert "GOLDEN_RUN_DATASET_REPO_ID=nvidia/GR00T-N2-YAM-Stack-Bowls-200Traj" in output
    assert "GOLDEN_RUN_DATASET_REVISION=main" in output
    assert "GOLDEN_RUN_DATASET_LOCAL_NAME=yam_workstation_stack_bowls_20260430_170700" in output
    assert "GOLDEN_RUN_FINAL_MODEL_REPO_ID=nvidia/GR00T-N2-MinWAM-Stack-Bowls-PostTrain" in output
    assert "GOLDEN_RUN_FINAL_MODEL_REVISION=main" in output
    assert "N" + "17" not in output


def test_print_env_includes_experiment_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    config = _experiment_config(tmp_path)
    monkeypatch.setenv(golden_train.CONFIG_PATH_ENV_VAR, str(config))

    golden_train._print_env("sample", "h100")

    output = capsys.readouterr().out
    assert "GOLDEN_RUN_EXPERIMENT=n2_yam_simulation_stack_bowls" in output
    assert "GOLDEN_RUN_EXPERIMENT_DIR" not in output


def test_print_env_supports_oss_sim_training_variant_h100(
    capsys: pytest.CaptureFixture[str],
):
    golden_train._print_env("n2-minwam-sim", "h100")

    output = capsys.readouterr().out
    assert "GOLDEN_RUN_VARIANT_LABEL='N2 Min-WAM Sim OSS Training'" in output
    assert "groot/core/training/scripts/n2/golden/launch_n2_minwam_sim_oss_golden.py" in output
    assert "NUM_GPUS=96" in output
    assert "NUM_NODES=13" in output
    assert "BATCH_SIZE=4" in output
    assert "GRADIENT_ACCUMULATION_STEPS=1" in output


def test_submit_experiment_variant_requires_internal_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _experiment_config(tmp_path)
    variant = load_variants(config)["sample"]
    args = SimpleNamespace(compute_type="h100")

    def reject_experiment_configs(**_kwargs):
        raise RuntimeError("internal golden-run submitter")

    monkeypatch.setattr(
        golden_train,
        "_resolve_experiment_configs",
        reject_experiment_configs,
    )

    with pytest.raises(RuntimeError, match="internal golden-run submitter"):
        golden_train._submit_experiment_backed_variant(
            args=args,
            variant=variant,
            env={},
            job_name="golden_run_sample_h100_sha",
        )


def test_checkpoint_path_uses_dashboard_checkpoint_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "ckpts_path.txt"
    monkeypatch.delenv("USER_SLUG", raising=False)

    golden_train._write_checkpoint_path(
        output=output,
        checkpoint_root="checkpoints/project/00075_golden_run_n2_minwam_sha",
    )

    assert (
        output.read_text() == "s3://checkpoints/project/"
        "00075_golden_run_n2_minwam_sha/checkpoint-10000\n"
    )


def test_checkpoint_path_falls_back_to_legacy_user_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "ckpts_path.txt"
    monkeypatch.setenv("USER_SLUG", "ci-user")

    golden_train._write_checkpoint_path(output=output, job_name="golden_run_n2_minwam_gb300_sha")

    assert output.read_text() == "s3://ci-user_golden_run_n2_minwam_gb300_sha/checkpoint-10000\n"


def test_checkpoint_path_honors_checkpoint_step_with_dashboard_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "ckpts_path.txt"
    monkeypatch.delenv("USER_SLUG", raising=False)

    golden_train._write_checkpoint_path(
        output=output,
        checkpoint_root="checkpoints/project/00075_golden_run_n16_robocasa365_sha",
        checkpoint_step=30000,
    )

    assert (
        output.read_text() == "s3://checkpoints/project/"
        "00075_golden_run_n16_robocasa365_sha/checkpoint-30000\n"
    )


def test_checkpoint_path_honors_checkpoint_step_with_legacy_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "ckpts_path.txt"
    monkeypatch.setenv("USER_SLUG", "ci-user")

    golden_train._write_checkpoint_path(
        output=output,
        job_name="golden_run_n16_robocasa365_h100_sha",
        checkpoint_step=30000,
    )

    assert (
        output.read_text() == "s3://ci-user_golden_run_n16_robocasa365_h100_sha/checkpoint-30000\n"
    )


def test_read_dashboard_checkpoint_root_from_submit_result(tmp_path: Path):
    result = tmp_path / "dashboard_submit_result.json"
    result.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_path": "checkpoints/project/00075_job",
            }
        )
    )

    assert golden_train._read_dashboard_checkpoint_root(result) == "checkpoints/project/00075_job"


def test_submit_prints_launcher_output_when_launcher_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    args = SimpleNamespace(
        run_type="n2-minwam-sim",
        compute_type="h100",
        cluster_id="sky://test-cluster",
        ckpts_path_output=tmp_path / "ckpts_path.txt",
        log_file=tmp_path / "ray_training.log",
        submission_id_output=tmp_path / "submission_id.txt",
        submission_id_wait_attempts=1,
        status_poll_interval=0.01,
    )

    def fail_launcher(*_args, **_kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["python", "launcher.py"],
            output="launcher inner error\n",
        )

    monkeypatch.setattr(golden_train.subprocess, "run", fail_launcher)

    with pytest.raises(subprocess.CalledProcessError):
        golden_train._submit(args)

    assert "launcher inner error\n" in capsys.readouterr().out


def test_tail_and_poll_reuses_one_ray_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    class FakeRayClient:
        def __init__(self) -> None:
            self.statuses = ["RUNNING", "SUCCEEDED"]
            self.status_calls = 0

        async def tail_job_logs(self, submission_id: str):
            yield f"{submission_id}: training log\n"

        def get_job_status(self, submission_id: str) -> str:
            self.status_calls += 1
            return self.statuses.pop(0)

    instances = []

    class FakeGearClient:
        def __init__(self, cluster_id: str) -> None:
            self.cluster_id = cluster_id
            self.client = FakeRayClient()
            instances.append(self)

    monkeypatch.setattr(golden_train, "_create_ray_client", FakeGearClient)
    monkeypatch.setattr(golden_train.time, "sleep", lambda _seconds: None)

    log_file = tmp_path / "ray_training.log"

    golden_train._tail_logs_and_poll_terminal_status(
        "sky://test-cluster",
        "submission-123",
        log_file,
        interval=60,
        status_error_limit=10,
    )

    assert [client.cluster_id for client in instances] == ["sky://test-cluster"]
    assert instances[0].client.status_calls == 2
    assert log_file.read_text() == "submission-123: training log\n"
    output = capsys.readouterr().out
    assert "Ray job submission-123 SUCCEEDED." in output


def test_tail_error_still_polls_status_with_same_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    class FakeRayClient:
        async def tail_job_logs(self, _submission_id: str):
            raise RuntimeError("tail disconnected")
            yield ""

        def get_job_status(self, _submission_id: str) -> str:
            return "SUCCEEDED"

    instances = []

    class FakeGearClient:
        def __init__(self, cluster_id: str) -> None:
            self.cluster_id = cluster_id
            self.client = FakeRayClient()
            instances.append(self)

    monkeypatch.setattr(golden_train, "_create_ray_client", FakeGearClient)

    golden_train._tail_logs_and_poll_terminal_status(
        "sky://test-cluster",
        "submission-123",
        tmp_path / "ray_training.log",
        interval=60,
        status_error_limit=10,
    )

    assert [client.cluster_id for client in instances] == ["sky://test-cluster"]
    output = capsys.readouterr().out
    assert "Ray log tail failed: RuntimeError('tail disconnected')" in output
    assert "Ray job submission-123 SUCCEEDED." in output


def test_poll_terminal_status_fails_after_repeated_status_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    class FakeRayClient:
        def __init__(self) -> None:
            self.status_calls = 0

        def get_job_status(self, _submission_id: str) -> str:
            self.status_calls += 1
            raise RuntimeError("dashboard unreachable")

    client = FakeRayClient()
    monkeypatch.setattr(golden_train.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="cluster may be gone or unreachable"):
        golden_train._poll_terminal_status_with_client(
            client,
            "submission-123",
            interval=60,
            status_error_limit=2,
        )

    assert client.status_calls == 2
    output = capsys.readouterr().out
    assert "Ray status query failed (1/2): RuntimeError('dashboard unreachable')" in output
    assert "Ray status query failed (2/2): RuntimeError('dashboard unreachable')" in output
