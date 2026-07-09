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

"""Tests for OSMO Ray cluster helpers."""

from __future__ import annotations

from importlib import resources
import subprocess

import pytest

from groot.infra.job import ClusterSpec, RayCluster
from groot.infra.job.osmo_client import OsmoClient
from groot.infra.job.workflow_config import WorkflowConfig


class FakeOsmoClient(OsmoClient):
    def __init__(self, status: str = "RUNNING") -> None:
        self.commands = []
        self.status = status

    def run(self, command):
        self.commands.append(command)
        if "--dry-run" in command:
            return """
workflow:
  name: test-ray_uuid_abcd
  resources:
    default:
      cpu: 1
      gpu: 4
      memory: 32Gi
      storage: 32Gi
"""
        return "Workflow ID - wf-123\n"

    def get_workflow_status(self, _workflow_id: str) -> str:
        return self.status


def _spec(**overrides) -> ClusterSpec:
    values = {
        "cluster_name": "test-ray",
        "pool": "example-pool",
        "gpu_type": "gb200",
        "num_nodes": 2,
        "image": "example/ray:latest",
    }
    values.update(overrides)
    return ClusterSpec(**values)


def test_render_config_uses_osmo_dry_run_and_gpu_type_defaults() -> None:
    osmo_client = FakeOsmoClient()

    rendered = RayCluster(_spec(user_command="python train.py"), osmo_client).render_config()

    assert rendered.data["workflow"]["name"] == "test-ray_uuid_abcd"
    command = osmo_client.commands[0]
    assert command[:3] == ["osmo", "workflow", "submit"]
    assert "--dry-run" in command
    assert "--set" in command
    assert "workflow_name=test-ray" in command
    assert "num_nodes=2" in command
    assert "image=example/ray:latest" in command
    assert "gpus=4" in command
    assert "cpus=1" in command
    assert "memory=32" in command
    assert "storage=32" in command
    assert "user_command=python train.py" in command


def test_launch_submits_rendered_yaml_and_returns_workflow_id(monkeypatch) -> None:
    osmo_client = FakeOsmoClient()
    wait_ready = []
    monkeypatch.setattr(
        RayCluster,
        "wait_until_ready",
        lambda self, workflow_id, **kwargs: wait_ready.append((workflow_id, kwargs)),
    )

    result = RayCluster(_spec(), osmo_client).launch(wait=True)

    assert result.workflow_id == "wf-123"
    assert result.cluster_id == "wf-123"
    assert "osmo workflow query wf-123" in result.followup_commands
    assert any(command[:3] == ["osmo", "workflow", "submit"] for command in osmo_client.commands)
    assert wait_ready[0][0] == "wf-123"


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        (["Workflow ID - wf-123"], "wf-123"),
        (["Workflow ID: wf-456"], "wf-456"),
    ],
)
def test_parse_workflow_id(outputs, expected) -> None:
    assert FakeOsmoClient().parse_workflow_id(outputs) == expected


def test_run_with_timeout_returns_partial_output(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["osmo", "workflow", "logs", "wf-123"],
            timeout=5,
            output=b"partial stdout\n",
            stderr=b"partial stderr\n",
        )

    monkeypatch.setattr("groot.infra.job.osmo_client.subprocess.run", fake_run)

    assert (
        OsmoClient().run_with_timeout(["osmo", "workflow", "logs", "wf-123"], 5)
        == "partial stdout\npartial stderr\n"
    )


def test_submit_workflow_retries_transient_submit_errors(monkeypatch) -> None:
    calls = []
    sleeps = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(
                command,
                10,
                stdout="Error message: Status Code: 503\nBody:\nupstream connect error\n",
                stderr="Server responded with status code 503\n",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="Workflow ID - wf-retry\n",
            stderr="",
        )

    monkeypatch.setattr("groot.infra.job.osmo_client.subprocess.run", fake_run)
    monkeypatch.setattr("groot.infra.job.osmo_client.time.sleep", sleeps.append)

    output = OsmoClient(submit_retry_base_delay_seconds=0.01).submit_workflow(
        WorkflowConfig({"workflow": {"name": "retry-test"}}),
        pool="example-pool",
        priority="HIGH",
    )

    assert output == "Workflow ID - wf-retry\n"
    assert len(calls) == 2
    assert sleeps == [0.01]
    assert calls[0][-4:] == ["--pool", "example-pool", "--priority", "HIGH"]


def test_submit_workflow_does_not_retry_non_transient_errors(monkeypatch) -> None:
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            10,
            stdout="Error message: Status Code: 401\n",
            stderr="Server responded with status code 401\n",
        )

    monkeypatch.setattr("groot.infra.job.osmo_client.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="status code 401"):
        OsmoClient(submit_retry_base_delay_seconds=0.01).submit_workflow(
            WorkflowConfig({"workflow": {"name": "auth-test"}}),
            pool="example-pool",
            priority="HIGH",
        )

    assert len(calls) == 1


def test_wait_until_ready_succeeds_when_dashboard_answers(monkeypatch) -> None:
    class FakePortForward:
        local_port = 48265

        @classmethod
        def init_for_dashboard(cls, _workflow_id, *, dashboard_port):
            assert dashboard_port == 8265
            return cls()

        def start(self):
            return None

        def stop(self):
            return None

    class FakeProbe:
        def __init__(self, local_port, timeout_seconds):
            self.local_port = local_port
            self.timeout_seconds = timeout_seconds

        def is_ready(self):
            return True, ""

    monkeypatch.setattr("groot.infra.job.ray_cluster.PortForward", FakePortForward)
    monkeypatch.setattr("groot.infra.job.ray_cluster.RayDashboardProbe", FakeProbe)

    RayCluster(_spec(), FakeOsmoClient()).wait_until_ready(
        "wf-123",
        timeout_seconds=1,
        interval_seconds=0.01,
    )


def test_wait_until_ready_fails_on_terminal_state() -> None:
    with pytest.raises(RuntimeError, match="FAILED_IMAGE_PULL"):
        RayCluster(_spec(), FakeOsmoClient(status="FAILED_IMAGE_PULL")).wait_until_ready(
            "wf-123",
            timeout_seconds=1,
            interval_seconds=0.01,
        )


def test_bundled_template_is_available_as_package_data() -> None:
    template = resources.files("groot.infra.job").joinpath("templates/ray_cluster.yaml")
    assert template.is_file()
    text = template.read_text()
    assert "ray start --head" in text
    assert "workdir_path" in text
    assert "osmo data download" in text
    assert "uv pip install" in text
    assert '".[training]"' in text
    assert "aws_credential_name" in text


@pytest.mark.parametrize(
    "overrides",
    [
        {"cluster_name": "Bad_Name"},
        {"cluster_name": "bad-"},
        {"pool": ""},
        {"num_nodes": 0},
        {"image": ""},
        {"gpu_type": "private-pool-name"},
        {"ray_port": 0},
        {"ray_dashboard_port": 70000},
        {"gpus": -1},
    ],
)
def test_invalid_specs_fail_fast(overrides) -> None:
    with pytest.raises(ValueError, match="must|non-empty|between"):
        RayCluster(_spec(**overrides), FakeOsmoClient()).render_config()
