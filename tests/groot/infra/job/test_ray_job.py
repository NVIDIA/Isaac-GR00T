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
"""Tests for public Ray job helpers."""

from __future__ import annotations

from typing import ClassVar

from groot.infra.job import STANDARD_PIP_PACKAGES, RayJob, RayJobSpec, RuntimeDeps
from groot.infra.job.ray_job import _DEFAULT_WORKING_DIR_EXCLUDES


class FakeJobSubmissionClient:
    instances: ClassVar[list[FakeJobSubmissionClient]] = []
    statuses: ClassVar[list[str]] = ["RUNNING", "SUCCEEDED"]

    def __init__(self, address: str) -> None:
        self.address = address
        self.submissions = []
        self.tailed = []
        FakeJobSubmissionClient.instances.append(self)

    def submit_job(self, **kwargs) -> str:
        self.submissions.append(kwargs)
        return "job-123"

    async def tail_job_logs(self, submission_id: str):
        self.tailed.append(submission_id)
        yield "logs\n"

    def get_job_status(self, _submission_id: str) -> str:
        return FakeJobSubmissionClient.statuses.pop(0)

    def get_job_logs(self, submission_id: str) -> str:
        return f"final logs for {submission_id}\n"

    def get_job_info(self, submission_id: str) -> str:
        return f"info for {submission_id}"


class FakePortForward:
    instances: ClassVar[list[FakePortForward]] = []

    def __init__(self, workflow_id: str, dashboard_port: int) -> None:
        self.workflow_id = workflow_id
        self.dashboard_port = dashboard_port
        self.local_port = 9876
        self.started = False
        self.stopped = False
        FakePortForward.instances.append(self)

    @classmethod
    def init_for_dashboard(cls, workflow_id: str, dashboard_port: int) -> FakePortForward:
        return cls(workflow_id, dashboard_port)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeRayDashboardProbe:
    def __init__(self, local_port: int, timeout_seconds: float) -> None:
        self.local_port = local_port
        self.timeout_seconds = timeout_seconds

    def is_ready(self) -> tuple[bool, str]:
        return True, ""


def _install_fake_ray(monkeypatch) -> None:
    monkeypatch.setattr("groot.infra.job.ray_job.JobSubmissionClient", FakeJobSubmissionClient)
    monkeypatch.setattr("groot.infra.job.ray_job.PortForward", FakePortForward)
    monkeypatch.setattr("groot.infra.job.ray_job.RayDashboardProbe", FakeRayDashboardProbe)
    FakeJobSubmissionClient.instances = []
    FakeJobSubmissionClient.statuses = ["RUNNING", "SUCCEEDED"]
    FakePortForward.instances = []


def test_ray_job_submit_forwards_runtime_env(monkeypatch) -> None:
    _install_fake_ray(monkeypatch)

    result = RayJob(
        RayJobSpec(
            cluster_id="wf-123",
            job_name="job-name",
            entrypoint="python train.py",
            env_vars={"A": "B"},
            pip_packages=("pkg==1",),
            working_dir=".",
        )
    ).submit()

    assert result.submission_id == "job-123"
    client = FakeJobSubmissionClient.instances[0]
    assert client.address == "http://127.0.0.1:9876"
    assert client.submissions == [
        {
            "entrypoint": "python train.py",
            "submission_id": "job-name",
            "runtime_env": {
                "env_vars": {"A": "B"},
                "pip": ["pkg==1"],
                "working_dir": ".",
                "excludes": list(_DEFAULT_WORKING_DIR_EXCLUDES),
            },
        }
    ]
    port_forward = FakePortForward.instances[0]
    assert port_forward.workflow_id == "wf-123"
    assert port_forward.started is True
    assert port_forward.stopped is True


def test_ray_job_tail_and_wait_until_terminal(monkeypatch, tmp_path) -> None:
    _install_fake_ray(monkeypatch)
    monkeypatch.setattr("groot.infra.job.ray_job.time.sleep", lambda _seconds: None)

    job = RayJob(RayJobSpec(cluster_id="wf-123", job_name="job", entrypoint="python eval.py"))
    log = tmp_path / "ray.log"
    with log.open("w") as output:
        job.tail_logs("job-123", output=output)

    assert log.read_text() == "logs\n"
    assert job.logs("job-123") == "final logs for job-123\n"
    assert job.info("job-123") == "'info for job-123'"
    assert job.wait_until_terminal("job-123", poll_interval_seconds=0.01) == "SUCCEEDED"


def test_ray_job_wait_raises_on_failed_status(monkeypatch) -> None:
    _install_fake_ray(monkeypatch)
    FakeJobSubmissionClient.statuses = ["FAILED"]

    job = RayJob(RayJobSpec(cluster_id="wf-123", job_name="job", entrypoint="python eval.py"))

    import pytest

    with pytest.raises(RuntimeError, match="FAILED"):
        job.wait_until_terminal("job-123", poll_interval_seconds=0.01)


def test_runtime_dependency_helpers_are_exported_from_infra_job() -> None:
    assert "wandb" in STANDARD_PIP_PACKAGES
    assert "transformers==4.57.6" in RuntimeDeps(model="n2", cluster_type="gb300").get_pip_packages()
    assert "transformers==4.57.6" in RuntimeDeps(
        model="n2",
        cluster_type="h100",
        package_overrides={"transformers": "4.57.6"},
    ).get_pip_packages()
