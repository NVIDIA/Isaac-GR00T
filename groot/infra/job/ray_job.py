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
"""Ray job submission helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import time
from typing import Any, TextIO

from ray.job_submission import JobSubmissionClient

from groot.infra.job.constants import (
    DEFAULT_RAY_DASHBOARD_PORT,
    DEFAULT_RAY_DASHBOARD_PROBE_TIMEOUT_SECONDS,
)
from groot.infra.job.port_forward import PortForward
from groot.infra.job.ray_dashboard_probe import RayDashboardProbe

_SUCCEEDED_STATUSES = {"SUCCEEDED"}
_FAILED_STATUSES = {"FAILED", "STOPPED"}
_DEFAULT_WORKING_DIR_EXCLUDES = (
    ".git/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".venv/**",
    "__pycache__/**",
    "*.egg-info/**",
    "build/**",
    "dist/**",
)


@dataclass(frozen=True)
class RayJobSpec:
    """Configuration for one Ray job submitted to an existing cluster."""

    cluster_id: str
    job_name: str
    entrypoint: str
    env_vars: Mapping[str, str] = field(default_factory=dict)
    pip_packages: Sequence[str] = field(default_factory=tuple)
    working_dir: str | None = "."
    excludes: Sequence[str] = _DEFAULT_WORKING_DIR_EXCLUDES
    dashboard_port: int = DEFAULT_RAY_DASHBOARD_PORT

    def runtime_env(self) -> dict[str, object]:
        env: dict[str, object] = {"env_vars": dict(self.env_vars)}
        if self.pip_packages:
            env["pip"] = list(self.pip_packages)
        if self.working_dir:
            env["working_dir"] = self.working_dir
        if self.excludes:
            env["excludes"] = list(self.excludes)
        return env


@dataclass(frozen=True)
class RayJobSubmissionResult:
    """Result returned after a Ray job is accepted by the dashboard."""

    cluster_id: str
    submission_id: str
    followup_commands: tuple[str, ...]


@dataclass(frozen=True)
class RayJob:
    """Submit and monitor Ray jobs through Ray's public Jobs API."""

    spec: RayJobSpec

    def submit(self) -> RayJobSubmissionResult:
        submission_id = self._with_client(
            lambda client: client.submit_job(
                entrypoint=self.spec.entrypoint,
                submission_id=self.spec.job_name,
                runtime_env=self.spec.runtime_env(),
            )
        )
        return RayJobSubmissionResult(
            cluster_id=self.spec.cluster_id,
            submission_id=str(submission_id),
            followup_commands=(
                "osmo workflow port-forward "
                f"{self.spec.cluster_id} master --port 8265:{self.spec.dashboard_port}",
                f"ray job status --address http://127.0.0.1:8265 {submission_id}",
                f"ray job logs --address http://127.0.0.1:8265 {submission_id}",
            ),
        )

    def tail_logs(self, submission_id: str, output: TextIO | None = None) -> None:
        self._with_client(
            lambda client: asyncio.run(self._tail_logs(client, submission_id, output))
        )

    def logs(self, submission_id: str) -> str:
        return str(self._with_client(lambda client: client.get_job_logs(submission_id)))

    def info(self, submission_id: str) -> str:
        return repr(self._with_client(lambda client: client.get_job_info(submission_id)))

    def status(self, submission_id: str) -> str:
        status = self._with_client(lambda client: client.get_job_status(submission_id))
        name = getattr(status, "name", "")
        if isinstance(name, str) and name:
            return name
        return str(status).rsplit(".", 1)[-1]

    def wait_until_terminal(
        self,
        submission_id: str,
        *,
        poll_interval_seconds: float = 30.0,
        timeout_seconds: float | None = None,
    ) -> str:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than 0")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        while True:
            status = self.status(submission_id)
            if status in _SUCCEEDED_STATUSES:
                return status
            if status in _FAILED_STATUSES:
                raise RuntimeError(f"Ray job {submission_id} reached terminal status {status}")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for Ray job {submission_id} to finish "
                    f"(last status: {status})"
            )
            time.sleep(poll_interval_seconds)

    async def _tail_logs(
        self,
        client: JobSubmissionClient,
        submission_id: str,
        output: TextIO | None,
    ) -> None:
        async for chunk in client.tail_job_logs(submission_id):
            if output is None:
                print(chunk, end="")  # noqa: T201
            else:
                output.write(chunk)

    def _with_client(self, operation: Callable[[JobSubmissionClient], Any]) -> Any:
        port_forward = PortForward.init_for_dashboard(
            self.spec.cluster_id,
            dashboard_port=self.spec.dashboard_port,
        )
        port_forward.start()
        try:
            ready, last_error = RayDashboardProbe(
                local_port=port_forward.local_port,
                timeout_seconds=DEFAULT_RAY_DASHBOARD_PROBE_TIMEOUT_SECONDS,
            ).is_ready()
            if not ready:
                raise ConnectionError(
                    "Ray dashboard port-forward did not become ready. "
                    f"Last readiness error: {last_error}"
                )
            return operation(
                JobSubmissionClient(address=f"http://127.0.0.1:{port_forward.local_port}")
            )
        finally:
            port_forward.stop()
