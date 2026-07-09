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

"""OSMO-backed Ray cluster launcher."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from groot.infra.job.cluster_launch_result import ClusterLaunchResult
from groot.infra.job.cluster_spec import ClusterSpec
from groot.infra.job.constants import (
    DEFAULT_RAY_CLUSTER_READY_INTERVAL_SECONDS,
    DEFAULT_RAY_CLUSTER_READY_TIMEOUT_SECONDS,
    DEFAULT_RAY_DASHBOARD_PROBE_TIMEOUT_SECONDS,
    OSMO_WORKFLOW_TERMINAL_STATES,
)
from groot.infra.job.osmo_client import OsmoClient
from groot.infra.job.port_forward import PortForward
from groot.infra.job.ray_dashboard_probe import RayDashboardProbe
from groot.infra.job.workflow_config import WorkflowConfig
from groot.infra.job.workflow_template import WorkflowTemplate


@dataclass(frozen=True)
class RayCluster:
    """Ray cluster lifecycle operations."""

    spec: ClusterSpec
    osmo_client: OsmoClient = field(default_factory=OsmoClient)

    def render_config(self) -> WorkflowConfig:
        self.spec.validate()
        return self.osmo_client.render_workflow(
            WorkflowTemplate.init_from_reference(self.spec.template),
            self.spec.to_workflow_overrides(),
        )

    def launch(
        self,
        *,
        wait: bool = True,
        readiness_timeout_seconds: float = DEFAULT_RAY_CLUSTER_READY_TIMEOUT_SECONDS,
        readiness_interval_seconds: float = DEFAULT_RAY_CLUSTER_READY_INTERVAL_SECONDS,
    ) -> ClusterLaunchResult:
        self.spec.validate()
        output = self.osmo_client.submit_workflow(
            self.render_config(),
            pool=self.spec.pool,
            priority=self.spec.priority,
        )
        outputs = (output,)
        workflow_id = self.osmo_client.parse_workflow_id(outputs)
        if wait:
            self.wait_until_ready(
                workflow_id,
                timeout_seconds=readiness_timeout_seconds,
                interval_seconds=readiness_interval_seconds,
            )
        return ClusterLaunchResult(
            cluster_name=self.spec.cluster_name,
            workflow_id=workflow_id,
            outputs=outputs,
            followup_commands=(
                f"osmo workflow query {workflow_id}",
                f"osmo workflow logs {workflow_id} --task master -n 100",
                f"osmo workflow port-forward {workflow_id} master --port "
                f"8265:{self.spec.ray_dashboard_port}",
                f"osmo workflow cancel {workflow_id}",
            ),
        )

    def wait_until_ready(
        self,
        workflow_id: str,
        *,
        timeout_seconds: float = DEFAULT_RAY_CLUSTER_READY_TIMEOUT_SECONDS,
        interval_seconds: float = DEFAULT_RAY_CLUSTER_READY_INTERVAL_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than 0")

        deadline = time.monotonic() + timeout_seconds
        last_status = "UNKNOWN"
        last_error = ""

        while True:
            last_status = self.osmo_client.get_workflow_status(workflow_id)
            if last_status in OSMO_WORKFLOW_TERMINAL_STATES:
                raise RuntimeError(
                    f"Workflow {workflow_id} reached terminal state {last_status} "
                    "before Ray was ready"
                )
            if last_status == "RUNNING":
                port_forward = PortForward.init_for_dashboard(
                    workflow_id,
                    dashboard_port=self.spec.ray_dashboard_port,
                )
                port_forward.start()
                try:
                    ready, last_error = RayDashboardProbe(
                        local_port=port_forward.local_port,
                        timeout_seconds=min(
                            DEFAULT_RAY_DASHBOARD_PROBE_TIMEOUT_SECONDS,
                            max(deadline - time.monotonic(), 0.0),
                        ),
                    ).is_ready()
                    if ready:
                        return
                finally:
                    port_forward.stop()

            if time.monotonic() >= deadline:
                detail = f" Last readiness error: {last_error}" if last_error else ""
                raise TimeoutError(
                    f"Timed out after {timeout_seconds:.0f}s waiting for Ray cluster "
                    f"{workflow_id} (last workflow status: {last_status}).{detail}"
                )
            time.sleep(interval_seconds)
