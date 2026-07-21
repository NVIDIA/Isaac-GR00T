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

"""Tests for ``python -m groot.infra.job`` CLI plumbing."""

from groot.infra.job.cluster_launch_result import ClusterLaunchResult
from groot.infra.job.main import start_cluster


def test_start_cluster_cli_builds_spec_and_prints_workflow_id(monkeypatch, capsys) -> None:
    calls = {}

    class FakeRayCluster:
        def __init__(self, spec):
            calls["spec"] = spec

        def launch(self, *, wait, readiness_timeout_seconds, readiness_interval_seconds):
            calls["wait"] = wait
            calls["readiness_timeout_seconds"] = readiness_timeout_seconds
            calls["readiness_interval_seconds"] = readiness_interval_seconds
            return ClusterLaunchResult(
                cluster_name=calls["spec"].cluster_name,
                workflow_id="wf-123",
                outputs=("Workflow ID - wf-123",),
                followup_commands=("osmo workflow query wf-123",),
            )

    monkeypatch.setattr(start_cluster, "RayCluster", FakeRayCluster)

    assert (
        start_cluster.main(
            [
                "start-cluster",
                "--cluster-name",
                "test-ray",
                "--pool",
                "example-pool",
                "--gpu-type",
                "gb200",
                "--num-nodes",
                "3",
                "--image",
                "example/ray:latest",
                "--priority",
                "high",
                "--gpus",
                "4",
                "--set",
                "extra=value",
                "--detach",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Cluster ID: wf-123" in output
    assert "Workflow ID: wf-123" in output
    assert calls["wait"] is False
    assert calls["spec"].cluster_name == "test-ray"
    assert calls["spec"].pool == "example-pool"
    assert calls["spec"].gpu_type == "gb200"
    assert calls["spec"].num_nodes == 3
    assert calls["spec"].image == "example/ray:latest"
    assert calls["spec"].priority == "high"
    assert calls["spec"].gpus == 4
    assert calls["spec"].extra_overrides == {"extra": "value"}
