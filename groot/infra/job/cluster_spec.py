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

"""Ray cluster launch specification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
import re

from groot.infra.job.constants import (
    DEFAULT_CPUS,
    DEFAULT_GPUS_BY_TYPE,
    DEFAULT_IMAGE,
    DEFAULT_MEMORY_GIB,
    DEFAULT_RAY_DASHBOARD_PORT,
    DEFAULT_RAY_PORT,
    DEFAULT_STORAGE_GIB,
    DEFAULT_WORKDIR,
)
from groot.infra.job.workflow_overrides import WorkflowOverrides

_CLUSTER_NAME_PATTERN = re.compile(r"^[a-z]([a-z0-9-]*[a-z0-9])?$")


@dataclass(frozen=True)
class ClusterSpec:
    """Configuration for an OSMO workflow that runs a Ray cluster."""

    cluster_name: str
    pool: str
    gpu_type: str | None = None
    num_nodes: int = 1
    image: str = DEFAULT_IMAGE
    template: str | Path | None = None
    priority: str = "high"
    workdir: str = DEFAULT_WORKDIR
    workdir_path: str = ""
    user_command: str = ""
    runtime_env_json: str = "{}"
    exec_timeout: str = "7d"
    queue_timeout: str = "1d"
    ray_port: int = DEFAULT_RAY_PORT
    ray_dashboard_port: int = DEFAULT_RAY_DASHBOARD_PORT
    cpus: int | None = None
    gpus: int | None = None
    memory_gib: int | None = None
    storage_gib: int | None = None
    extra_overrides: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not _CLUSTER_NAME_PATTERN.match(self.cluster_name):
            raise ValueError(
                "cluster_name must match [a-z]([-a-z0-9]*[a-z0-9])? "
                "(lowercase letters, numbers, and dashes)"
            )
        if not self.pool.strip():
            raise ValueError("pool must be non-empty")
        if self.gpu_type is not None and self.gpu_type.lower() not in DEFAULT_GPUS_BY_TYPE:
            valid = ", ".join(sorted(DEFAULT_GPUS_BY_TYPE))
            raise ValueError(f"gpu_type must be one of: {valid}")
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be greater than 0")
        if not self.image.strip():
            raise ValueError("image must be non-empty")
        if not self.priority.strip():
            raise ValueError("priority must be non-empty")
        if not self.exec_timeout.strip():
            raise ValueError("exec_timeout must be non-empty")
        if not self.queue_timeout.strip():
            raise ValueError("queue_timeout must be non-empty")
        for field_name in ("cpus", "gpus", "memory_gib", "storage_gib"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be greater than or equal to 0")
        for name, port in (
            ("ray_port", self.ray_port),
            ("ray_dashboard_port", self.ray_dashboard_port),
        ):
            if port <= 0 or port > 65535:
                raise ValueError(f"{name} must be between 1 and 65535")

    def to_workflow_overrides(self) -> WorkflowOverrides:
        self.validate()
        values = (
            ("workflow_name", self.cluster_name),
            ("num_nodes", str(self.num_nodes)),
            ("image", self.image),
            ("workdir", os.path.abspath(self.workdir)),
            ("workdir_path", self.workdir_path),
            ("user_command", self.get_normalized_user_command()),
            ("runtime_env_json", self.runtime_env_json),
            ("exec_timeout", self.exec_timeout),
            ("queue_timeout", self.queue_timeout),
            ("ray_port", str(self.ray_port)),
            ("ray_dashboard_port", str(self.ray_dashboard_port)),
            ("cpus", str(self.get_cpus())),
            ("gpus", str(self.get_gpus())),
            ("memory", str(self.get_memory_gib())),
            ("storage", str(self.get_storage_gib())),
            *((str(key), str(value)) for key, value in self.extra_overrides.items()),
        )
        return WorkflowOverrides(values=values)

    def get_cpus(self) -> int:
        return self.cpus if self.cpus is not None else DEFAULT_CPUS

    def get_gpus(self) -> int:
        if self.gpus is not None:
            return self.gpus
        if self.gpu_type is None:
            return 0
        return DEFAULT_GPUS_BY_TYPE[self.gpu_type.lower()]

    def get_memory_gib(self) -> int:
        return self.memory_gib if self.memory_gib is not None else DEFAULT_MEMORY_GIB

    def get_storage_gib(self) -> int:
        return self.storage_gib if self.storage_gib is not None else DEFAULT_STORAGE_GIB

    def get_normalized_user_command(self) -> str:
        normalized_lines = []
        for raw_line in self.user_command.splitlines():
            line = raw_line.strip()
            if line.endswith("\\"):
                normalized_lines.append(line[:-1].strip())
            else:
                normalized_lines.append(line)
        return " ".join(line for line in normalized_lines if line).strip()
