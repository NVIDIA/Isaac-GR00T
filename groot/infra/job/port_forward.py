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

"""OSMO workflow port forwarding."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import subprocess


@dataclass
class PortForward:
    """A local OSMO workflow port-forward process."""

    workflow_id: str
    local_port: int
    dashboard_port: int
    process: subprocess.Popen[str] | None = None

    @classmethod
    def init_for_dashboard(cls, workflow_id: str, dashboard_port: int) -> PortForward:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            local_port = int(sock.getsockname()[1])
        return cls(
            workflow_id=workflow_id,
            local_port=local_port,
            dashboard_port=dashboard_port,
        )

    def start(self) -> None:
        self.process = subprocess.Popen(
            [
                "osmo",
                "workflow",
                "port-forward",
                self.workflow_id,
                "master",
                "--port",
                f"{self.local_port}:{self.dashboard_port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
