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

"""OSMO CLI client."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import subprocess
import tempfile
import time

from groot.infra.job.osmo_command_error import OsmoCommandError
from groot.infra.job.workflow_config import WorkflowConfig
from groot.infra.job.workflow_overrides import WorkflowOverrides
from groot.infra.job.workflow_template import WorkflowTemplate

_WORKFLOW_ID_PATTERN = re.compile(r"Workflow ID\s*[-:]\s*(\S+)")
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OsmoClient:
    """Small wrapper around the OSMO CLI."""

    submit_retry_attempts: int = 3
    submit_retry_base_delay_seconds: float = 5.0
    workflow_status_timeout_seconds: float = 75.0

    def render_workflow(
        self,
        template: WorkflowTemplate,
        overrides: WorkflowOverrides,
    ) -> WorkflowConfig:
        output = self.run(
            [
                "osmo",
                "workflow",
                "submit",
                str(template.path),
                "--dry-run",
                *overrides.to_osmo_args(),
            ]
        )
        return WorkflowConfig.init_from_yaml_text(output)

    def submit_workflow(self, workflow_config: WorkflowConfig, *, pool: str, priority: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
            temp_file.write(workflow_config.to_yaml_text())
            temp_path = temp_file.name
        try:
            command = [
                "osmo",
                "workflow",
                "submit",
                temp_path,
                "--pool",
                pool,
                "--priority",
                priority,
            ]
            for attempt in range(self.submit_retry_attempts + 1):
                try:
                    return self.run(command)
                except RuntimeError as exc:
                    if attempt >= self.submit_retry_attempts or not self.is_transient_submit_error(
                        exc
                    ):
                        raise
                    delay_seconds = self.submit_retry_base_delay_seconds * (2**attempt)
                    _LOGGER.warning(
                        "OSMO workflow submit hit a transient error; "
                        "retrying attempt %s/%s in %.1fs.",
                        attempt + 2,
                        self.submit_retry_attempts + 1,
                        delay_seconds,
                    )
                    time.sleep(delay_seconds)
            raise RuntimeError("OSMO workflow submit retry loop exited unexpectedly")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def is_transient_submit_error(self, error: RuntimeError) -> bool:
        text = str(error).lower()
        transient_markers = (
            "status code: 502",
            "status code: 503",
            "status code: 504",
            "server responded with status code 502",
            "server responded with status code 503",
            "server responded with status code 504",
            "upstream connect error",
            "connection termination",
            "disconnect/reset before headers",
            "reset reason",
        )
        return any(marker in text for marker in transient_markers)

    def get_workflow_status(self, workflow_id: str) -> str:
        output = self.run(
            ["osmo", "workflow", "query", workflow_id],
            timeout_seconds=self.workflow_status_timeout_seconds,
        )
        for line in output.splitlines():
            if line.strip().startswith("Status"):
                return line.split(":", 1)[1].strip()
        raise ValueError(f"Could not parse workflow status from OSMO output:\n{output}")

    def parse_workflow_id(self, outputs: Sequence[str]) -> str:
        match = _WORKFLOW_ID_PATTERN.search("\n".join(outputs))
        if match:
            return match.group(1)
        raise ValueError("Could not parse workflow id from OSMO submit output")

    def run(self, command: Sequence[str], timeout_seconds: float | None = None) -> str:
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Command timed out after {timeout_seconds:g}s: {' '.join(command)}\n"
                f"stdout:\n{self.output_text(exc.stdout)}\n"
                f"stderr:\n{self.output_text(exc.stderr)}"
            ) from exc
        if process.returncode != 0:
            raise OsmoCommandError(
                command=tuple(command),
                returncode=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
            )
        return process.stdout

    def run_with_timeout(self, command: Sequence[str], timeout_seconds: float) -> str:
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return self.output_text(exc.stdout) + self.output_text(exc.stderr)
        if process.returncode != 0:
            raise OsmoCommandError(
                command=tuple(command),
                returncode=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
            )
        return process.stdout

    def output_text(self, output: str | bytes | None) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode(errors="replace")
        return output
