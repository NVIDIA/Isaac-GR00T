# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Local and SSH execution targets for AutoDebug probes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import shlex
from typing import Any


@dataclass(frozen=True)
class ExecutionTarget:
    name: str
    kind: str
    workspace: str
    host: str | None = None

    @classmethod
    def init_from_data(
        cls,
        name: str,
        data: dict[str, Any],
        *,
        config_directory: Path,
    ) -> ExecutionTarget:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
            raise ValueError(f"Invalid execution target name: {name!r}")
        kind = data.get("kind")
        workspace = data.get("workspace")
        if kind not in {"local", "ssh"}:
            raise ValueError(f"Execution target {name!r} must use kind 'local' or 'ssh'")
        if not isinstance(workspace, str) or not workspace.strip():
            raise ValueError(f"Execution target {name!r} needs a workspace")
        host = data.get("host")
        if kind == "local":
            if host is not None:
                raise ValueError(f"Local execution target {name!r} cannot set host")
            workspace_path = Path(workspace).expanduser()
            if not workspace_path.is_absolute():
                workspace_path = config_directory / workspace_path
            workspace = str(workspace_path.resolve())
        else:
            if not isinstance(host, str) or not host.strip() or host.startswith("-"):
                raise ValueError(f"SSH execution target {name!r} needs a valid host or SSH alias")
            if not PurePosixPath(workspace).is_absolute():
                raise ValueError(f"SSH execution target {name!r} needs an absolute workspace")
        return cls(name=name, kind=kind, workspace=workspace, host=host)

    def to_dict(self) -> dict[str, str]:
        data = {"kind": self.kind, "workspace": self.workspace}
        if self.host is not None:
            data["host"] = self.host
        return data

    def command_argv(self, command: str) -> tuple[list[str], Path | None]:
        argv = shlex.split(command)
        if not argv:
            raise ValueError("Command cannot be empty")
        if self.kind == "local":
            return argv, Path(self.workspace)
        remote_command = f"cd -- {shlex.quote(self.workspace)} && exec {shlex.join(argv)}"
        return ["ssh", "-o", "BatchMode=yes", self.host or "", remote_command], None


def load_execution_targets(
    config_path: Path | None,
    *,
    default_workspace: Path,
) -> dict[str, dict[str, str]]:
    if config_path is None:
        target = ExecutionTarget(
            name="local",
            kind="local",
            workspace=str(default_workspace.resolve()),
        )
        return {target.name: target.to_dict()}

    config_path = config_path.expanduser().resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError(f"Execution target config must use version 1: {config_path}")
    raw_targets = data.get("targets")
    if not isinstance(raw_targets, dict) or not raw_targets:
        raise ValueError(f"Execution target config has no targets: {config_path}")
    targets: dict[str, dict[str, str]] = {}
    for name, raw_target in raw_targets.items():
        if not isinstance(name, str) or not isinstance(raw_target, dict):
            raise ValueError(f"Invalid execution target in {config_path}")  # noqa: TRY004
        target = ExecutionTarget.init_from_data(
            name,
            raw_target,
            config_directory=config_path.parent,
        )
        targets[name] = target.to_dict()
    return targets
