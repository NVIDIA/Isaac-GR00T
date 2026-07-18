# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed failure raised by an OSMO CLI subprocess."""

from __future__ import annotations


class OsmoCommandError(RuntimeError):
    """Preserve command output and exit status for retry decisions."""

    def __init__(
        self,
        command: tuple[str, ...],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(command, returncode, stdout, stderr)

    @property
    def is_temporary(self) -> bool:
        return self.returncode == 75

    @property
    def is_rate_limited(self) -> bool:
        output = f"{self.stdout}\n{self.stderr}".lower()
        return self.is_temporary and "429" in output

    def __str__(self) -> str:
        return (
            f"Command failed with exit code {self.returncode}: {' '.join(self.command)}\n"
            f"stdout:\n{self.stdout}\n"
            f"stderr:\n{self.stderr}"
        )
