# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Deterministic, sanitized benchmark report helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

from groot.core.policy.benchmark_contract import artifact_identity, write_json_atomic

REPORT_SCHEMA_VERSION = "groot-n2-benchmark-report-v1"
REPORT_FILENAME = "benchmark-report.json"
MAX_IDENTITY_BYTES = 64 * 1024 * 1024

_HOME_PATH = re.compile(r"/(?:home|Users)/[^/\s]+")
_INTERNAL_URL = re.compile(
    r"https?://[^\s]*(?:gitlab-master[.]nvidia[.]com|osmo[.]nvidia[.]com)[^\s]*",
    re.IGNORECASE,
)
_INTERNAL_LOCATION = re.compile(r"(?:s3|msc)://[^\s]+|/m[n]t(?:/[^\s]+)?", re.IGNORECASE)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?P<name>(?:api[_-]?key|auth|credential|password|secret|token))"
    r"(?P<separator>\s*[=:]\s*)[^\s,;]+"
)
_SECRET_VALUE = re.compile(
    r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|\b(?:glpat-|nvapi-)[A-Za-z0-9_-]{20,}\b|"
    r"\bgh[opurs]_[A-Za-z0-9_]{20,}\b"
)
_URI_CREDENTIALS = re.compile(r"(?P<scheme>https?://)[^\s/:@]+:[^\s/@]+@")
_SENSITIVE_OPTION = re.compile(
    r"^--?(?:api[-_]?key|auth|credential|password|secret|token)(?:$|[=-])",
    re.IGNORECASE,
)


def to_json_text(payload: Mapping[str, Any]) -> str:
    """Serialize a sanitized report with stable ordering and finite numbers."""
    return json.dumps(_public_json(payload), indent=2, allow_nan=False) + "\n"


def write_benchmark_report(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a deterministically ordered report through the shared atomic writer."""
    write_json_atomic(path, _public_json(payload))


def bounded_artifact_identity(path: Path, *, report_root: Path) -> dict[str, Any]:
    """Identify a retained metadata artifact while refusing unexpectedly large files."""
    resolved = path.expanduser().resolve()
    size = resolved.stat().st_size
    if size > MAX_IDENTITY_BYTES:
        raise ValueError(
            f"Benchmark metadata artifact exceeds {MAX_IDENTITY_BYTES} bytes: {resolved}"
        )
    identity = artifact_identity(resolved)
    identity["path"] = safe_path(resolved, report_root=report_root)
    return identity


def safe_path(path: Path, *, report_root: Path) -> str:
    """Return an output-relative or user-neutral path for a public report."""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(report_root.expanduser().resolve()))
    except ValueError:
        return sanitize_text(str(resolved))


def sanitize_command(command: Sequence[str]) -> str:
    """Render a shell command without credentials, usernames, or internal locations."""
    import shlex

    sanitized: list[str] = []
    redact_next = False
    for token in command:
        if redact_next:
            sanitized.append("[REDACTED]")
            redact_next = False
            continue
        if _SENSITIVE_OPTION.match(token):
            if "=" in token:
                sanitized.append(f"{token.split('=', maxsplit=1)[0]}=[REDACTED]")
            else:
                sanitized.append(token)
                redact_next = True
            continue
        sanitized.append(sanitize_text(token))
    return shlex.join(sanitized)


def sanitize_text(value: str) -> str:
    """Remove credential-like values, user homes, and non-public locations."""
    sanitized = _URI_CREDENTIALS.sub(r"\g<scheme>[REDACTED]@", value)
    sanitized = _INTERNAL_URL.sub("[REDACTED_INTERNAL_URL]", sanitized)
    sanitized = _INTERNAL_LOCATION.sub("[REDACTED_INTERNAL_LOCATION]", sanitized)
    sanitized = _SECRET_ASSIGNMENT.sub(
        lambda match: f"{match.group('name')}{match.group('separator')}[REDACTED]",
        sanitized,
    )
    sanitized = _SECRET_VALUE.sub("[REDACTED]", sanitized)
    current_home = str(Path.home())
    if current_home and current_home != "/":
        sanitized = sanitized.replace(current_home, "$HOME")
    return _HOME_PATH.sub("$HOME", sanitized)


def _public_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _public_json(value[key]) for key in sorted(value)}
    if isinstance(value, list | tuple):
        return [_public_json(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value
