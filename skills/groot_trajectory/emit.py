# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Write a trajectory record to disk (file hand-off), optionally upload to xagents.

Writing is unconditional and offline: every finished run leaves its record next to its
other output. Uploading is opt-in through environment variables so a public checkout never
reaches for an internal service. An upload failure is logged, never raised: the agent's
real work is already finished and the record is already on disk.

``safe_export_record`` is the shared producer hook: both AutoResearch and AutoDebug build
an agent-specific record, then call this so a mapping bug cannot undo a finished run.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
from urllib import error, request

from skills.groot_trajectory.record import TrajectoryRecord, now_utc

TRAJECTORY_FILENAME = "groot_trajectory.json"

TRAJECTORY_URL_ENV = "XAGENTS_TRAJECTORY_URL"
TRAJECTORY_TOKEN_ENV = "XAGENTS_TRAJECTORY_TOKEN"

UPLOAD_TIMEOUT_S = 30

logger = logging.getLogger(__name__)


def write_record(record: TrajectoryRecord, directory: Path) -> Path:
    """Atomically write the record into ``directory`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / TRAJECTORY_FILENAME
    payload = json.dumps(record.to_dict(), indent=4, sort_keys=True, allow_nan=False)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=directory,
        prefix=f".{TRAJECTORY_FILENAME}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def upload_record(record: TrajectoryRecord) -> bool:
    """Post the record when both URL and token env vars are set; else no-op."""
    url = os.environ.get(TRAJECTORY_URL_ENV, "").strip()
    token = os.environ.get(TRAJECTORY_TOKEN_ENV, "").strip()
    if not url or not token:
        return False

    payload = json.dumps(record.to_dict(), allow_nan=False).encode("utf-8")
    post = request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": record.run_id,
        },
    )
    with request.urlopen(post, timeout=UPLOAD_TIMEOUT_S) as response:
        logger.info(
            "uploaded trajectory for run %s to %s (HTTP %s)", record.run_id, url, response.status
        )
    return True


def export_record(record: TrajectoryRecord, directory: Path) -> Path:
    """Write the record, then upload if configured; return the written path."""
    destination = write_record(record, directory)
    try:
        upload_record(record)
    except (error.URLError, OSError, ValueError) as export_error:
        logger.warning(
            "trajectory upload failed for run %s; the record remains at %s: %s",
            record.run_id,
            destination,
            export_error,
        )
    return destination


def started_at_from_path(path: Path) -> datetime:
    """Best-effort start time from a durable file's ctime (timezone-aware UTC).

    Agent ledgers do not yet persist true start timestamps; this is the least-wrong
    proxy for a single-invocation run. Resumed runs report the final leg.
    """
    try:
        return datetime.fromtimestamp(path.stat().st_ctime, tz=timezone.utc)
    except OSError:
        return now_utc()


def safe_export_record(
    build: Callable[[], TrajectoryRecord],
    directory: Path,
    *,
    run_label: str,
    log: logging.Logger | None = None,
) -> Path | None:
    """Build and export a record; log and return ``None`` on mapping/IO errors.

    Producers call this after their real work is finished so a reporting failure cannot
    discard a completed study or incident.
    """
    active_log = log or logger
    try:
        return export_record(build(), directory)
    except (OSError, TypeError, ValueError) as exc:
        active_log.warning("trajectory export skipped for %s: %s", run_label, exc)
        return None
