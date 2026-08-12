# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Durable PDX outbox for dashboard writes that cannot go through HTTP.

Failed fire-and-forget dashboard calls are persisted as JSON files under:

    {root}/{experiment_ref}/{training_run_id}/events/{event}_{epoch_ns}.json

Default ``root`` is ``msc://GearHome/dashboard/expids`` and can be
overridden with ``GEAR_DASHBOARD_OUTBOX_URL``. The outbox writer never
raises; failures are logged and the caller continues.
"""

from __future__ import annotations

from datetime import datetime, timezone
import getpass
import json
import logging
import os
import re
import socket
import time
from typing import Any
import uuid

import multistorageclient as msc

logger = logging.getLogger(__name__)

OUTBOX_URL_ENV = "GEAR_DASHBOARD_OUTBOX_URL"
DEFAULT_OUTBOX_URL = "msc://GearHome/dashboard/expids"
SCHEMA_VERSION = 1


def _outbox_root() -> str:
    """Resolve the outbox bucket URL (env override -> default)."""
    return os.environ.get(OUTBOX_URL_ENV, "").strip() or DEFAULT_OUTBOX_URL


def _safe_segment(s: str) -> str:
    """Make ``s`` safe to use as a path segment."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._-")
    return cleaned or "unknown"


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return ""


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:  # noqa: BLE001
        return ""


def _filename(event: str) -> str:
    """Return a collision-resistant event filename."""
    return f"{_safe_segment(event)}_{time.time_ns()}.json"


def write_event(
    *,
    event: str,
    verb: str,
    url: str,
    body: Any,
    experiment_id: str | None = None,
    experiment_ref: str | None = None,
    training_run_id: str | None = None,
) -> bool:
    """Persist one dashboard call to the PDX outbox.

    Returns True on success, False on any failure. Never raises.

    Args:
        event: Domain-specific event name (e.g. ``training_run_status_update``).
            Surfaces in the filename and at the top of the JSON for human
            inspection / poller-side filtering.
        verb: HTTP method the poller will replay (``POST`` or ``PATCH``).
        url: Full URL the live HTTP call targeted. The poller can replay
            against this URL directly, or rewrite the host first if it
            wants to centralize against its local dashboard.
        body: JSON-serializable payload that would have been sent.
        experiment_id: UUID. Falls back to env ``GEAR_EXPERIMENT_ID``.
        experiment_ref: 5-digit human-friendly ref (e.g. ``"00005"``).
            Falls back to env ``GEAR_EXPERIMENT_REF``. Used as the
            directory name in the outbox path so the bucket is human-
            browseable; falls back to ``experiment_id`` if absent.
        training_run_id: Same env-fallback pattern as experiment_id,
            via ``GEAR_TRAINING_RUN_ID``.
    """
    exp_id = experiment_id or os.environ.get("GEAR_EXPERIMENT_ID", "")
    exp_ref = experiment_ref or os.environ.get("GEAR_EXPERIMENT_REF", "")
    run_id = training_run_id or os.environ.get("GEAR_TRAINING_RUN_ID", "")

    # Prefer the human-readable ref, then UUID, then a stable sentinel.
    exp_scope = exp_ref or exp_id or "no-experiment"
    run_scope = run_id or "no-run"

    target = (
        f"{_outbox_root().rstrip('/')}"
        f"/{_safe_segment(exp_scope)}"
        f"/{_safe_segment(run_scope)}"
        f"/events"
        f"/{_filename(event)}"
    )

    payload: dict[str, Any] = {
        "event_id": str(uuid.uuid4()),
        "schema_version": SCHEMA_VERSION,
        "event": event,
        "experiment_id": exp_id or None,
        "experiment_ref": exp_ref or None,
        "training_run_id": run_id or None,
        "verb": verb,
        "url": url,
        "body": body,
        "emitted_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "submitter": os.environ.get("USER", "") or _safe_user() or None,
            "node": _hostname() or None,
            "groot_commit": os.environ.get("GROOT_COMMIT_HASH", "") or None,
        },
    }

    try:
        msc.write(  # type: ignore[union-attr]
            url=target,
            body=json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "GEAR Dashboard event-outbox write to %s failed (non-fatal): %s",
            target,
            exc,
        )
        return False

    logger.info("GEAR Dashboard event-outbox: wrote %s -> %s", event, target)
    return True
