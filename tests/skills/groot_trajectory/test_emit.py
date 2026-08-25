# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for writing and shipping a trajectory record."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace
from urllib import error, request

import pytest
from skills.groot_trajectory.emit import (
    TRAJECTORY_FILENAME,
    TRAJECTORY_TOKEN_ENV,
    TRAJECTORY_URL_ENV,
    export_record,
    upload_record,
    write_record,
)
from skills.groot_trajectory.record import RESEARCH_AGENT, TrajectoryRecord

STARTED = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)


def _record(run_id: str = "yam-smoke") -> TrajectoryRecord:
    """Build a minimal valid record."""
    return TrajectoryRecord(
        agent_type=RESEARCH_AGENT,
        run_id=run_id,
        task="minimize results.mse",
        started_at=STARTED,
    )


def test_the_record_lands_as_readable_json(tmp_path: Path):
    destination = write_record(_record(), tmp_path / "study")

    assert destination.name == TRAJECTORY_FILENAME
    assert json.loads(destination.read_text(encoding="utf-8"))["run_id"] == "yam-smoke"
    # A leftover temporary file would be picked up by anything scanning the output dir.
    assert [path.name for path in destination.parent.iterdir()] == [TRAJECTORY_FILENAME]


def test_a_failed_write_leaves_the_previous_record_intact(tmp_path: Path, monkeypatch):
    write_record(_record("first"), tmp_path)

    def _fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _fail)
    with pytest.raises(OSError, match="disk full"):
        write_record(_record("second"), tmp_path)

    destination = tmp_path / TRAJECTORY_FILENAME
    assert json.loads(destination.read_text(encoding="utf-8"))["run_id"] == "first"
    assert [path.name for path in tmp_path.iterdir()] == [TRAJECTORY_FILENAME]


def test_an_unconfigured_checkout_never_reaches_for_the_network(monkeypatch):
    monkeypatch.delenv(TRAJECTORY_URL_ENV, raising=False)
    monkeypatch.delenv(TRAJECTORY_TOKEN_ENV, raising=False)

    assert upload_record(_record()) is False


def test_a_url_without_a_token_is_not_an_anonymous_upload(monkeypatch):
    monkeypatch.setenv(TRAJECTORY_URL_ENV, "https://xagents.example/gr00t-trajectories")
    monkeypatch.delenv(TRAJECTORY_TOKEN_ENV, raising=False)

    assert upload_record(_record()) is False


def test_a_configured_upload_identifies_the_run_it_repeats(monkeypatch):
    monkeypatch.setenv(TRAJECTORY_URL_ENV, "https://xagents.example/gr00t-trajectories")
    monkeypatch.setenv(TRAJECTORY_TOKEN_ENV, "secret")
    sent: list[request.Request] = []
    monkeypatch.setattr(request, "urlopen", _capture(sent))

    assert upload_record(_record()) is True

    posted = sent[0]
    # Resuming a study re-sends the same run, so the key has to be the run's identity
    # rather than the attempt's, or a resumed study lands twice.
    assert posted.get_header("Idempotency-key") == "yam-smoke"
    assert posted.get_header("Authorization") == "Bearer secret"
    assert json.loads(posted.data)["run_id"] == "yam-smoke"


def test_an_upload_failure_does_not_lose_the_finished_study(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(TRAJECTORY_URL_ENV, "https://xagents.example/gr00t-trajectories")
    monkeypatch.setenv(TRAJECTORY_TOKEN_ENV, "secret")

    def _refuse(*_args, **_kwargs):
        raise error.URLError("connection refused")

    monkeypatch.setattr(request, "urlopen", _refuse)

    destination = export_record(_record(), tmp_path)

    assert json.loads(destination.read_text(encoding="utf-8"))["run_id"] == "yam-smoke"


def test_safe_export_swallows_mapping_errors(tmp_path: Path):
    from skills.groot_trajectory.emit import safe_export_record

    def _build():
        raise ValueError("missing change")

    assert safe_export_record(_build, tmp_path, run_label="broken") is None
    assert list(tmp_path.iterdir()) == []


def test_safe_export_writes_on_success(tmp_path: Path):
    from skills.groot_trajectory.emit import safe_export_record

    path = safe_export_record(lambda: _record(), tmp_path, run_label="yam-smoke")
    assert path is not None
    assert path.name == TRAJECTORY_FILENAME


def _capture(sent: list):
    """Build a urlopen stand-in that records the request instead of sending it."""

    @contextmanager
    def _urlopen(post, **_kwargs):
        sent.append(post)
        yield SimpleNamespace(status=202)

    return _urlopen
