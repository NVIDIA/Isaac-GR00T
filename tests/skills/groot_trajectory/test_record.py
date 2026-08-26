# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for the shared trajectory record."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from skills.groot_trajectory.record import (
    DEBUG_AGENT,
    FAQ_AGENT,
    PHASE_VOCABULARY,
    RESEARCH_AGENT,
    SCHEMA_VERSION,
    TrajectoryRecord,
    TrajectoryStep,
)

STARTED = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)


def _record(**values) -> TrajectoryRecord:
    """Build a minimal valid record."""
    defaults = {
        "agent_type": RESEARCH_AGENT,
        "run_id": "yam-smoke",
        "task": "minimize results.mse",
        "started_at": STARTED,
    }
    return TrajectoryRecord(**{**defaults, **values})


def test_the_wire_shape_is_the_documented_contract():
    record = _record(
        ended_at=STARTED,
        terminal_status="budget_exhausted",
        steps=(TrajectoryStep(phase="observe", summary="mse=0.02"),),
    )

    data = record.to_dict()

    assert data["schema_version"] == SCHEMA_VERSION
    # The reader on the far side rejects an unknown key outright rather than dropping it,
    # so these two sets are the contract and not merely this writer's current output.
    assert set(data) == {
        "schema_version",
        "agent_type",
        "run_id",
        "task",
        "inputs",
        "started_at",
        "ended_at",
        "terminal_status",
        "steps",
    }
    assert set(data["steps"][0]) == {
        "phase",
        "summary",
        "status",
        "unit_id",
        "parent_unit_id",
        "inputs",
        "outputs",
        "evidence_refs",
        "timestamp",
    }


def test_the_phase_vocabularies_are_the_shared_ones():
    # These name the loops the far side places a step in. AutoDebug's `implementation` is
    # the wait while a separate agent applies the proposed fix, and it has to be here
    # rather than folded into a phase meaning "propose one".
    assert PHASE_VOCABULARY[RESEARCH_AGENT] == ("hypothesize", "execute", "observe", "decide")
    assert PHASE_VOCABULARY[DEBUG_AGENT] == (
        "intake",
        "reproduce",
        "diagnose",
        "implementation",
        "verify",
        "terminal",
    )
    assert PHASE_VOCABULARY[FAQ_AGENT] == ("intake", "retrieve", "read", "compose", "terminal")


def test_a_phase_outside_the_agent_vocabulary_is_rejected():
    # "reproduce" is a debug phase; a research run naming it would produce steps the
    # pipeline cannot place.
    with pytest.raises(ValueError, match="not in the groot_research vocabulary"):
        _record(steps=(TrajectoryStep(phase="reproduce", summary="ran it"),))

    _record(agent_type=DEBUG_AGENT, steps=(TrajectoryStep(phase="reproduce", summary="ran it"),))


def test_a_run_a_human_stopped_is_not_a_failure():
    # `cancelled` is its own outcome: reporting it as failed would teach the far side that
    # the work was wrong rather than abandoned.
    _record(terminal_status="cancelled")


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"agent_type": "groot_unknown"}, "unknown agent_type"),
        ({"terminal_status": "gave_up"}, "unknown terminal_status"),
        ({"run_id": "  "}, "run_id must not be empty"),
        # A naive datetime is the point of this case, so DTZ001 does not apply.
        ({"started_at": datetime(2026, 7, 1, 12, 0)}, "must be timezone-aware"),  # noqa: DTZ001
    ],
)
def test_invalid_records_are_rejected(values: dict, message: str):
    with pytest.raises(ValueError, match=message):
        _record(**values)


def test_record_inputs_must_be_a_mapping():
    with pytest.raises(TypeError, match="inputs must be a dict"):
        _record(inputs=["base_commit"])


def test_a_step_needs_a_phase_and_a_summary():
    with pytest.raises(ValueError, match="summary must not be empty"):
        TrajectoryStep(phase="observe", summary="   ")


def test_a_step_status_outside_the_vocabulary_is_rejected():
    # `status` is what separates a hypothesis rejected from a job that died, so a value
    # the far side cannot place would erase that distinction rather than degrade it.
    with pytest.raises(ValueError, match="unknown step status 'errored'"):
        TrajectoryStep(phase="observe", summary="mse=0.02", status="errored")


def test_a_unit_cannot_descend_from_one_no_earlier_step_declares():
    # Caught here so a mapping bug fails this repository's tests instead of the ingest on
    # the far side, which rejects the whole record for it.
    with pytest.raises(ValueError, match="descends from unit 'trial-0'"):
        _record(
            steps=(
                TrajectoryStep(
                    phase="hypothesize",
                    summary="a longer warmup",
                    unit_id="trial-1",
                    parent_unit_id="trial-0",
                ),
            )
        )


def test_a_unit_cannot_vouch_for_its_own_lineage():
    # Naming itself would satisfy a check made after its own declaration, and close a
    # cycle the far side would walk as ancestry.
    with pytest.raises(ValueError, match="descends from unit 'trial-1'"):
        _record(
            steps=(
                TrajectoryStep(
                    phase="hypothesize",
                    summary="a longer warmup",
                    unit_id="trial-1",
                    parent_unit_id="trial-1",
                ),
            )
        )


def test_a_declared_parent_is_accepted():
    record = _record(
        steps=(
            TrajectoryStep(phase="hypothesize", summary="baseline recipe", unit_id="trial-1"),
            TrajectoryStep(
                phase="hypothesize",
                summary="a longer warmup on top",
                unit_id="trial-2",
                parent_unit_id="trial-1",
            ),
        )
    )

    assert [step.unit_id for step in record.steps] == ["trial-1", "trial-2"]
