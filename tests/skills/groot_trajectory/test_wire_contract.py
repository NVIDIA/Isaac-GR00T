# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests that this writer's field sets are still the consumer's."""

from __future__ import annotations

from datetime import datetime, timezone

from skills.groot_trajectory.record import RESEARCH_AGENT, TrajectoryRecord, TrajectoryStep
from skills.groot_trajectory.wire_contract import schema_violations

STARTED = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)


def _full_record() -> TrajectoryRecord:
    """Populate every field, so an omission is a violation rather than a default."""
    return TrajectoryRecord(
        agent_type=RESEARCH_AGENT,
        run_id="yam-smoke",
        task="minimize results.mse",
        inputs={"base_commit": "a" * 40, "eval_metric": "results.mse"},
        started_at=STARTED,
        ended_at=STARTED,
        terminal_status="budget_exhausted",
        steps=(
            TrajectoryStep(
                phase="hypothesize",
                summary="lower the learning rate",
                status="ok",
                unit_id="yam-smoke_wave0_trial1",
                parent_unit_id=None,
                inputs={"parent_score": 0.5},
                outputs={"idea_type": "hyperparameter"},
                evidence_refs=("yam-smoke_wave0_trial1/train.stdout.log",),
                timestamp=STARTED,
            ),
        ),
    )


def test_a_fully_populated_record_is_one_the_consumer_accepts():
    assert schema_violations(_full_record().to_dict()) == []


def test_a_field_the_consumer_removed_is_caught_here_instead_of_in_production():
    # This is the step_id regression: the consumer forbids unknown fields, so a key this
    # side keeps after the other side drops it rejects the whole record, every step at
    # once, and says so for the first time on a real run.
    data = _full_record().to_dict()
    for index, step in enumerate(data["steps"]):
        step["step_id"] = index

    assert schema_violations(data) == ["steps[0]: 'step_id' is not a field the consumer accepts"]


def test_a_required_field_this_side_stops_sending_is_caught_too():
    data = _full_record().to_dict()
    del data["run_id"]
    del data["steps"][0]["summary"]

    assert schema_violations(data) == [
        "record: 'run_id' is required and missing",
        "steps[0]: 'summary' is required and missing",
    ]
