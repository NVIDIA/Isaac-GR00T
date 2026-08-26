# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The ``groot_agent_trajectory.v0`` record that GR00T agents emit about their own runs.

An AutoResearch study or an AutoDebug session is one agent run. Downstream, the xagents
knowledge pipeline distills those runs into a reviewed knowledge graph, and it reads one
normalized shape instead of carrying a separate reader per agent. This module is this
repository's implementation of that shape.

The field names, the per-agent phase vocabulary, and the terminal-status values are a
wire contract shared with a service outside this repository. They are spelled out here
rather than imported because these skills ship publicly and must not depend on an
internal service; the cost is that both copies have to move together. Treat this module
as a **verbatim mirror** of ``xagents/trajectory/models.py`` — any difference between
the two is a bug, not a local dialect, because the reader on the far side rejects a
record carrying a field it does not know rather than dropping it. Bump
``SCHEMA_VERSION`` when the shape itself changes.

The contract for what each agent must *populate* is agentics/xagents#18 §3.1; this
module only enforces what every record must be shaped like.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "groot_agent_trajectory.v0"

RESEARCH_AGENT = "groot_research"
DEBUG_AGENT = "groot_debug"
FAQ_AGENT = "groot_faq"

COMPLETED = "completed"
FAILED = "failed"
BUDGET_EXHAUSTED = "budget_exhausted"
AWAITING_EVIDENCE = "awaiting_evidence"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
CANCELLED = "cancelled"

TERMINAL_STATUSES = frozenset(
    {
        COMPLETED,
        FAILED,
        BUDGET_EXHAUSTED,
        AWAITING_EVIDENCE,
        INSUFFICIENT_EVIDENCE,
        CANCELLED,
    }
)

STEP_STATUSES = frozenset({"ok", "waiting", "blocked", "failed"})

PHASE_VOCABULARY: dict[str, tuple[str, ...]] = {
    RESEARCH_AGENT: ("hypothesize", "execute", "observe", "decide"),
    DEBUG_AGENT: (
        "intake",
        "reproduce",
        "diagnose",
        "implementation",
        "verify",
        "terminal",
    ),
    FAQ_AGENT: ("intake", "retrieve", "read", "compose", "terminal"),
}


def now_utc() -> datetime:
    """Return the current instant, timezone-aware so it serializes unambiguously."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """One phase-tagged step of an agent run."""

    phase: str
    summary: str
    status: str = "ok"
    unit_id: str | None = None
    parent_unit_id: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    timestamp: datetime = field(default_factory=now_utc)

    def __post_init__(self) -> None:
        """Validate dataclass invariants before the value is persisted."""
        if not self.phase.strip():
            raise ValueError("step phase must not be empty")
        if not self.summary.strip():
            raise ValueError("step summary must not be empty")
        if self.status not in STEP_STATUSES:
            known = ", ".join(sorted(STEP_STATUSES))
            raise ValueError(f"unknown step status {self.status!r}; expected one of: {known}")
        for name in ("inputs", "outputs"):
            if not isinstance(getattr(self, name), dict):
                raise TypeError(f"step {name} must be a dict")
        if self.timestamp.tzinfo is None:
            raise ValueError("step timestamp must be timezone-aware")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this step. Order is the list position, so steps carry no identifier."""
        return {
            "phase": self.phase,
            "summary": self.summary,
            "status": self.status,
            "unit_id": self.unit_id,
            "parent_unit_id": self.parent_unit_id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "evidence_refs": list(self.evidence_refs),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    """One complete agent run in the shared cross-plane shape."""

    agent_type: str
    run_id: str
    task: str
    inputs: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=now_utc)
    ended_at: datetime | None = None
    terminal_status: str | None = None
    steps: tuple[TrajectoryStep, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate dataclass invariants before the value is persisted."""
        if self.agent_type not in PHASE_VOCABULARY:
            known = ", ".join(sorted(PHASE_VOCABULARY))
            raise ValueError(f"unknown agent_type {self.agent_type!r}; expected one of: {known}")
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not self.task.strip():
            raise ValueError("task must not be empty")
        if not isinstance(self.inputs, dict):
            raise TypeError("record inputs must be a dict")
        if self.started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")
        if self.ended_at is not None and self.ended_at.tzinfo is None:
            raise ValueError("ended_at must be timezone-aware")
        if self.terminal_status is not None and self.terminal_status not in TERMINAL_STATUSES:
            known = ", ".join(sorted(TERMINAL_STATUSES))
            raise ValueError(
                f"unknown terminal_status {self.terminal_status!r}; expected one of: {known}"
            )
        allowed = PHASE_VOCABULARY[self.agent_type]
        for step in self.steps:
            if step.phase not in allowed:
                raise ValueError(
                    f"phase {step.phase!r} is not in the {self.agent_type} vocabulary {allowed}"
                )
        self._check_lineage()

    def _check_lineage(self) -> None:
        """Reject a parent no earlier step declares, the way the tip reader does."""
        seen: set[str] = set()
        for position, step in enumerate(self.steps):
            if step.parent_unit_id is not None and step.parent_unit_id not in seen:
                raise ValueError(
                    f"step {position} descends from unit {step.parent_unit_id!r}, "
                    "which no earlier step declares"
                )
            if step.unit_id is not None:
                seen.add(step.unit_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this object into durable JSON-compatible data."""
        return {
            "schema_version": self.schema_version,
            "agent_type": self.agent_type,
            "run_id": self.run_id,
            "task": self.task,
            "inputs": self.inputs,
            "started_at": self.started_at.isoformat(),
            "ended_at": None if self.ended_at is None else self.ended_at.isoformat(),
            "terminal_status": self.terminal_status,
            "steps": [step.to_dict() for step in self.steps],
        }
