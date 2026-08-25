# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Check an emitted record against the consumer's own schema.

``record.py`` calls itself a verbatim mirror of ``xagents/trajectory/models.py``. It said
so before it emitted a ``step_id`` the consumer had removed, and nothing failed: the
consumer forbids unknown fields, so every record was rejected whole, one error per step,
and the first place anyone found out was a real run (agentics/xagents#18 §13.2). A mirror
nothing compares is a comment. This compares it.

Only the field sets are checked, because that is the drift that costs a whole record:
the consumer's models forbid extra fields and require a few, so a key this side invents
and a key it drops are both total rejections, while a wrong *value* is one bad field in
an otherwise accepted document. Phase and status vocabularies are asserted separately in
``test_record.py``.

The schema beside this file is a copy of the one the consumer validates with, taken from
xagents ``ae5581f08ee0e63294bb95cda17c34789d7fa62f``. A copy can itself fall behind, and
this catches drift between *this repository's code* and that snapshot rather than between
the snapshot and xagents today; refreshing it is part of taking a new contract version.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("groot_agent_trajectory.v0.schema.json")


def _schema() -> dict[str, Any]:
    """Load the vendored consumer schema."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _field_violations(obj: Mapping[str, Any], spec: Mapping[str, Any], where: str) -> list[str]:
    """Report keys the schema does not declare and required keys the object omits."""
    declared = set(spec.get("properties", {}))
    present = set(obj)
    problems = []
    # additionalProperties is false throughout this schema, so anything undeclared is a
    # rejection rather than an ignored extra.
    for name in sorted(present - declared):
        problems.append(f"{where}: {name!r} is not a field the consumer accepts")
    for name in sorted(set(spec.get("required", ())) - present):
        problems.append(f"{where}: {name!r} is required and missing")
    return problems


def schema_violations(record: Mapping[str, Any]) -> list[str]:
    """Return every way ``record``'s field sets depart from the consumer's schema."""
    schema = _schema()
    problems = _field_violations(record, schema, "record")
    step_spec = schema["$defs"]["step"]
    for index, step in enumerate(record.get("steps", ())):
        problems.extend(_field_violations(step, step_spec, f"steps[{index}]"))
    return problems
