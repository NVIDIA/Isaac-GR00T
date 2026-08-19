# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Retrieve and render prior Insights for one AutoResearch wave."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skills.autoresearch.config import StudyConfig
from skills.autoresearch.study_state import StudyState, Trial
from skills.insights.infra.insight_projection import search_insights

_IDEAS_QUERY = "successful experiments improvements useful recommendations ideas to build on"


def prior_insights_for_wave(
    *,
    config: StudyConfig,
    state: StudyState,
    wave_index: int,
) -> str:
    """Retrieve and render the Insights handoff used by one wave."""
    return ""

    context = _query_context(config, state, wave_index=wave_index)
    lookup = {
        "purpose": "Ideas to build on",
        "result": search_insights(
            f"{_IDEAS_QUERY}. {context}",
            start=Path(__file__),
        ),
    }
    return _render_handoff([lookup])


def _query_context(config: StudyConfig, state: StudyState, *, wave_index: int) -> str:
    """Build stable study context shared by every lookup in a wave."""
    wave_idea_types = sorted(
        {
            trial.idea_type
            for trial in state.trials
            if trial.wave_index == wave_index and trial.idea_type != "composition"
        }
    )
    fields = [
        f"AutoResearch study {config.study_id}",
        f"metric {config.objective.result_path}",
        f"objective {config.objective.direction}",
    ]
    if config.train_dataset:
        fields.append(f"training dataset {config.train_dataset}")
    if config.eval_dataset:
        fields.append(f"evaluation dataset {config.eval_dataset}")
    if wave_idea_types:
        fields.append(f"research areas {' '.join(wave_idea_types)}")
    if config.user_guidance:
        fields.append(f"user guidance {config.user_guidance}")

    relevant_trials = _relevant_completed_trials(state, wave_index=wave_index)
    for trial in relevant_trials:
        fields.append(
            " ".join(
                part
                for part in (
                    f"prior hypothesis {trial.hypothesis}" if trial.hypothesis else "",
                    f"result {trial.result_summary}" if trial.result_summary else "",
                    f"next guidance {trial.next_guidance}" if trial.next_guidance else "",
                )
                if part
            )
        )
    return ". ".join(field for field in fields if field)


def _relevant_completed_trials(state: StudyState, *, wave_index: int) -> tuple[Trial, ...]:
    """Use the global best and most recent completed wave to refine later lookups."""
    completed = tuple(
        trial
        for trial in state.trials
        if trial.wave_index < wave_index and (trial.result_summary or trial.next_guidance)
    )
    if not completed:
        return ()
    latest_wave = max(trial.wave_index for trial in completed)
    selected = [trial for trial in completed if trial.wave_index == latest_wave]
    if state.best_trial_id:
        best = state.get_trial(state.best_trial_id)
        if best not in selected:
            selected.insert(0, best)
    return tuple(selected)


def _render_handoff(lookups: list[dict[str, Any]]) -> str:
    """Render complete, deduplicated findings and chunks for the coding agent."""
    lines = [
        "Treat these published records as directional historical knowledge only. User guidance ",
        "and current-study train/eval evidence take precedence.",
    ]
    seen_keys: set[str] = set()
    projection: dict[str, Any] | None = None
    matched = False
    unavailable_errors: list[str] = []

    for lookup in lookups:
        if not isinstance(lookup, dict) or not isinstance(lookup.get("result"), dict):
            continue
        result = lookup["result"]
        if projection is None and isinstance(result.get("projection"), dict):
            projection = result["projection"]
        if not result.get("available"):
            if isinstance(result.get("error"), str):
                unavailable_errors.append(result["error"])
            continue

        section: list[str] = []
        for item in result.get("results", []):
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            if not isinstance(key, str) or key in seen_keys:
                continue
            seen_keys.add(key)
            section.extend(_render_item(item))
        if section:
            matched = True
            lines.extend(
                ("", f"### Retrieved for {lookup.get('purpose', 'prior research')}", *section)
            )

    if projection is not None:
        lines.extend(
            (
                "",
                "Projection provenance: "
                f"generated_at={projection.get('generated_at', 'unknown')}, "
                f"sha256={projection.get('sha256', 'unknown')}",
            )
        )
    if not matched:
        if unavailable_errors:
            lines.extend(("", "- The published Insights projection was unavailable for this wave."))
        else:
            lines.extend(("", "- No relevant published Insights matched this wave's queries."))
    return "\n".join(lines)


def _render_item(item: dict[str, Any]) -> list[str]:
    """Keep the complete returned finding and supporting source chunk together."""
    lines: list[str] = []
    finding = item.get("finding")
    chunk = item.get("chunk")
    if isinstance(finding, dict):
        lines.extend(_render_record("Finding", finding))
    if isinstance(chunk, dict):
        lines.extend(_render_record("Supporting chunk", chunk))
    return lines


def _render_record(label: str, record: dict[str, Any]) -> list[str]:
    """Render one normalized projection record without truncating its evidence."""
    record_id = record.get("id", "unknown")
    title = record.get("title", "")
    text = record.get("text", "")
    return [
        f"- {label} `{record_id}`: {title}",
        *(f"  {line}" for line in str(text).splitlines() or [""]),
    ]
