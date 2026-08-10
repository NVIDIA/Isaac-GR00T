# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resumable wave-based AutoResearch study controller."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from skills.autoresearch.codex_editor import CodexEditor
from skills.autoresearch.config import StudyConfig
from skills.autoresearch.git_workspace import GitWorkspaceManager
from skills.autoresearch.study_state import StudyState, Trial, make_trial_id
from skills.autoresearch.trial_controller import StepResult, TrialController
from skills.autoresearch.trial_prompt import idea_type_for_index


# WaveController owns the saved state machine; runners and editors do side effects.
class WaveController:
    """Create, schedule, and settle waves of independent trials."""

    def __init__(
        self,
        *,
        config: StudyConfig,
        workspace_manager: GitWorkspaceManager,
        editor: CodexEditor,
        runner: Any,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """Store constructor dependencies without performing work."""
        self.config = config
        self.state_path = config.output_dir / "state.json"
        self._progress = progress or _ignore_progress
        self.trial_controller = TrialController(
            config=config,
            workspace_manager=workspace_manager,
            editor=editor,
            runner=runner,
            save_state=self._save_state,
            progress=self._progress,
        )

    def initialize(self) -> StudyState:
        """Create initial study state from the config."""
        state = StudyState.create(
            study_id=self.config.study_id,
            baseline_score=self.config.objective.baseline,
            base_commit=self.config.base_commit,
            eval_metric=self.config.objective.result_path,
            direction=self.config.objective.direction,
            min_delta=self.config.objective.min_delta,
        )
        self._save_state(state)
        return state

    def run_once(self) -> StepResult:
        """Advance the study by exactly one saved transition."""
        state = self.load_state()

        # Each trial owns its next action; the wave only schedules the group.
        if state.active_wave_index is not None:
            wave_trials = _wave_trials(state)
            result = self.trial_controller.advance(state, wave_trials)
            if result is not None:
                return result
            state = self.load_state()
            return self._settle_wave(state, _wave_trials(state))

        # Once no work is active, either stop or create the next frozen-parent wave.
        if len(state.trials) >= self.config.max_trials:
            message = f"Maximum trials reached ({self.config.max_trials})"
            return StepResult("stopped", message, done=True)
        return self._create_wave(state)

    def load_state(self) -> StudyState:
        """Load current state for this study."""
        state = StudyState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))
        return self._validate_state_config(state)

    def _save_state(self, state: StudyState) -> None:
        """Write the current study state as plain JSON."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            state.to_dict(),
            indent=4,
            sort_keys=True,
            allow_nan=False,
        )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.state_path.parent,
            prefix=f".{self.state_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.state_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _validate_state_config(self, state: StudyState) -> StudyState:
        """Reject a save directory that belongs to a different study contract."""
        identity_values = {
            "study_id": self.config.study_id,
            "baseline_score": self.config.objective.baseline,
        }
        for name, value in identity_values.items():
            if getattr(state, name) != value:
                raise ValueError(
                    f"Existing state {name}={getattr(state, name)!r} does not match "
                    f"config value {value!r}"
                )

        objective_values = {
            "base_commit": self.config.base_commit,
            "eval_metric": self.config.objective.result_path,
            "direction": self.config.objective.direction,
            "min_delta": self.config.objective.min_delta,
        }
        for name, value in objective_values.items():
            if getattr(state, name) != value:
                raise ValueError(
                    f"Existing state {name}={getattr(state, name)!r} does not match "
                    f"config value {value!r}"
                )
        return state

    def _create_wave(self, state: StudyState) -> StepResult:
        """Wave scheduling creates a frozen-parent batch and queues compositions later."""
        # Size the wave by remaining trial budget and configured parallelism.
        remaining = self.config.max_trials - len(state.trials)
        wave_size = min(self.config.max_parallel_trials, remaining)
        wave_index = state.next_wave_index
        # Freeze every trial in the wave to the same current-best parent.
        parent = _best_trial(state)
        parent_revision = parent.code_revision if parent else self.config.base_commit
        queued_sources = list(state.pending_change_trial_ids[:wave_size])
        pending = state.pending_change_trial_ids[len(queued_sources) :]
        # Fill queued compositions first, then use fresh idea types for open slots.
        trials: list[Trial] = []
        for offset in range(wave_size):
            index = state.next_trial_index + offset
            trial_id = make_trial_id(state.study_id, wave_index, index)
            source_id = queued_sources[offset] if offset < len(queued_sources) else None
            source = state.get_trial(source_id) if source_id else None
            trials.append(
                Trial(
                    trial_id=trial_id,
                    index=index,
                    wave_index=wave_index,
                    parent_trial_id=parent.trial_id if parent else None,
                    parent_revision=parent_revision,
                    composition_source_trial_id=source_id,
                    idea_type=(
                        "composition" if source else idea_type_for_index(self.config, index)
                    ),
                    hypothesis="",
                    parent_score=state.best_score,
                )
            )
        # Save the new active wave before editor or runner side effects happen.
        updated = replace(
            state,
            trials=(*state.trials, *trials),
            active_wave_index=wave_index,
            pending_change_trial_ids=pending,
        )
        self._save_state(updated)
        parent_name = parent.trial_id if parent else "baseline"
        self._progress(
            f"[Wave {wave_index}] started | trials={trials[0].index}-"
            f"{trials[-1].index}/{self.config.max_trials} | count={len(trials)} | "
            f"parent={parent_name} | parent_score={state.best_score}"
        )
        return StepResult(
            "wave_created",
            f"Created wave {wave_index} with {len(trials)} trials "
            f"({len(queued_sources)} compositions)",
        )

    def _settle_wave(self, state: StudyState, wave_trials: list[Trial]) -> StepResult:
        """Select the wave winner and queue useful non-winners."""
        # Select the best scored trial only if it beats the previous global best.
        scored = [trial for trial in wave_trials if trial.score is not None]
        winner = _best_scored_trial(scored, direction=self.config.objective.direction)
        if winner is not None and not self.config.objective.is_better(
            winner.score, state.best_score
        ):
            winner = None

        # Merge recovered and scored wave records back into the full trial list.
        trials_by_id = {trial.trial_id: trial for trial in state.trials}
        for trial in wave_trials:
            updated_trial = trial
            if trial.score is not None:
                updated_trial = replace(
                    trial,
                    is_new_best=(winner is not None and trial.trial_id == winner.trial_id),
                )
            updated_trial = _with_result_feedback(updated_trial)
            trials_by_id[updated_trial.trial_id] = updated_trial

        # Useful non-winners become queued composition sources for later waves.
        queued = list(state.pending_change_trial_ids)
        for trial in wave_trials:
            if not trial.improved_over_parent or (winner and trial.trial_id == winner.trial_id):
                continue
            source_id = trial.composition_source_trial_id or trial.trial_id
            if source_id not in queued:
                queued.append(source_id)

        best_score = state.best_score
        best_trial_id = state.best_trial_id
        # A winning composition consumes any source changes it already includes.
        if winner is not None:
            best_score = winner.score
            best_trial_id = winner.trial_id
            included = _included_change_ids(state, winner)
            queued = [source_id for source_id in queued if source_id not in included]

        updated = replace(
            state,
            trials=tuple(trials_by_id[trial.trial_id] for trial in state.trials),
            best_score=best_score,
            best_trial_id=best_trial_id,
            active_wave_index=None,
            pending_change_trial_ids=tuple(queued),
        )
        self._save_state(updated)
        winner_message = winner.trial_id if winner else "none"
        best_name = best_trial_id or "baseline"
        self._progress(
            f"[Wave {state.active_wave_index}] settled | winner={winner_message} | "
            f"best={best_name} | best_score={best_score} | queued_changes={len(queued)}"
        )
        return StepResult(
            "wave_settled",
            f"Settled wave {state.active_wave_index}; winner={winner_message}; "
            f"queued_changes={len(queued)}",
        )


def _with_result_feedback(trial: Trial) -> Trial:
    """Attach the completed experiment result used by later prompts."""
    if trial.score is None:
        return replace(
            trial,
            result_summary="The experiment produced no valid metric result.",
            next_guidance=(
                "Do not draw a conclusion about the hypothesis from this run; inspect the "
                "recorded stage error before retrying or adapting it."
            ),
        )

    if trial.improved_over_parent:
        result_summary = (
            f"Improved the metric from parent score {trial.parent_score} to {trial.score}."
        )
        next_guidance = (
            "Build on this new global-best result with one isolated follow-up hypothesis."
            if trial.is_new_best
            else (
                "Preserve this parent improvement as a composition candidate; it did not "
                "become the global best."
            )
        )
    else:
        result_summary = (
            f"Did not improve the metric over parent score {trial.parent_score}; "
            f"observed {trial.score}."
        )
        next_guidance = (
            "Avoid repeating this exact hypothesis unless a distinct mechanism addresses "
            "the observed result."
        )
    return replace(
        trial,
        result_summary=result_summary,
        next_guidance=next_guidance,
    )


def _ignore_progress(message: str) -> None:
    """Keep library use quiet unless the caller requests progress events."""
    del message


def _wave_trials(state: StudyState) -> list[Trial]:
    """Scheduling helpers keep the main state machine readable and deterministic."""
    return [trial for trial in state.trials if trial.wave_index == state.active_wave_index]


def _best_scored_trial(trials: list[Trial], *, direction: str) -> Trial | None:
    """Select the best scored trial for the configured direction."""
    if not trials:
        return None
    if direction == "minimize":
        return min(trials, key=lambda trial: (trial.score, trial.index))
    return max(trials, key=lambda trial: (trial.score, -trial.index))


def _included_change_ids(state: StudyState, trial: Trial) -> set[str]:
    """Trace the isolated changes already included in a composed recipe."""
    included: set[str] = set()
    current: Trial | None = trial
    while current is not None:
        included.add(current.composition_source_trial_id or current.trial_id)
        current = state.get_trial(current.parent_trial_id) if current.parent_trial_id else None
    return included


def _best_trial(state: StudyState) -> Trial | None:
    """Select the best scored trial for the configured direction."""
    if state.best_trial_id is None:
        return None
    return state.get_trial(state.best_trial_id)
