# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""N1.7-style bounded research prompts for standalone AutoResearch."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

from skills.autoresearch.config import TRIAL_EVAL_RESULT, StudyConfig
from skills.autoresearch.study_state import StudyState, Trial

# Idea-type guidance constrains each Codex turn to one research style.
IDEA_TYPE_GUIDANCE = {
    "history_ablation": (
        "Investigate how temporal context is represented across visual observations, "
        "proprioception, actions, or tokens. Consider history length and stride, past/current "
        "grouping, current-frame emphasis, temporal embeddings, history masking, and history "
        "dropout while keeping the dataset, evaluation, and compute budget fixed."
    ),
    "inference_path_ablation": (
        "Investigate consistency between training and the model-side inference path. Consider "
        "action-chunk construction and decoding, conditioning, action horizons, normalization, "
        "diffusion or flow parameterization, sampler schedules, and train/inference code-path "
        "mismatches without editing the evaluator, metric, or evaluation budget."
    ),
    "recipe_tuning": (
        "Investigate the training recipe: learning rate, optimizer, scheduler, warmup, weight "
        "decay, gradient clipping, dropout, augmentation, trainable "
        "modules, and resource-compatible micro-batch or accumulation settings that preserve "
        "the effective global batch and optimizer-step count. Preserve the fixed "
        "training and evaluation budget."
    ),
    "free_roam": (
        "Choose the strongest single hypothesis supported by the code and prior results across "
        "training recipes, model architecture, temporal or inference paths, implementation "
        "correctness, and configuration plumbing. A substantial change is allowed when it "
        "directly and cleanly tests that one hypothesis."
    ),
}


def idea_type_for_index(config: StudyConfig, index: int) -> str:
    """Prompt builders turn durable study state into bounded Codex instructions."""
    if index < 1:
        raise ValueError("trial index must be >= 1")
    return config.idea_types[(index - 1) % len(config.idea_types)]


def build_prompt(
    *,
    config: StudyConfig,
    state: StudyState,
    trial_id: str,
    trial_index: int,
    parent_revision: str | None = None,
    parent_score: float | None = None,
) -> str:
    """Build one Codex prompt for an isolated, scoreable candidate."""
    idea_type = idea_type_for_index(config, trial_index)
    direction = "lower" if config.objective.direction == "minimize" else "higher"
    editable = "\n".join(f"- `{path}`" for path in config.editable_paths)
    frozen = "\n".join(f"- `{path}`" for path in config.frozen_paths)
    feedback = _all_result_feedback(state, current_trial_id=trial_id)
    safe_runner_context = _safe_runner_context(config)
    runner_context = json.dumps(safe_runner_context, indent=2, sort_keys=True)
    protocol = _protocol_summary(safe_runner_context)
    current_trial = next((trial for trial in state.trials if trial.trial_id == trial_id), None)
    parent_trial_id = (
        current_trial.parent_trial_id if current_trial is not None else state.best_trial_id
    )
    best_trial = _best_trial(state)
    resolved_parent_revision = parent_revision or (
        best_trial.code_revision if best_trial and best_trial.code_revision else config.base_commit
    )
    resolved_parent_score = state.best_score if parent_score is None else parent_score
    parent_label = parent_trial_id or "baseline"
    guidance = IDEA_TYPE_GUIDANCE.get(
        idea_type,
        "Follow this configured idea category while making one isolated training-side change.",
    )
    user_guidance = config.user_guidance or "No additional user guidance was provided."
    return f"""\
You are AutoResearch running trial {trial_index} (`{trial_id}`) for study
`{config.study_id}`.

Goal:
- Improve `{config.objective.result_path}`; {direction} is better.
- Current best score: {state.best_score}.
- Starting baseline score: {config.objective.baseline}.
- Pinned baseline commit: `{config.base_commit}`. Do not rebase or update upstream code.
- Candidate parent: `{parent_label}` at revision `{resolved_parent_revision}` with score
  {resolved_parent_score}.
- Assigned idea type: `{idea_type}`.
- Generic idea-type guidance: {guidance}

User guidance:
{user_guidance}

Rules:
- When User guidance is provided, treat it as the required research scope. It takes precedence
  over the broader generic idea-type guidance. Obey its requested changes and exclusions.
- Make exactly one isolated, evaluable training-side hypothesis. The implementation may be
  substantial when that is necessary to test the one hypothesis cleanly.
- The training and evaluation compute budget is fixed; user-provided wrapper scripts, dataset
  contents and splits, pretrained/base model, evaluation code, metric, and task set are fixed too.
  Do not change the maximum training steps or epochs, evaluation sample or episode count,
  accelerator count, or other budget settings to obtain a better score.
- Files listed under Editable surfaces are the training-side search surface. You may change the
  inner training script/config called by the wrapper, training recipe, model code, and
  training-side implementation as long as the fixed surfaces above remain unchanged.
- Preserve the configured execution contract ({protocol}).
- Do not change or bypass the pinned pretrained/base model, dataset/script inputs, or source
  revision checks. When the runner context lists `locked_train_config`, the controller rejects
  any candidate whose rendered locked fields or effective global batch differ from the baseline,
  or whose training datasets overlap an evaluation dataset.
- Use prior results below to avoid repeating failed changes and to build on useful evidence.
- Do not commit, push, create branches, or submit training/evaluation jobs. The controller owns
  those operations.
- Run focused local validation when practical.
- Your final structured response must accurately describe the hypothesis and validation
  commands actually run.

Editable surfaces:
{editable}

Frozen surfaces:
{frozen}

Sanitized runner context (read-only; do not edit these settings):
{runner_context}

Completed experiment feedback:
{feedback}
"""


def _all_result_feedback(state: StudyState, *, current_trial_id: str) -> str:
    """Return every completed trial result in chronological order."""
    completed = [
        trial
        for trial in state.trials
        if trial.trial_id != current_trial_id and (trial.result_summary or trial.next_guidance)
    ]
    if not completed:
        return "- No completed trials."
    return "\n".join(
        _trial_feedback_line(
            trial,
            is_current_best=trial.trial_id == state.best_trial_id,
        )
        for trial in completed
    )


def _trial_feedback_line(trial: Trial, *, is_current_best: bool) -> str:
    """Format one compact completed-trial result for the next prompt."""
    parent = trial.parent_trial_id or "baseline"
    score = "no score" if trial.score is None else str(trial.score)
    if trial.score is None:
        outcome = "failed"
    elif is_current_best:
        outcome = "current global best"
    elif trial.improved_over_parent:
        outcome = "improved over parent"
    else:
        outcome = "did not improve parent"
    pieces = [
        f"trial `{trial.trial_id}`",
        f"wave {trial.wave_index}",
        f"idea `{trial.idea_type}`",
        f"parent `{parent}` at score {trial.parent_score}",
        f"score {score}",
        f"outcome {outcome}",
        f"hypothesis {_compact(trial.hypothesis)}",
        f"result_summary {_compact(trial.result_summary)}",
        f"next_guidance {_compact(trial.next_guidance)}",
    ]
    error = trial.evaluation.error or trial.train.error
    if error:
        pieces.append(f"error {_compact(error)}")
    return "- " + ", ".join(pieces)


def _best_trial(state: StudyState) -> Trial | None:
    """Select the best scored trial for the configured direction."""
    if state.best_trial_id is None:
        return None
    return next(
        (trial for trial in state.trials if trial.trial_id == state.best_trial_id),
        None,
    )


def _safe_runner_context(config: StudyConfig) -> dict[str, Any]:
    """Runner context is sanitized before being shown to Codex as read-only protocol."""
    context: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "train_script": str(config.train_script),
        "eval_script": str(config.eval_script),
        "eval_result": TRIAL_EVAL_RESULT,
        "eval_metric": config.objective.result_path,
        "direction": config.objective.direction,
        "train_output_dir": config.output_dir_name,
    }
    if config.train_dataset:
        context["train_dataset"] = config.train_dataset
    if config.eval_dataset:
        context["eval_dataset"] = config.eval_dataset
    if config.checkpoint_path is not None:
        context["train_checkpoint"] = config.checkpoint_path
    if config.train_config_lock is not None:
        lock = config.train_config_lock
        context["locked_train_config"] = {
            "path": lock.path,
            "fields": list(lock.fields),
            "effective_global_batch": "per_device_batch_size * num_gpus * "
            "gradient_accumulation_steps must match the baseline",
            "eval_dataset_paths": list(lock.eval_dataset_paths),
        }
    return context


def _protocol_summary(runner_context: Mapping[str, Any]) -> str:
    """Summarize context for prompts without overwhelming Codex."""
    del runner_context
    return (
        "user-provided train/eval scripts are fixed wrappers and their artifact paths are fixed; "
        "editable training-side files they invoke remain the search surface"
    )


def _compact(text: str, *, limit: int = 300) -> str:
    """Summarize context for prompts without overwhelming Codex."""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."
