# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for the AutoResearch operator skill."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from skills.autoresearch.study_state import StudyState
import yaml

_REPOSITORY = Path(__file__).resolve().parents[3]
_AUTORESEARCH_SKILL = _REPOSITORY / "skills/autoresearch/SKILL.md"
_BASELINE_SKILL = _REPOSITORY / "skills/prepare-autoresearch-baseline/SKILL.md"
_SKILL_PATHS = (
    _REPOSITORY / "skills/autoresearch/SKILL.md",
    _REPOSITORY / ".codex/skills/autoresearch/SKILL.md",
    _REPOSITORY / ".agents/skills/autoresearch/SKILL.md",
)

_CHECK = _REPOSITORY / "skills/autoresearch/scripts/check.py"


def test_preflight_is_read_only_and_detects_resume(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=workspace, check=True)
    (workspace / "README.md").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "branch", "-M", "baseline"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.test/repo.git"],
        cwd=workspace,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    marker = tmp_path / "user-script-ran"
    wrappers = []
    for name in ("train.sh", "eval.sh"):
        wrapper = tmp_path / name
        wrapper.write_text(
            f"#!/usr/bin/env bash\ntouch {marker}\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        wrappers.append(wrapper)
    codex = tmp_path / "codex"
    codex.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    codex.chmod(0o755)
    baseline_result = tmp_path / "baseline/results.json"
    baseline_result.parent.mkdir()
    baseline_result.write_text('{"score": 0.25}\n', encoding="utf-8")
    output_dir = tmp_path / "output"
    config_path = tmp_path / "autoresearch.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "study_id": "skill-test",
                "workspace": str(workspace),
                "output_dir": str(output_dir),
                "max_trials": 1,
                "max_parallel_trials": 1,
                "max_retry": 0,
                "baseline": {
                    "train_script": str(wrappers[0]),
                    "eval_script": str(wrappers[1]),
                    "eval_result": str(baseline_result),
                    "eval_metric": "score",
                    "direction": "maximize",
                    "baseline_result": 0.25,
                    "min_delta": 0.0,
                    "script_timeout_s": 10,
                },
                "git": {
                    "git_remote": "origin",
                    "base_branch": "baseline",
                },
                "editable_paths": ["groot/"],
                "frozen_paths": ["tests/"],
                "idea_types": ["recipe_tuning"],
                "agent": {"executable": str(codex), "model": "gpt-5.6"},
            }
        ),
        encoding="utf-8",
    )

    fresh = subprocess.run(
        [sys.executable, str(_CHECK), str(config_path)],
        cwd=_REPOSITORY,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert fresh.returncode == 0, fresh.stderr
    assert "mode=new" in fresh.stdout
    assert not marker.exists()
    assert not output_dir.exists()

    baseline_result.unlink()
    incomplete = subprocess.run(
        [sys.executable, str(_CHECK), str(config_path)],
        cwd=_REPOSITORY,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert incomplete.returncode == 1
    assert "Invalid evaluation result JSON" in incomplete.stderr
    assert not marker.exists()
    assert not output_dir.exists()
    baseline_result.write_text('{"score": 0.25}\n', encoding="utf-8")

    output_dir.mkdir()
    state = StudyState.create(
        study_id="skill-test",
        baseline_score=0.25,
        base_commit=commit,
        eval_metric="score",
        direction="maximize",
        min_delta=0.0,
    )
    (output_dir / "state.json").write_text(
        json.dumps(state.to_dict()),
        encoding="utf-8",
    )
    resumed = subprocess.run(
        [sys.executable, str(_CHECK), str(config_path)],
        cwd=_REPOSITORY,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert resumed.returncode == 0, resumed.stderr
    assert "mode=resume" in resumed.stdout
    assert "state=wave:none trials:0 best_trial:baseline best_score:0.25" in resumed.stdout
    assert not marker.exists()


def test_all_canonical_skills_are_discoverable() -> None:
    """Every non-empty canonical skill must resolve through all agent surfaces."""
    canonical_skills = sorted(
        path.parent
        for path in (_REPOSITORY / "skills").glob("*/SKILL.md")
        if path.read_text(encoding="utf-8").strip()
    )
    assert canonical_skills

    agent_skills = _REPOSITORY / ".agents/skills"
    assert agent_skills.is_symlink()

    for canonical in canonical_skills:
        for root in (".claude/skills", ".codex/skills", ".agents/skills"):
            discovered = _REPOSITORY / root / canonical.name
            assert discovered.exists(), f"{canonical.name} is missing from {root}"
            if root != ".agents/skills":
                assert discovered.is_symlink()
            assert discovered.resolve() == canonical.resolve()


def test_autoresearch_skill_routes_only_incomplete_new_baselines() -> None:
    """Lock the handoff and its execution/resume safety boundaries."""
    text = _AUTORESEARCH_SKILL.read_text(encoding="utf-8")

    assert "Do not require a config path before" in text
    assert "baseline preparation." in text
    assert "`$prepare-autoresearch-baseline` when any of these are missing" in text
    assert "authorization to launch AutoResearch alone does" in text
    assert "not imply that permission" in text
    assert "Do not use the preparation skill when output_dir contains state.json" in text
    assert "normal trial failure" in text


def test_skills_preserve_user_guidance_in_yaml() -> None:
    """User research constraints must survive the handoff into trial prompts."""
    autoresearch = _AUTORESEARCH_SKILL.read_text(encoding="utf-8")
    baseline = _BASELINE_SKILL.read_text(encoding="utf-8")

    assert "do not leave that intent only in the\nchat history" in autoresearch
    assert "Distill the user's requested research ideas" in baseline
    assert "`user_guidance`" in baseline
    assert "do not invent requirements" in baseline


def test_autoresearch_skill_copies_match() -> None:
    """The installed Codex and agent copies must match the canonical skill."""
    expected = _AUTORESEARCH_SKILL.read_text(encoding="utf-8")
    for path in (
        _REPOSITORY / ".codex/skills/autoresearch/SKILL.md",
        _REPOSITORY / ".agents/skills/autoresearch/SKILL.md",
    ):
        assert path.read_text(encoding="utf-8") == expected


def test_skill_routes_out_of_scope_requests() -> None:
    """Related study work stays local; other requests route to the right surface."""
    canonical = _SKILL_PATHS[0].read_text(encoding="utf-8")
    for path in _SKILL_PATHS[1:]:
        assert path.read_text(encoding="utf-8") == canonical
    normalized = " ".join(canonical.split())

    assert "Before any AutoResearch action" in normalized
    assert "local OSS setup, dry-runs, training, evaluation, or policy serving" in normalized
    assert "Use the onboarding skill to handle this local OSS workflow" in normalized
    assert "product, support, concept, or documentation questions" in normalized
    assert "Use the faq agent to answer this question" in normalized
    assert "unrelated GR00T failure diagnosis" in normalized
    assert "`$autodebug`" in normalized
    assert '"Resume `/work/study/autoresearch.yaml`."' in normalized
    assert '"Dry-run local post-training on `/data/task`."' in normalized
    assert '"Importing GR00T fails in a clean checkout."' in normalized
    assert "in-study failure diagnosis or `state.json` status" in normalized
    assert (
        "Do not inspect state.json, run preflight, modify files, or execute scripts" in normalized
    )
    assert "Failures within an AutoResearch study remain" in normalized
