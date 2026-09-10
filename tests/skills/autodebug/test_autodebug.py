# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tests for the minimal GR00T AutoDebug loop."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
from skills.autodebug.infra import AutoDebugIncident, ExecutionTarget, SolvedCase
from skills.autodebug.infra.autodebug_incident import (
    DEFAULT_CASE_DIRECTORY,
)

REPORT = """# GR00T bug report

## Problem

Dataset loading fails with a shard error.

## Expected behavior

The dataset initializes.

## Observed behavior

All shards must have length greater than 0.

## Reproduction

Run the focused check.

## Acceptance check

The focused check and its regression both pass.
"""


def _report(tmp_path: Path, body: str = REPORT) -> Path:
    path = tmp_path / "report.md"
    path.write_text(body, encoding="utf-8")
    return path


def _case(tmp_path: Path, *, status: str = "verified") -> Path:
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    data = {
        "version": 1,
        "id": "case-shard",
        "status": status,
        "title": "Shard construction failure",
        "source": "https://example.invalid/case",
        "scope": ["test"],
        "tags": ["dataset"],
        "match_terms": ["all shards"],
        "symptom": "A shard is empty.",
        "first_test": "Inspect shard assignment.",
        "cause": "The assignment omitted a shard.",
        "fix": "Assign one item to each shard first.",
        "compute": "Local CPU.",
        "verification": "Focused and regression checks passed.",
    }
    path = case_dir / "case.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return case_dir


def _incident(tmp_path: Path, *, ignore_fixed: bool = False) -> AutoDebugIncident:
    workspace = tmp_path / "diagnosis"
    workspace.mkdir()
    (workspace / "README.md").write_text("diagnosis checkout\n", encoding="utf-8")
    if ignore_fixed:
        (workspace / ".gitignore").write_text("fixed\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AutoDebug Test",
            "-c",
            "user.email=autodebug@example.invalid",
            "commit",
            "-m",
            "Diagnosis baseline",
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    return AutoDebugIncident.init_from_report(
        _report(tmp_path),
        workspace=workspace,
        run_root=tmp_path / "runs",
        case_directory=_case(tmp_path),
    )


def _implementation_workspace(
    incident: AutoDebugIncident, tmp_path: Path, *, fixed: bool
) -> tuple[Path, str]:
    workspace = tmp_path / ("implementation-fixed" if fixed else "implementation-unfixed")
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(workspace), incident.repo_commit],
        cwd=incident.workspace,
        check=True,
        capture_output=True,
    )
    if fixed:
        (workspace / "fixed").write_text("fixed\n", encoding="utf-8")
    else:
        (workspace / "implementation-note").write_text("no effective fix\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AutoDebug Test",
            "-c",
            "user.email=autodebug@example.invalid",
            "commit",
            "-m",
            "Implementation under test",
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return workspace, revision


def _failing_reproducer(tmp_path: Path) -> tuple[str, str]:
    signature = "EXPECTED BUG: marker is missing"
    reproducer = tmp_path / "reproducer.py"
    reproducer.write_text(
        "from pathlib import Path\n"
        f"signature = {signature!r}\n"
        "if not Path('fixed').exists():\n"
        "    print(signature)\n"
        "    raise SystemExit(1)\n",
        encoding="utf-8",
    )
    return f"{sys.executable} {reproducer}", signature


def _support_and_defend(
    incident: AutoDebugIncident,
    command: str,
    hypothesis: str,
) -> str:
    defense = f"{sys.executable} -c 'print(\"matched control\")'"
    incident.run_diagnostic(command, hypothesis, "supported", 30)
    incident.run_diagnostic(
        defense,
        f"Root-cause defense for: {hypothesis}",
        "supported",
        30,
    )
    return defense


def _prepare_handoff(incident: AutoDebugIncident, tmp_path: Path) -> str:
    failing, signature = _failing_reproducer(tmp_path)
    passing = f"{sys.executable} -c 'raise SystemExit(0)'"
    incident.run_reproducer(failing, signature, 30)
    _support_and_defend(incident, passing, "A focused factor is supported.")
    incident.handoff(
        "The focused check isolates the failing factor.",
        "Apply the smallest change in the implementation checkout.",
        ["fixed", "implementation-note"],
        [passing],
    )
    return passing


def _nested_incident(tmp_path: Path) -> AutoDebugIncident:
    repository = tmp_path / "repository"
    workspace = repository / "oss"
    workspace.mkdir(parents=True)
    (workspace / "README.md").write_text("nested diagnosis checkout\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AutoDebug Test",
            "-c",
            "user.email=autodebug@example.invalid",
            "commit",
            "-m",
            "Nested diagnosis baseline",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    incident = AutoDebugIncident.init_from_report(
        _report(tmp_path),
        workspace=workspace,
        run_root=tmp_path / "runs",
        case_directory=_case(tmp_path),
    )
    _prepare_handoff(incident, tmp_path)
    return incident


def _nested_implementation(
    incident: AutoDebugIncident, tmp_path: Path, *, change_outside_oss: bool
) -> tuple[Path, str]:
    root = tmp_path / ("implementation-outside" if change_outside_oss else "implementation-oss")
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(root), incident.repo_commit],
        cwd=incident.workspace,
        check=True,
        capture_output=True,
    )
    (root / "oss" / "fixed").write_text("fixed\n", encoding="utf-8")
    if change_outside_oss:
        (root / "outside.txt").write_text("outside diagnosed scope\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AutoDebug Test",
            "-c",
            "user.email=autodebug@example.invalid",
            "commit",
            "-m",
            "Nested implementation",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return root / "oss", revision


def test_new_incident_matches_a_verified_case_and_resumes(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    assert incident.phase == "reproduce"
    assert incident.related_case_ids == ["case-shard"]
    assert incident.runs[0]["kind"] == "insights"
    assert incident.runs[0]["available"] is False
    assert incident.runs[0]["record_ids"] == []

    same_incident = AutoDebugIncident.init_from_report(
        incident.report_path,
        workspace=incident.workspace,
        run_root=tmp_path / "runs",
        case_directory=incident.case_directory,
    )
    assert same_incident.path == incident.path


def test_incident_runs_a_probe_on_a_declared_local_target(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    config = tmp_path / "targets.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "targets": {
                    "reference": {
                        "kind": "local",
                        "workspace": str(incident.workspace),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    incident = AutoDebugIncident.init_from_report(
        incident.report_path,
        workspace=incident.workspace,
        run_root=tmp_path / "runs",
        case_directory=incident.case_directory,
        target_config=config,
    )

    incident.run_probe(f"{sys.executable} -c 'print(42)'", "reference", 30)

    assert incident.phase == "reproduce"
    assert incident.runs[-1]["target"] == "reference"
    assert incident.runs[-1]["target_kind"] == "local"
    assert incident.runs[-1]["summary"] == "probe; target=reference; exit=0"
    assert "42" in Path(incident.runs[-1]["log"]).read_text(encoding="utf-8")
    assert json.loads(incident.path.read_text(encoding="utf-8"))["runs"][-1]["summary"] == (
        "probe; target=reference; exit=0"
    )
    last_run = json.loads(incident.to_summary_json_text())["last_run"]
    assert last_run["exit_code"] == 0
    assert last_run["summary"] == "probe; target=reference; exit=0"


def test_ssh_target_builds_a_noninteractive_remote_command(tmp_path: Path) -> None:
    target = ExecutionTarget.init_from_data(
        "remote-gpu",
        {
            "kind": "ssh",
            "host": "gpu-workstation.example.com",
            "workspace": "/workspace/project",
        },
        config_directory=tmp_path,
    )

    argv, workspace = target.command_argv("python3 -c 'print(42)'")

    assert argv[:4] == ["ssh", "-o", "BatchMode=yes", "gpu-workstation.example.com"]
    assert "cd -- /workspace/project" in argv[-1]
    assert workspace is None


def test_same_version_case_is_preferred_without_excluding_older_case(tmp_path: Path) -> None:
    workspace = tmp_path / "diagnosis"
    workspace.mkdir()
    (workspace / "README.md").write_text("diagnosis checkout\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AutoDebug Test",
            "-c",
            "user.email=autodebug@example.invalid",
            "commit",
            "-m",
            "Diagnosis baseline",
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    base_case = {
        "version": 1,
        "status": "verified",
        "title": "Shared preprocessing failure",
        "source": "https://example.invalid/case",
        "tags": ["preprocessing"],
        "match_terms": ["letterbox"],
        "symptom": "Letterbox preprocessing differs.",
        "first_test": "Compare preprocessed tensors.",
        "cause": "The preprocessing path diverged.",
        "fix": "Use the verified preprocessing path.",
        "compute": "One GPU.",
        "verification": "Tensor comparison passed.",
    }
    for case_id, scope in (("old-n17", ["GR00T N1.7"]), ("current-n2", ["GR00T N2"])):
        data = {**base_case, "id": case_id, "scope": scope}
        (case_dir / f"{case_id}.json").write_text(json.dumps(data), encoding="utf-8")
    report = REPORT.replace(
        "Dataset loading fails with a shard error.",
        "GR00T N2 letterbox preprocessing differs across platforms.",
    )

    incident = AutoDebugIncident.init_from_report(
        _report(tmp_path, report),
        workspace=workspace,
        run_root=tmp_path / "runs",
        case_directory=case_dir,
    )

    assert incident.related_case_ids == ["current-n2", "old-n17"]


def test_report_requires_the_common_sections(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Acceptance check"):
        AutoDebugIncident.init_from_report(
            _report(tmp_path, "## Problem\n\nMissing most fields.\n"),
            workspace=tmp_path,
            run_root=tmp_path / "runs",
        )


def test_only_verified_cases_enter_runtime_knowledge(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="status 'verified'"):
        SolvedCase.get_all(_case(tmp_path, status="candidate"))


def test_checked_in_runtime_cases_are_valid() -> None:
    cases = SolvedCase.get_all(DEFAULT_CASE_DIRECTORY)
    assert DEFAULT_CASE_DIRECTORY.is_dir()
    assert all(case.status == "verified" for case in cases)


def test_agent_skill_discovery_surfaces_are_complete() -> None:
    oss_root = Path(__file__).resolve().parents[3]
    for runtime in (".codex", ".claude"):
        for skill in ("autodebug", "insights"):
            discovered = oss_root / runtime / "skills" / skill / "SKILL.md"
            canonical = oss_root / "skills" / skill / "SKILL.md"
            assert discovered.read_bytes() == canonical.read_bytes()
        assert not (oss_root / runtime / "skills" / "autodebug-verify").exists()
    assert (oss_root / ".claude" / "agents" / "autodebug.md").read_bytes() == (
        oss_root / ".agents" / "autodebug.md"
    ).read_bytes()
    assert (oss_root / ".codex" / "agents" / "autodebug.toml").read_bytes() == (
        oss_root / ".agents" / "autodebug.toml"
    ).read_bytes()


def test_shipped_skill_wrapper_starts() -> None:
    oss_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            str(oss_root / "skills" / "autodebug" / "scripts" / "autodebug.py"),
            "--help",
        ],
        cwd=oss_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Run the GR00T AutoDebug workflow" in result.stdout


def test_nested_oss_workspace_uses_workspace_relative_targets(tmp_path: Path) -> None:
    incident = _nested_incident(tmp_path)
    workspace, revision = _nested_implementation(incident, tmp_path, change_outside_oss=False)

    incident.resume("Applied inside OSS.", workspace, revision)
    incident.verify(30)

    summary = next(
        run for run in reversed(incident.runs) if run.get("kind") == "verification-summary"
    )
    assert incident.status == "resolved"
    assert summary["changed_paths"] == ["fixed"]


def test_nested_oss_workspace_rejects_changes_outside_scope(tmp_path: Path) -> None:
    incident = _nested_incident(tmp_path)
    workspace, revision = _nested_implementation(incident, tmp_path, change_outside_oss=True)

    with pytest.raises(ValueError, match="outside the diagnosed workspace"):
        incident.resume("Changed OSS and monorepo files.", workspace, revision)


def test_nested_implementation_must_use_the_same_subdirectory(tmp_path: Path) -> None:
    incident = _nested_incident(tmp_path)
    workspace, revision = _nested_implementation(incident, tmp_path, change_outside_oss=False)

    with pytest.raises(ValueError, match="same repository subdirectory"):
        incident.resume(
            "Supplied the worktree root instead of its OSS directory.", workspace.parent, revision
        )


def test_incident_resolves_only_after_original_and_regression_pass(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    command, signature = _failing_reproducer(tmp_path)
    passing = f"{sys.executable} -c 'raise SystemExit(0)'"

    incident.run_reproducer(command, signature, 30)
    assert incident.phase == "diagnose"
    _support_and_defend(
        incident,
        passing,
        "The missing marker represents the localized defect.",
    )
    assert "Report the verified root cause" in incident.next_action
    incident.handoff(
        "The focused predicate confirms the marker is the failing boundary.",
        "Create the marker in the implementation checkout.",
        ["fixed"],
        [passing],
    )
    fix_artifact = Path(str(incident.fix_artifact))
    fix_artifact_text = fix_artifact.read_text(encoding="utf-8")
    workspace, revision = _implementation_workspace(incident, tmp_path, fixed=True)
    incident.resume("Applied by downstream test agent.", workspace, revision)
    incident.verify(30)

    assert incident.status == "resolved"
    assert incident.phase == "done"
    assert fix_artifact.read_text(encoding="utf-8") == fix_artifact_text
    assert Path(str(incident.verification_artifact)).is_file()
    assert "All checks passed: `True`" in Path(str(incident.verification_artifact)).read_text(
        encoding="utf-8"
    )


def test_failed_verification_returns_to_diagnosis(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    failing, signature = _failing_reproducer(tmp_path)
    passing = f"{sys.executable} -c 'raise SystemExit(0)'"
    incident.run_reproducer(failing, signature, 30)
    _support_and_defend(incident, passing, "A candidate factor is supported.")
    incident.handoff(
        "The predicate supports one candidate factor.",
        "One proposed change.",
        ["implementation-note"],
        [passing],
    )
    workspace, revision = _implementation_workspace(incident, tmp_path, fixed=False)
    incident.resume("Applied without the required marker.", workspace, revision)
    incident.verify(30)

    assert incident.status == "active"
    assert incident.phase == "diagnose"
    with pytest.raises(ValueError, match="new passing root-cause defense"):
        incident.handoff(
            "Reuse old evidence.",
            "Try the old change again.",
            ["candidate"],
            [passing],
        )


def test_zero_exit_with_failure_signature_is_not_resolved(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    signature = "EXPECTED BUG: failure remains visible"
    reproducer = tmp_path / "signature_reproducer.py"
    reproducer.write_text(
        "from pathlib import Path\n"
        f"print({signature!r})\n"
        "raise SystemExit(0 if Path('fixed').exists() else 1)\n",
        encoding="utf-8",
    )
    command = f"{sys.executable} {reproducer}"
    passing = f"{sys.executable} -c 'raise SystemExit(0)'"
    incident.run_reproducer(command, signature, 30)
    _support_and_defend(incident, passing, "The marker is the focused factor.")
    incident.handoff(
        "The focused check supports the marker change.",
        "Create the marker.",
        ["fixed"],
        [passing],
    )
    workspace, revision = _implementation_workspace(incident, tmp_path, fixed=True)
    incident.resume("Created the marker.", workspace, revision)

    incident.verify(30)

    summary = next(
        run for run in reversed(incident.runs) if run.get("kind") == "verification-summary"
    )
    assert incident.status == "active"
    assert incident.phase == "diagnose"
    assert summary["failure_signature_present"] is True


def test_changed_auto_frozen_reproducer_is_rejected(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    _prepare_handoff(incident, tmp_path)
    reproducer = tmp_path / "reproducer.py"
    assert str(reproducer) in incident.verification_asset_paths
    reproducer.write_text("raise SystemExit(0)\n", encoding="utf-8")
    workspace, revision = _implementation_workspace(incident, tmp_path, fixed=True)

    with pytest.raises(ValueError, match="Frozen verification asset changed"):
        incident.resume("Claimed implementation after changing the oracle.", workspace, revision)


def test_directly_executable_external_reproducer_is_frozen(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    signature = "EXPECTED BUG: direct executable"
    reproducer = tmp_path / "direct-reproducer"
    reproducer.write_text(
        f"#!/usr/bin/env python3\nprint({signature!r})\nraise SystemExit(1)\n",
        encoding="utf-8",
    )
    reproducer.chmod(0o755)
    passing = f"{sys.executable} -c 'raise SystemExit(0)'"
    incident.run_reproducer(f"{reproducer} --mode fail", signature, 30)
    assert str(reproducer) in incident.verification_asset_paths
    _support_and_defend(incident, passing, "The direct oracle reproduces.")
    incident.handoff(
        "The focused check supports one repository change.",
        "Apply the proposed implementation change.",
        ["implementation-note"],
        [passing],
    )
    workspace, revision = _implementation_workspace(incident, tmp_path, fixed=False)
    reproducer.write_text("#!/usr/bin/env python3\nraise SystemExit(0)\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Frozen verification asset changed"):
        incident.resume("Changed the direct oracle.", workspace, revision)


def test_external_reproducer_symlink_identity_is_frozen(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    signature = "EXPECTED BUG: symlink executable"
    failing_target = tmp_path / "failing-target"
    failing_target.write_text(
        f"#!/usr/bin/env python3\nprint({signature!r})\nraise SystemExit(1)\n",
        encoding="utf-8",
    )
    failing_target.chmod(0o755)
    passing_target = tmp_path / "passing-target"
    passing_target.write_text("#!/usr/bin/env python3\nraise SystemExit(0)\n", encoding="utf-8")
    passing_target.chmod(0o755)
    middle = tmp_path / "middle-reproducer"
    middle.symlink_to(failing_target.name)
    reproducer = tmp_path / "symlink-reproducer"
    reproducer.symlink_to(middle.name)
    passing = f"{sys.executable} -c 'raise SystemExit(0)'"
    incident.run_reproducer(f"{reproducer} --mode fail", signature, 30)
    assert str(reproducer) in incident.verification_asset_paths
    assert str(middle) in incident.verification_asset_paths
    _support_and_defend(incident, passing, "The symlinked oracle reproduces.")
    incident.handoff(
        "The focused check supports one repository change.",
        "Apply the proposed implementation change.",
        ["implementation-note"],
        [passing],
    )
    workspace, revision = _implementation_workspace(incident, tmp_path, fixed=False)
    middle.unlink()
    middle.symlink_to(passing_target.name)

    with pytest.raises(ValueError, match="Frozen verification asset changed"):
        incident.resume("Retargeted the external oracle.", workspace, revision)


def test_broken_external_reproducer_symlink_is_rejected(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    reproducer = tmp_path / "broken-reproducer"
    reproducer.symlink_to("missing-target")

    with pytest.raises(ValueError, match="External command asset is missing"):
        incident.run_reproducer(str(reproducer), "EXPECTED BUG", 30)


def test_external_file_under_a_symlinked_parent_is_frozen(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    signature = "EXPECTED BUG: symlinked parent"
    actual_directory = tmp_path / "actual-suite"
    actual_directory.mkdir()
    reproducer = actual_directory / "reproducer.py"
    reproducer.write_text(f"print({signature!r})\nraise SystemExit(1)\n", encoding="utf-8")
    linked_directory = tmp_path / "linked-suite"
    linked_directory.symlink_to(actual_directory.name)
    lexical_reproducer = linked_directory / reproducer.name

    incident.run_reproducer(f"{sys.executable} {lexical_reproducer}", signature, 30)

    assert incident.phase == "diagnose"
    assert str(linked_directory) in incident.verification_asset_paths
    assert str(reproducer) in incident.verification_asset_paths


def test_external_directory_verdict_input_is_rejected(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    signature = "EXPECTED BUG: external directory"
    reproducer = tmp_path / "directory_reproducer.py"
    reproducer.write_text(f"print({signature!r})\nraise SystemExit(1)\n", encoding="utf-8")
    external_directory = tmp_path / "external-suite"
    external_directory.mkdir()
    (external_directory / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot name an external directory"):
        incident.run_reproducer(
            f"{sys.executable} {reproducer} {external_directory}", signature, 30
        )


def test_implementation_cannot_rewrite_a_repository_verification_path(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "diagnosis"
    workspace.mkdir()
    signature = "EXPECTED BUG: repository oracle"
    oracle = workspace / "check.py"
    oracle.write_text(f"print({signature!r})\nraise SystemExit(1)\n", encoding="utf-8")
    (workspace / "passing.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    (workspace / "oracle.py").symlink_to("check.py")
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AutoDebug Test",
            "-c",
            "user.email=autodebug@example.invalid",
            "commit",
            "-m",
            "Diagnosis baseline",
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    incident = AutoDebugIncident.init_from_report(
        _report(tmp_path),
        workspace=workspace,
        run_root=tmp_path / "runs",
        case_directory=_case(tmp_path),
    )
    passing = f"{sys.executable} -c 'raise SystemExit(0)'"
    incident.run_reproducer(f"{sys.executable} oracle.py", signature, 30)
    _support_and_defend(incident, passing, "The repository check reproduces.")
    incident.handoff(
        "The repository check isolates the reported symptom.",
        "Change the implementation without weakening the check.",
        ["oracle.py"],
        [passing],
    )
    implementation = tmp_path / "implementation-oracle"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(implementation), incident.repo_commit],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    (implementation / "oracle.py").unlink()
    (implementation / "oracle.py").symlink_to("passing.py")
    subprocess.run(["git", "add", "oracle.py"], cwd=implementation, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AutoDebug Test",
            "-c",
            "user.email=autodebug@example.invalid",
            "commit",
            "-m",
            "Weaken oracle",
        ],
        cwd=implementation,
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=implementation,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    with pytest.raises(ValueError, match="protected verification path"):
        incident.resume("Changed only the check.", implementation, revision)


@pytest.mark.parametrize(
    ("command_template", "oracle_path"),
    [
        ("bash -c 'python3 check.py'", "check.py"),
        ("bash -lc 'python3 check.py'", "check.py"),
        ("sh -ec 'python3 check.py'", "check.py"),
        ("env bash -c 'python3 check.py'", "check.py"),
        ("/usr/bin/env bash -c 'python3 check.py'", "check.py"),
        ("/usr/bin/env -S \"bash -c 'python3 check.py'\"", "check.py"),
        ("{python} -m checks.fail", "checks/fail.py"),
    ],
)
def test_wrapped_repository_verification_paths_are_protected(
    tmp_path: Path, command_template: str, oracle_path: str
) -> None:
    workspace = tmp_path / "diagnosis"
    workspace.mkdir()
    signature = "EXPECTED BUG: wrapped repository oracle"
    (workspace / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    (workspace / "check.py").write_text(
        f"print({signature!r})\nraise SystemExit(1)\n", encoding="utf-8"
    )
    (workspace / "checks").mkdir()
    (workspace / "checks" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "checks" / "fail.py").write_text(
        f"print({signature!r})\nraise SystemExit(1)\n", encoding="utf-8"
    )
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AutoDebug Test",
            "-c",
            "user.email=autodebug@example.invalid",
            "commit",
            "-m",
            "Diagnosis baseline",
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    incident = AutoDebugIncident.init_from_report(
        _report(tmp_path),
        workspace=workspace,
        run_root=tmp_path / "runs",
        case_directory=_case(tmp_path),
    )
    command = command_template.format(python=sys.executable)
    passing = f"{sys.executable} -c 'raise SystemExit(0)'"
    incident.run_reproducer(command, signature, 30)
    _support_and_defend(incident, passing, "The wrapped check reproduces.")
    incident.handoff(
        "The wrapped check isolates the reported symptom.",
        "Change the implementation without weakening the check.",
        [oracle_path],
        [passing],
    )
    implementation = tmp_path / "implementation-wrapped"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(implementation), incident.repo_commit],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    (implementation / oracle_path).write_text("raise SystemExit(0)\n", encoding="utf-8")
    subprocess.run(["git", "add", oracle_path], cwd=implementation, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AutoDebug Test",
            "-c",
            "user.email=autodebug@example.invalid",
            "commit",
            "-m",
            "Weaken wrapped oracle",
        ],
        cwd=implementation,
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=implementation,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    with pytest.raises(ValueError, match="protected verification path"):
        incident.resume("Changed only the wrapped check.", implementation, revision)


def test_reproduction_must_hold_in_a_clean_baseline(tmp_path: Path) -> None:
    workspace = tmp_path / "diagnosis"
    workspace.mkdir()
    signature = "EXPECTED BUG: ignored trigger"
    (workspace / ".gitignore").write_text("bug-trigger\n", encoding="utf-8")
    (workspace / "check.py").write_text(
        "from pathlib import Path\n"
        f"if Path('bug-trigger').exists():\n    print({signature!r})\n    raise SystemExit(1)\n",
        encoding="utf-8",
    )
    (workspace / "bug-trigger").write_text("diagnosis-only state\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AutoDebug Test",
            "-c",
            "user.email=autodebug@example.invalid",
            "commit",
            "-m",
            "Diagnosis baseline",
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    incident = AutoDebugIncident.init_from_report(
        _report(tmp_path),
        workspace=workspace,
        run_root=tmp_path / "runs",
        case_directory=_case(tmp_path),
    )

    incident.run_reproducer(f"{sys.executable} check.py", signature, 30)

    assert incident.status == "waiting"
    assert incident.phase == "reproduce"
    assert incident.reproducer is None
    assert incident.runs[-1]["kind"] == "reproduce-clean"
    assert incident.runs[-1]["signature_matched"] is False


def test_ignored_implementation_file_cannot_supply_the_fix(tmp_path: Path) -> None:
    incident = _incident(tmp_path, ignore_fixed=True)
    _prepare_handoff(incident, tmp_path)
    workspace, revision = _implementation_workspace(incident, tmp_path, fixed=False)
    (workspace / "fixed").write_text("ignored local state\n", encoding="utf-8")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status == ""
    incident.resume("Only an ignored local marker exists.", workspace, revision)

    incident.verify(30)

    assert incident.status == "active"
    assert incident.phase == "diagnose"


def test_changed_diagnosis_checkout_blocks_implementation_resume(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    _prepare_handoff(incident, tmp_path)
    (incident.workspace / "README.md").write_text("changed while waiting\n", encoding="utf-8")
    workspace, revision = _implementation_workspace(incident, tmp_path, fixed=True)

    with pytest.raises(ValueError, match="diagnosis checkout changed"):
        incident.resume("Implementation arrived after diagnosis changed.", workspace, revision)

    assert incident.status == "blocked"


def test_nonzero_diagnostic_cannot_support_a_proposal(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    failing, signature = _failing_reproducer(tmp_path)
    incident.run_reproducer(failing, signature, 30)
    incident.run_diagnostic(failing, "This check itself fails.", "supported", 30)

    assert incident.runs[-1]["outcome"] == "inconclusive"
    with pytest.raises(ValueError, match="root-cause defense"):
        incident.handoff(
            "Unsupported analysis.",
            "Unsupported change.",
            ["candidate"],
            [failing],
        )


def test_root_cause_defense_requires_and_records_a_distinct_supported_check(
    tmp_path: Path,
) -> None:
    incident = _incident(tmp_path)
    failing, signature = _failing_reproducer(tmp_path)
    candidate_check = f"{sys.executable} -c 'raise SystemExit(0)'"
    defense_check = f"{sys.executable} -c 'print(\"matched control\")'"
    candidate = "The focused factor causes the reported failure."
    incident.run_reproducer(failing, signature, 30)
    incident.run_diagnostic(candidate_check, candidate, "supported", 30)

    with pytest.raises(ValueError, match="root-cause defense"):
        incident.handoff(
            "A candidate alone is insufficient.",
            "Do not implement this yet.",
            ["fixed"],
            [defense_check],
        )

    with pytest.raises(ValueError, match="distinct command"):
        incident.run_diagnostic(
            candidate_check,
            f"Root-cause defense for: {candidate}",
            "supported",
            30,
        )

    incident.run_diagnostic(
        defense_check,
        f"Root-cause defense for: {candidate}",
        "supported",
        30,
    )

    assert incident.runs[-1]["defends_hypothesis"] == candidate
    assert incident.runs[-1]["outcome"] == "supported"
    assert "Report the verified root cause" in incident.next_action
    incident.handoff(
        "The matched control defended the candidate cause.",
        "Apply the recommended repository change.",
        ["fixed"],
        [defense_check],
    )
    assert incident.hypothesis == candidate


def test_pause_resumes_and_user_can_cancel(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    incident.pause("a captured failing input")
    assert incident.status == "waiting"
    incident.resume()
    assert incident.phase == "reproduce"
    incident.block("Required robot hardware is not available in this environment.")
    assert incident.status == "blocked"
    incident.resume("Robot access was restored.")
    assert incident.status == "active"
    incident.cancel()
    assert incident.status == "cancelled"


def test_timeout_output_is_saved_as_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    incident = _incident(tmp_path)
    real_run = subprocess.run

    def raise_timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess[object]:
        if args and args[0] == ["check"]:
            raise subprocess.TimeoutExpired(
                cmd=["check"], timeout=1, output=b"partial output", stderr=b"partial error"
            )
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    incident.run_reproducer("check", "Timed out after 1 seconds", 1)

    log = Path(incident.runs[-1]["log"]).read_text(encoding="utf-8")
    assert incident.runs[-1]["exit_code"] == 124
    assert "partial output" in log
    assert "partial error" in log
    assert "Timed out after 1 seconds" in log


def test_command_launch_failure_is_logged_and_waits(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    signature = "definitely-not-an-autodebug-command"
    incident.run_reproducer(signature, signature, 30)

    assert incident.status == "waiting"
    assert incident.reproducer is None
    assert "FileNotFoundError" in incident.runs[-1]["launch_error"]
    assert "Command could not start" in Path(incident.runs[-1]["log"]).read_text(encoding="utf-8")


def test_nonzero_command_without_reported_signature_does_not_reproduce(
    tmp_path: Path,
) -> None:
    incident = _incident(tmp_path)
    failing = f"{sys.executable} -c 'print(\"different failure\"); raise SystemExit(1)'"
    incident.run_reproducer(failing, "reported failure", 30)

    assert incident.status == "waiting"
    assert incident.phase == "reproduce"
    assert incident.reproducer is None
    assert incident.failure_signature is None
    assert incident.runs[-1]["signature_matched"] is False

    incident.resume("Correcting the reproduction command.")
    corrected, signature = _failing_reproducer(tmp_path)
    incident.run_reproducer(corrected, signature, 30)
    assert incident.phase == "diagnose"
    assert incident.reproducer == corrected
    assert incident.failure_signature == signature


def test_handoff_rejects_an_unrelated_repository(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    _prepare_handoff(incident, tmp_path)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "README.md").write_text("different repository\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=unrelated, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=unrelated, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AutoDebug Test",
            "-c",
            "user.email=autodebug@example.invalid",
            "commit",
            "-m",
            "Unrelated repository",
        ],
        cwd=unrelated,
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=unrelated,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    with pytest.raises(ValueError, match="worktree of the diagnosed repository"):
        incident.resume("Unrelated implementation.", unrelated, revision)


def test_verification_rejects_a_moved_or_dirty_implementation(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    _prepare_handoff(incident, tmp_path)
    workspace, revision = _implementation_workspace(incident, tmp_path, fixed=True)
    incident.resume("Applied by downstream test agent.", workspace, revision)
    (workspace / "README.md").write_text("changed after handoff\n", encoding="utf-8")

    with pytest.raises(ValueError, match="uncommitted or untracked changes"):
        incident.verify(30)


def test_handoff_rejects_untracked_implementation_state(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    _prepare_handoff(incident, tmp_path)
    workspace, revision = _implementation_workspace(incident, tmp_path, fixed=False)
    (workspace / "fixed").write_text("untracked false pass\n", encoding="utf-8")

    with pytest.raises(ValueError, match="uncommitted or untracked changes"):
        incident.resume("Fix exists only as an untracked file.", workspace, revision)


def test_handoff_rejects_out_of_scope_change_and_accepts_corrected_commit(
    tmp_path: Path,
) -> None:
    incident = _incident(tmp_path)
    _prepare_handoff(incident, tmp_path)
    workspace, _ = _implementation_workspace(incident, tmp_path, fixed=True)
    (workspace / "README.md").write_text("oracle weakened\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AutoDebug Test",
            "-c",
            "user.email=autodebug@example.invalid",
            "commit",
            "-m",
            "Change outside proposed scope",
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(ValueError, match="outside the fix package"):
        incident.resume("Implementation also changed an oracle file.", workspace, revision)
    assert incident.status == "waiting"
    assert incident.phase == "implementation"

    (workspace / "README.md").write_text("diagnosis checkout\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AutoDebug Test",
            "-c",
            "user.email=autodebug@example.invalid",
            "commit",
            "-m",
            "Restore out-of-scope file",
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    corrected_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    incident.resume("Corrected implementation scope.", workspace, corrected_revision)
    incident.verify(30)
    assert incident.status == "resolved"


def test_changed_fix_package_is_rejected(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    _prepare_handoff(incident, tmp_path)
    fix_artifact = Path(str(incident.fix_artifact))
    fix_artifact.write_text(
        fix_artifact.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8"
    )
    workspace, revision = _implementation_workspace(incident, tmp_path, fixed=True)

    with pytest.raises(ValueError, match="changed after handoff"):
        incident.resume("Applied by downstream test agent.", workspace, revision)


def test_changed_verification_asset_is_rejected(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    failing, signature = _failing_reproducer(tmp_path)
    passing = f"{sys.executable} -c 'raise SystemExit(0)'"
    asset = incident.path.parent / "acceptance.py"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text("raise SystemExit(0)\n", encoding="utf-8")
    incident.run_reproducer(failing, signature, 30)
    _support_and_defend(incident, passing, "A focused factor is supported.")
    incident.handoff(
        "The focused check isolates the failing factor.",
        "Apply the smallest change in the implementation checkout.",
        ["fixed"],
        [passing],
        [str(asset)],
    )
    workspace, revision = _implementation_workspace(incident, tmp_path, fixed=True)
    incident.resume("Applied by downstream test agent.", workspace, revision)
    asset.write_text("print('weakened check')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Frozen verification asset changed"):
        incident.verify(30)


def test_verification_blocks_if_a_check_changes_the_implementation(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    failing, signature = _failing_reproducer(tmp_path)
    passing = f"{sys.executable} -c 'raise SystemExit(0)'"
    mutating_regression = (
        f"{sys.executable} -c 'from pathlib import Path; "
        'Path("README.md").write_text("changed by check")\''
    )
    incident.run_reproducer(failing, signature, 30)
    _support_and_defend(incident, passing, "A focused factor is supported.")
    incident.handoff(
        "The focused check isolates the failing factor.",
        "Apply the smallest change in the implementation checkout.",
        ["fixed"],
        [mutating_regression],
    )
    workspace, revision = _implementation_workspace(incident, tmp_path, fixed=True)
    incident.resume("Applied by downstream test agent.", workspace, revision)
    incident.verify(30)

    assert incident.status == "blocked"
    assert incident.phase == "verify"
    assert incident.runs[-1]["integrity_error"] is not None
    assert "no fix decision was made" in Path(str(incident.verification_artifact)).read_text(
        encoding="utf-8"
    )


def test_verification_blocks_if_a_check_changes_a_frozen_asset(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    failing, signature = _failing_reproducer(tmp_path)
    passing = f"{sys.executable} -c 'raise SystemExit(0)'"
    asset = incident.path.parent / "self_mutating_check.py"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text(
        "from pathlib import Path\nPath(__file__).write_text('raise SystemExit(0)\\n')\n",
        encoding="utf-8",
    )
    incident.run_reproducer(failing, signature, 30)
    _support_and_defend(incident, passing, "A focused factor is supported.")
    incident.handoff(
        "The focused check isolates the failing factor.",
        "Apply the smallest change in the implementation checkout.",
        ["fixed"],
        [f"{sys.executable} {asset}"],
        [str(asset)],
    )
    workspace, revision = _implementation_workspace(incident, tmp_path, fixed=True)
    incident.resume("Applied by downstream test agent.", workspace, revision)
    incident.verify(30)

    assert incident.status == "blocked"
    assert "Frozen verification asset changed" in incident.runs[-1]["integrity_error"]


def test_diagnosis_command_cannot_change_tracked_files(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    signature = "EXPECTED BUG: diagnosis command mutated source"
    mutating = (
        f"{sys.executable} -c 'from pathlib import Path; "
        f'Path("README.md").write_text("changed"); print("{signature}"); '
        "raise SystemExit(1)'"
    )

    with pytest.raises(ValueError, match="changed the recorded revision"):
        incident.run_reproducer(mutating, signature, 30)

    persisted = AutoDebugIncident.init_from_path(incident.path)
    assert persisted.status == "blocked"
    assert persisted.runs[-1]["isolation_violation"] is True


def test_diagnosis_command_cannot_commit_source_changes(tmp_path: Path) -> None:
    incident = _incident(tmp_path)
    signature = "EXPECTED BUG: diagnosis command committed source"
    commit_script = tmp_path / "commit_diagnosis.py"
    commit_script.write_text(
        "from pathlib import Path\n"
        "import subprocess\n"
        "Path('README.md').write_text('committed change\\n')\n"
        "subprocess.run(['git', 'add', 'README.md'], check=True)\n"
        "subprocess.run([\n"
        "    'git', '-c', 'user.name=AutoDebug Test',\n"
        "    '-c', 'user.email=autodebug@example.invalid',\n"
        "    'commit', '-m', 'Forbidden diagnosis commit'\n"
        "])\n"
        f"print({signature!r})\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed the recorded revision"):
        incident.run_reproducer(f"{sys.executable} {commit_script}", signature, 30)

    persisted = AutoDebugIncident.init_from_path(incident.path)
    assert persisted.runs[-1]["recorded_revision"] != persisted.runs[-1]["observed_revision"]
