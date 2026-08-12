# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The persisted incident and code-enforced AutoDebug loop."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tempfile
import time
from typing import Any

from skills.autodebug.infra.execution_target import (
    ExecutionTarget,
    load_execution_targets,
)
from skills.autodebug.infra.git_checkout import GitCheckout
from skills.autodebug.infra.solved_case import SolvedCase
from skills.insights.infra.insight_projection import search_insights

REQUIRED_REPORT_HEADINGS = (
    "Problem",
    "Expected behavior",
    "Observed behavior",
    "Reproduction",
    "Acceptance check",
)
TERMINAL_STATUSES = {"resolved", "cancelled"}
DEFAULT_CASE_DIRECTORY = Path(__file__).resolve().parent / "cases"


@dataclass
class AutoDebugIncident:
    path: Path
    workspace: Path
    report_path: Path
    report_sha256: str
    report: str
    repo_commit: str
    repo_common_dir: Path
    diagnosis_tree_sha256: str
    case_directory: Path
    related_case_ids: list[str]
    id: str
    execution_targets: dict[str, dict[str, str]] = field(default_factory=dict)
    status: str = "active"
    phase: str = "reproduce"
    reproducer: str | None = None
    failure_signature: str | None = None
    hypothesis: str | None = None
    analysis: str | None = None
    proposed_change: str | None = None
    implementation_targets: list[str] = field(default_factory=list)
    regression_commands: list[str] = field(default_factory=list)
    verification_asset_paths: list[str] = field(default_factory=list)
    verification_asset_sha256: list[str] = field(default_factory=list)
    verification_repo_paths: list[str] = field(default_factory=list)
    fix_artifact: str | None = None
    implementation_workspace: Path | None = None
    implementation_ref: str | None = None
    verification_artifact: str | None = None
    blocked_reason: str | None = None
    next_action: str = "Run the smallest command that demonstrates the reported failure."
    runs: list[dict[str, Any]] = field(default_factory=list)
    version: int = 1

    @classmethod
    def init_from_report(
        cls,
        report_path: Path,
        workspace: Path,
        run_root: Path,
        case_directory: Path = DEFAULT_CASE_DIRECTORY,
        target_config: Path | None = None,
        insights_root: Path | None = None,
    ) -> AutoDebugIncident:
        report_path = report_path.resolve()
        workspace = workspace.resolve()
        case_directory = case_directory.resolve()
        if not workspace.is_dir():
            raise ValueError(f"Workspace is not a directory: {workspace}")
        report = report_path.read_text(encoding="utf-8")
        if not report.strip():
            raise ValueError("Bug report cannot be empty")
        missing = [
            heading
            for heading in REQUIRED_REPORT_HEADINGS
            if re.search(
                rf"^##\s+{re.escape(heading)}\s*$",
                report,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            is None
        ]
        if missing:
            raise ValueError("Bug report is missing sections: " + ", ".join(missing))

        checkout = GitCheckout.init_from_path(workspace)
        if checkout.tracked_status:
            raise ValueError(
                "Diagnosis workspace has tracked changes; commit or stash them before AutoDebug"
            )
        report_sha256 = hashlib.sha256(report.encode("utf-8")).hexdigest()
        identity = hashlib.sha256(
            f"{report_sha256}\0{workspace}\0{checkout.head}".encode()
        ).hexdigest()[:12]
        incident_path = run_root.resolve() / f"incident-{identity}" / "incident.json"
        if incident_path.exists():
            incident = cls.init_from_path(incident_path)
            if target_config is not None:
                incident.execution_targets = load_execution_targets(
                    target_config, default_workspace=workspace
                )
                incident.save()
            return incident

        ranked_cases = sorted(
            (
                (case.match_score(report), case.version_affinity(report), case.id)
                for case in SolvedCase.get_all(case_directory)
            ),
            key=lambda item: (-item[0], -item[1], item[2]),
        )
        incident = cls(
            path=incident_path,
            workspace=workspace,
            report_path=report_path,
            report_sha256=report_sha256,
            report=report,
            repo_commit=checkout.head,
            repo_common_dir=checkout.common_dir,
            diagnosis_tree_sha256=checkout.tracked_tree_sha256,
            case_directory=case_directory,
            related_case_ids=[case_id for score, _, case_id in ranked_cases if score > 0][:3],
            id=f"incident-{identity}",
            execution_targets=load_execution_targets(target_config, default_workspace=workspace),
        )
        incident.save()
        incident.lookup_insights(
            incident._initial_insight_query(incident.report),
            insights_root=insights_root,
        )
        return incident

    @classmethod
    def init_from_path(cls, path: Path) -> AutoDebugIncident:
        path = path.resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError(f"Unsupported AutoDebug incident: {path}")
        return cls(
            path=path,
            workspace=Path(data["workspace"]),
            report_path=Path(data["report_path"]),
            report_sha256=data["report_sha256"],
            report=data["report"],
            repo_commit=data["repo_commit"],
            repo_common_dir=Path(data["repo_common_dir"]),
            diagnosis_tree_sha256=data["diagnosis_tree_sha256"],
            case_directory=Path(data["case_directory"]),
            related_case_ids=list(data["related_case_ids"]),
            id=data["id"],
            execution_targets=dict(
                data.get(
                    "execution_targets",
                    {
                        "local": {
                            "kind": "local",
                            "workspace": data["workspace"],
                        }
                    },
                )
            ),
            status=data["status"],
            phase=data["phase"],
            reproducer=data.get("reproducer"),
            failure_signature=data.get("failure_signature"),
            hypothesis=data.get("hypothesis"),
            analysis=data.get("analysis"),
            proposed_change=data.get("proposed_change"),
            implementation_targets=list(data.get("implementation_targets", [])),
            regression_commands=list(data.get("regression_commands", [])),
            verification_asset_paths=list(data.get("verification_asset_paths", [])),
            verification_asset_sha256=list(data.get("verification_asset_sha256", [])),
            verification_repo_paths=list(data.get("verification_repo_paths", [])),
            fix_artifact=data.get("fix_artifact"),
            implementation_workspace=(
                Path(data["implementation_workspace"])
                if data.get("implementation_workspace")
                else None
            ),
            implementation_ref=data.get("implementation_ref"),
            verification_artifact=data.get("verification_artifact"),
            blocked_reason=data.get("blocked_reason"),
            next_action=data["next_action"],
            runs=list(data.get("runs", [])),
            version=data["version"],
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._refresh_run_summaries()
        self.path.write_text(self.to_json_text(), encoding="utf-8")

    def _refresh_run_summaries(self) -> None:
        """Keep a compact ledger while preserving each full command log separately."""
        for run in self.runs:
            parts = [str(run.get("kind", "unknown"))]
            if run.get("target"):
                parts.append(f"target={run['target']}")
            if run.get("kind") == "insights":
                parts.append(f"available={run.get('available', False)}")
                parts.append(f"records={len(run.get('record_ids', []))}")
            if "exit_code" in run:
                parts.append(f"exit={run['exit_code']}")
            if run.get("outcome"):
                parts.append(f"outcome={run['outcome']}")
            if "signature_matched" in run:
                parts.append(f"signature_matched={run['signature_matched']}")
            if "passed" in run:
                parts.append(f"passed={run['passed']}")
            if run.get("hypothesis"):
                hypothesis = " ".join(str(run["hypothesis"]).split())
                if len(hypothesis) > 120:
                    hypothesis = hypothesis[:117] + "..."
                parts.append(f"hypothesis={hypothesis}")
            if run.get("launch_error"):
                parts.append("launch_error=true")
            if run.get("integrity_error"):
                parts.append("integrity_error=true")
            run["summary"] = "; ".join(parts)

    def to_json_text(self) -> str:
        self._refresh_run_summaries()
        data = {
            "version": self.version,
            "id": self.id,
            "status": self.status,
            "phase": self.phase,
            "next_action": self.next_action,
            "workspace": str(self.workspace),
            "repo_commit": self.repo_commit,
            "repo_common_dir": str(self.repo_common_dir),
            "diagnosis_tree_sha256": self.diagnosis_tree_sha256,
            "report_path": str(self.report_path),
            "report_sha256": self.report_sha256,
            "report": self.report,
            "case_directory": str(self.case_directory),
            "related_case_ids": self.related_case_ids,
            "execution_targets": self.execution_targets,
            "reproducer": self.reproducer,
            "failure_signature": self.failure_signature,
            "hypothesis": self.hypothesis,
            "analysis": self.analysis,
            "proposed_change": self.proposed_change,
            "implementation_targets": self.implementation_targets,
            "regression_commands": self.regression_commands,
            "verification_asset_paths": self.verification_asset_paths,
            "verification_asset_sha256": self.verification_asset_sha256,
            "verification_repo_paths": self.verification_repo_paths,
            "fix_artifact": self.fix_artifact,
            "implementation_workspace": (
                str(self.implementation_workspace) if self.implementation_workspace else None
            ),
            "implementation_ref": self.implementation_ref,
            "verification_artifact": self.verification_artifact,
            "blocked_reason": self.blocked_reason,
            "runs": self.runs,
        }
        return json.dumps(data, indent=2, sort_keys=True) + "\n"

    def to_summary_json_text(self) -> str:
        data = {
            "incident": str(self.path),
            "id": self.id,
            "status": self.status,
            "phase": self.phase,
            "next": self.next_action,
            "related_cases": self.related_case_ids,
            "execution_targets": sorted(self.execution_targets),
            "fix_artifact": self.fix_artifact,
            "verification_artifact": self.verification_artifact,
        }
        if self.runs:
            last_run = self.runs[-1]
            data["last_run"] = {
                "kind": last_run.get("kind"),
                "target": last_run.get("target"),
                "log": last_run.get("log"),
                "summary": last_run.get("summary"),
            }
            if "exit_code" in last_run:
                data["last_run"]["exit_code"] = last_run["exit_code"]
        return json.dumps(data, indent=2, sort_keys=True) + "\n"

    def related_cases_text(self) -> str:
        cases = {case.id: case for case in SolvedCase.get_all(self.case_directory)}
        selected = [cases[case_id] for case_id in self.related_case_ids if case_id in cases]
        if not selected:
            return "No verified solved case matched this report.\n"
        return "\n".join(case.to_reference_text() for case in selected)

    def execution_targets_text(self) -> str:
        return (
            json.dumps(
                {"version": 1, "targets": self.execution_targets},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    @staticmethod
    def _initial_insight_query(report: str) -> str:
        sections = []
        for heading in ("Problem", "Observed behavior"):
            match = re.search(
                rf"^##\s+{re.escape(heading)}\s*$\n(?P<body>.*?)(?=^##\s+|\Z)",
                report,
                flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
            )
            if match and match.group("body").strip():
                sections.append(match.group("body").strip())
        return " ".join(sections) or report

    def lookup_insights(
        self,
        query: str,
        *,
        insights_root: Path | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Record a deterministic, non-authoritative knowledge lookup."""
        if self.status in TERMINAL_STATUSES:
            raise ValueError(f"Terminal incident cannot be changed: {self.status}")
        started = time.monotonic()
        result = search_insights(
            query,
            start=self.workspace,
            insights_root=insights_root,
            limit=limit,
        )
        duration_seconds = time.monotonic() - started
        log_directory = self.path.parent / "logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        index = len(self.runs) + 1
        log_path = log_directory / f"{index:03d}-insights.log"
        log_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        log_path.write_text(log_text, encoding="utf-8")
        projection = result.get("projection") or {}
        self.runs.append(
            {
                "index": index,
                "kind": "insights",
                "query": result["query"],
                "available": result["available"],
                "record_ids": list(result["record_ids"]),
                "projection_schema_version": projection.get("schema_version"),
                "projection_generated_at": projection.get("generated_at"),
                "projection_sha256": projection.get("sha256"),
                "projection_root": projection.get("root"),
                "error": result.get("error"),
                "duration_seconds": duration_seconds,
                "log": str(log_path),
                "log_sha256": hashlib.sha256(log_text.encode()).hexdigest(),
            }
        )
        self.save()
        return result

    def run_probe(
        self,
        command: str,
        target: str,
        timeout_seconds: int,
    ) -> AutoDebugIncident:
        """Run a read-only evidence probe without advancing the incident phase."""
        self._ensure_active()
        self._execute(
            command,
            "probe",
            timeout_seconds,
            target_name=target,
        )
        self.save()
        return self

    def run_reproducer(
        self, command: str, failure_signature: str, timeout_seconds: int
    ) -> AutoDebugIncident:
        self._ensure_active()
        if self.phase != "reproduce":
            raise ValueError(f"Reproduction requires phase 'reproduce', not {self.phase!r}")
        if self.reproducer is not None and command != self.reproducer:
            raise ValueError(
                "The original reproducer is frozen; create a new incident to change it"
            )
        if not failure_signature.strip():
            raise ValueError("Failure signature cannot be empty")
        if self.failure_signature is not None and failure_signature != self.failure_signature:
            raise ValueError("The original failure signature is frozen")
        reproducer_assets = self._external_command_files(command, self.workspace)
        reproducer_asset_digests = {
            str(path): self._verification_asset_digest(path) for path in reproducer_assets
        }
        exit_code, _, output = self._execute(command, "reproduce", timeout_seconds)
        signature_matched = failure_signature in output
        self.runs[-1]["failure_signature"] = failure_signature
        self.runs[-1]["signature_matched"] = signature_matched
        launch_error = self.runs[-1].get("launch_error")
        diagnosis_reproduced = exit_code != 0 and signature_matched and launch_error is None
        clean_reproduced = False
        clean_launch_error = None
        if diagnosis_reproduced:
            with self._clean_repository_workspace(self.repo_commit) as clean_workspace:
                clean_exit_code, _, clean_output = self._execute(
                    command,
                    "reproduce-clean",
                    timeout_seconds,
                    workspace=clean_workspace,
                )
                clean_signature_matched = failure_signature in clean_output
                self.runs[-1]["failure_signature"] = failure_signature
                self.runs[-1]["signature_matched"] = clean_signature_matched
                clean_launch_error = self.runs[-1].get("launch_error")
                clean_reproduced = (
                    clean_exit_code != 0 and clean_signature_matched and clean_launch_error is None
                )
                try:
                    self._validate_clean_runner_checkout(
                        GitCheckout.init_from_path(clean_workspace), self.repo_commit
                    )
                except ValueError as exc:
                    self.status = "blocked"
                    self.blocked_reason = str(exc)
                    self.next_action = self.blocked_reason
                    self.runs[-1]["isolation_violation"] = True
                    self.save()
                    raise
        if diagnosis_reproduced and clean_reproduced:
            try:
                self._freeze_verification_assets(
                    reproducer_assets, expected_digests=reproducer_asset_digests
                )
            except ValueError as exc:
                self.status = "blocked"
                self.blocked_reason = str(exc)
                self.next_action = self.blocked_reason
                self.save()
                raise
            self.reproducer = command
            self.failure_signature = failure_signature
            self.status = "active"
            self.phase = "diagnose"
            self.next_action = (
                "Locate the earliest failing boundary and run one discriminating diagnostic check."
            )
        else:
            self.status = "waiting"
            self.phase = "reproduce"
            if launch_error is not None:
                self.next_action = (
                    "The reproduction command could not start. Correct the command or environment, "
                    "then resume this incident."
                )
            elif diagnosis_reproduced and clean_launch_error is not None:
                self.next_action = (
                    "The command could not start in the clean baseline runner. Correct the "
                    "command or environment, then resume this incident."
                )
            elif diagnosis_reproduced:
                self.next_action = (
                    "The reported failure does not reproduce in the clean baseline runner. "
                    "Provide the required external assets or LFS inputs, then resume."
                )
            elif exit_code == 0:
                self.next_action = (
                    "The failure did not reproduce. Request the most useful missing artifact or "
                    "condition, then resume this incident."
                )
            else:
                self.next_action = (
                    "The command failed for a different reason. Repair the reproducer or request "
                    "the evidence needed to match the reported failure signature, then resume."
                )
        self.save()
        return self

    def run_diagnostic(
        self,
        command: str,
        hypothesis: str,
        outcome: str,
        timeout_seconds: int,
        target: str | None = None,
    ) -> AutoDebugIncident:
        self._ensure_active()
        if self.phase != "diagnose":
            raise ValueError(f"Diagnostic checks require phase 'diagnose', not {self.phase!r}")
        if outcome not in {"supported", "rejected", "inconclusive"}:
            raise ValueError("Diagnostic outcome must be supported, rejected, or inconclusive")
        if not hypothesis.strip():
            raise ValueError("Diagnostic hypothesis cannot be empty")
        hypothesis_text = hypothesis.strip()
        defense_prefix = "Root-cause defense for:"
        defended_hypothesis = None
        if hypothesis_text.casefold().startswith(defense_prefix.casefold()):
            defended_hypothesis = hypothesis_text[len(defense_prefix) :].strip()
            candidate = next(
                (
                    run
                    for run in reversed(self.runs)
                    if run.get("kind") == "diagnose"
                    and run.get("outcome") == "supported"
                    and run.get("hypothesis") == defended_hypothesis
                ),
                None,
            )
            if not defended_hypothesis or candidate is None:
                raise ValueError(
                    "A root-cause defense must name a prior supported diagnostic hypothesis"
                )
            if candidate.get("command") == command:
                raise ValueError("A root-cause defense must use a distinct command")
        exit_code, _, _ = self._execute(
            command,
            "diagnose",
            timeout_seconds,
            target_name=target,
        )
        recorded_outcome = outcome if exit_code == 0 else "inconclusive"
        self.runs[-1]["hypothesis"] = hypothesis_text
        self.runs[-1]["outcome"] = recorded_outcome
        if defended_hypothesis is not None:
            self.runs[-1]["defends_hypothesis"] = defended_hypothesis
        if recorded_outcome == "supported":
            if defended_hypothesis is not None:
                self.next_action = (
                    "The distinct root-cause defense passed. Report the verified root cause, "
                    "evidence, uncertainty, and recommended action. Prepare a handoff only if "
                    "implementation is authorized."
                )
            else:
                self.next_action = (
                    "Challenge this candidate with a distinct control or counterfactual and "
                    "record the result before claiming a verified root cause."
                )
        else:
            self.next_action = (
                "Update the evidence and test the next highest-value hypothesis at the earliest "
                "failing boundary."
            )
        self.save()
        return self

    def handoff(
        self,
        analysis: str,
        change: str,
        implementation_targets: list[str],
        regressions: list[str],
        verification_assets: list[str] | None = None,
    ) -> AutoDebugIncident:
        self._ensure_active()
        last_verification = max(
            (
                index
                for index, run in enumerate(self.runs)
                if run.get("kind") == "verification-summary"
            ),
            default=-1,
        )
        supported = next(
            (
                run
                for run in reversed(self.runs[last_verification + 1 :])
                if run.get("kind") == "diagnose"
                and run.get("exit_code") == 0
                and run.get("outcome") == "supported"
                and run.get("defends_hypothesis")
            ),
            None,
        )
        if self.phase != "diagnose" or supported is None:
            raise ValueError("A new passing root-cause defense must support the handoff first")
        if not analysis.strip() or not change.strip():
            raise ValueError("Handoff analysis and proposed change cannot be empty")
        if not implementation_targets:
            raise ValueError("Handoff requires at least one implementation target")
        normalized_targets = []
        for target in implementation_targets:
            normalized_target = target.strip().replace("\\", "/")
            target_path = PurePosixPath(normalized_target)
            if (
                not normalized_target
                or target_path.is_absolute()
                or normalized_target == "."
                or ".." in target_path.parts
            ):
                raise ValueError(
                    "Implementation targets must be relative to the diagnosis workspace"
                )
            normalized_targets.append(normalized_target.rstrip("/"))
        if not regressions or any(not command.strip() for command in regressions):
            raise ValueError("Handoff requires at least one non-empty regression command")
        self._assert_verification_assets_unchanged()
        self.hypothesis = str(supported["defends_hypothesis"])
        self.analysis = analysis.strip()
        self.proposed_change = change.strip()
        self.implementation_targets = normalized_targets
        self.regression_commands = [command.strip() for command in regressions]
        assets_to_freeze: list[Path] = []
        for asset in verification_assets or []:
            asset_path = Path(asset).expanduser().absolute()
            if not asset_path.is_symlink() and not asset_path.is_file():
                raise ValueError(f"Verification asset is not a file: {asset_path}")
            try:
                asset_path.relative_to(self.path.parent)
            except ValueError as exc:
                raise ValueError("Verification assets must live in the incident directory") from exc
            assets_to_freeze.append(asset_path)
            if asset_path.is_symlink():
                assets_to_freeze.append(asset_path.resolve())
        for regression in self.regression_commands:
            assets_to_freeze.extend(self._external_command_files(regression, self.workspace))
        self._freeze_verification_assets(assets_to_freeze)
        verdict_commands = [self.reproducer, *self.regression_commands]
        self.verification_repo_paths = sorted(
            {
                path
                for command in verdict_commands
                if command is not None
                for path in self._repository_command_paths(command, self.workspace)
            }
        )
        self.implementation_workspace = None
        self.implementation_ref = None
        self.verification_artifact = None
        handoff_index = 1 + sum(run.get("kind") == "handoff" for run in self.runs)
        self.fix_artifact = str(self.path.parent / f"fix-package-{handoff_index:03d}.md")
        self.runs.append(
            {
                "index": len(self.runs) + 1,
                "kind": "handoff",
                "hypothesis": self.hypothesis,
                "analysis": self.analysis,
                "change": self.proposed_change,
                "implementation_targets": self.implementation_targets,
                "regressions": self.regression_commands,
                "verification_assets": self.verification_asset_paths,
                "verification_repo_paths": self.verification_repo_paths,
                "artifact": self.fix_artifact,
            }
        )
        self.status = "waiting"
        self.phase = "implementation"
        self.next_action = (
            "Give the fix package to a separate implementation agent. Resume with its commit or "
            "worktree after it applies and commits the proposed change."
        )
        self._write_fix_artifact()
        self.runs[-1]["artifact_sha256"] = hashlib.sha256(
            Path(self.fix_artifact).read_bytes()
        ).hexdigest()
        self.save()
        return self

    def verify(self, timeout_seconds: int) -> AutoDebugIncident:
        self._ensure_active()
        if (
            self.phase != "verify"
            or self.reproducer is None
            or self.failure_signature is None
            or self.proposed_change is None
        ):
            raise ValueError("Verification requires a reproduced failure and implemented handoff")
        if not self.regression_commands:
            raise ValueError("Verification requires frozen regression commands from the handoff")
        if self.implementation_workspace is None or self.implementation_ref is None:
            raise ValueError("Verification requires the downstream workspace and commit ref")
        self._assert_diagnosis_checkout_unchanged()
        self._assert_fix_artifact_unchanged()
        self._assert_verification_assets_unchanged()
        implementation_checkout = GitCheckout.init_from_path(self.implementation_workspace)
        self._validate_implementation_checkout(implementation_checkout, self.implementation_ref)
        implementation_diff, implementation_diff_stat, changed_paths = (
            self._get_implementation_diff(implementation_checkout, self.implementation_ref)
        )

        baseline_integrity_error = None
        with self._clean_repository_workspace(self.repo_commit) as baseline_workspace:
            baseline_exit_code, _, baseline_output = self._execute(
                self.reproducer,
                "verify-baseline",
                timeout_seconds,
                workspace=baseline_workspace,
            )
            baseline_signature_present = self.failure_signature in baseline_output
            self.runs[-1]["failure_signature"] = self.failure_signature
            self.runs[-1]["signature_matched"] = baseline_signature_present
            try:
                self._validate_clean_runner_checkout(
                    GitCheckout.init_from_path(baseline_workspace), self.repo_commit
                )
                self._assert_verification_assets_unchanged()
            except ValueError as exc:
                baseline_integrity_error = str(exc)
        baseline_reproduced = (
            baseline_exit_code != 0
            and baseline_signature_present
            and self.runs[-1].get("launch_error") is None
            and baseline_integrity_error is None
        )

        original_exit_code = None
        failure_signature_present = False
        regression_results = []
        integrity_error = baseline_integrity_error
        if integrity_error is None:
            with self._clean_repository_workspace(
                self.implementation_ref
            ) as verification_workspace:
                original_exit_code, _, original_output = self._execute(
                    self.reproducer,
                    "verify-original",
                    timeout_seconds,
                    workspace=verification_workspace,
                )
                failure_signature_present = self.failure_signature in original_output
                self.runs[-1]["failure_signature"] = self.failure_signature
                self.runs[-1]["signature_matched"] = failure_signature_present
                for command in self.regression_commands:
                    exit_code, log_path, _ = self._execute(
                        command,
                        "verify-regression",
                        timeout_seconds,
                        workspace=verification_workspace,
                    )
                    regression_results.append(
                        {"command": command, "exit_code": exit_code, "log": str(log_path)}
                    )
                try:
                    self._validate_implementation_checkout(
                        GitCheckout.init_from_path(verification_workspace),
                        self.implementation_ref,
                    )
                    self._assert_diagnosis_checkout_unchanged()
                    self._assert_verification_assets_unchanged()
                except ValueError as exc:
                    integrity_error = str(exc)
        passed = (
            baseline_reproduced
            and original_exit_code == 0
            and not failure_signature_present
            and all(result["exit_code"] == 0 for result in regression_results)
            and integrity_error is None
        )
        self.runs.append(
            {
                "index": len(self.runs) + 1,
                "kind": "verification-summary",
                "baseline_exit_code": baseline_exit_code,
                "baseline_failure_signature_present": baseline_signature_present,
                "baseline_reproduced": baseline_reproduced,
                "original_exit_code": original_exit_code,
                "failure_signature_present": failure_signature_present,
                "regressions": regression_results,
                "passed": passed,
                "hypothesis": self.hypothesis,
                "change": self.proposed_change,
                "implementation_diff_sha256": hashlib.sha256(implementation_diff).hexdigest(),
                "implementation_diff_stat": implementation_diff_stat,
                "changed_paths": changed_paths,
                "integrity_error": integrity_error,
            }
        )
        if integrity_error is not None:
            self.status = "blocked"
            self.phase = "verify"
            self.blocked_reason = integrity_error
            self.next_action = (
                "Restore the recorded implementation commit to a clean worktree, then resume "
                "verification."
            )
        elif not baseline_reproduced:
            self.status = "waiting"
            self.phase = "verify"
            self.next_action = (
                "The clean original revision no longer reproduces the frozen failure. Restore "
                "the required environment or assets, then resume verification."
            )
        elif passed:
            self.status = "resolved"
            self.phase = "done"
            self.next_action = (
                "The original failure and regression checks pass. Review the evidence and "
                "contribute a solved case if it is reusable."
            )
        else:
            self.status = "active"
            self.phase = "diagnose"
            self.next_action = (
                "Verification failed. Preserve the result, update priorities, and run the next "
                "diagnostic check."
            )
        verification_index = sum(run.get("kind") == "verification-summary" for run in self.runs)
        self.verification_artifact = str(
            self.path.parent / f"verification-result-{verification_index:03d}.md"
        )
        self._write_verification_artifact()
        self.runs[-1]["artifact"] = self.verification_artifact
        self.runs[-1]["artifact_sha256"] = hashlib.sha256(
            Path(self.verification_artifact).read_bytes()
        ).hexdigest()
        self.save()
        return self

    def pause(self, need: str) -> AutoDebugIncident:
        self._ensure_active()
        if not need.strip():
            raise ValueError("Pause condition cannot be empty")
        self.status = "waiting"
        self.next_action = f"Resume when available: {need.strip()}"
        self.runs.append(
            {
                "index": len(self.runs) + 1,
                "kind": "pause",
                "need": need.strip(),
                "resume_phase": self.phase,
            }
        )
        self.save()
        return self

    def resume(
        self,
        note: str | None = None,
        implementation_workspace: Path | None = None,
        implementation_ref: str | None = None,
    ) -> AutoDebugIncident:
        if self.status not in {"waiting", "blocked"}:
            raise ValueError("Only a waiting or blocked incident can be resumed")
        self._assert_diagnosis_checkout_unchanged()
        self._assert_verification_assets_unchanged()
        if self.phase == "implementation":
            if note is None or not note.strip():
                raise ValueError(
                    "Resuming an implementation handoff requires an implementation note"
                )
            if implementation_workspace is None or implementation_ref is None:
                raise ValueError(
                    "Resuming an implementation handoff requires a workspace and commit ref"
                )
            resolved_workspace = implementation_workspace.resolve()
            if not resolved_workspace.is_dir():
                raise ValueError(
                    f"Implementation workspace is not a directory: {resolved_workspace}"
                )
            if resolved_workspace == self.workspace:
                raise ValueError("Implementation workspace must differ from diagnosis workspace")
            self._assert_fix_artifact_unchanged()
            implementation_checkout = GitCheckout.init_from_path(resolved_workspace)
            resolved_ref = implementation_checkout.get_text(
                "rev-parse", f"{implementation_ref}^{{commit}}"
            )
            self._validate_implementation_checkout(implementation_checkout, resolved_ref)
            self._get_implementation_diff(implementation_checkout, resolved_ref)
            self.implementation_workspace = resolved_workspace
            self.implementation_ref = resolved_ref
        elif implementation_workspace is not None or implementation_ref is not None:
            raise ValueError(
                "Implementation workspace and ref apply only to an implementation handoff"
            )
        previous_status = self.status
        previous_phase = self.phase
        self.status = "active"
        self.blocked_reason = None
        if previous_phase == "implementation":
            self.phase = "verify"
            self.next_action = "Rerun the frozen original failure and regression checks."
        elif previous_phase == "verify":
            self.next_action = "Rerun the frozen original failure and regression checks."
        elif previous_phase == "diagnose":
            self.next_action = (
                "Locate the earliest failing boundary and run one discriminating diagnostic check."
            )
        else:
            self.next_action = "Run the smallest command that demonstrates the reported failure."
        self.runs.append(
            {
                "index": len(self.runs) + 1,
                "kind": "resume",
                "from_status": previous_status,
                "from_phase": previous_phase,
                "note": note.strip() if note else None,
                "implementation_workspace": (
                    str(self.implementation_workspace) if self.implementation_workspace else None
                ),
                "implementation_ref": self.implementation_ref,
            }
        )
        self.save()
        return self

    def block(self, reason: str) -> AutoDebugIncident:
        self._ensure_active()
        if not reason.strip():
            raise ValueError("Blocked reason cannot be empty")
        self.status = "blocked"
        self.blocked_reason = reason.strip()
        self.next_action = (
            "Resume this incident if the stated access, hardware, dependency, or scope constraint "
            "changes."
        )
        self.runs.append(
            {
                "index": len(self.runs) + 1,
                "kind": "block",
                "reason": self.blocked_reason,
                "resume_phase": self.phase,
            }
        )
        self.save()
        return self

    def cancel(self) -> AutoDebugIncident:
        if self.status in TERMINAL_STATUSES:
            raise ValueError(f"Terminal incident cannot be changed: {self.status}")
        self.status = "cancelled"
        self.phase = "done"
        self.next_action = "The user cancelled this incident."
        self.save()
        return self

    def _write_fix_artifact(self) -> None:
        if self.fix_artifact is None:
            raise ValueError("No fix artifact has been prepared")
        artifact_path = Path(self.fix_artifact)

        lines = [
            "# GR00T AutoDebug fix package",
            "",
            f"- Incident: `{self.id}`",
            "- Decision: `ready_for_implementation`",
            f"- Debugged repository revision: `{self.repo_commit}`",
            f"- Debugged workspace: `{self.workspace}`",
            f"- Incident record: `{self.path}`",
            "- Ownership: AutoDebug investigated and verified evidence; a separate agent owns "
            "implementation.",
            "",
            "## Supported diagnosis",
            "",
            f"**Hypothesis:** {self.hypothesis}",
            "",
            self.analysis or "No analysis summary was recorded.",
            "",
            "## Proposed implementation",
            "",
            "Implementation targets:",
            "",
        ]
        lines.extend(f"- `{target}`" for target in self.implementation_targets)
        lines.extend(
            [
                "",
                self.proposed_change or "No proposed change was recorded.",
                "",
                "## Original problem report",
                "",
            ]
        )
        lines.extend(f"    {line}" if line else "" for line in self.report.splitlines())
        lines.extend(
            [
                "",
                "## Verified precedents considered",
                "",
                (
                    ", ".join(f"`{case_id}`" for case_id in self.related_case_ids)
                    if self.related_case_ids
                    else "No verified solved case matched this report."
                ),
                "",
                "## Investigation and command results",
                "",
            ]
        )
        for run in self.runs:
            kind = str(run.get("kind", "unknown"))
            lines.extend([f"### Step {run.get('index', '?')}: {kind}", ""])
            if "hypothesis" in run:
                lines.append(f"- Hypothesis: {run['hypothesis']}")
            if "outcome" in run:
                lines.append(f"- Outcome: `{run['outcome']}`")
            if "exit_code" in run:
                lines.append(f"- Exit code: `{run['exit_code']}`")
            if "duration_seconds" in run:
                lines.append(f"- Duration: `{run['duration_seconds']:.3f}` seconds")
            if "target" in run:
                lines.append(f"- Execution target: `{run['target']}`")
            if "log_sha256" in run:
                lines.append(f"- Log SHA-256: `{run['log_sha256']}`")
            if "failure_signature" in run:
                lines.append(f"- Expected failure signature: `{run['failure_signature']}`")
            if "signature_matched" in run:
                lines.append(f"- Failure signature matched: `{run['signature_matched']}`")
            if "passed" in run:
                lines.append(f"- Passed: `{run['passed']}`")
            if run.get("note"):
                lines.append(f"- Downstream or resume note: {run['note']}")
            if kind == "insights":
                lines.append(f"- Knowledge available: `{run.get('available', False)}`")
                lines.append(f"- Query: {run.get('query')}")
                lines.append(
                    f"- Projection SHA-256: `{run.get('projection_sha256') or 'unavailable'}`"
                )
                records = run.get("record_ids", [])
                lines.append(
                    "- Retrieved records: "
                    + (", ".join(f"`{record_id}`" for record_id in records) if records else "none")
                )
                if run.get("error"):
                    lines.append(f"- Grounding gap: {run['error']}")
            if run.get("command"):
                lines.extend(["- Command:", "", f"    {run['command']}"])
            if run.get("log"):
                log_path = Path(str(run["log"]))
                try:
                    log_reference = log_path.relative_to(artifact_path.parent)
                except ValueError:
                    log_reference = log_path
                lines.append(f"- Log: [{log_reference}]({log_reference})")
            lines.append("")

        lines.extend(
            [
                "## Frozen verification plan",
                "",
                "Original failure command:",
                "",
                f"    {self.reproducer}",
                "",
                f"Expected failure signature: `{self.failure_signature}`",
                "",
                "Regression commands:",
                "",
            ]
        )
        lines.extend(f"- `{command}`" for command in self.regression_commands)
        lines.extend(["", "Runner-owned verification assets:", ""])
        if self.verification_asset_paths:
            lines.extend(
                f"- `{path}` — SHA-256 `{digest}`"
                for path, digest in zip(
                    self.verification_asset_paths,
                    self.verification_asset_sha256,
                    strict=True,
                )
            )
        else:
            lines.append("- None; all frozen checks are self-contained commands.")
        lines.extend(["", "Protected repository test paths:", ""])
        if self.verification_repo_paths:
            lines.extend(f"- `{path}`" for path in self.verification_repo_paths)
        else:
            lines.append("- None; verdict commands do not directly name repository paths.")
        if self.blocked_reason:
            lines.extend(["", "## Current constraint", "", self.blocked_reason])
        lines.extend(
            [
                "",
                "## Downstream instructions",
                "",
                "1. Review the evidence and proposed scope in this package.",
                "2. Apply the change in a separate Git worktree, using the same repository "
                "subdirectory as the diagnosis workspace; do not edit the diagnosis workspace "
                "or alter the frozen verification commands.",
                "3. Commit the implementation and return the worktree path and commit SHA.",
                "4. Let AutoDebug run the frozen checks in that worktree and produce a separate "
                "verification result.",
                "",
            ]
        )
        with artifact_path.open("x", encoding="utf-8") as stream:
            stream.write("\n".join(lines))

    def _write_verification_artifact(self) -> None:
        if self.verification_artifact is None or self.fix_artifact is None:
            raise ValueError("Verification and fix artifact paths are required")
        artifact_path = Path(self.verification_artifact)
        summary = next(
            run for run in reversed(self.runs) if run.get("kind") == "verification-summary"
        )
        lines = [
            "# GR00T AutoDebug verification result",
            "",
            f"- Incident: `{self.id}`",
            f"- Decision: `{self.status}`",
            f"- Fix package: [{Path(self.fix_artifact).name}]({Path(self.fix_artifact).name})",
            f"- Diagnosis revision: `{self.repo_commit}`",
            f"- Implementation revision: `{self.implementation_ref}`",
            f"- Implementation workspace: `{self.implementation_workspace}`",
            f"- Baseline failure reproduced: `{summary['baseline_reproduced']}`",
            f"- Baseline check exit code: `{summary['baseline_exit_code']}`",
            f"- Implementation check exit code: `{summary['original_exit_code']}`",
            f"- Failure signature still present: `{summary['failure_signature_present']}`",
            f"- All checks passed: `{summary['passed']}`",
            f"- Implementation diff SHA-256: `{summary['implementation_diff_sha256']}`",
            f"- Checkout integrity error: `{summary['integrity_error']}`",
            "- Changed paths: " + ", ".join(f"`{path}`" for path in summary["changed_paths"]),
            "",
            "## Implementation diff summary",
            "",
            "```text",
            summary["implementation_diff_stat"].rstrip(),
            "```",
            "",
            "## Frozen checks and results",
            "",
        ]
        verification_runs = [
            run
            for run in self.runs
            if run.get("kind") in {"verify-baseline", "verify-original", "verify-regression"}
        ]
        for run in verification_runs[-(2 + len(self.regression_commands)) :]:
            log_path = Path(str(run["log"]))
            try:
                log_reference = log_path.relative_to(artifact_path.parent)
            except ValueError:
                log_reference = log_path
            lines.extend(
                [
                    f"### {run['kind']}",
                    "",
                    f"- Command: `{run['command']}`",
                    f"- Exit code: `{run['exit_code']}`",
                    f"- Duration: `{run['duration_seconds']:.3f}` seconds",
                    f"- Log SHA-256: `{run['log_sha256']}`",
                    f"- Log: [{log_reference}]({log_reference})",
                    "",
                ]
            )
        lines.extend(
            [
                "## Interpretation",
                "",
                self._verification_interpretation(summary),
                "",
            ]
        )
        with artifact_path.open("x", encoding="utf-8") as stream:
            stream.write("\n".join(lines))

    def _verification_interpretation(self, summary: dict[str, Any]) -> str:
        if summary.get("integrity_error"):
            return (
                "A frozen check changed the clean verification worktree or another recorded "
                "artifact. Verification is blocked; no fix decision was made."
            )
        if not summary.get("baseline_reproduced"):
            return (
                "The frozen failure did not reproduce on a clean copy of the original revision. "
                "More environment or input evidence is required; no fix decision was made."
            )
        if summary["passed"]:
            return (
                "The reported failure and all frozen regressions pass in a clean worktree at "
                "the recorded implementation commit."
            )
        if summary.get("failure_signature_present"):
            return (
                "The original check still emitted the frozen failure signature. A zero exit "
                "code alone is not accepted as a fix."
            )
        return (
            "At least one frozen check failed. The incident returns to diagnosis; this "
            "implementation is not accepted."
        )

    def _execute(
        self,
        command: str,
        kind: str,
        timeout_seconds: int,
        workspace: Path | None = None,
        target_name: str | None = None,
    ) -> tuple[int, Path, str]:
        if not command.strip():
            raise ValueError("Command cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be greater than zero")
        if workspace is not None and target_name is not None:
            raise ValueError("Cannot combine an internal workspace with an execution target")
        target = None
        local_execution_workspace: Path
        recorded_workspace: str
        if target_name is not None:
            target_data = self.execution_targets.get(target_name)
            if target_data is None:
                raise ValueError(f"Unknown execution target: {target_name!r}")
            target = ExecutionTarget.init_from_data(
                target_name,
                target_data,
                config_directory=self.path.parent,
            )
            argv, target_workspace = target.command_argv(command)
            local_execution_workspace = target_workspace or self.workspace
            recorded_workspace = target.workspace
        else:
            argv = shlex.split(command)
            if not argv:
                raise ValueError("Command cannot be empty")
            local_execution_workspace = workspace or self.workspace
            recorded_workspace = str(local_execution_workspace)
        started = time.monotonic()
        launch_error = None
        try:
            result = subprocess.run(
                argv,
                cwd=local_execution_workspace,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            exit_code = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            stdout = (
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else exc.stdout or ""
            )
            stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else exc.stderr or ""
            )
            stderr += f"\nTimed out after {timeout_seconds} seconds.\n"
        except OSError as exc:
            exit_code = 126
            stdout = ""
            launch_error = f"{type(exc).__name__}: {exc}"
            stderr = f"Command could not start: {launch_error}\n"
        duration_seconds = time.monotonic() - started

        log_directory = self.path.parent / "logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        index = len(self.runs) + 1
        log_path = log_directory / f"{index:03d}-{kind}.log"
        target_text = target_name or "diagnosis-workspace"
        log_text = (
            f"$ {command}\n\n[target]\n{target_text}\n\n[workspace]\n{recorded_workspace}"
            f"\n\n[stdout]\n{stdout}\n[stderr]\n{stderr}"
        )
        log_path.write_text(log_text, encoding="utf-8")
        self.runs.append(
            {
                "index": index,
                "kind": kind,
                "command": command,
                "exit_code": exit_code,
                "workspace": recorded_workspace,
                "duration_seconds": duration_seconds,
                "log": str(log_path),
                "log_sha256": hashlib.sha256(log_text.encode()).hexdigest(),
            }
        )
        if target is not None:
            self.runs[-1]["target"] = target.name
            self.runs[-1]["target_kind"] = target.kind
        if launch_error is not None:
            self.runs[-1]["launch_error"] = launch_error
        diagnosis_checkout = (
            GitCheckout.init_from_path(self.workspace)
            if local_execution_workspace.resolve() == self.workspace
            else None
        )
        if diagnosis_checkout is not None and (
            diagnosis_checkout.head != self.repo_commit
            or diagnosis_checkout.common_dir != self.repo_common_dir
            or diagnosis_checkout.tracked_tree_sha256 != self.diagnosis_tree_sha256
        ):
            self.status = "blocked"
            self.blocked_reason = (
                "A diagnosis command changed the recorded revision or tracked files. Restore "
                "the diagnosis checkout to its recorded state before resuming."
            )
            self.next_action = self.blocked_reason
            self.runs[-1]["isolation_violation"] = True
            self.runs[-1]["recorded_revision"] = self.repo_commit
            self.runs[-1]["observed_revision"] = diagnosis_checkout.head
            self.save()
            raise ValueError(self.blocked_reason)
        return exit_code, log_path, f"{stdout}\n{stderr}"

    def _assert_fix_artifact_unchanged(self) -> None:
        if self.fix_artifact is None:
            raise ValueError("No fix package is recorded")
        handoff = next(
            (
                run
                for run in reversed(self.runs)
                if run.get("kind") == "handoff" and run.get("artifact") == self.fix_artifact
            ),
            None,
        )
        if handoff is None or not handoff.get("artifact_sha256"):
            raise ValueError("Fix package does not have a recorded digest")
        artifact_path = Path(self.fix_artifact)
        if not artifact_path.is_file():
            raise ValueError("Fix package is missing")
        actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual != handoff["artifact_sha256"]:
            raise ValueError("Fix package changed after handoff")

    @staticmethod
    def _command_paths(command: str, workspace: Path) -> list[Path]:
        """Return existing path arguments, including an explicitly addressed executable."""
        workspace = workspace.resolve()
        paths: list[Path] = []

        def add_candidate(value: str) -> None:
            value = value.split("::", 1)[0]
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = workspace / candidate
            candidate = candidate.absolute()
            chain = [candidate]
            current = Path(candidate.anchor)
            symlink_seen = False
            visited_links: set[Path] = set()
            for part in candidate.parts[1:]:
                current = current / part
                while current.is_symlink():
                    symlink_seen = True
                    if current in visited_links:
                        raise ValueError(f"Command path contains a symlink cycle: {candidate}")
                    visited_links.add(current)
                    if current not in chain:
                        chain.append(current)
                    target = Path(os.readlink(current))
                    current = (
                        target if target.is_absolute() else current.parent / target
                    ).absolute()
            if not candidate.exists() and not symlink_seen:
                return
            if current not in chain:
                chain.append(current)
            for path in chain:
                if path not in paths:
                    paths.append(path)

        def collect(command_text: str, depth: int = 0) -> None:
            if depth > 3:
                return
            argv = shlex.split(command_text)
            if not argv:
                return
            tokens = argv[1:]
            if Path(argv[0]).is_absolute() or "/" in argv[0] or "\\" in argv[0]:
                tokens = argv
            for token in tokens:
                value = token.split("=", 1)[1] if token.startswith("-") and "=" in token else token
                add_candidate(value)

            for index, token in enumerate(argv[:-1]):
                if token != "-m":
                    continue
                module_parts = argv[index + 1].split(".")
                if not module_parts or any(not part.isidentifier() for part in module_parts):
                    continue
                module_base = workspace.joinpath(*module_parts)
                if module_base.with_suffix(".py").is_file():
                    add_candidate(str(module_base.with_suffix(".py")))
                    for package_depth in range(1, len(module_parts)):
                        package_init = workspace.joinpath(
                            *module_parts[:package_depth], "__init__.py"
                        )
                        if package_init.is_file():
                            add_candidate(str(package_init))
                elif module_base.is_dir():
                    add_candidate(str(module_base))

            effective_argv = argv
            launcher = Path(effective_argv[0]).name
            if launcher == "env":
                command_index = 1
                while command_index < len(effective_argv):
                    value = effective_argv[command_index]
                    if value in {"-S", "--split-string"}:
                        if command_index + 1 < len(effective_argv):
                            collect(effective_argv[command_index + 1], depth + 1)
                        return
                    if value.startswith("--split-string="):
                        collect(value.split("=", 1)[1], depth + 1)
                        return
                    if value in {"-i", "--ignore-environment"} or "=" in value:
                        command_index += 1
                        continue
                    if value in {"-u", "--unset"}:
                        command_index += 2
                        continue
                    if value.startswith("-"):
                        command_index += 1
                        continue
                    break
                effective_argv = effective_argv[command_index:]
                if not effective_argv:
                    return
                launcher = Path(effective_argv[0]).name
            if launcher in {"bash", "dash", "sh", "zsh"}:
                for index, token in enumerate(effective_argv[:-1]):
                    short_flags = (
                        token[1:] if token.startswith("-") and not token.startswith("--") else ""
                    )
                    if token == "--command" or (short_flags and "c" in short_flags):
                        collect(effective_argv[index + 1], depth + 1)
                        break

        collect(command)
        return paths

    @classmethod
    def _external_command_files(cls, command: str, workspace: Path) -> list[Path]:
        """Return the external file closure that can influence a verdict command."""
        workspace = workspace.resolve()
        files: list[Path] = []
        for candidate in cls._command_paths(command, workspace):
            try:
                candidate.relative_to(workspace)
                continue
            except ValueError:
                pass
            if candidate.is_dir() and not candidate.is_symlink():
                raise ValueError(
                    "Verdict commands cannot name an external directory; pass explicit files "
                    "or a runner-owned manifest instead"
                )
            if not candidate.is_symlink() and not candidate.is_file():
                raise ValueError(f"External command asset is missing: {candidate}")
            candidates = [candidate]
            for path in candidates:
                if path not in files:
                    files.append(path)
        return files

    @classmethod
    def _repository_command_paths(cls, command: str, workspace: Path) -> list[str]:
        """Return workspace-relative files or directories used by a verdict command."""
        workspace = workspace.resolve()
        paths: list[str] = []
        for candidate in cls._command_paths(command, workspace):
            try:
                relative = candidate.relative_to(workspace).as_posix() or "."
            except ValueError:
                continue
            if relative not in paths:
                paths.append(relative)
        return paths

    def _freeze_verification_assets(
        self,
        paths: list[Path],
        *,
        expected_digests: dict[str, str] | None = None,
    ) -> None:
        existing = dict(
            zip(
                self.verification_asset_paths,
                self.verification_asset_sha256,
                strict=True,
            )
        )
        for path in paths:
            asset_path = path.absolute()
            if not asset_path.is_symlink() and not asset_path.is_file():
                raise ValueError(f"Verification asset is missing: {asset_path}")
            digest = self._verification_asset_digest(asset_path)
            expected = (expected_digests or {}).get(str(asset_path))
            if expected is not None and digest != expected:
                raise ValueError(
                    f"Verification asset changed while reproducing the failure: {asset_path}"
                )
            previous = existing.get(str(asset_path))
            if previous is not None:
                if previous != digest:
                    raise ValueError(f"Frozen verification asset changed: {asset_path}")
                continue
            self.verification_asset_paths.append(str(asset_path))
            self.verification_asset_sha256.append(digest)
            existing[str(asset_path)] = digest

    @staticmethod
    def _verification_asset_digest(path: Path) -> str:
        if path.is_symlink():
            target = os.readlink(path)
            return hashlib.sha256(b"symlink\0" + os.fsencode(target)).hexdigest()
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _assert_verification_assets_unchanged(self) -> None:
        if len(self.verification_asset_paths) != len(self.verification_asset_sha256):
            raise ValueError("Verification asset ledger is inconsistent")
        for path, expected in zip(
            self.verification_asset_paths,
            self.verification_asset_sha256,
            strict=True,
        ):
            asset_path = Path(path)
            if not asset_path.is_symlink() and not asset_path.is_file():
                raise ValueError(f"Frozen verification asset is missing: {asset_path}")
            actual = self._verification_asset_digest(asset_path)
            if actual != expected:
                raise ValueError(f"Frozen verification asset changed: {asset_path}")

    def _assert_diagnosis_checkout_unchanged(self) -> GitCheckout:
        checkout = GitCheckout.init_from_path(self.workspace)
        if (
            checkout.common_dir != self.repo_common_dir
            or checkout.head != self.repo_commit
            or checkout.tracked_tree_sha256 != self.diagnosis_tree_sha256
        ):
            self.status = "blocked"
            self.blocked_reason = (
                "The diagnosis checkout changed after the incident was created. Restore its "
                "recorded revision and tracked files before resuming."
            )
            self.next_action = self.blocked_reason
            self.save()
            raise ValueError(self.blocked_reason)
        return checkout

    def _validate_clean_runner_checkout(self, checkout: GitCheckout, revision: str) -> None:
        diagnosis_checkout = self._assert_diagnosis_checkout_unchanged()
        if checkout.common_dir != self.repo_common_dir:
            raise ValueError("Clean runner is not a worktree of the diagnosed repository")
        if checkout.head != revision:
            raise ValueError("A verdict command changed the clean runner revision")
        if checkout.prefix != diagnosis_checkout.prefix:
            raise ValueError("Clean runner does not use the diagnosed repository subdirectory")
        if checkout.worktree_status:
            raise ValueError("A verdict command changed files in the clean runner")

    @contextmanager
    def _clean_repository_workspace(self, revision: str) -> Iterator[Path]:
        """Materialize a detached runner workspace with a fixed LFS pointer policy."""
        diagnosis_checkout = self._assert_diagnosis_checkout_unchanged()
        with tempfile.TemporaryDirectory(prefix="gr00t-autodebug-runner-") as temporary:
            root = Path(temporary) / "checkout"
            result = subprocess.run(
                [
                    "git",
                    "worktree",
                    "add",
                    "--detach",
                    str(root),
                    revision,
                ],
                cwd=diagnosis_checkout.root,
                env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
                raise ValueError(f"Cannot create clean runner worktree: {detail}")
            workspace = root / diagnosis_checkout.prefix if diagnosis_checkout.prefix else root
            try:
                self._validate_clean_runner_checkout(
                    GitCheckout.init_from_path(workspace), revision
                )
                yield workspace
            finally:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(root)],
                    cwd=diagnosis_checkout.root,
                    capture_output=True,
                    text=True,
                    check=False,
                )

    def _validate_implementation_checkout(self, checkout: GitCheckout, revision: str) -> None:
        if checkout.common_dir != self.repo_common_dir:
            raise ValueError(
                "Implementation workspace must be a worktree of the diagnosed repository"
            )
        if checkout.head != revision:
            raise ValueError("Implementation workspace HEAD does not match the supplied ref")
        if checkout.worktree_status:
            raise ValueError("Implementation workspace has uncommitted or untracked changes")
        if not checkout.contains_revision(self.repo_commit):
            raise ValueError("Implementation commit does not descend from the diagnosis revision")
        diagnosis_checkout = self._assert_diagnosis_checkout_unchanged()
        if checkout.prefix != diagnosis_checkout.prefix:
            raise ValueError(
                "Implementation workspace must use the same repository subdirectory as the "
                "diagnosis workspace"
            )

    def _get_implementation_diff(
        self, checkout: GitCheckout, revision: str
    ) -> tuple[bytes, str, list[str]]:
        root_changed_paths = checkout.get_text(
            "diff",
            "--no-renames",
            "--name-only",
            f"{self.repo_commit}..{revision}",
            "--",
        ).splitlines()
        if not root_changed_paths:
            raise ValueError("Implementation commit contains no repository changes")

        prefix = checkout.prefix
        if prefix:
            prefix_with_separator = f"{prefix}/"
            outside_workspace = [
                path for path in root_changed_paths if not path.startswith(prefix_with_separator)
            ]
            if outside_workspace:
                raise ValueError(
                    "Implementation changed paths outside the diagnosed workspace: "
                    + ", ".join(outside_workspace)
                )
            changed_paths = [
                path.removeprefix(prefix_with_separator) for path in root_changed_paths
            ]
        else:
            changed_paths = root_changed_paths

        implementation_diff = checkout.get_bytes(
            "diff",
            "--no-ext-diff",
            "--binary",
            "--relative",
            f"{self.repo_commit}..{revision}",
            "--",
            ".",
        )
        implementation_diff_stat = checkout.get_text(
            "diff", "--stat", "--relative", f"{self.repo_commit}..{revision}", "--", "."
        )
        unexpected_paths = [
            path
            for path in changed_paths
            if not any(
                path == target or path.startswith(f"{target}/")
                for target in self.implementation_targets
            )
        ]
        if unexpected_paths:
            raise ValueError(
                "Implementation changed paths outside the fix package: "
                + ", ".join(unexpected_paths)
            )
        changed_oracles = [
            path
            for path in changed_paths
            if any(
                path == oracle
                or path.startswith(f"{oracle}/")
                or oracle.startswith(f"{path}/")
                or oracle == "."
                for oracle in self.verification_repo_paths
            )
        ]
        if changed_oracles:
            raise ValueError(
                "Implementation changed a protected verification path: "
                + ", ".join(changed_oracles)
            )
        return implementation_diff, implementation_diff_stat, changed_paths

    def _ensure_active(self) -> None:
        if self.status != "active":
            raise ValueError(f"Only an active incident can run this action: {self.status}")
