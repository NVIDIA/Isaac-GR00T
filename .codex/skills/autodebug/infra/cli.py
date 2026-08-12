# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""CLI for the GR00T AutoDebug incident loop."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from skills.autodebug.infra.autodebug_incident import (
    DEFAULT_CASE_DIRECTORY,
    AutoDebugIncident,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Run the GR00T AutoDebug workflow.")
    commands = root.add_subparsers(dest="action", required=True)

    new = commands.add_parser("new", help="Create or resume an incident")
    new.add_argument("--report", type=Path, required=True)
    new.add_argument("--workspace", type=Path, default=Path.cwd())
    new.add_argument("--run-root", type=Path, default=Path(".autodebug-runs"))
    new.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIRECTORY)
    new.add_argument("--target-config", type=Path)
    new.add_argument("--insights-root", type=Path)

    for name in ("show", "cases", "targets", "cancel"):
        command = commands.add_parser(name)
        command.add_argument("--incident", type=Path, required=True)

    resume = commands.add_parser("resume")
    resume.add_argument("--incident", type=Path, required=True)
    resume.add_argument("--note")
    resume.add_argument("--implementation-workspace", type=Path)
    resume.add_argument("--implementation-ref")

    reproduce = commands.add_parser("reproduce")
    reproduce.add_argument("--incident", type=Path, required=True)
    reproduce.add_argument("--command", dest="execution_command", required=True)
    reproduce.add_argument("--failure-signature", required=True)
    reproduce.add_argument("--timeout", type=int, default=300)

    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("--incident", type=Path, required=True)
    diagnose.add_argument("--command", dest="execution_command", required=True)
    diagnose.add_argument("--hypothesis", required=True)
    diagnose.add_argument(
        "--outcome", choices=("supported", "rejected", "inconclusive"), required=True
    )
    diagnose.add_argument("--target")
    diagnose.add_argument("--timeout", type=int, default=300)

    probe = commands.add_parser("probe")
    probe.add_argument("--incident", type=Path, required=True)
    probe.add_argument("--target", required=True)
    probe.add_argument("--command", dest="execution_command", required=True)
    probe.add_argument("--timeout", type=int, default=300)

    insights = commands.add_parser("insights")
    insights.add_argument("--incident", type=Path, required=True)
    insights.add_argument("--query", required=True)
    insights.add_argument("--insights-root", type=Path)
    insights.add_argument("--limit", type=int, default=5)

    handoff = commands.add_parser("handoff")
    handoff.add_argument("--incident", type=Path, required=True)
    handoff.add_argument("--analysis", required=True)
    handoff.add_argument("--change", required=True)
    handoff.add_argument("--target", action="append", default=[])
    handoff.add_argument("--regression", action="append", default=[])
    handoff.add_argument("--asset", action="append", default=[])

    verify = commands.add_parser("verify")
    verify.add_argument("--incident", type=Path, required=True)
    verify.add_argument("--timeout", type=int, default=300)

    pause = commands.add_parser("pause")
    pause.add_argument("--incident", type=Path, required=True)
    pause.add_argument("--need", required=True)

    block = commands.add_parser("block")
    block.add_argument("--incident", type=Path, required=True)
    block.add_argument("--reason", required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.action == "new":
            incident = AutoDebugIncident.init_from_report(
                report_path=args.report,
                workspace=args.workspace,
                run_root=args.run_root,
                case_directory=args.case_dir,
                target_config=args.target_config,
                insights_root=args.insights_root,
            )
        else:
            incident = AutoDebugIncident.init_from_path(args.incident)
            if args.action == "show":
                sys.stdout.write(incident.to_json_text())
                return 0
            if args.action == "cases":
                sys.stdout.write(incident.related_cases_text())
                return 0
            if args.action == "targets":
                sys.stdout.write(incident.execution_targets_text())
                return 0
            if args.action == "insights":
                result = incident.lookup_insights(
                    args.query,
                    insights_root=args.insights_root,
                    limit=args.limit,
                )
                result["incident"] = str(incident.path)
                sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
                return 0
            if args.action == "reproduce":
                incident.run_reproducer(
                    args.execution_command,
                    args.failure_signature,
                    args.timeout,
                )
            elif args.action == "diagnose":
                incident.run_diagnostic(
                    args.execution_command,
                    args.hypothesis,
                    args.outcome,
                    args.timeout,
                    args.target,
                )
            elif args.action == "probe":
                incident.run_probe(args.execution_command, args.target, args.timeout)
            elif args.action == "handoff":
                incident.handoff(
                    args.analysis,
                    args.change,
                    args.target,
                    args.regression,
                    args.asset,
                )
            elif args.action == "verify":
                incident.verify(args.timeout)
            elif args.action == "pause":
                incident.pause(args.need)
            elif args.action == "block":
                incident.block(args.reason)
            elif args.action == "resume":
                incident.resume(
                    args.note,
                    args.implementation_workspace,
                    args.implementation_ref,
                )
            elif args.action == "cancel":
                incident.cancel()
            else:
                raise ValueError(f"Unknown command: {args.action}")
        sys.stdout.write(incident.to_summary_json_text())
        return 0
    except (OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"AutoDebug error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
