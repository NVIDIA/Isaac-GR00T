# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backfill, check, and extract owners for OSMO workflow templates."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from groot.infra.ci.osmo_workflow_owner_template import OsmoWorkflowOwnerTemplate

COMMAND_BACKFILL = "backfill"
COMMAND_CHECK = "check"
COMMAND_EXTRACT = "extract"
COMMAND_STRIP_RUNNER_YAML = "strip-runner-yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    backfill = sub.add_parser(COMMAND_BACKFILL, help="Insert missing owner fields")
    backfill.add_argument("templates", nargs="+", type=Path)
    backfill.add_argument("--repo-root", type=Path, default=Path("."))
    backfill.add_argument("--codeowners", type=Path, default=Path("CODEOWNERS"))
    backfill.add_argument("--dry-run", action="store_true")

    check = sub.add_parser(COMMAND_CHECK, help="Validate owner fields")
    check.add_argument("templates", nargs="+", type=Path)

    extract = sub.add_parser(COMMAND_EXTRACT, help="Print workflow owner map as JSON")
    extract.add_argument("template", type=Path)

    strip_runner_yaml = sub.add_parser(
        COMMAND_STRIP_RUNNER_YAML,
        help="Write an OSMO runner-compatible YAML file without owner fields",
    )
    strip_runner_yaml.add_argument("template", type=Path)
    strip_runner_yaml.add_argument("output", type=Path)

    args = parser.parse_args(argv)

    if args.command == COMMAND_BACKFILL:
        total = 0
        templates = OsmoWorkflowOwnerTemplate.init_all_from_paths(
            paths=args.templates,
            repo_root=args.repo_root.resolve(),
            codeowners_path=args.codeowners,
        )
        for template in templates:
            count = template.backfill(dry_run=args.dry_run)
            total += count
            action = "would update" if args.dry_run else "updated"
            sys.stdout.write(f"{template.path}: {action} {count} workflow owner field(s)\n")
        sys.stdout.write(f"Total: {total}\n")
        return 0

    if args.command == COMMAND_CHECK:
        errors: list[str] = []
        for template in OsmoWorkflowOwnerTemplate.init_all_from_paths(paths=args.templates):
            errors.extend(template.validate())
        for error in errors:
            sys.stderr.write(f"{error}\n")
        return 1 if errors else 0

    if args.command == COMMAND_EXTRACT:
        sys.stdout.write(OsmoWorkflowOwnerTemplate.init_from_path(args.template).to_json() + "\n")
        return 0

    if args.command == COMMAND_STRIP_RUNNER_YAML:
        count = OsmoWorkflowOwnerTemplate.init_from_path(args.template).write_osmo_runner_yaml(
            args.output
        )
        sys.stdout.write(f"{args.output}: stripped {count} workflow owner field(s)\n")
        return 0

    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
