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

"""CI state filter — query tests in zombie/sloth state from DB.

Usage:
  # Unit test mode: print --deselect args for pytest (two lines per test: flag then nodeid)
  uv run -m groot.infra.ci.main.ci_state_filter \
      --test-type unit

  # Unit test mode: print test names for zombie/sloth unit tests (for postmerge zombie jobs)
  uv run -m groot.infra.ci.main.ci_state_filter \
      --test-type unit --zombie-mode

  # Integration test mode: filter a workflow YAML (exclude zombie/sloth)
  uv run -m groot.infra.ci.main.ci_state_filter \
      --test-type integration \
      --input list_rendered.yaml --output list_filtered.yaml

  # Integration test mode: keep ONLY zombie/sloth (for postmerge zombie jobs)
  uv run -m groot.infra.ci.main.ci_state_filter \
      --test-type integration --zombie-mode \
      --input list_rendered.yaml --output list_filtered.yaml

Exit codes:
  0  - Success
  2  - Nothing to run (no workflows remain, or zombie-mode with no zombie/sloth tests)
  1  - Unexpected error
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from groot.infra.ci.test import Test, TestType


def _filter_yaml(
    input_path: Path,
    output_path: Path,
    excluded_names: set[str],
    zombie_mode: bool,
) -> int:
    """Filter YAML workflow blocks by top-level key.

    In normal mode: exclude zombie/sloth workflows (keep passing ones).
    In zombie_mode: keep ONLY zombie/sloth workflows (for postmerge zombie jobs).

    Returns the number of remaining (non-environment) workflows after filtering.
    Uses the same block-by-block line parser as the existing osmo_runner_execute.sh.
    """
    with open(input_path) as f:
        lines = f.readlines()

    output: list[str] = []
    skip = False
    workflow_count = 0
    included: list[str] = []
    for line in lines:
        if line and not line[0].isspace() and not line.startswith("#") and ":" in line:
            key = line.split(":")[0].strip()
            if key == "environment":
                skip = False
            elif zombie_mode:
                # zombie mode: keep only zombie/sloth, skip everything else
                skip = key not in excluded_names
                if not skip:
                    workflow_count += 1
                    included.append(key)
            else:
                # normal mode: skip zombie/sloth, keep everything else
                skip = key in excluded_names
                if not skip:
                    workflow_count += 1
                    included.append(key)
        if not skip:
            output.append(line)

    with open(output_path, "w") as f:
        f.write("".join(output))

    if included:
        mode = "zombie/sloth only" if zombie_mode else "excluding zombie/sloth"
        print(
            f"[ci-state-filter] workflows to run ({mode}): {', '.join(included)}",
            file=sys.stderr,
        )

    return workflow_count


def _junit_to_pytest_nodeid(name: str) -> str:
    """Convert JUnit dot-notation classname to pytest file path nodeid.

    JUnit format (from pytest XML): tests.module.path.ClassName::test_method
    Pytest format:                  tests/module/path/test_file.py::ClassName::test_method
    """
    parts = name.split("::")
    if len(parts) < 2:
        return name  # not classname format, return as-is

    classname = parts[0]
    rest = "::".join(parts[1:])

    # Find where the class name begins (first segment starting with uppercase)
    segments = classname.split(".")
    class_idx = next((i for i, s in enumerate(segments) if s and s[0].isupper()), None)

    if class_idx is not None:
        module_path = "/".join(segments[:class_idx]) + ".py"
        class_part = ".".join(segments[class_idx:])
        return f"{module_path}::{class_part}::{rest}"
    else:
        # No class (standalone function), just module::method
        module_path = "/".join(segments) + ".py"
        return f"{module_path}::{rest}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter zombie/sloth tests using CI state DB.")
    parser.add_argument(
        "--test-type",
        required=True,
        choices=[TestType.UNIT.value, TestType.INTEGRATION.value],
        help="Type of tests to query (unit or integration).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input workflow YAML file to filter (integration mode).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for filtered workflow YAML (integration mode).",
    )
    parser.add_argument(
        "--zombie-mode",
        action="store_true",
        default=False,
        help=(
            "Zombie mode: keep ONLY zombie/sloth workflows (for postmerge zombie jobs). "
            "Default: exclude zombie/sloth, keep passing workflows."
        ),
    )
    parser.add_argument(
        "--run-zombie-tests",
        default="",
        help=(
            "Comma-separated test names to force-include in normal mode "
            "(removes them from the zombie/sloth excluded list). "
            "Example: --run-zombie-tests=test_a,test_b"
        ),
    )
    args = parser.parse_args()

    test_type = TestType(args.test_type)

    # Fetch tests in zombie/sloth state from DB
    excluded = Test.get_in_zombie_states(test_type)

    # --run-zombie-tests: in normal mode, force-include specific tests that are currently
    # excluded as zombie/sloth so they run in the normal pipeline (e.g. CIPriest fix branch).
    # Caller extracts "RUN_ZOMBIE_TESTS=test01,test02" from $CI_COMMIT_MESSAGE and passes it here.
    run_zombie_tests = args.run_zombie_tests
    if not args.zombie_mode and run_zombie_tests:
        force_include = {n.strip() for n in run_zombie_tests.split(",") if n.strip()}
        if force_include:
            print(
                f"[ci-state-filter] RUN_ZOMBIE_TESTS force-includes: {', '.join(force_include)}",
                file=sys.stderr,
            )
            excluded = [n for n in excluded if n not in force_include]

    if test_type == TestType.UNIT:
        if args.zombie_mode:
            if not excluded:
                print(
                    "[ci-state-filter] no unit tests in zombie/sloth state — nothing to run",
                    file=sys.stderr,
                )
                sys.exit(2)
            for name in excluded:
                print(_junit_to_pytest_nodeid(name))
        else:
            for name in excluded:
                print("--deselect")
                print(_junit_to_pytest_nodeid(name))
        sys.exit(0)

    # Integration mode
    if args.input is None or args.output is None:
        print("❌  --input and --output are required for integration mode", file=sys.stderr)
        sys.exit(1)

    if args.zombie_mode and not excluded:
        print(
            f"[ci-state-filter] no {test_type.value} tests in zombie/sloth state — nothing to run",
            file=sys.stderr,
        )
        sys.exit(2)

    remaining = _filter_yaml(args.input, args.output, set(excluded), args.zombie_mode)

    if remaining == 0:
        mode_label = "zombie/sloth" if args.zombie_mode else "non-zombie/sloth"
        print(
            f"[ci-state-filter] no {mode_label} workflows remaining for {test_type.value} — nothing to run",
            file=sys.stderr,
        )
        sys.exit(2)

    mode_label = "zombie/sloth only" if args.zombie_mode else "excluding zombie/sloth"
    print(
        f"[ci-state-filter] {remaining} workflow(s) remaining ({mode_label})",
        file=sys.stderr,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
