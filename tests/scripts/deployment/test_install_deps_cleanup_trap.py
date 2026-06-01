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

"""Regression tests for the /tmp source-build cleanup trap.

The thor and spark install_deps.sh scripts source-build flash-attn /
torchcodec under /tmp. Before this guard, a failure in `pip install`
left /tmp/flash-attn or /tmp/torchcodec behind because `set -e` aborted
the script before the explicit cleanup ran. The next install on the
same host (CI runner / Docker build cache / dev machine) would then
silently reuse the stale clone.

These tests extract the exact trap prelude from the real installer
scripts and replay it inside a controlled subprocess, asserting that
registered build dirs are removed on both successful exit AND on
forced mid-build failure.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALLER_SCRIPTS = {
    "thor": REPO_ROOT / "scripts/deployment/thor/install_deps.sh",
    "spark": REPO_ROOT / "scripts/deployment/spark/install_deps.sh",
}


def _extract_trap_prelude(script_path: Path) -> str:
    """Pull the trap-cleanup prelude bytes verbatim from the real installer.

    Anchors on the substring between `set -euo pipefail` and the first
    `SCRIPT_DIR=` line — both installers follow that layout.
    """
    text = script_path.read_text()
    match = re.search(r"(set -euo pipefail.*?)^SCRIPT_DIR=", text, re.DOTALL | re.MULTILINE)
    assert match is not None, f"Expected the standard prelude in {script_path}"
    return match.group(1)


def _run_with_prelude(prelude: str, body: str, tmp_path: Path) -> subprocess.CompletedProcess:
    """Run `prelude + body` under bash and capture the result.

    The prelude registers the trap; the body is the test-specific scenario
    (success / failure / multiple dirs). The body is responsible for
    appending any dirs it cares about to TMP_BUILD_DIRS.
    """
    script = tmp_path / "harness.sh"
    script.write_text("#!/bin/bash\n" + prelude + "\n" + body + "\n")
    script.chmod(0o755)
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("name", list(INSTALLER_SCRIPTS))
def test_trap_cleans_tmp_dir_on_success(name: str, tmp_path: Path) -> None:
    """Healthy path: build dir is registered, work succeeds, trap removes it."""
    prelude = _extract_trap_prelude(INSTALLER_SCRIPTS[name])
    build_dir = tmp_path / "stale-build"
    body = f"""
mkdir -p "{build_dir}"
TMP_BUILD_DIRS+=("{build_dir}")
echo "build complete"
"""
    result = _run_with_prelude(prelude, body, tmp_path)
    assert result.returncode == 0, result.stderr
    assert not build_dir.exists(), (
        f"Build dir {build_dir} survived a successful run — trap did not fire"
    )


@pytest.mark.parametrize("name", list(INSTALLER_SCRIPTS))
def test_trap_cleans_tmp_dir_on_set_e_abort(name: str, tmp_path: Path) -> None:
    """The bug this MR fixes: `set -e` aborts mid-build, dir must still be cleaned."""
    prelude = _extract_trap_prelude(INSTALLER_SCRIPTS[name])
    build_dir = tmp_path / "failed-build"
    body = f"""
mkdir -p "{build_dir}"
TMP_BUILD_DIRS+=("{build_dir}")
false   # simulate `pip install` blowing up under `set -e`
echo "this line never runs"
"""
    result = _run_with_prelude(prelude, body, tmp_path)
    assert result.returncode != 0, "Harness should have aborted on `false`"
    assert not build_dir.exists(), (
        f"Build dir {build_dir} survived a failed run — exactly the leak this MR fixes"
    )


@pytest.mark.parametrize("name", list(INSTALLER_SCRIPTS))
def test_trap_handles_empty_array(name: str, tmp_path: Path) -> None:
    """Trap fires at script exit even when no dirs were ever registered.

    Guards against an early-exit path (e.g. arch validation, prebuilt wheel
    branch) leaving TMP_BUILD_DIRS empty. The trap must not crash on an
    unset/empty array under `set -u`.
    """
    prelude = _extract_trap_prelude(INSTALLER_SCRIPTS[name])
    body = 'echo "early exit, never appended"'
    result = _run_with_prelude(prelude, body, tmp_path)
    assert result.returncode == 0, (
        f"Empty-array exit crashed under `set -u`:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_trap_cleans_multiple_dirs_on_failure(tmp_path: Path) -> None:
    """Spark builds two /tmp dirs (flash-attn + torchcodec); both must clean."""
    prelude = _extract_trap_prelude(INSTALLER_SCRIPTS["spark"])
    dir_a = tmp_path / "flash-attn"
    dir_b = tmp_path / "torchcodec"
    body = f"""
mkdir -p "{dir_a}"
TMP_BUILD_DIRS+=("{dir_a}")
echo "flash-attn build done"
mkdir -p "{dir_b}"
TMP_BUILD_DIRS+=("{dir_b}")
false   # second build fails; first build dir must still get cleaned
"""
    result = _run_with_prelude(prelude, body, tmp_path)
    assert result.returncode != 0
    assert not dir_a.exists(), f"{dir_a} leaked when later step failed"
    assert not dir_b.exists(), f"{dir_b} leaked when its own step failed"
