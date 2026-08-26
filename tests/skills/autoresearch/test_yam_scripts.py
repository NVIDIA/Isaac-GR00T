# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

_REPOSITORY = Path(__file__).resolve().parents[3]
_TRAIN_WRAPPER = _REPOSITORY / "skills/autoresearch/examples/yam_stack_bowl/train_baseline.sh"
_LAUNCHER = _REPOSITORY / "skills/autoresearch/scripts/launch_autoresearch.sh"
_CONFIG_RENDERER = _REPOSITORY / "skills/autoresearch/scripts/render_config.py"


def test_config_renderer_expands_environment_paths(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    destination = tmp_path / "rendered.yaml"
    root = tmp_path / "autoresearch"
    source.write_text("dataset: ${AUTORESEARCH_ROOT}/datasets/train\n", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(_CONFIG_RENDERER), str(source), str(destination)],
        env={**os.environ, "AUTORESEARCH_ROOT": str(root)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert destination.read_text(encoding="utf-8") == f"dataset: {root}/datasets/train\n"


def test_launcher_exports_selected_python(tmp_path: Path) -> None:
    fake_repo = tmp_path / "repo"
    fake_launcher = fake_repo / "skills/autoresearch/scripts/launch_autoresearch.sh"
    fake_launcher.parent.mkdir(parents=True)
    shutil.copy2(_LAUNCHER, fake_launcher)

    fake_python = fake_repo / ".venv/bin/python"
    fake_python.parent.mkdir(parents=True)
    capture_path = tmp_path / "python.txt"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "${PYTHON:-}" "$@" > "${CAPTURE_PATH}"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    config_path = tmp_path / "autoresearch.yaml"
    completed = subprocess.run(
        ["bash", str(fake_launcher), str(config_path)],
        env={**os.environ, "CAPTURE_PATH": str(capture_path), "PYTHON": ""},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert capture_path.read_text(encoding="utf-8").splitlines() == [
        str(fake_python),
        "-m",
        "skills.autoresearch",
        str(config_path),
    ]


@pytest.mark.skipif(
    shutil.which("flock") is None,
    reason=(
        "train_baseline.sh serializes concurrent trials with flock, which is util-linux "
        "and absent on macOS. Skipped rather than worked around: the lock is what stops "
        "parallel trials writing one output directory, so a version without it would not "
        "be the script this asserts on."
    ),
)
def test_yam_wrapper_prunes_checkpoints_only_after_success(tmp_path: Path) -> None:
    fake_repo = tmp_path / "repo"
    fake_train = fake_repo / "skills/autoresearch/scripts/train_local.sh"
    fake_train.parent.mkdir(parents=True)
    fake_train.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
output_dir="${@: -1}"
mkdir -p "${output_dir}/checkpoint-5" "${output_dir}/checkpoint-10"
printf 'resume data\n' > "${output_dir}/checkpoint-10/trainer_state.json"
if [[ "${FAKE_EXIT_CODE}" != 0 ]]; then
  exit "${FAKE_EXIT_CODE}"
fi
printf '{}\n' > "${output_dir}/config.json"
printf 'final weights\n' > "${output_dir}/model.safetensors"
""",
        encoding="utf-8",
    )

    # train_baseline.sh resolves CONFIG_PATH under PACKAGE_ROOT, which follows
    # REPO_ROOT. The fake repo only exists so train_local.sh can be stubbed, so
    # the example yaml has to live there too or render_config.py exits 1.
    fake_config = fake_repo / "skills/autoresearch/examples/yam_stack_bowl/train_config.yaml"
    fake_config.parent.mkdir(parents=True)
    fake_config.write_text("training: {}\n", encoding="utf-8")

    for exit_code in (0, 7):
        run_root = tmp_path / f"baseline-{exit_code}"
        output_dir = run_root / "training"
        environment = {
            **os.environ,
            "RUN_ROOT": str(run_root),
            "AUTORESEARCH_ROOT": str(tmp_path),
            "FAKE_EXIT_CODE": str(exit_code),
            "GROOT_REVISION": "a" * 40,
            "OUTPUT_DIR": str(output_dir),
            "REPO_ROOT": str(fake_repo),
            "PYTHON": sys.executable,
        }

        completed = subprocess.run(
            ["bash", str(_TRAIN_WRAPPER)],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == exit_code, completed.stderr
        checkpoints = list(output_dir.glob("checkpoint-*"))
        if exit_code == 0:
            assert not checkpoints
            assert (output_dir / "model.safetensors").is_file()
