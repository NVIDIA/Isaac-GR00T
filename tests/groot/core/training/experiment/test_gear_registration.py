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

import importlib.util
from pathlib import Path


def _oss_registration():
    path = (
        Path(__file__).resolve().parents[5] / "groot/core/training/experiment/gear_registration.py"
    )
    spec = importlib.util.spec_from_file_location("oss_gear_registration", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


def test_oss_registration_is_standalone() -> None:
    path, registration = _oss_registration()

    assert not path.is_symlink()
    assert registration._get_api_url() == ""  # noqa: SLF001
    assert registration.register_experiment(experiment_name="test") == {}
    assert registration.register_training_run(experiment_id="test") == ""
    source = path.read_text(encoding="utf-8")
    assert "gear-dashboard.nvidia.com" not in source
    assert "gear-dashboard-01.nvidia.com" not in source
    assert "urlopen" not in source


def test_oss_registration_preserves_output_and_resume_contracts(
    monkeypatch, tmp_path: Path
) -> None:
    _, registration = _oss_registration()
    training_output = tmp_path / "training-output"
    checkpoint = tmp_path / "checkpoint"
    configured = tmp_path / "configured"
    monkeypatch.setenv("TRAINING_OUTPUT_DIR", str(training_output))
    monkeypatch.setenv("GROOT_OL_RESUME_DECIDED", "true")
    monkeypatch.setenv("GROOT_OL_RESUME_CHECKPOINT_PATH", str(checkpoint))

    assert registration.resolve_training_output_dir(str(configured)) == str(training_output)
    assert registration.one_launcher_resume_checkpoint() == str(checkpoint)
    assert registration.dashboard_checkpoint_path_to_msc_url("bucket/key") == "msc://bucket/key"
