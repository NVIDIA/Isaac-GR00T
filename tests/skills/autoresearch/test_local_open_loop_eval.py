# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from pathlib import Path

import pytest
from skills.autoresearch.local_open_loop_eval import _load_config, _parse_config

# Templates that eval_local.sh feeds to OssOpenLoopEvalConfig. The genie_sim_osmo
# eval_config.yaml is excluded: it configures the OSMO policy server, not this loader.
SHIPPED_EVAL_TEMPLATES = (
    "skills/autoresearch/scripts/eval_config.yaml",
    "skills/autoresearch/examples/yam_stack_bowl/eval_config.yaml",
)


def test_load_config_sets_environment(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "environment:\n  AUTORESEARCH_EVAL_TEST: enabled\nsteps: 10\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("AUTORESEARCH_EVAL_TEST", raising=False)

    config = _load_config(config_path)

    assert config.steps == 10
    assert os.environ["AUTORESEARCH_EVAL_TEST"] == "enabled"


def test_load_config_expands_environment_paths(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "autoresearch"
    monkeypatch.setenv("AUTORESEARCH_ROOT", str(root))
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "dataset_path: ${AUTORESEARCH_ROOT}/datasets/eval\n",
        encoding="utf-8",
    )

    config = _load_config(config_path)

    assert config.dataset_path == str(root / "datasets/eval")


def test_parse_config_applies_cli_override(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("steps: 10\n", encoding="utf-8")

    config = _parse_config(["--load-config-path", str(config_path), "--steps", "20"])

    assert config.steps == 20


@pytest.mark.parametrize("template", SHIPPED_EVAL_TEMPLATES)
def test_load_config_accepts_shipped_templates(template: str, tmp_path: Path, monkeypatch) -> None:
    package_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("AUTORESEARCH_ROOT", str(tmp_path))
    # _load_config exports the template's `environment:` block into os.environ; seeding
    # the key through monkeypatch makes that write roll back with the fixture.
    monkeypatch.setenv("ENABLE_TENSORRT", os.environ.get("ENABLE_TENSORRT", ""))

    config = _load_config(package_root / template)

    assert config.checkpoint
