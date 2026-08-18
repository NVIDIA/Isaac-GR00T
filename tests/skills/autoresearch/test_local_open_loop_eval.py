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

from skills.autoresearch.local_open_loop_eval import _load_config, _parse_config


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


def test_parse_config_applies_cli_override(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("steps: 10\n", encoding="utf-8")

    config = _parse_config(["--load-config-path", str(config_path), "--steps", "20"])

    assert config.steps == 20
