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
"""Tests for the OSS evaluation skill trigger helpers."""

from importlib import util
from pathlib import Path
import sys


def _load_open_loop_eval_module():
    repo_root = next(
        parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()
    )
    script = next(
        path for path in (
            repo_root / "oss/.codex/skills/evaluation/scripts/evaluation.py",
            repo_root / ".codex/skills/evaluation/scripts/evaluation.py",
        )
        if path.exists()
    )
    spec = util.spec_from_file_location("evaluation_skill", script)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_spec_uses_open_loop_eval_variables() -> None:
    module = _load_open_loop_eval_module()

    run_type, compute_type, label, variables = module._run_spec("n2-minwam-sim-h100")

    assert run_type == "n2-minwam-sim"
    assert compute_type == "h100"
    assert label == "N2 Min-WAM Sim OSS Training"
    assert variables == {
        "OPEN_LOOP_EVAL_E2E_LEAF": "true",
        "OPEN_LOOP_EVAL_VARIANT": "n2-minwam-sim",
        "OPEN_LOOP_EVAL_RUN_TYPE": "n2-minwam-sim",
        "OPEN_LOOP_EVAL_COMPUTE_TYPE": "h100",
    }


def test_post_train_variables_enable_combined_pipeline() -> None:
    module = _load_open_loop_eval_module()

    assert module._post_train_variables("n2-minwam-sim", "h100") == {
        "POST_TRAIN_E2E_LEAF": "true",
        "POST_TRAIN_VARIANT": "n2-minwam-sim",
        "POST_TRAIN_RUN_TYPE": "n2-minwam-sim",
        "POST_TRAIN_COMPUTE_TYPE": "h100",
        "OPEN_LOOP_EVAL_AFTER_POST_TRAIN": "true",
    }


def test_pipeline_env_overrides_forward_optional_pool(monkeypatch) -> None:
    module = _load_open_loop_eval_module()

    monkeypatch.setenv("OSMO_POOL", "groot-l40-ci-02")

    assert module._pipeline_env_overrides() == {"OSMO_POOL": "groot-l40-ci-02"}


def test_parse_rejects_non_h100_open_loop_eval_argument() -> None:
    module = _load_open_loop_eval_module()

    try:
        module._parse_run_type_arg("n2-minwam-sim-gb300")
    except ValueError as exc:
        assert "<run-type>-h100" in str(exc)
    else:
        raise AssertionError("expected ValueError")
