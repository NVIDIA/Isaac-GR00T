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
"""Tests for the OSS post-train skill trigger helpers."""

from importlib import util
from pathlib import Path
import sys


def _repo_root() -> Path:
    return next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())


def _load_post_train_module():
    repo_root = _repo_root()
    script = next(
        path for path in (
            repo_root / "oss/.codex/skills/post-train/scripts/post_train.py",
            repo_root / ".codex/skills/post-train/scripts/post_train.py",
        )
        if path.exists()
    )
    spec = util.spec_from_file_location("post_train_skill", script)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _post_train_ci_text() -> str:
    repo_root = _repo_root()
    config = next(
        path
        for path in (
            repo_root / "oss/ci/staging.posttrain.h100.gitlab-ci.yml",
            repo_root / "ci/staging.posttrain.h100.gitlab-ci.yml",
        )
        if path.exists()
    )
    return config.read_text()


def test_run_spec_uses_post_train_variables() -> None:
    module = _load_post_train_module()

    _run_type, compute_type, label, variables = module._run_spec("n2-minwam-sim", "h100")

    assert compute_type == "h100"
    assert label == "N2 Min-WAM Sim OSS Training"
    assert variables == {
        "POST_TRAIN_E2E_LEAF": "true",
        "POST_TRAIN_VARIANT": "n2-minwam-sim",
        "POST_TRAIN_RUN_TYPE": "n2-minwam-sim",
        "POST_TRAIN_COMPUTE_TYPE": "h100",
    }


def test_open_loop_eval_vars_enable_eval_after_post_train() -> None:
    module = _load_post_train_module()

    assert module._open_loop_eval_vars("n2-minwam-sim", "h100") == {
        "OPEN_LOOP_EVAL_E2E_LEAF": "true",
        "OPEN_LOOP_EVAL_AFTER_POST_TRAIN": "true",
        "OPEN_LOOP_EVAL_VARIANT": "n2-minwam-sim",
        "OPEN_LOOP_EVAL_RUN_TYPE": "n2-minwam-sim",
        "OPEN_LOOP_EVAL_COMPUTE_TYPE": "h100",
    }


def test_pipeline_env_overrides_omit_unset_values(monkeypatch) -> None:
    module = _load_post_train_module()

    monkeypatch.delenv("OSMO_POOL", raising=False)
    monkeypatch.delenv("POST_TRAIN_CLUSTER_WAIT_TIMEOUT", raising=False)

    assert module._pipeline_env_overrides() == {}


def test_pipeline_env_overrides_forward_optional_values(monkeypatch) -> None:
    module = _load_post_train_module()

    monkeypatch.setenv("OSMO_POOL", "groot-h100-01")
    monkeypatch.setenv("POST_TRAIN_CLUSTER_WAIT_TIMEOUT", "5400")

    assert module._pipeline_env_overrides() == {
        "OSMO_POOL": "groot-h100-01",
        "POST_TRAIN_CLUSTER_WAIT_TIMEOUT": "5400",
    }


def test_post_train_ci_configures_wait_timeout_and_workflow_cleanup() -> None:
    config = _post_train_ci_text()

    assert '--wait-timeout "${POST_TRAIN_CLUSTER_WAIT_TIMEOUT:-3600}"' in config
    assert r'grep -oP "(?:Cluster ID:|Ray cluster|Workflow)\s+\K\S+"' in config
    assert 'echo "$WORKFLOW_NAME" > "$CI_PROJECT_DIR/cluster_id.txt"' not in config
    assert 'osmo workflow cancel "$CLUSTER_ID" || true' in config


def test_parse_rejects_non_h100_post_train_argument() -> None:
    module = _load_post_train_module()

    try:
        module._parse_run_type_arg("n2-minwam-sim-gb300")
    except ValueError as exc:
        assert "<run-type>-h100" in str(exc)
    else:
        raise AssertionError("expected ValueError")
