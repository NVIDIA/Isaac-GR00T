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

"""Pin the per-episode step budget used by ``rollout_policy``.

``VideoConfig``, ``MultiStepConfig`` and the ``RolloutConfig`` CLI all default
the same ``max_episode_steps`` knob; they used to carry independent literals
(504 on the CLI, 720 on the wrappers) so the default env (LIBERO) was evaluated
with a different cap depending on the entry point. They now share one module
constant, which must also match the canonical LIBERO cap in
``_sim_eval_defaults``. Parsed via AST so the check stays torch-free.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_ROLLOUT_POLICY = Path(__file__).resolve().parents[3] / "gr00t" / "eval" / "rollout_policy.py"
_CONST_NAME = "DEFAULT_MAX_EPISODE_STEPS"
_DATACLASSES = ("VideoConfig", "MultiStepConfig", "RolloutConfig")


def _module() -> ast.Module:
    return ast.parse(_ROLLOUT_POLICY.read_text())


def _module_constant(tree: ast.Module, name: str) -> int:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if name in targets and isinstance(node.value, ast.Constant):
                return node.value.value
    raise AssertionError(f"module-level constant {name!r} not found in rollout_policy.py")


def _class_field(tree: ast.Module, class_name: str, field_name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id == field_name
                ):
                    return stmt.value
    raise AssertionError(f"{class_name}.{field_name} not found in rollout_policy.py")


def _string_default(tree: ast.Module, class_name: str, field_name: str) -> str:
    value = _class_field(tree, class_name, field_name)
    assert isinstance(value, ast.Constant) and isinstance(value.value, str)
    return value.value


def test_all_defaults_reference_the_single_constant():
    tree = _module()
    _module_constant(tree, _CONST_NAME)  # must exist
    for class_name in _DATACLASSES:
        value = _class_field(tree, class_name, "max_episode_steps")
        assert isinstance(value, ast.Name) and value.id == _CONST_NAME, (
            f"{class_name}.max_episode_steps default should be {_CONST_NAME}, "
            "not an independent literal"
        )


def test_constant_matches_canonical_libero_cap():
    # The canonical table lives under ``ci/`` (stripped from the OSS mirror and
    # absent in torch-free/OSS environments), so guard this cross-check while the
    # pure-AST test above still runs anywhere.
    sim_eval_defaults = pytest.importorskip("ci.metrics.utils._sim_eval_defaults")
    tree = _module()
    constant = _module_constant(tree, _CONST_NAME)
    # RolloutConfig's default env is a LIBERO task, so its cap must match the
    # canonical LIBERO backend default rather than drifting from it.
    assert _string_default(tree, "RolloutConfig", "env_name").startswith("libero")
    assert constant == sim_eval_defaults.SIM_EVAL_DEFAULTS["libero"].max_episode_steps
