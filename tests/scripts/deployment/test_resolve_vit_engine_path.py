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

"""Regression tests for ``_resolve_vit_engine_path``.

Older builds named the ViT engine ``vit_bf16.engine`` regardless of
the source ONNX dtype — misleading whenever the FP32 ONNX path was
taken. New builds emit ``vit.engine``; the resolver bridges both names
during the rollout so existing engine directories keep working.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import types

import pytest


DEPLOY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts/deployment"))


@pytest.fixture
def resolve_vit_engine_path(monkeypatch):
    """Yield ``_resolve_vit_engine_path`` with heavy deps stubbed in ``sys.modules``.

    Every side effect goes through ``monkeypatch`` so it is reverted at
    teardown and never leaks across tests collected by the same pytest-xdist
    worker. Installing these stubs at module top-level (the previous form of
    this file) replaces ``sys.modules['trt_torch'].Engine`` with ``object``;
    if the same worker later runs ``test_trt_full_pipeline``, ``Engine(path)``
    raises ``TypeError: object() takes no arguments``. The same lesson is
    spelled out in ``tests/scripts/deployment/test_build_tensorrt_engine.py``.

    Forcing a fresh import of ``trt_model_forward`` is part of the contract:
    if a prior test imported it against the real ``trt_torch``, the cached
    module would still hold the real ``Engine`` symbol; if a prior test left
    it cached against a stub, the cached module would still hold ``object``.
    Re-importing under our currently-installed stub keeps the binding honest.
    """
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        # ``trt_model_forward`` annotates helpers with ``torch.Tensor``;
        # the annotation is evaluated at import time, so the stub has to
        # expose *something* at that name even when we never call into it.
        torch_stub.Tensor = type("Tensor", (), {})
        monkeypatch.setitem(sys.modules, "torch", torch_stub)
    if "transformers" not in sys.modules:
        monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
        feat = types.ModuleType("transformers.feature_extraction_utils")
        feat.BatchFeature = object
        monkeypatch.setitem(sys.modules, "transformers.feature_extraction_utils", feat)
    if "trt_torch" not in sys.modules:
        trt_torch_stub = types.ModuleType("trt_torch")
        trt_torch_stub.Engine = object
        monkeypatch.setitem(sys.modules, "trt_torch", trt_torch_stub)

    monkeypatch.syspath_prepend(DEPLOY_DIR)
    monkeypatch.delitem(sys.modules, "trt_model_forward", raising=False)

    from trt_model_forward import _resolve_vit_engine_path as fn

    yield fn


def test_prefers_new_name_when_present(tmp_path: Path, resolve_vit_engine_path) -> None:
    """If both files exist, the precision-neutral name wins."""
    (tmp_path / "vit.engine").write_bytes(b"new")
    (tmp_path / "vit_bf16.engine").write_bytes(b"legacy")
    assert resolve_vit_engine_path(str(tmp_path)) == str(tmp_path / "vit.engine")


def test_falls_back_to_legacy_with_warning(tmp_path: Path, caplog, resolve_vit_engine_path) -> None:
    """Existing engine dirs built before this MR still load, with a nudge to rebuild."""
    (tmp_path / "vit_bf16.engine").write_bytes(b"legacy")

    with caplog.at_level(logging.WARNING):
        path = resolve_vit_engine_path(str(tmp_path))

    assert path == str(tmp_path / "vit_bf16.engine")
    assert any(
        "legacy" in rec.message and "rebuild" in rec.message.lower() for rec in caplog.records
    ), "Expected a warning prompting a rebuild; got: " + repr(
        [rec.message for rec in caplog.records]
    )


def test_returns_canonical_path_when_neither_exists(
    tmp_path: Path, resolve_vit_engine_path
) -> None:
    """No engine yet → return the new-style name so any 'not found' error is canonical."""
    assert resolve_vit_engine_path(str(tmp_path)) == str(tmp_path / "vit.engine")


@pytest.mark.parametrize("present", ["vit.engine", "vit_bf16.engine"])
def test_returns_existing_file_path(present: str, tmp_path: Path, resolve_vit_engine_path) -> None:
    """Either filename, alone, returns its own path."""
    (tmp_path / present).write_bytes(b"x")
    assert resolve_vit_engine_path(str(tmp_path)) == str(tmp_path / present)
