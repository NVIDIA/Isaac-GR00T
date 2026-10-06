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

"""CPU-only lifecycle tests for ``scripts.deployment.trt_torch.Engine``.

Pins the regression that ``__init__`` no longer registers an ``atexit``
callback, plus the ``close()`` / ``__enter__`` / ``__exit__`` / ``__del__``
contracts the fix introduces. ``tensorrt`` is stubbed in ``sys.modules``
so the tests run on any host without a GPU.
"""

from __future__ import annotations

import atexit
import os
import sys
import types
from unittest import mock

import pytest


def _make_trt_stub() -> types.ModuleType:
    """Minimal ``tensorrt`` stub sufficient for ``Engine.__init__`` and
    ``Engine.load`` to run on CPU."""
    trt_stub = types.ModuleType("tensorrt")

    class _Logger:
        ERROR = 1

        def __init__(self, level=ERROR):
            self.level = level

    trt_stub.Logger = _Logger

    def _init_plugins(_logger, _ns):
        return None

    trt_stub.init_libnvinfer_plugins = _init_plugins

    class _IOMode:
        INPUT = 0
        OUTPUT = 1

    trt_stub.TensorIOMode = _IOMode

    for name in ("float32", "float16", "bfloat16", "int8", "int32", "bool", "uint8", "int64"):
        setattr(trt_stub, name, name)

    class _Handle:
        def __init__(self):
            self._tensors = []

        def __iter__(self):
            return iter(self._tensors)

        def create_execution_context(self):
            return mock.MagicMock(name="execution_context")

    class _Runtime:
        def __init__(self, _logger):
            self._logger = _logger

        def deserialize_cuda_engine(self, _blob):
            return _Handle()

    trt_stub.Runtime = _Runtime
    return trt_stub


@pytest.fixture
def engine_module(monkeypatch, tmp_path):
    """Yield ``(Engine, engine_file_path)`` with ``tensorrt`` stubbed."""
    monkeypatch.setitem(sys.modules, "tensorrt", _make_trt_stub())

    deploy_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../scripts/deployment")
    )
    monkeypatch.syspath_prepend(deploy_dir)
    monkeypatch.delitem(sys.modules, "trt_torch", raising=False)

    from trt_torch import Engine

    engine_file = tmp_path / "fake.engine"
    engine_file.write_bytes(b"not-a-real-engine")

    yield Engine, str(engine_file)


def test_init_does_not_register_atexit_callback(engine_module):
    """Regression: pre-fix ``__init__`` registered an ``atexit`` callback
    that segfaulted at shutdown and pinned the engine alive for the life
    of the process. It must stay gone."""
    Engine, engine_file = engine_module

    with mock.patch.object(atexit, "register") as fake_register:
        eng = Engine(engine_file)

    fake_register.assert_not_called()
    eng.close()


def test_close_clears_context_and_handle(engine_module):
    """After ``close()`` both attributes are ``None`` (state check; the
    source enforces context-before-handle ordering by line order)."""
    Engine, engine_file = engine_module
    eng = Engine(engine_file)
    assert eng.execution_context is not None
    assert eng.handle is not None

    eng.close()

    assert eng.execution_context is None
    assert eng.handle is None
    assert eng._closed is True


def test_close_is_idempotent(engine_module):
    """Repeat ``close()`` calls must be a noop so ``__del__`` can layer on
    top of explicit ``with``-block teardown safely."""
    Engine, engine_file = engine_module
    eng = Engine(engine_file)
    eng.close()
    eng.close()
    assert eng._closed is True


def test_del_does_not_raise_after_normal_close(engine_module):
    """After explicit ``close()`` the destructor must be a clean noop."""
    Engine, engine_file = engine_module
    eng = Engine(engine_file)
    eng.close()
    eng.__del__()


def test_del_swallows_errors_from_close(engine_module):
    """The defensive ``__del__`` shim must not propagate errors raised by
    ``close()`` — that path covers interpreter shutdown when the
    ``tensorrt`` module is already gone."""
    Engine, engine_file = engine_module
    eng = Engine(engine_file)
    with mock.patch.object(type(eng), "close", side_effect=RuntimeError("shutdown")):
        eng.__del__()


def test_del_does_not_raise_on_partially_initialized_engine(engine_module):
    """If ``__init__`` raises before ``load()`` completes, the destructor
    still runs. The pre-init of ``_closed`` / ``execution_context`` /
    ``handle`` keeps it safe."""
    Engine, engine_file = engine_module

    captured: list = []
    real_load = Engine.load

    def boom(self, file):
        captured.append(self)
        raise RuntimeError("simulated load failure")

    Engine.load = boom
    try:
        with pytest.raises(RuntimeError, match="simulated load failure"):
            Engine(engine_file)
    finally:
        Engine.load = real_load

    assert captured
    captured[0].__del__()
