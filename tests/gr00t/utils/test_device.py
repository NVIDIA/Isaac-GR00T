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

"""Tests for default inference device selection and its use by the inference entry points."""

from __future__ import annotations

import importlib

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.utils.device import get_default_device
import pytest
from test_support.runtime import get_root
import torch


def _patch_availabilities(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    xpu_available: bool,
) -> None:
    """Fake torch availability probes."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    # Patch the probe only: torch.manual_seed reaches into the real torch.xpu module.
    monkeypatch.setattr(torch.xpu, "is_available", lambda: xpu_available)


class _PolicyBuilt(Exception):
    """Raised by the fake policy to stop an entry point right after it picks the device."""


def _fake_policy(monkeypatch: pytest.MonkeyPatch, module: object) -> dict:
    """Replace ``module.Gr00tPolicy`` with a stub that records the device it was given."""
    seen: dict = {}

    def fake(*args, **kwargs):
        seen["device"] = kwargs["device"]
        raise _PolicyBuilt

    monkeypatch.setattr(module, "Gr00tPolicy", fake)
    return seen


@pytest.mark.parametrize(
    ("cuda_available", "xpu_available", "expected"),
    [
        (True, True, "cuda"),
        (True, False, "cuda"),
        (False, True, "xpu"),
        (False, False, "cpu"),
    ],
)
def test_get_default_device_priority(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    xpu_available: bool,
    expected: str,
) -> None:
    """CUDA wins over XPU, and XPU wins over CPU."""
    _patch_availabilities(monkeypatch, cuda_available, xpu_available)
    assert get_default_device() == expected


def test_server_main_resolves_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_gr00t_server picks XPU when it is the only GPU."""
    from gr00t.eval import run_gr00t_server

    _patch_availabilities(monkeypatch, cuda_available=False, xpu_available=True)
    seen = _fake_policy(monkeypatch, run_gr00t_server)
    with pytest.raises(_PolicyBuilt):
        run_gr00t_server.main(run_gr00t_server.ServerConfig(model_path="nvidia/fake"))
    assert seen["device"] == "xpu"


def test_open_loop_eval_uses_default_device(monkeypatch: pytest.MonkeyPatch) -> None:
    from gr00t.eval import open_loop_eval

    _patch_availabilities(monkeypatch, cuda_available=False, xpu_available=True)
    seen = _fake_policy(monkeypatch, open_loop_eval)
    with pytest.raises(_PolicyBuilt):
        open_loop_eval.main(open_loop_eval.ArgsConfig(model_path="nvidia/fake"))
    assert seen["device"] == "xpu"


def test_rollout_sim_policy_uses_default_device(monkeypatch: pytest.MonkeyPatch) -> None:
    from gr00t.eval import rollout_policy
    import gr00t.policy.gr00t_policy as gr00t_policy

    _patch_availabilities(monkeypatch, cuda_available=False, xpu_available=True)
    seen = _fake_policy(monkeypatch, gr00t_policy)
    with pytest.raises(_PolicyBuilt):
        rollout_policy.create_gr00t_sim_policy(
            model_path="nvidia/fake", embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
        )
    assert seen["device"] == "xpu"


@pytest.fixture
def standalone_script(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(get_root() / "scripts" / "deployment"))
    return importlib.import_module("standalone_inference_script")


@pytest.mark.parametrize(
    ("xpu_available", "device_arg", "expected"),
    [(True, None, "xpu"), (False, "cpu", "cpu")],
    ids=["auto-xpu", "explicit-cpu-no-gpu"],
)
def test_standalone_script_device(
    monkeypatch: pytest.MonkeyPatch,
    standalone_script,
    xpu_available: bool,
    device_arg: str | None,
    expected: str,
) -> None:
    _patch_availabilities(monkeypatch, cuda_available=False, xpu_available=xpu_available)
    seen = _fake_policy(monkeypatch, standalone_script)
    with pytest.raises(_PolicyBuilt):
        standalone_script.main(
            standalone_script.ArgsConfig(model_path="nvidia/fake", device=device_arg)
        )
    assert seen["device"] == expected
