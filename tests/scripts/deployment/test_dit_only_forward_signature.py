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

"""Pin the ``--export-mode dit_only`` forward-method signature contract.

The TRT ``dit_only`` setup monkey-patches ``action_head.get_action_with_features``.
``Gr00tN1d7.get_action`` invokes that method with the *full* keyword set
(``action_input`` / ``options`` included); a drift where the patch drops those
keywords compiles fine but explodes at verify/inference time with a
``TypeError: ... got an unexpected keyword argument 'action_input'`` — the only
TRT mode documented for Orin. This test reproduces the contract without a GPU or
TRT engine by stubbing ``Engine``, so the signature can't silently drift again.
"""

from __future__ import annotations

import inspect
import os
import sys
import types

import pytest


DEPLOY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts/deployment"))
if DEPLOY_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_DIR)

# Keywords Gr00tN1d7.get_action() forwards to get_action_with_features().
GET_ACTION_WITH_FEATURES_KWARGS = (
    "backbone_features",
    "state_features",
    "embodiment_id",
    "backbone_output",
    "action_input",
    "options",
)


@pytest.fixture(scope="module")
def trt_model_forward():
    try:
        import trt_model_forward as mod  # noqa: E402
    except (ImportError, OSError) as e:  # torch/tensorrt or native CUDA libs absent on CPU CI
        pytest.skip(f"trt_model_forward not importable in this env: {e}")
    return mod


def test_dit_only_patch_binds_get_action_caller_kwargs(trt_model_forward, monkeypatch):
    """``_setup_dit_only`` must install a forward that binds every keyword
    ``get_action`` passes — otherwise verify/inference crashes on Orin."""
    monkeypatch.setattr(trt_model_forward, "Engine", lambda *a, **k: object())
    monkeypatch.setattr(trt_model_forward.torch.cuda, "empty_cache", lambda: None)

    action_head = types.SimpleNamespace(model=object())
    policy = types.SimpleNamespace(model=types.SimpleNamespace(action_head=action_head))

    trt_model_forward._setup_dit_only(policy, "/tmp/does-not-exist")

    sig = inspect.signature(action_head.get_action_with_features)
    # Raises TypeError on the pre-fix 4-arg signature; binds cleanly once fixed.
    sig.bind(**{name: None for name in GET_ACTION_WITH_FEATURES_KWARGS})
