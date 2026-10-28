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

"""Pin the deployment-side contract for ``strict_action_padding_mask``.

Two things can drift apart here. The TensorRT paths must *refuse* a checkpoint
trained with the strict mask, because the engines cannot reproduce the masking
and would quietly serve a different policy. They must also still *accept* the
``hidden_attention_mask`` keyword, because the action head passes it on every
call, strict or not, so a wrapper that drops it breaks the default path too.
"""

from __future__ import annotations

import inspect
import os
import sys
import types

from gr00t.deployment.action_padding import (
    STRICT_ACTION_PADDING_MASK_UNSUPPORTED_MODES,
    assert_strict_action_padding_mask_disabled,
)
from gr00t.deployment.modes import InferenceMode
import pytest


DEPLOY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts/deployment"))
if DEPLOY_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_DIR)

# Keywords Gr00tN1d7ActionHead passes to the DiT it may have swapped for TRT.
DIT_FORWARD_KWARGS = (
    "hidden_states",
    "encoder_hidden_states",
    "encoder_attention_mask",
    "timestep",
    "return_all_hidden_states",
    "image_mask",
    "backbone_attention_mask",
    "hidden_attention_mask",
)


def test_guard_rejects_a_strict_mask_checkpoint():
    config = types.SimpleNamespace(strict_action_padding_mask=True)
    with pytest.raises(NotImplementedError, match="strict_action_padding_mask"):
        assert_strict_action_padding_mask_disabled(config, "some TensorRT path")


def test_guard_accepts_the_default_checkpoint():
    config = types.SimpleNamespace(strict_action_padding_mask=False)
    assert_strict_action_padding_mask_disabled(config, "some TensorRT path")


def test_every_mode_that_replaces_action_head_compute_is_guarded():
    """``vit_llm_only`` keeps the action head in PyTorch and is the only mode
    that can serve a strict-mask checkpoint. A new mode must pick a side."""
    assert set(InferenceMode) - STRICT_ACTION_PADDING_MASK_UNSUPPORTED_MODES == {
        InferenceMode.vit_llm_only
    }


def test_standalone_trt_dit_binds_every_action_head_keyword(monkeypatch):
    try:
        import standalone_inference_script as mod
    except (ImportError, OSError) as e:  # torch/tensorrt absent on CPU CI
        pytest.skip(f"standalone_inference_script not importable in this env: {e}")

    monkeypatch.setattr(mod, "TensorRTDiTWrapper", lambda *a, **k: object())
    action_head = types.SimpleNamespace(
        model=types.SimpleNamespace(),
        config=types.SimpleNamespace(strict_action_padding_mask=False),
    )
    policy = types.SimpleNamespace(model=types.SimpleNamespace(action_head=action_head))

    mod.replace_dit_with_tensorrt(policy, "/does-not-exist")

    sig = inspect.signature(action_head.model.forward)
    sig.bind(**{name: None for name in DIT_FORWARD_KWARGS})
