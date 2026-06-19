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

"""Unit tests for the no-weight load harness buffer handling.

``_zero_no_weight_model_parameters`` zeros uninitialized params/persistent
buffers under the ``GROOT_SKIP_HF_MODEL_WEIGHTS`` path, but must NOT zero
non-persistent derived buffers (e.g. RoPE ``inv_freq``), which the module
recomputes analytically in ``__init__`` and which are not in ``state_dict``.
Zeroing them would collapse RoPE to cos≡1/sin≡0.
"""

from gr00t import _zero_no_weight_model_parameters
import torch


class _TinyRotary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.register_buffer("persistent_buf", torch.ones(4), persistent=True)
        # analytic, derived, not in state_dict — must survive the harness
        self.register_buffer("inv_freq", torch.tensor([1.0, 0.5, 0.25, 0.125]), persistent=False)


class _Nested(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rotary = _TinyRotary()
        self.lin = torch.nn.Linear(4, 4)


def test_zeros_params_and_persistent_buffers():
    m = _TinyRotary()
    _zero_no_weight_model_parameters(m)
    assert torch.all(m.weight == 0), "parameters should be zeroed"
    assert torch.all(m.persistent_buf == 0), "persistent buffers should be zeroed"


def test_preserves_non_persistent_buffers():
    m = _TinyRotary()
    expected = m.inv_freq.clone()
    _zero_no_weight_model_parameters(m)
    assert torch.equal(m.inv_freq, expected), "non-persistent inv_freq must be preserved (analytic)"
    assert bool((m.inv_freq != 0).any()), "inv_freq must not be zeroed (RoPE would degenerate)"


def test_preserves_non_persistent_in_nested_modules():
    m = _Nested()
    expected = m.rotary.inv_freq.clone()
    _zero_no_weight_model_parameters(m)
    assert torch.equal(m.rotary.inv_freq, expected), (
        "nested non-persistent buffer must be preserved"
    )
    assert torch.all(m.lin.weight == 0), "nested params should still be zeroed"
