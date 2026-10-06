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

"""CPU-only oracle tests for the ViT ONNX export rotary path.

Covers two silent-failure risks in ``scripts/deployment/export_onnx_n1d7.py``
whose only existing coverage is a backend-vs-backend cosine fingerprint (blind to
common-mode error):

- ``_apply_rotary_real`` re-implements the vision rotary application with
  real-valued ops. An independent closed-form (complex-arithmetic) oracle pins it,
  so a rotate-half / sign / precision drift is caught.
- The exporter freezes ``rot_pos_emb``-derived cos/sin built from a non-persistent
  ``inv_freq`` buffer. ``_assert_vision_rotary_matches_analytic`` must abort the
  export when that buffer drifts from the analytic value.

No checkpoint download or GPU is required.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch


DEPLOY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts/deployment"))
if DEPLOY_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_DIR)

export = pytest.importorskip("export_onnx_n1d7")
qwen3_vl = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
qwen3_vl_config = pytest.importorskip("transformers.models.qwen3_vl.configuration_qwen3_vl")


Qwen3VLVisionConfig = qwen3_vl_config.Qwen3VLVisionConfig
Qwen3VLVisionRotaryEmbedding = qwen3_vl.Qwen3VLVisionRotaryEmbedding


# --- _apply_rotary_real closed-form parity --------------------------------


def _complex_oracle_apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Independent rotary application via complex multiplication (float64).

    The production path is ``out = x * cos + rotate_half(x) * sin`` with
    ``rotate_half(x) = [-x2, x1]``. For the vision embeddings (``emb =
    cat(freqs, freqs)``, so the two cos/sin halves are equal) that is exactly the
    complex product ``(x1 + i x2) * (cos + i sin)`` re-split into ``[real, imag]``
    -- a genuinely different implementation, so a shared bug cannot hide.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half].double()
    x2 = x[..., half:].double()
    c = cos[..., :half].double().unsqueeze(1)
    s = sin[..., :half].double().unsqueeze(1)
    z = torch.complex(x1, x2) * torch.complex(c, s)
    return torch.cat([z.real, z.imag], dim=-1)


def test_apply_rotary_real_matches_complex_oracle():
    torch.manual_seed(0)
    seq, heads, half = 7, 3, 4
    x = torch.randn(seq, heads, 2 * half)
    freqs = torch.randn(seq, half)
    emb = torch.cat([freqs, freqs], dim=-1)  # how the exporter builds cos/sin
    cos, sin = emb.cos(), emb.sin()

    got = export._apply_rotary_real(x, cos, sin).double()
    expected = _complex_oracle_apply_rotary(x, cos, sin)

    assert got.shape == x.shape
    # _apply_rotary_real computes in float32 (to match the exported path), while the
    # oracle is float64, so tolerate float32-level rounding rather than exact equality.
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_apply_rotary_real_identity_at_zero_angle():
    # cos=1, sin=0 (freqs=0) must be a no-op, whatever the rotate_half layout is.
    x = torch.randn(5, 2, 8)
    cos = torch.ones(5, 8)
    sin = torch.zeros(5, 8)
    assert torch.allclose(export._apply_rotary_real(x, cos, sin), x, atol=1e-6)


# --- export-time rotary analytic oracle -----------------------------------


def _vision_config() -> Qwen3VLVisionConfig:
    return Qwen3VLVisionConfig(hidden_size=32, num_heads=4)  # head_dim = 8


def _fake_vision(dim: int) -> torch.nn.Module:
    vision = torch.nn.Module()
    vision.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(dim)
    return vision


def test_rotary_oracle_passes_for_correct_inv_freq():
    cfg = _vision_config()
    dim = (cfg.hidden_size // cfg.num_heads) // 2
    vision = _fake_vision(dim)  # fresh module -> analytic inv_freq
    export._assert_vision_rotary_matches_analytic(vision, cfg)  # must not raise


def test_rotary_oracle_raises_on_corrupt_inv_freq():
    cfg = _vision_config()
    dim = (cfg.hidden_size // cfg.num_heads) // 2
    vision = _fake_vision(dim)
    rotary = vision.rotary_pos_emb
    rotary.inv_freq = rotary.inv_freq + 1.0  # drift from the analytic value
    with pytest.raises(RuntimeError, match="diverges from the analytic oracle"):
        export._assert_vision_rotary_matches_analytic(vision, cfg)


def test_rotary_oracle_raises_when_layout_missing():
    vision = torch.nn.Module()  # no rotary_pos_emb attribute at all
    with pytest.raises(RuntimeError, match="not found"):
        export._assert_vision_rotary_matches_analytic(vision, _vision_config())
