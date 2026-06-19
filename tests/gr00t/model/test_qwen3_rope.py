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

"""CPU-only oracle tests for Qwen3-VL RoPE ``inv_freq`` determinism.

Qwen3-VL registers its vision and text RoPE ``inv_freq`` with
``persistent=False``, so the buffers are not restored from a checkpoint and can
be left uninitialized when the model is loaded under a ``meta`` / no-init
context (the silent backend-divergence bug). ``Qwen3Backbone`` repairs them once
at load time by re-deriving the analytic frequencies via each module's own
constructor.

These tests verify that repair against an *independent* closed-form oracle (the
RoPE math written out here, not re-using transformers' helper), assert the
repair actually fires when the buffer is corrupt, and assert idempotence when it
is already correct. The oracle is carried all the way to the **cos/sin** the
attention kernels consume (including interleaved mRoPE and the configured
``rope_type`` branch), not just ``inv_freq`` -- a backend-vs-backend fingerprint
agrees on a common-mode error in that shared assembly, an absolute oracle does
not. No checkpoint download or GPU is required.
"""

import logging
import types

import pytest
import torch


qwen3_vl = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
from gr00t.model.modules.qwen3_backbone import (  # noqa: E402
    Qwen3Backbone,
    _assign_inv_freq,
    recompute_text_rotary_inv_freq,
    recompute_vision_rotary_inv_freq,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import (  # noqa: E402
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)


Qwen3VLVisionRotaryEmbedding = qwen3_vl.Qwen3VLVisionRotaryEmbedding
Qwen3VLTextRotaryEmbedding = qwen3_vl.Qwen3VLTextRotaryEmbedding

# Qwen3VLVisionRotaryEmbedding hardcodes theta=10000.0 in its constructor.
_VISION_THETA = 10000.0


def _vision_config() -> Qwen3VLVisionConfig:
    return Qwen3VLVisionConfig(hidden_size=32, num_heads=4)


def _text_config(rope_scaling: dict | None = None, head_dim: int = 8) -> Qwen3VLTextConfig:
    return Qwen3VLTextConfig(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=head_dim,
        rope_theta=10000.0,
        rope_scaling=rope_scaling or {"rope_type": "default", "mrope_section": [2, 1, 1]},
        max_position_embeddings=64,
    )


def _vision_head_dim_half(cfg: Qwen3VLVisionConfig) -> int:
    return (cfg.hidden_size // cfg.num_heads) // 2


def _oracle_inv_freq(dim: int, base: float) -> torch.Tensor:
    """Independent closed-form RoPE inverse frequencies (external-truth oracle).

    ``inv_freq[i] = 1 / base ** ((2 * i) / dim)`` for ``i in [0, dim/2)``. This is
    written out by hand so it does not share code with the production path.
    """
    exponents = torch.arange(0, dim, 2, dtype=torch.float32) / dim
    return 1.0 / (base**exponents)


def _corrupt(rotary: torch.nn.Module, name: str = "inv_freq") -> None:
    """Simulate an uninitialized non-persistent buffer (NaN garbage)."""
    current = getattr(rotary, name)
    setattr(rotary, name, torch.full_like(current, float("nan")))


def _oracle_text_cos_sin(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    mrope_section: list[int],
    attention_scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent closed-form mRoPE cos/sin (external-truth oracle).

    Re-derives Qwen3-VL's interleaved-mRoPE cos/sin from ``position_ids`` and a
    closed-form ``inv_freq`` without calling the production forward or
    ``apply_interleaved_mrope``: per-section frequencies are the outer product
    ``position * inv_freq``; the merged track starts as the T section and the H
    (offset 1) and W (offset 2) sections overwrite every third channel up to
    ``mrope_section[axis] * 3``. A drift in the reset frequencies, the section
    interleave, or the channel layout that all three backends share would change
    these values, which a backend-vs-backend fingerprint cannot see.

    ``position_ids`` is ``(3, seq)`` (T/H/W); the return is ``(seq, 2 * half)``.
    """
    half = inv_freq.shape[0]
    pos = position_ids.to(torch.float32)
    freqs = pos[:, :, None] * inv_freq[None, None, :]  # (3, seq, half)
    merged = freqs[0].clone()  # T section
    for axis, offset in ((1, 1), (2, 2)):  # H then W
        upper = min(mrope_section[axis] * 3, half)
        cols = list(range(offset, upper, 3))
        if cols:
            merged[:, cols] = freqs[axis][:, cols]
    emb = torch.cat([merged, merged], dim=-1)
    return emb.cos() * attention_scaling, emb.sin() * attention_scaling


def _oracle_vision_cos_sin(
    inv_freq: torch.Tensor, seqlen: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent closed-form vision RoPE cos/sin for positions ``0..seqlen-1``.

    Mirrors the vision tower contract: ``freqs = outer(arange(seqlen), inv_freq)``
    then ``emb = cat(freqs, freqs); (emb.cos(), emb.sin())``.
    """
    seq = torch.arange(seqlen, dtype=torch.float32)
    freqs = torch.outer(seq, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _corrupted_backbone(vis_cfg: Qwen3VLVisionConfig, txt_cfg: Qwen3VLTextConfig):
    """Build a Qwen3Backbone shell whose vision+text RoPE buffers are corrupt.

    Wires real Qwen3-VL rotary modules under the attribute layout
    ``_reset_rotary_inv_freq`` expects, with both ``inv_freq`` buffers NaN-filled
    so a successful reset is observable. No checkpoint / GPU is loaded.
    """
    backbone = Qwen3Backbone.__new__(Qwen3Backbone)
    torch.nn.Module.__init__(backbone)

    dim = _vision_head_dim_half(vis_cfg)
    vision_rotary = Qwen3VLVisionRotaryEmbedding(dim)
    text_rotary = Qwen3VLTextRotaryEmbedding(config=txt_cfg)
    _corrupt(vision_rotary)
    _corrupt(text_rotary)

    visual = torch.nn.Module()
    visual.rotary_pos_emb = vision_rotary
    language_model = torch.nn.Module()
    language_model.rotary_emb = text_rotary

    model = torch.nn.Module()
    model.visual = visual
    model.language_model = language_model
    model.config = types.SimpleNamespace(vision_config=vis_cfg)
    backbone.model = model
    return backbone


class TestVisionRotaryInvFreq:
    def test_matches_closed_form_oracle_after_repair(self):
        cfg = _vision_config()
        dim = _vision_head_dim_half(cfg)
        rotary = Qwen3VLVisionRotaryEmbedding(dim)
        _corrupt(rotary)

        new = recompute_vision_rotary_inv_freq(rotary, dim, torch.device("cpu"))
        fired = _assign_inv_freq(rotary, "inv_freq", new, persistent=False)

        assert fired, "corrupt vision inv_freq must be repaired, not silently skipped"
        assert rotary.inv_freq.dtype == torch.float32
        assert torch.isfinite(rotary.inv_freq).all()
        assert torch.equal(rotary.inv_freq, _oracle_inv_freq(dim, _VISION_THETA))

    def test_repaired_after_meta_construction(self):
        cfg = _vision_config()
        dim = _vision_head_dim_half(cfg)
        with torch.device("meta"):
            rotary = Qwen3VLVisionRotaryEmbedding(dim)
        assert rotary.inv_freq.device.type == "meta"

        new = recompute_vision_rotary_inv_freq(rotary, dim, torch.device("cpu"))

        assert new.device.type == "cpu"
        assert torch.isfinite(new).all()
        assert torch.equal(new, _oracle_inv_freq(dim, _VISION_THETA))

    def test_repair_is_idempotent(self):
        cfg = _vision_config()
        dim = _vision_head_dim_half(cfg)
        rotary = Qwen3VLVisionRotaryEmbedding(dim)
        _corrupt(rotary)

        _assign_inv_freq(
            rotary,
            "inv_freq",
            recompute_vision_rotary_inv_freq(rotary, dim, torch.device("cpu")),
            persistent=False,
        )
        fired_again = _assign_inv_freq(
            rotary,
            "inv_freq",
            recompute_vision_rotary_inv_freq(rotary, dim, torch.device("cpu")),
            persistent=False,
        )

        assert fired_again is False, "second repair on a correct buffer must be a no-op"


class TestTextRotaryInvFreq:
    def test_matches_closed_form_oracle_after_repair(self):
        cfg = _text_config()
        rotary = Qwen3VLTextRotaryEmbedding(config=cfg)
        _corrupt(rotary)

        inv_freq, scaling = recompute_text_rotary_inv_freq(rotary, cfg, torch.device("cpu"))
        fired = _assign_inv_freq(rotary, "inv_freq", inv_freq, persistent=False)

        assert fired, "corrupt text inv_freq must be repaired, not silently skipped"
        assert rotary.inv_freq.dtype == torch.float32
        assert torch.isfinite(rotary.inv_freq).all()
        assert torch.equal(rotary.inv_freq, _oracle_inv_freq(cfg.head_dim, cfg.rope_theta))
        assert scaling == pytest.approx(1.0)

    def test_repaired_after_meta_construction(self):
        cfg = _text_config()
        with torch.device("meta"):
            rotary = Qwen3VLTextRotaryEmbedding(config=cfg)
        assert rotary.inv_freq.device.type == "meta"

        inv_freq, _scaling = recompute_text_rotary_inv_freq(rotary, cfg, torch.device("cpu"))

        assert inv_freq.device.type == "cpu"
        assert torch.isfinite(inv_freq).all()
        assert torch.equal(inv_freq, _oracle_inv_freq(cfg.head_dim, cfg.rope_theta))


class TestBackboneRotaryReset:
    """Exercise the backbone wiring end-to-end without loading a checkpoint."""

    def _fake_backbone(self):
        vis_cfg = _vision_config()
        txt_cfg = _text_config()
        dim = _vision_head_dim_half(vis_cfg)
        backbone = _corrupted_backbone(vis_cfg, txt_cfg)
        vision_rotary = backbone.model.visual.rotary_pos_emb
        text_rotary = backbone.model.language_model.rotary_emb
        return backbone, vision_rotary, text_rotary, vis_cfg, txt_cfg, dim

    def test_reset_repairs_both_paths(self):
        backbone, vision_rotary, text_rotary, vis_cfg, txt_cfg, dim = self._fake_backbone()

        backbone._reset_rotary_inv_freq()

        assert torch.equal(vision_rotary.inv_freq, _oracle_inv_freq(dim, _VISION_THETA))
        assert torch.equal(
            text_rotary.inv_freq, _oracle_inv_freq(txt_cfg.head_dim, txt_cfg.rope_theta)
        )
        # original_inv_freq (used by dynamic-RoPE updates) must track inv_freq.
        assert torch.equal(text_rotary.original_inv_freq, text_rotary.inv_freq)

    def test_reset_warns_when_layout_missing(self, caplog):
        backbone = Qwen3Backbone.__new__(Qwen3Backbone)
        torch.nn.Module.__init__(backbone)
        model = torch.nn.Module()
        model.config = types.SimpleNamespace()
        backbone.model = model

        with caplog.at_level(logging.WARNING):
            backbone._reset_rotary_inv_freq()

        assert any("rotary" in record.message.lower() for record in caplog.records), (
            "missing Qwen3-VL layout must warn loudly, not fail silently"
        )


class TestTextRotaryCosSinOracle:
    """Pin the *cos/sin* the attention kernels consume, not just ``inv_freq``.

    The inv_freq tests stop one layer above the value attention actually uses;
    the interleaved-mRoPE assembly (``inv_freq`` -> per-section freqs -> channel
    interleave -> cos/sin) sits between them and is shared by every backend, so a
    backend-vs-backend fingerprint is blind to a common-mode error there. These
    drive the reset module's own ``forward`` and compare against an independent
    closed-form oracle, across several ``mrope_section`` layouts and with a
    distinct position per T/H/W section so the interleave is observable.
    """

    @pytest.mark.parametrize("mrope_section", [[4, 2, 2], [2, 4, 2], [2, 2, 4], [8, 0, 0]])
    def test_cos_sin_matches_closed_form_oracle(self, mrope_section):
        head_dim = 16  # half = 8: large enough for section layouts to differ
        cfg = _text_config(
            rope_scaling={"rope_type": "default", "mrope_section": mrope_section},
            head_dim=head_dim,
        )
        backbone = _corrupted_backbone(_vision_config(), cfg)
        backbone._reset_rotary_inv_freq()
        rotary = backbone.model.language_model.rotary_emb

        seq = 6
        # Distinct positions per section; identical sections would mask interleave.
        position_ids = torch.stack(
            [torch.arange(seq), torch.arange(seq) + 17, torch.arange(seq) + 41]
        )[:, None, :]  # (3, bs=1, seq)
        cos, sin = rotary.forward(torch.zeros(1, seq, 1), position_ids)

        inv_freq = _oracle_inv_freq(head_dim, cfg.rope_theta)
        exp_cos, exp_sin = _oracle_text_cos_sin(inv_freq, position_ids[:, 0, :], mrope_section)

        assert cos.shape == (1, seq, head_dim)
        assert torch.allclose(cos[0], exp_cos, atol=1e-6, rtol=0)
        assert torch.allclose(sin[0], exp_sin, atol=1e-6, rtol=0)


class TestTextRopeTypeBranches:
    """The recompute must honor ``rope_scaling.rope_type``, not assume default.

    Each branch is checked against a closed form re-derived from the documented
    rope-init math (default: base inv_freq; linear: base / factor; dynamic: base
    inv_freq at construction-time seq_len), so a future change that silently
    routes every config through the default initializer is caught.
    """

    @pytest.mark.parametrize(
        "rope_scaling, freq_divisor",
        [
            ({"rope_type": "default", "mrope_section": [2, 1, 1]}, 1.0),
            ({"rope_type": "linear", "factor": 4.0, "mrope_section": [2, 1, 1]}, 4.0),
            ({"rope_type": "dynamic", "factor": 4.0, "mrope_section": [2, 1, 1]}, 1.0),
        ],
    )
    def test_recompute_honors_rope_type(self, rope_scaling, freq_divisor):
        cfg = _text_config(rope_scaling=rope_scaling)
        rotary = Qwen3VLTextRotaryEmbedding(config=cfg)
        _corrupt(rotary)

        inv_freq, scaling = recompute_text_rotary_inv_freq(rotary, cfg, torch.device("cpu"))
        expected = _oracle_inv_freq(cfg.head_dim, cfg.rope_theta) / freq_divisor

        assert torch.equal(inv_freq, expected)
        assert scaling == pytest.approx(1.0)


class TestVisionRotaryCosSin:
    """Reach the vision cos/sin contract, not just ``inv_freq``."""

    def test_cos_sin_matches_closed_form_oracle(self):
        vis_cfg = _vision_config()
        backbone = _corrupted_backbone(vis_cfg, _text_config())
        backbone._reset_rotary_inv_freq()
        rotary = backbone.model.visual.rotary_pos_emb

        seqlen = 7
        freqs = rotary.forward(seqlen)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()

        inv_freq = _oracle_inv_freq(_vision_head_dim_half(vis_cfg), _VISION_THETA)
        exp_cos, exp_sin = _oracle_vision_cos_sin(inv_freq, seqlen)

        assert torch.allclose(cos, exp_cos, atol=1e-6, rtol=0)
        assert torch.allclose(sin, exp_sin, atol=1e-6, rtol=0)
