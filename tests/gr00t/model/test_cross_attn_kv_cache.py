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

"""Cross-attention K/V are projected once per inference, not once per denoising step.

``encoder_hidden_states`` (the backbone's vision-language embedding) is fixed for
the whole flow-matching loop, so ``to_k``/``to_v`` recompute identical tensors on
every timestep. ``CrossAttnKVCache`` projects them once and reuses them.

This is a memory-traffic optimisation aimed at bandwidth-bound edge targets
(Orin/Thor/Spark), where weight streaming dominates. On a discrete GPU the model
sits far above its bandwidth floor and the saving is invisible, so these tests
assert **bytes and call counts**, which are device-independent, rather than wall
time. All CPU-only.

The safety property that distinguishes this from the backbone-caching dead end is
lifetime: the cache is a local inside ``get_action_with_features`` and dies with
the call, so it can never pair fresh proprioception with stale vision.
"""

import pytest
import torch


diffusers = pytest.importorskip("diffusers")

from diffusers.models.attention_processor import AttnProcessor2_0  # noqa: E402
from gr00t.model.modules.dit import (  # noqa: E402
    AlternateVLDiT,
    BasicTransformerBlock,
    CachedCrossAttnProcessor,
    CrossAttnKVCache,
)


INNER, HEADS, HEAD_DIM, XDIM, LAYERS = 128, 4, 32, 96, 4
CTX_TOKENS, ACT_TOKENS, BATCH = 11, 7, 2


def _block(cross: bool = True) -> BasicTransformerBlock:
    torch.manual_seed(0)
    return BasicTransformerBlock(
        INNER,
        HEADS,
        HEAD_DIM,
        cross_attention_dim=XDIM if cross else None,
        norm_type="ada_norm",
        activation_fn="gelu-approximate",
        attention_bias=True,
        dropout=0.0,
        final_dropout=False,
    ).eval()


def _inputs():
    torch.manual_seed(1)
    return (
        torch.randn(BATCH, ACT_TOKENS, INNER),
        torch.randn(BATCH, CTX_TOKENS, XDIM),
        torch.randn(BATCH, INNER),
    )


def _count_projections(module):
    """Count to_k / to_v invocations across every attention in ``module``."""
    counts = {"to_k": 0, "to_v": 0}
    handles = []
    for name, sub in module.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in ("to_k", "to_v"):
            handles.append(
                sub.register_forward_hook(
                    lambda _m, _i, _o, k=leaf: counts.__setitem__(k, counts[k] + 1)
                )
            )
    return counts, handles


def test_processor_matches_diffusers_reference():
    """With no cache the processor must be bitwise identical to AttnProcessor2_0.

    Guards against drift from the diffusers implementation this mirrors.
    """
    block = _block()
    hidden, ctx, temb = _inputs()

    block.attn1.set_processor(AttnProcessor2_0())
    with torch.no_grad():
        reference = block(hidden, encoder_hidden_states=ctx, temb=temb)

    block.attn1.set_processor(CachedCrossAttnProcessor())
    with torch.no_grad():
        ours = block(hidden, encoder_hidden_states=ctx, temb=temb)

    assert torch.equal(reference, ours)


def test_cached_output_is_bitwise_identical():
    """Reusing cached K/V must not change the block output at all."""
    block = _block()
    hidden, ctx, temb = _inputs()

    with torch.no_grad():
        uncached = block(hidden, encoder_hidden_states=ctx, temb=temb)

    cache = CrossAttnKVCache()
    with torch.no_grad():
        first = block(hidden, encoder_hidden_states=ctx, temb=temb, kv_cache=cache, cache_key=0)
        second = block(hidden, encoder_hidden_states=ctx, temb=temb, kv_cache=cache, cache_key=0)

    assert torch.equal(uncached, first), "cache-populating pass diverged"
    assert torch.equal(uncached, second), "cache-hit pass diverged"


def test_cache_populates_once_and_is_reused():
    """A second call with the same key must not re-run to_k / to_v."""
    block = _block()
    hidden, ctx, temb = _inputs()
    cache = CrossAttnKVCache()
    counts, handles = _count_projections(block)
    try:
        with torch.no_grad():
            for _ in range(4):  # four denoising steps
                block(hidden, encoder_hidden_states=ctx, temb=temb, kv_cache=cache, cache_key=0)
    finally:
        for h in handles:
            h.remove()

    assert counts == {"to_k": 1, "to_v": 1}, f"expected one projection each, got {counts}"
    assert set(cache) == {0}


def test_self_attention_is_never_cached():
    """Self-attention K/V come from hidden_states, which changes every step."""
    block = _block(cross=False)
    hidden, _, temb = _inputs()
    cache = CrossAttnKVCache()
    counts, handles = _count_projections(block)
    try:
        with torch.no_grad():
            for _ in range(3):
                block(hidden, encoder_hidden_states=None, temb=temb, kv_cache=cache, cache_key=0)
    finally:
        for h in handles:
            h.remove()

    assert counts == {"to_k": 3, "to_v": 3}, f"self-attention must not be cached, got {counts}"
    assert len(cache) == 0


def test_full_dit_saves_projections_across_denoising_steps():
    """End-to-end over AlternateVLDiT: cross-attn projections drop from 4x to 1x."""
    torch.manual_seed(0)
    dit = AlternateVLDiT(
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        num_layers=LAYERS,
        output_dim=INNER,
        cross_attention_dim=XDIM,
        interleave_self_attention=True,
        positional_embeddings=None,
        dropout=0.0,
        final_dropout=False,
        norm_type="ada_norm",
    ).eval()

    torch.manual_seed(2)
    hidden = torch.randn(BATCH, ACT_TOKENS, dit.inner_dim)
    ctx = torch.randn(BATCH, CTX_TOKENS, XDIM)
    timestep = torch.zeros(BATCH, dtype=torch.long)
    image_mask = torch.zeros(BATCH, CTX_TOKENS, dtype=torch.bool)
    image_mask[:, : CTX_TOKENS // 2] = True
    attn_mask = torch.ones(BATCH, CTX_TOKENS, dtype=torch.bool)
    kwargs = dict(
        encoder_hidden_states=ctx,
        timestep=timestep,
        image_mask=image_mask,
        backbone_attention_mask=attn_mask,
    )

    steps = 4
    base_counts, handles = _count_projections(dit)
    try:
        with torch.no_grad():
            baseline = [dit(hidden, **kwargs) for _ in range(steps)]
    finally:
        for h in handles:
            h.remove()

    counts, handles = _count_projections(dit)
    cache = CrossAttnKVCache()
    try:
        with torch.no_grad():
            cached = [dit(hidden, kv_cache=cache, **kwargs) for _ in range(steps)]
    finally:
        for h in handles:
            h.remove()

    for i, (a, b) in enumerate(zip(baseline, cached)):
        assert torch.equal(a, b), f"denoising step {i} diverged with the cache"

    # Self-attention layers still project every step (their source changes); only
    # the cross-attention layers collapse to a single projection.
    cross_layers = LAYERS // 2
    saved = (steps - 1) * cross_layers
    for proj in ("to_k", "to_v"):
        assert base_counts[proj] == steps * LAYERS, (
            f"baseline {proj} should be one per layer per step, got {base_counts[proj]}"
        )
        assert counts[proj] == base_counts[proj] - saved, (
            f"{proj}: expected {base_counts[proj]} - {saved}, got {counts[proj]}"
        )
    assert len(cache) == cross_layers


def test_cache_is_not_shared_across_inferences():
    """A fresh cache must re-project: stale vision would be a correctness bug.

    Mirrors the real call structure, where the cache is a local in
    ``get_action_with_features`` and cannot outlive one inference.
    """
    block = _block()
    hidden, ctx, temb = _inputs()

    with torch.no_grad():
        out_a = block(
            hidden, encoder_hidden_states=ctx, temb=temb, kv_cache=CrossAttnKVCache(), cache_key=0
        )

    torch.manual_seed(99)
    new_ctx = torch.randn(BATCH, CTX_TOKENS, XDIM)  # next control step: vision changed
    with torch.no_grad():
        out_b = block(
            hidden,
            encoder_hidden_states=new_ctx,
            temb=temb,
            kv_cache=CrossAttnKVCache(),
            cache_key=0,
        )
        expected = block(hidden, encoder_hidden_states=new_ctx, temb=temb)

    assert not torch.equal(out_a, out_b), "new vision must change the output"
    assert torch.equal(expected, out_b), "fresh cache must match the uncached path"


def test_denoising_loop_threads_the_cache_through():
    """The real ``get_action_with_features`` loop must reuse K/V across timesteps.

    The block-level tests above exercise ``dit.py`` directly and pass whether or
    not the action head wires a cache in, so this is the test that pins the
    integration.
    """
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
    from transformers.feature_extraction_utils import BatchFeature

    steps, layers, seq = 4, 4, 8
    config = Gr00tN1d7Config(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=steps,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=True,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": layers,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    torch.manual_seed(0)
    head = Gr00tN1d7ActionHead(config).eval()

    backbone_output = BatchFeature(
        data={
            "backbone_features": torch.randn(2, seq, config.backbone_embedding_dim),
            # real backbone emits bool masks (attention_mask == 1)
            "backbone_attention_mask": torch.ones(2, seq, dtype=torch.bool),
            "image_mask": torch.ones(2, seq, dtype=torch.bool),
        }
    )
    action_input = BatchFeature(
        data={
            "state": torch.randn(2, config.state_history_length, config.max_state_dim),
            "embodiment_id": torch.zeros(2, dtype=torch.long),
        }
    )

    counts, handles = _count_projections(head.model)
    try:
        with torch.no_grad():
            torch.manual_seed(7)
            head.get_action(backbone_output, action_input)
    finally:
        for h in handles:
            h.remove()

    cross_layers = layers // 2
    # cross layers project once for the whole loop; self layers project every step
    expected = cross_layers + cross_layers * steps
    for proj in ("to_k", "to_v"):
        assert counts[proj] == expected, (
            f"{proj}: expected {expected} "
            f"({cross_layers} cached cross + {cross_layers * steps} self), got {counts[proj]}"
        )
