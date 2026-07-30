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

"""``Qwen3Backbone`` must not pay for language-model logits it never reads.

The backbone consumes only ``hidden_states[-1]``, but
``Qwen3VLForConditionalGeneration.forward`` unconditionally evaluates
``lm_head`` over the *whole* sequence and the *full* vocabulary (151936 for
Cosmos-Reason2-2B) and the result is discarded. Passing ``logits_to_keep=1``
confines that matmul to a single position.

These tests pin both halves of the contract:

1. the optimisation happens -- ``lm_head`` sees one position, not the sequence;
2. it is lossless -- every entry of ``hidden_states`` is bitwise unchanged.

They also pin the *reason* the obvious-looking alternative is wrong. Calling the
inner ``Qwen3VLModel`` and reading ``last_hidden_state`` would skip ``lm_head``
entirely, but it is **not** equivalent: that output class carries
``last_hidden_state``, so ``check_model_inputs`` ties ``hidden_states[-1]`` to the
POST-final-norm tensor, while ``Qwen3VLCausalLMOutputWithPast`` has no such field
and ``hidden_states[-1]`` is the PRE-final-norm decoder output this checkpoint was
trained against. ``test_pre_norm_semantics_are_load_bearing`` fails if anyone
"simplifies" the backbone that way.

CPU-only: builds a tiny randomly-initialised Qwen3-VL, so no gated checkpoint
download and no GPU.
"""

import pytest
import torch


transformers = pytest.importorskip("transformers")


def _tiny_qwen3vl():
    """A minimal randomly-initialised Qwen3VLForConditionalGeneration."""
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    torch.manual_seed(0)
    vision_config = dict(
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        depth=2,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=64,
        num_position_embeddings=64,
        deepstack_visual_indexes=[0],
    )
    text_config = dict(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=3,
        vocab_size=1000,
        head_dim=16,
        max_position_embeddings=512,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [4, 2, 2],
            "mrope_interleaved": True,
        },
    )
    # NOTE: Qwen3VLConfig only unpacks sub-configs given as dicts; passing config
    # objects leaves vision_config/text_config unset.
    config = Qwen3VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=100,
        video_token_id=101,
        vision_start_token_id=102,
        vision_end_token_id=103,
    )
    return Qwen3VLForConditionalGeneration(config).eval()


def _tiny_inputs(model):
    """One 32x32 image plus a couple of text tokens, matching the tiny config."""
    config = model.config
    vision_config = config.vision_config
    grid = torch.tensor([[1, 2, 2]])
    num_patches = int(grid.prod())
    pixel_values = torch.randn(
        num_patches,
        3 * vision_config.temporal_patch_size * vision_config.patch_size**2,
    )
    num_image_tokens = num_patches // vision_config.spatial_merge_size**2
    input_ids = torch.tensor([[5, 6] + [config.image_token_id] * num_image_tokens + [7, 8]])
    return dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        pixel_values=pixel_values,
        image_grid_thw=grid,
    )


@pytest.fixture(scope="module")
def tiny_model():
    return _tiny_qwen3vl()


@pytest.fixture(scope="module")
def tiny_inputs(tiny_model):
    return _tiny_inputs(tiny_model)


def _lm_head_input_shapes(model):
    """Record the shape of every tensor entering lm_head."""
    shapes = []
    handle = model.lm_head.register_forward_pre_hook(
        lambda _mod, args: shapes.append(tuple(args[0].shape))
    )
    return shapes, handle


def test_logits_to_keep_confines_lm_head_to_one_position(tiny_model, tiny_inputs):
    """lm_head must see a single position, not the whole sequence."""
    seq_len = tiny_inputs["input_ids"].shape[1]
    shapes, handle = _lm_head_input_shapes(tiny_model)
    try:
        with torch.no_grad():
            tiny_model(**tiny_inputs, output_hidden_states=True)
            tiny_model(**tiny_inputs, output_hidden_states=True, logits_to_keep=1)
    finally:
        handle.remove()

    baseline_shape, optimised_shape = shapes
    assert baseline_shape[1] == seq_len, "baseline should run lm_head over the full sequence"
    assert optimised_shape[1] == 1, f"expected one position, got {optimised_shape}"


def test_logits_to_keep_is_lossless_for_hidden_states(tiny_model, tiny_inputs):
    """Every hidden-state entry must be bitwise unchanged -- this is the safety claim."""
    with torch.no_grad():
        baseline = tiny_model(**tiny_inputs, output_hidden_states=True)
        optimised = tiny_model(**tiny_inputs, output_hidden_states=True, logits_to_keep=1)

    assert len(baseline.hidden_states) == len(optimised.hidden_states)
    for layer, (before, after) in enumerate(zip(baseline.hidden_states, optimised.hidden_states)):
        assert torch.equal(before, after), f"hidden_states[{layer}] changed"


def test_only_the_discarded_logits_shrink(tiny_model, tiny_inputs):
    """The logits we never read shrink from the full sequence to one position."""
    seq_len = tiny_inputs["input_ids"].shape[1]
    vocab = tiny_model.config.text_config.vocab_size
    with torch.no_grad():
        baseline = tiny_model(**tiny_inputs, output_hidden_states=True)
        optimised = tiny_model(**tiny_inputs, output_hidden_states=True, logits_to_keep=1)

    assert baseline.logits.shape == (1, seq_len, vocab)
    assert optimised.logits.shape == (1, 1, vocab)


def test_pre_norm_semantics_are_load_bearing(tiny_model, tiny_inputs):
    """``hidden_states[-1]`` is PRE-final-norm; the inner model's output is POST-norm.

    Guards against "simplifying" the backbone to
    ``self.model.model(...).last_hidden_state``, which silently inserts the final
    RMSNorm and changes what the action head consumes.
    """
    with torch.no_grad():
        wrapper_feature = tiny_model(**tiny_inputs, output_hidden_states=True).hidden_states[-1]
        inner_feature = tiny_model.model(**tiny_inputs).last_hidden_state

    assert not torch.equal(wrapper_feature, inner_feature), (
        "expected the wrapper's hidden_states[-1] to differ from the inner "
        "last_hidden_state; if these are now equal the norm placement changed"
    )
    final_norm = tiny_model.model.language_model.norm
    assert torch.equal(final_norm(wrapper_feature), inner_feature), (
        "inner last_hidden_state should be exactly final_norm(hidden_states[-1])"
    )


def test_backbone_forward_requests_only_one_logit_position(monkeypatch, tiny_model, tiny_inputs):
    """``Qwen3Backbone.forward`` itself must pass logits_to_keep=1 through."""
    from gr00t.model.modules import qwen3_backbone as backbone_module

    captured = {}
    real_forward = type(tiny_model).forward

    def spy(self, *args, **kwargs):
        captured.update(kwargs)
        return real_forward(self, *args, **kwargs)

    monkeypatch.setattr(type(tiny_model), "forward", spy)
    monkeypatch.setattr(
        backbone_module.Qwen3VLForConditionalGeneration,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: tiny_model),
    )

    backbone = backbone_module.Qwen3Backbone(
        model_name="tiny",
        select_layer=tiny_model.config.text_config.num_hidden_layers,
    )
    with torch.no_grad():
        out = backbone(transformers.feature_extraction_utils.BatchFeature(data=tiny_inputs))

    assert captured.get("logits_to_keep") == 1, f"logits_to_keep not forwarded: {captured}"
    assert out["backbone_features"].shape[:2] == tiny_inputs["input_ids"].shape
