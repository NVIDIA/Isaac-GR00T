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

"""Test the transformer modules in gr00t.model.modules.dit."""

from gr00t.model.modules.dit import SelfAttentionTransformer
import pytest
import torch


HEADS, HEAD_DIM, LAYERS = 2, 8, 3
HIDDEN = HEADS * HEAD_DIM
REAL = 5


@pytest.fixture
def self_attention():
    torch.manual_seed(0)
    model = SelfAttentionTransformer(
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        num_layers=LAYERS,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
    )
    return model.eval()


@pytest.mark.parametrize("pad_len", [1, 7, 23])
def test_masked_positions_do_not_reach_real_tokens(self_attention, pad_len):
    """Left padding must not change the tokens it is padding.

    VL tokens are left-padded to the longest prompt in the batch, so an unmasked stack makes
    one observation's answer depend on which other observations share its batch.
    """
    torch.manual_seed(1)
    real = torch.randn(1, REAL, HIDDEN)
    mask = torch.zeros(1, pad_len + REAL, dtype=torch.bool)
    mask[:, -REAL:] = True

    def run(pad):
        return self_attention(torch.cat([pad, real], dim=1), attention_mask=mask)[:, -REAL:]

    with torch.no_grad():
        quiet = run(torch.randn(1, pad_len, HIDDEN))
        loud = run(torch.randn(1, pad_len, HIDDEN) * 1000.0)

    torch.testing.assert_close(quiet, loud)


def test_all_true_mask_matches_no_mask(self_attention):
    """A batch without padding, such as a single sample, is unaffected by the mask."""
    torch.manual_seed(2)
    hidden_states = torch.randn(2, REAL, HIDDEN)

    with torch.no_grad():
        unmasked = self_attention(hidden_states)
        masked = self_attention(hidden_states, attention_mask=torch.ones(2, REAL, dtype=torch.bool))

    torch.testing.assert_close(unmasked, masked)
