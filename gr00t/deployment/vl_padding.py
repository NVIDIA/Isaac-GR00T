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

"""Guard for TensorRT paths that run the plain DiT, whose engine has no VL mask input.

With ``strict_vl_padding_mask`` on, PyTorch masks padded VL tokens out of the plain
DiT's cross-attention, but the exported engine cannot. The two agree whenever a batch
has no VL padding, which is always the case for a single sample, so only padded
batches are refused. With the flag off, PyTorch attends to the padding as the engine
does, so nothing is refused.
"""

from __future__ import annotations

import torch


def assert_plain_dit_sees_no_vl_padding(
    config, backbone_attention_mask: torch.Tensor | None, context: str
) -> None:
    """Refuse a padded batch when a plain-DiT TensorRT engine would ignore the VL mask.

    Args:
        config: Action-head config carrying ``use_alternate_vl_dit`` and
            ``strict_vl_padding_mask``.
        backbone_attention_mask: ``(B, S)`` mask, True for real VL tokens.
        context: Name of the path, interpolated into the error.
    """
    if (
        config.use_alternate_vl_dit
        or not config.strict_vl_padding_mask
        or backbone_attention_mask is None
    ):
        return
    if not bool(backbone_attention_mask.all()):
        raise NotImplementedError(
            f"{context} runs the plain DiT engine, which has no VL mask input, so a batch "
            "whose prompts differ in length would be served differently than in PyTorch. "
            "Run one sample per call, or serve it with the PyTorch action head."
        )
