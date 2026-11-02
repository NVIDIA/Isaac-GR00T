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

"""Capability guard for ``strict_action_padding_mask`` on the TensorRT paths.

The exported engines have no ``hidden_attention_mask`` input and the TensorRT
action-head loops do not apply the action mask, so a checkpoint trained with the
strict mask would silently be served with its padding visible again. The export
CLI, the engine-swap CLI, and the standalone script all refuse such a checkpoint,
so the check and its wording live here.
"""

from __future__ import annotations

from gr00t.deployment.modes import InferenceMode


# Modes that run the denoising loop on TensorRT engines. ``vit_llm_only`` leaves
# the whole action head in PyTorch, so it serves a strict-mask checkpoint fine.
# This lives next to the guard rather than in ``trt_model_forward`` so the set can
# be asserted without importing TensorRT.
STRICT_ACTION_PADDING_MASK_UNSUPPORTED_MODES = frozenset(
    {InferenceMode.n17_full_pipeline, InferenceMode.action_head, InferenceMode.dit_only}
)


def assert_strict_action_padding_mask_disabled(config, context: str) -> None:
    """Refuse a checkpoint whose config enables the strict action padding mask.

    Args:
        config: Model config carrying ``strict_action_padding_mask``.
        context: Name of the unsupported path, interpolated into the error.
    """
    if config.strict_action_padding_mask:
        raise NotImplementedError(
            f"{context} does not support strict_action_padding_mask. This checkpoint "
            "was trained with the padded action dimensions masked out, but the "
            "TensorRT path cannot reproduce that masking, so serving it here would "
            "change the policy. Serve it with the PyTorch action head instead."
        )
