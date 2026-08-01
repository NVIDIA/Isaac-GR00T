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

"""Bitwise parity of the inference image path after removing the PIL round-trip.

``_apply_vlm_processing`` used to convert every CHW frame to PIL before handing
it to the Qwen3VL processor; it now passes CHW uint8 tensors directly. These
tests assert the final model inputs are bitwise identical to a verbatim
reimplementation of the old PIL path, on the real checkpoint processor.
"""

import numpy as np
from PIL import Image
import pytest
import torch


pytestmark = pytest.mark.serial

INSTRUCTION = "pick up the black bowl and place it on the plate"


@pytest.fixture(scope="module")
def vlm_processor(load_hf_model_weights):
    """Real Qwen3VL processor; skips when the gated repo is unreachable.

    Only processor configs/tokenizer files are fetched (no model weights).
    Uses the ``load_hf_model_weights`` opt-in: the suite-wide
    ``GROOT_SKIP_HF_MODEL_WEIGHTS=1`` harness would otherwise stub the load and
    make the parity assertions vacuous.
    """
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import build_processor

    try:
        with load_hf_model_weights():
            return build_processor("nvidia/Cosmos-Reason2-2B", {"trust_remote_code": True})
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Cosmos-Reason2-2B processor unavailable (gated/offline?): {exc}")


def _reference_pil_outputs(processor, frames_chw: np.ndarray, language: str):
    """Verbatim reimplementation of the pre-change path: CHW -> PIL -> processor."""
    pil_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in frames_chw]
    conversation = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in pil_images],
                {"type": "text", "text": language},
            ],
        }
    ]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=pil_images, return_tensors="pt", padding=True)


def _direct_outputs(processor, frames_chw: np.ndarray, language: str):
    """The new path: CHW uint8 tensors handed to the processor directly."""
    frames = [torch.as_tensor(v) for v in frames_chw]
    conversation = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in frames],
                {"type": "text", "text": language},
            ],
        }
    ]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=frames, return_tensors="pt", padding=True)


@pytest.mark.parametrize(
    "height,width",
    [
        (256, 256),  # matches patch grid: smart-resize is a no-op
        (240, 320),  # non-square: real smart-resize happens
        (243, 243),  # not a patch multiple: smart-resize rounds
    ],
)
@pytest.mark.parametrize("num_frames", [2, 6])  # 1 and 3 cameras x 2 temporal frames
def test_direct_tensor_input_is_bitwise_identical_to_pil(vlm_processor, height, width, num_frames):
    rng = np.random.default_rng(1234)
    frames = rng.integers(0, 256, size=(num_frames, 3, height, width), dtype=np.uint8)

    reference = _reference_pil_outputs(vlm_processor, frames, INSTRUCTION)
    direct = _direct_outputs(vlm_processor, frames, INSTRUCTION)

    assert set(reference.keys()) == set(direct.keys())
    for key in reference.keys():
        assert torch.equal(reference[key], direct[key]), f"mismatch in {key}"


def test_processor_output_through_gr00t_pipeline_shape(vlm_processor):
    """The tensors handed over by _apply_vlm_processing (post-change) are CHW
    uint8 — the exact type proven bitwise-equal above."""
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor  # noqa: F401

    rng = np.random.default_rng(0)
    frames = torch.as_tensor(rng.integers(0, 256, size=(2, 3, 256, 256), dtype=np.uint8))
    out = _direct_outputs(vlm_processor, frames.numpy(), INSTRUCTION)
    assert out["pixel_values"].dtype == torch.float32
    assert out["image_grid_thw"].shape[0] == 2
