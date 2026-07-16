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

"""Bitwise-parity gates for the restructured eval-path image preprocessing.

The inference fast path in ``Gr00tN1d7Processor._get_vlm_inputs`` /
``_eval_transform_frames`` removes the per-frame numpy copies, the
HWC->CHW->HWC torch round trip and the numpy->PIL conversion of the original
code. Every test here asserts *bitwise* equality (``torch.equal``) of the
final model inputs (``pixel_values``, ``image_grid_thw``, ``input_ids``,
``attention_mask``) between the ORIGINAL code path — reimplemented verbatim in
``_reference_vlm_content`` below — and the current production path, using the
real Qwen3VL processor.

Also documents the (non-bitwise) divergence bounds of the opt-in
``use_fast_image_processor`` flag, and guards that the training/augmentation
path still runs through the original ``apply_with_replay`` code.
"""

import json
from pathlib import Path

from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processor_module
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor, build_processor
import numpy as np
from PIL import Image
import pytest
import torch


FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "processor_config"
MODEL_NAME = "nvidia/Cosmos-Reason2-2B"
LANGUAGE = "pick up the red block and place it in the bin"


def _resolve_model_name() -> str:
    """Prefer a locally cached Cosmos-Reason2-2B snapshot over a hub download.

    The test conftest redirects HF cache env vars to a test-shared cache that
    may not contain the VLM, so also look in the user's default HF hub cache.
    Falls back to the repo id (tests skip if that cannot be loaded either).
    """
    import os

    hubs = [
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        str(Path.home() / ".cache" / "huggingface" / "hub"),
    ]
    for hub in hubs:
        if not hub:
            continue
        snapshots = Path(hub) / "models--nvidia--Cosmos-Reason2-2B" / "snapshots"
        if snapshots.is_dir():
            for snap in sorted(snapshots.iterdir()):
                if (snap / "preprocessor_config.json").exists():
                    return str(snap)
    return MODEL_NAME


def _load_processor(**overrides) -> Gr00tN1d7Processor:
    """Real Gr00tN1d7Processor from the fixture config (real VLM processor)."""
    with open(FIXTURE_DIR / "processor_config.json") as f:
        kwargs = json.load(f)["processor_kwargs"]
    with open(FIXTURE_DIR / "statistics.json") as f:
        kwargs["statistics"] = json.load(f)
    kwargs["model_name"] = _resolve_model_name()
    kwargs.update(overrides)
    proc = Gr00tN1d7Processor(**kwargs)
    proc.eval()
    return proc


@pytest.fixture(scope="module")
def real_processor() -> Gr00tN1d7Processor:
    """Fixture-config processor with the REAL Qwen3VL processor (albumentations eval)."""
    try:
        return _load_processor()
    except Exception as exc:  # pragma: no cover - environment without the HF cache
        pytest.skip(f"real {MODEL_NAME} processor unavailable: {exc}")


@pytest.fixture(scope="module")
def real_processor_torchvision() -> Gr00tN1d7Processor:
    """Same, but with the torchvision (non-albumentations) transform flavor."""
    try:
        return _load_processor(
            use_albumentations=False,
            image_target_size=[256, 256],
            image_crop_size=[230, 230],
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"real {MODEL_NAME} processor unavailable: {exc}")


def _make_view_frames(n_cameras: int, hw: tuple[int, int], frames: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    image_keys = [f"cam{i}" for i in range(n_cameras)]
    images = {
        k: rng.integers(0, 256, (frames, hw[0], hw[1], 3), dtype=np.uint8) for k in image_keys
    }
    return image_keys, images


# ---------------------------------------------------------------------------
# Reference implementation: the ORIGINAL (pre-optimization) eval path, copied
# verbatim from _get_vlm_inputs + _apply_vlm_processing before this change.
# ---------------------------------------------------------------------------


def _reference_vlm_content(proc: Gr00tN1d7Processor, image_keys, images, language):
    temporal_stacked_images = {}
    if proc.use_albumentations:
        for view in image_keys:
            transformed_tensors = []
            for img in images[view]:
                img_array = np.array(img)
                augmented = proc.eval_image_transform(image=img_array)
                img_array = augmented["image"]
                if img_array.dtype == np.float32:
                    img_array = (img_array * 255).astype(np.uint8)
                elif img_array.dtype != np.uint8:
                    raise ValueError(f"Unexpected data type: {img_array.dtype}")
                transformed_tensors.append(torch.from_numpy(img_array).permute(2, 0, 1))
            temporal_stacked_images[view] = torch.stack(transformed_tensors)  # (T, C, H, W)
    else:
        for view in image_keys:
            temporal_stacked_images[view] = torch.stack(
                [proc.eval_image_transform(img) for img in images[view]]
            )  # (T, C, H, W)

    stacked_images = (
        torch.stack([temporal_stacked_images[view] for view in image_keys], dim=1)
        .flatten(0, 1)
        .numpy()
    )  # (T*V, C, H, W), "processor expects numpy array"

    # Original _apply_vlm_processing: numpy (C, H, W) -> PIL round trip.
    pil_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in stacked_images]
    conversation = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in pil_images],
                {"type": "text", "text": language},
            ],
        }
    ]
    text = proc.processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    return {"text": text, "images": pil_images}


def _model_inputs(proc: Gr00tN1d7Processor, vlm_content) -> dict:
    """Single full processor call — exactly what the collator does per batch."""
    return dict(
        proc.processor(
            text=[vlm_content["text"]],
            images=vlm_content["images"],
            return_tensors="pt",
            padding=True,
        )
    )


CASES = [
    pytest.param(1, (256, 256), id="1cam-256x256"),
    pytest.param(3, (256, 256), id="3cam-256x256"),
    pytest.param(1, (320, 240), id="1cam-320x240"),
    pytest.param(3, (320, 240), id="3cam-320x240"),
]


class TestEvalPathBitwiseParity:
    """Final model inputs of the fast eval path must equal the original path bitwise."""

    @pytest.mark.parametrize(("n_cameras", "hw"), CASES)
    def test_albumentations_eval_path(self, real_processor, n_cameras, hw):
        proc = real_processor
        image_keys, images = _make_view_frames(n_cameras, hw)

        ref = _model_inputs(proc, _reference_vlm_content(proc, image_keys, images, LANGUAGE))
        new_content = proc._get_vlm_inputs(
            image_keys=image_keys,
            images=images,
            masks=None,
            image_transform=proc.eval_image_transform,
            language=LANGUAGE,
        )["vlm_content"]
        # The fast path must actually be active: raw frames, not PIL.
        assert all(isinstance(img, np.ndarray) for img in new_content["images"])
        new = _model_inputs(proc, new_content)

        assert set(ref) == set(new)
        for key in ("pixel_values", "image_grid_thw", "input_ids", "attention_mask"):
            assert key in ref, key
            assert ref[key].dtype == new[key].dtype, key
            assert torch.equal(ref[key], new[key]), f"bitwise mismatch for {key!r}"

    @pytest.mark.parametrize(("n_cameras", "hw"), CASES)
    def test_torchvision_eval_path(self, real_processor_torchvision, n_cameras, hw):
        proc = real_processor_torchvision
        image_keys, images = _make_view_frames(n_cameras, hw)
        # torchvision path consumes CHW tensors via ToImage(); feed HWC uint8 frames.
        ref = _model_inputs(proc, _reference_vlm_content(proc, image_keys, images, LANGUAGE))
        new_content = proc._get_vlm_inputs(
            image_keys=image_keys,
            images=images,
            masks=None,
            image_transform=proc.eval_image_transform,
            language=LANGUAGE,
        )["vlm_content"]
        assert all(isinstance(img, torch.Tensor) for img in new_content["images"])
        new = _model_inputs(proc, new_content)
        for key in ("pixel_values", "image_grid_thw", "input_ids", "attention_mask"):
            assert torch.equal(ref[key], new[key]), f"bitwise mismatch for {key!r}"

    def test_mismatched_view_shapes_raise(self, real_processor):
        proc = real_processor
        rng = np.random.default_rng(0)
        images = {
            "cam0": rng.integers(0, 256, (2, 256, 256, 3), dtype=np.uint8),
            "cam1": rng.integers(0, 256, (2, 240, 320, 3), dtype=np.uint8),
        }
        with pytest.raises(ValueError, match="letter_box_transform"):
            proc._get_vlm_inputs(
                image_keys=["cam0", "cam1"],
                images=images,
                masks=None,
                image_transform=proc.eval_image_transform,
                language=LANGUAGE,
            )


class TestHFProcessorInputRepresentationParity:
    """The gate that let the fast path drop the PIL round trip: numpy HWC and
    torch CHW inputs to the real Qwen3VL processor must produce bitwise-identical
    outputs to PIL input. If this ever fails on a new transformers version, the
    eval fast path must go back to PIL frames."""

    @pytest.mark.parametrize("hw", [(256, 256), (320, 240), (243, 243)])
    def test_numpy_and_torch_match_pil_bitwise(self, real_processor, hw):
        vlm = real_processor.processor
        rng = np.random.default_rng(0)
        arrays = [rng.integers(0, 256, (hw[0], hw[1], 3), dtype=np.uint8) for _ in range(2)]
        pils = [Image.fromarray(a) for a in arrays]
        tensors = [torch.from_numpy(a).permute(2, 0, 1) for a in arrays]
        text = (
            "<|vision_start|><|image_pad|><|vision_end|>"
            "<|vision_start|><|image_pad|><|vision_end|>" + LANGUAGE
        )
        out_pil = vlm(text=[text], images=pils, return_tensors="pt", padding=True)
        out_np = vlm(text=[text], images=arrays, return_tensors="pt", padding=True)
        out_pt = vlm(text=[text], images=tensors, return_tensors="pt", padding=True)
        for key in out_pil:
            assert torch.equal(out_pil[key], out_np[key]), f"numpy != PIL for {key!r}"
            assert torch.equal(out_pil[key], out_pt[key]), f"torch != PIL for {key!r}"


class TestTrainingPathUntouched:
    """Training/augmentation must keep the exact original code path."""

    def test_training_uses_apply_with_replay_and_pil(self, real_processor, monkeypatch):
        proc = real_processor
        calls = {"n": 0}
        original = processor_module.apply_with_replay

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(processor_module, "apply_with_replay", counting)
        image_keys, images = _make_view_frames(2, (256, 256))
        try:
            proc.train()
            content = proc._get_vlm_inputs(
                image_keys=image_keys,
                images=images,
                masks=None,
                image_transform=proc.train_image_transform,
                language=LANGUAGE,
            )["vlm_content"]
        finally:
            proc.eval()
        assert calls["n"] == len(image_keys), "training must run through apply_with_replay"
        assert all(isinstance(img, Image.Image) for img in content["images"]), (
            "training path must keep the original PIL conversion"
        )

    def test_eval_does_not_use_apply_with_replay(self, real_processor, monkeypatch):
        proc = real_processor

        def boom(*args, **kwargs):  # pragma: no cover - assertion helper
            raise AssertionError("eval fast path must not call apply_with_replay")

        monkeypatch.setattr(processor_module, "apply_with_replay", boom)
        image_keys, images = _make_view_frames(1, (256, 256))
        proc._get_vlm_inputs(
            image_keys=image_keys,
            images=images,
            masks=None,
            image_transform=proc.eval_image_transform,
            language=LANGUAGE,
        )

    def test_process_observation_keeps_pil_path(self, real_processor):
        """process_observation still routes through the original _apply_vlm_processing."""
        proc = real_processor
        images = np.random.default_rng(0).integers(0, 256, (2, 3, 256, 256), dtype=np.uint8)
        content = proc._apply_vlm_processing(images, LANGUAGE)["vlm_content"]
        assert all(isinstance(img, Image.Image) for img in content["images"])


class TestUseFastImageProcessorFlag:
    """Opt-in flag semantics and documented divergence bounds (NOT a bitwise path).

    ``use_fast_image_processor=None`` (default) keeps the checkpoint's own
    image-processor class — bitwise-identical to the historical behavior.
    Forcing fast-vs-slow changes resize numerics whenever a real resize
    happens; this test DOCUMENTS the measured tolerance rather than asserting
    equality. Closed-loop validation is required before flipping the flag in
    production.
    """

    @pytest.fixture(scope="class")
    def processors(self):
        model_name = _resolve_model_name()
        try:
            default = build_processor(model_name, {"trust_remote_code": True})
            fast = build_processor(
                model_name, {"trust_remote_code": True}, use_fast_image_processor=True
            )
            slow = build_processor(
                model_name, {"trust_remote_code": True}, use_fast_image_processor=False
            )
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"real {MODEL_NAME} processor unavailable: {exc}")
        return default, fast, slow

    def test_flag_selects_image_processor_class_only(self, processors):
        default, fast, slow = processors
        assert type(fast.image_processor).__name__.endswith("Fast")
        assert not type(slow.image_processor).__name__.endswith("Fast")
        # Default (None): the Cosmos-Reason2-2B checkpoint config declares the
        # fast class, so the historical default already resolves to it.
        assert type(default.image_processor).__name__ == "Qwen2VLImageProcessorFast"
        # Forcing the slow IMAGE processor must not silently slow the tokenizer.
        assert type(slow.tokenizer).__name__ == type(default.tokenizer).__name__

    @pytest.mark.parametrize("hw", [(256, 256), (320, 240)])
    def test_fast_slow_divergence_bounds(self, processors, hw):
        _, fast, slow = processors
        rng = np.random.default_rng(0)
        pils = [
            Image.fromarray(rng.integers(0, 256, (hw[0], hw[1], 3), dtype=np.uint8))
            for _ in range(2)
        ]
        out_fast = fast.image_processor(images=pils, return_tensors="pt")
        out_slow = slow.image_processor(images=pils, return_tensors="pt")
        # The token grid (and thus input_ids) never diverges.
        assert torch.equal(out_fast["image_grid_thw"], out_slow["image_grid_thw"])
        diff = (out_fast["pixel_values"] - out_slow["pixel_values"]).abs()
        if hw == (256, 256):
            # 256x256 is already patch-aligned: smart-resize is a no-op, so
            # only rescale/normalize arithmetic order differs -> float-rounding
            # level divergence (measured max-abs ~5.9e-8).
            assert diff.max().item() <= 1e-6
        else:
            # A real resize happens (bicubic PIL vs antialiased torchvision):
            # measured max-abs 2/255 (~0.0078), mean-abs ~1.5e-5 on 320x240.
            assert diff.max().item() <= 4.0 / 255.0
            assert diff.mean().item() <= 1e-3
