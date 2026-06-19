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

"""Test N1.7 image transform sizing behavior across torchvision and albumentations paths."""

from pathlib import Path

from gr00t.model.gr00t_n1d7.image_augmentations import (
    apply_with_replay,
    build_image_transformations,
    build_image_transformations_albumentations,
)
import numpy as np
from PIL import Image
import pytest
import torch


FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "processor_config"


# ---- Transform-level tests ----


class TestTorchvisionTransforms:
    """Test that torchvision eval transform produces consistent sizes."""

    def setup_method(self):
        self.image_target_size = [256, 256]
        self.image_crop_size = [224, 224]
        self.train_transform, self.eval_transform = build_image_transformations(
            image_target_size=self.image_target_size,
            image_crop_size=self.image_crop_size,
            random_rotation_angle=None,
            color_jitter_params=None,
        )

    def test_letterbox_transform_is_disabled_by_default(self):
        transform_names = [
            type(transform).__name__
            for transform in [*self.train_transform.transforms, *self.eval_transform.transforms]
        ]
        assert "LetterBoxTransform" not in transform_names

    def test_letterbox_transform_can_be_enabled(self):
        train_transform, eval_transform = build_image_transformations(
            image_target_size=self.image_target_size,
            image_crop_size=self.image_crop_size,
            random_rotation_angle=None,
            color_jitter_params=None,
            letter_box_transform=True,
        )
        transform_names = [
            type(transform).__name__
            for transform in [*train_transform.transforms, *eval_transform.transforms]
        ]
        assert "LetterBoxTransform" in transform_names

    def test_same_size_images(self):
        img1 = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        out1 = self.eval_transform(img1)
        out2 = self.eval_transform(img2)
        assert out1.shape == out2.shape, f"Shape mismatch: {out1.shape} vs {out2.shape}"
        torch.stack([out1, out2])  # should not raise

    def test_variable_size_images(self):
        img_4_3 = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        img_16_9 = Image.fromarray(np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8))
        out_4_3 = self.eval_transform(img_4_3)
        out_16_9 = self.eval_transform(img_16_9)
        assert out_4_3.shape == out_16_9.shape, (
            f"Shape mismatch for different aspect ratios: {out_4_3.shape} vs {out_16_9.shape}"
        )
        torch.stack([out_4_3, out_16_9])  # should not raise

    def test_square_and_wide_images(self):
        img_square = Image.fromarray(np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8))
        img_wide = Image.fromarray(np.random.randint(0, 255, (240, 640, 3), dtype=np.uint8))
        out_sq = self.eval_transform(img_square)
        out_wide = self.eval_transform(img_wide)
        assert out_sq.shape == out_wide.shape, f"Shape mismatch: {out_sq.shape} vs {out_wide.shape}"
        torch.stack([out_sq, out_wide])  # should not raise


class TestAlbumentationsTransforms:
    """Test that albumentations preserves aspect ratio without letterboxing."""

    def setup_method(self):
        self.train_transform, self.eval_transform = build_image_transformations_albumentations(
            image_target_size=None,
            image_crop_size=None,
            random_rotation_angle=None,
            color_jitter_params=None,
            shortest_image_edge=256,
            crop_fraction=0.95,
        )

    def _apply(self, pil_img):
        result = self.eval_transform(image=np.array(pil_img))
        return torch.from_numpy(result["image"]).permute(2, 0, 1)

    def test_letterbox_pad_is_not_in_transform_pipeline(self):
        transform_names = [
            type(transform).__name__
            for transform in [*self.train_transform.transforms, *self.eval_transform.transforms]
        ]
        assert "LetterBoxPad" not in transform_names

    def test_uses_gear_groot_aspect_preserving_pipeline(self):
        train_names = [type(transform).__name__ for transform in self.train_transform.transforms]
        eval_names = [type(transform).__name__ for transform in self.eval_transform.transforms]
        assert train_names[:3] == [
            "SmallestMaxSize",
            "FractionalRandomCrop",
            "SmallestMaxSize",
        ]
        assert eval_names == [
            "SmallestMaxSize",
            "FractionalCenterCrop",
            "SmallestMaxSize",
        ]

    def test_train_replay_with_same_aspect_variable_size_images(self):
        img_4_3 = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        img_4_3_small = Image.fromarray(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
        transformed, _ = apply_with_replay(self.train_transform, [img_4_3, img_4_3_small])
        torch.stack(transformed)

    def test_same_size_images(self):
        img1 = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        out1 = self._apply(img1)
        out2 = self._apply(img2)
        assert out1.shape == out2.shape, f"Shape mismatch: {out1.shape} vs {out2.shape}"
        torch.stack([out1, out2])

    def test_same_aspect_variable_size_images(self):
        img_4_3 = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        img_4_3_small = Image.fromarray(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
        out_4_3 = self._apply(img_4_3)
        out_4_3_small = self._apply(img_4_3_small)
        assert out_4_3.shape == out_4_3_small.shape, (
            f"Shape mismatch for same aspect ratios: {out_4_3.shape} vs {out_4_3_small.shape}"
        )
        torch.stack([out_4_3, out_4_3_small])

    def test_mixed_aspect_images_preserve_different_shapes(self):
        img_square = Image.fromarray(np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8))
        img_wide = Image.fromarray(np.random.randint(0, 255, (240, 640, 3), dtype=np.uint8))
        out_sq = self._apply(img_square)
        out_wide = self._apply(img_wide)
        assert out_sq.shape != out_wide.shape


# ---- Processor-level tests (using fixture config, no checkpoint needed) ----


@pytest.fixture
def processor():
    from unittest.mock import MagicMock, patch

    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    mock_vlm = MagicMock()
    mock_vlm.apply_chat_template.return_value = "mock text"
    mock_vlm.tokenizer.padding_side = "left"

    with patch(
        "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor",
        return_value=mock_vlm,
    ):
        proc = Gr00tN1d7Processor.from_pretrained(FIXTURE_DIR)
    proc.eval()
    return proc


class TestProcessorVariableImageSize:
    """Test full _get_vlm_inputs path with variable image sizes."""

    def test_variable_size_vlm_inputs(self, processor):
        """Test _get_vlm_inputs with same-aspect variable-size images across views."""
        embodiment_tag = "libero_sim"
        image_keys = processor.modality_configs[embodiment_tag]["video"].modality_keys

        # Albumentations preserves aspect ratio, so views in one sample must share aspect ratio
        # before the processor stacks them.
        mock_images = {}
        for i, key in enumerate(image_keys):
            if i % 2 == 0:
                mock_images[key] = [
                    Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
                ]
            else:
                mock_images[key] = [
                    Image.fromarray(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
                ]

        vlm_inputs = processor._get_vlm_inputs(
            image_keys=image_keys,
            images=mock_images,
            image_transform=processor.eval_image_transform,
            language="pick up the object",
            masks=None,
        )

        assert "vlm_content" in vlm_inputs
        assert len(vlm_inputs["vlm_content"]["images"]) == len(image_keys)

    def test_mixed_aspect_vlm_inputs_raise(self, processor):
        """Albumentations preserves aspect ratio, so mixed-aspect views are not stackable."""
        embodiment_tag = "libero_sim"
        image_keys = processor.modality_configs[embodiment_tag]["video"].modality_keys

        mock_images = {}
        for i, key in enumerate(image_keys):
            if i % 2 == 0:
                mock_images[key] = [
                    Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
                ]
            else:
                mock_images[key] = [
                    Image.fromarray(np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8))
                ]

        with pytest.raises(RuntimeError, match="stack expects each tensor to be equal size"):
            processor._get_vlm_inputs(
                image_keys=image_keys,
                images=mock_images,
                image_transform=processor.eval_image_transform,
                language="pick up the object",
                masks=None,
            )

    def test_same_size_vlm_inputs(self, processor):
        """Test _get_vlm_inputs with same size images (regression test)."""
        embodiment_tag = "libero_sim"
        image_keys = processor.modality_configs[embodiment_tag]["video"].modality_keys

        mock_images = {}
        for key in image_keys:
            mock_images[key] = [
                Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            ]

        vlm_inputs = processor._get_vlm_inputs(
            image_keys=image_keys,
            images=mock_images,
            image_transform=processor.eval_image_transform,
            language="pick up the object",
            masks=None,
        )

        assert "vlm_content" in vlm_inputs
        assert len(vlm_inputs["vlm_content"]["images"]) == len(image_keys)
