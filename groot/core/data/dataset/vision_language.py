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

import json

from PIL import Image
import torch

from groot.core.data.types import MessageType
from groot.core.utils.io.read_lmdb import LMDBImageReader


class VisionLanguageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: dict[str, str],
        image_max_size=224,
        **kwargs,
    ):

        self.dataset_path = dataset_path
        self.image_max_size = image_max_size

        # Initialize LMDB reader with data_root for resolving paths
        self.lmdb_reader = LMDBImageReader(data_root=dataset_path["data_root"])

        self.data = []
        with open(dataset_path["jsonl_path"]) as f:
            for row in f:
                data = json.loads(row)
                self.data += [data]

        super().__init__()

    def __len__(self):
        return len(self.data)

    def maybe_resize_image(self, image):
        # Check aspect ratio and crop to square if it's worse than 4:1
        width, height = image.size
        aspect_ratio = max(width, height) / min(width, height)

        if aspect_ratio > 4:
            # Crop to square by taking the center portion
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            image = image.crop((left, top, right, bottom))
            width, height = image.size  # Update dimensions after cropping

        if self.image_max_size is not None:
            # Resize only if the shortest side is longer than image_max_size
            min_dim = min(width, height)

            # Calculate new dimensions maintaining aspect ratio
            if width < height:
                new_width = self.image_max_size
                new_height = int(height * (self.image_max_size / width))
            else:
                new_height = self.image_max_size
                new_width = int(width * (self.image_max_size / height))

            image = image.resize((new_width, new_height))
        return image

    def __getitem__(self, idx):

        messages = []

        value = self.data[idx]
        image_data = value.get("image", None)

        # Use LMDB reader to parse image data (supports both file paths and LMDB metadata)
        if image_data is not None:

            # TODO: support > 1 images
            if isinstance(image_data, list):
                image_data = image_data[0]

            try:
                image = self.lmdb_reader.parse_image_data(image_data)
                image = self.maybe_resize_image(image)
            except Exception as e:
                print(f"Error parsing image data: {e}")
                image = Image.new("RGB", (224, 224), color=(128, 128, 128))
            messages += [{"type": MessageType.IMAGE.value, "content": image, "role": "user"}]

        conv = value["conversations"]

        num_images = 0
        for message in conv:
            role = "user" if message["from"] == "human" else "assistant"
            text = message["value"]

            if not isinstance(text, str):
                assert isinstance(text, list)
                assert all(isinstance(t, str) for t in text)
                text = "\n".join(text)

            if text.startswith("<image>\n"):
                text = text[8:]
                num_images += 1

            if text.endswith("\n<image>"):
                text = text[:-8]
                num_images += 1

            # keep just one image for now, TODO: support multiple images
            if num_images > 1:
                break

            messages.append({"type": MessageType.TEXT.value, "content": text, "role": role})

        return messages

    def __del__(self):
        """Clean up LMDB resources when dataset is destroyed."""
        if hasattr(self, "lmdb_reader"):
            self.lmdb_reader.close()
