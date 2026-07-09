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
"""Pass-through processor + collator for :class:`IdentityModel`.

No tokenizer, no transforms — useful for isolating data-loading cost from
model compute in the GPU dry-run.
"""
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

from groot.core.data import types
from groot.core.data.utils import parse_modality_configs


class IdentityCollator:
    """Companion collator for :class:`IdentityModel`."""

    def __call__(self, features: list[dict[str, Any]]) -> BatchFeature:
        return BatchFeature({
            "__identity__": torch.zeros(len(features)),
            "step_data_id": [f.get("step_data_id") for f in features],
        })

    def get_datapoint_ids(self, batch_samples: list) -> list[types.VLAStepDataId]:
        ids = []
        for batch in batch_samples:
            ids.extend(v for v in batch.get("step_data_id", []) if isinstance(v, types.VLAStepDataId))
        return ids


class IdentityProcessor(types.BaseProcessor):
    """Companion processor for :class:`IdentityModel` — pass-through, no transforms."""

    attributes: list[str] = []
    # ProcessorMixin.save_pretrained reads self.chat_template; we never set
    # one (no tokenizer), and __init__ skips super().__init__() to avoid
    # ProcessorMixin's tokenizer-required init path.
    chat_template = None
    # transformers >= 4.57 unconditionally probes processor.audio_tokenizer
    # during save_pretrained; we don't have one.
    audio_tokenizer = None
    # Same generation, for safety against future ProcessorMixin probes.
    tokenizer = None
    image_processor = None
    feature_extractor = None

    def __init__(self, modality_configs: dict):
        self.modality_configs = parse_modality_configs(modality_configs)
        self.training = True
        self._collator = IdentityCollator()

    def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        content = messages[0]["content"] if messages else None
        step_data_id = (
            getattr(content, "metadata", {}).get(types.VLAStepDataId.STEP_DATA_ID_KEY)
            if content is not None
            else None
        )
        return {"__identity__": 0.0, "step_data_id": step_data_id}

    @property
    def collator(self) -> IdentityCollator:
        return self._collator

    def set_statistics(self, statistics: dict[str, Any], override: bool = False) -> None:
        pass

    def decode_action(self, action, embodiment_tag, state=None):
        raise NotImplementedError

    def to_dict(self, **_kwargs) -> dict:
        return {}

    def to_json_string(self, **_kwargs) -> str:
        # ProcessorMixin.save_pretrained on transformers >= 4.57 passes
        # ``legacy_serialization=…`` here. Swallow any forward-compat kwargs.
        return "{}"
