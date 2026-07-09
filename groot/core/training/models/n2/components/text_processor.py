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
import ast
import html
import re

import ftfy
import torch
from transformers import AutoTokenizer


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class HuggingfaceTokenizer:
    def __init__(self, path: str, seq_len: int, **transformers_loading_kwargs):
        self.path = path
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(path, **transformers_loading_kwargs)
        self.system_prompt = (
            "You are a helpful assistant who will provide prompts to an image generator."
        )

    def __call__(self, sequence: list[str], **kwargs):
        # arguments
        _kwargs: dict[str, str | int | bool] = {
            "return_tensors": "pt",
            "padding": "max_length",
            "truncation": True,
            "max_length": self.seq_len,
            "add_vision_id": False,
            "tokenize": True,
            "add_generation_prompt": False,
        }
        _kwargs.update(**kwargs)

        # tokenization
        sequence = [whitespace_clean(basic_clean(u)) for u in sequence]
        out_ids = []
        for seq in sequence:
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": seq,
                        }
                    ],
                },
            ]
            ids = self.tokenizer.apply_chat_template(conversations, **_kwargs)
            if hasattr(ids, "input_ids"):
                ids = ids.input_ids
            elif isinstance(ids, dict) and "input_ids" in ids:
                ids = ids["input_ids"]
            ids = torch.as_tensor(ids, dtype=torch.long)
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            out_ids.append(ids[0])
        out_ids = torch.stack(out_ids, dim=0)
        return out_ids

    def save_pretrained(self, *args, **kwargs):
        return self.tokenizer.save_pretrained(*args, **kwargs)


def process_text(
    text_features: list[str], tokenizer: HuggingfaceTokenizer
) -> tuple[torch.Tensor, torch.Tensor]:
    processed_text_features = []
    for text in text_features:
        try:
            parsed_item = ast.literal_eval(text)
            # Handle different return types from ast.literal_eval
            if isinstance(parsed_item, list | tuple):
                processed_text = str(parsed_item[0])
            else:
                processed_text = str(parsed_item)
        except (ValueError, SyntaxError, TypeError):
            processed_text = str(text)

        processed_text_features.append(processed_text)

    return tokenizer(processed_text_features, add_special_tokens=True)


def process_text_negative(
    text_features: list[str], tokenizer: HuggingfaceTokenizer
) -> tuple[torch.Tensor, torch.Tensor]:
    return tokenizer(text_features, add_special_tokens=True)
