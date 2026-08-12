# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Cosmos Predict 2.5 text encoder.

- Cosmos Reason 1.1 7B model
- use transformers library to load model and encode prompt
"""

import torch
from torch import nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from groot.core.utils.storage.msc import resolve_model_path


class CosmosTextEncoder(nn.Module):
    def __init__(self, text_encoder_pretrained_path: str):
        super().__init__()
        runtime_path = resolve_model_path(
            text_encoder_pretrained_path,
            "model.text_encoder_cfg.text_encoder_pretrained_path",
        )
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            runtime_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.tokenizer = AutoProcessor.from_pretrained(runtime_path)
        self.max_seq_len = 512
        self.system_prompt = (
            "You are a helpful assistant who will provide prompts to an image generator."
        )

    def prepare_input_ids(self, prompts: list[str]):
        input_ids_batch = []
        for prompt in prompts:
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
                            "text": prompt,
                        }
                    ],
                },
            ]
            input_ids = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                max_length=self.max_seq_len,
                truncation=True,
                padding="max_length",
            )
            input_ids = torch.LongTensor(input_ids)
            input_ids_batch.append(input_ids[0])
        input_ids_batch = torch.stack(input_ids_batch, dim=0)
        return input_ids_batch

    @torch.compiler.disable()
    def forward(self, input_ids):
        with torch.no_grad():
            outputs = self.text_encoder(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        normalized_hidden_states = []
        for layer_idx in range(1, len(hidden_states)):
            normalized_state = (
                hidden_states[layer_idx] - hidden_states[layer_idx].mean(dim=-1, keepdim=True)
            ) / (hidden_states[layer_idx].std(dim=-1, keepdim=True) + 1e-8)
            normalized_hidden_states.append(normalized_state)
        prompt_embeds = torch.cat(normalized_hidden_states, dim=-1)
        return prompt_embeds
