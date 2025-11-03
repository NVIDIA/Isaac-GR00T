# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from functools import partial
from typing import Dict, Any
import torch
from deployment_scripts import trt_torch as trt
from transformers.feature_extraction_utils import BatchFeature
import typing as Tuple
import tree


def eagle_tensorrt_forward(self, vl_input):
    eagle_prefix = "eagle_"
    eagle_input = {
        k.removeprefix(eagle_prefix): v for k, v in vl_input.items() if k.startswith(eagle_prefix)
    }
    del eagle_input["image_sizes"]
    vl_input = eagle_input

    self.set_frozen_modules_to_eval_mode()
    batch_size = vl_input["pixel_values"].shape[0]
    position_ids = torch.arange(self.num_patches, device="cuda").expand((batch_size, -1))
    if vl_input["pixel_values"].dtype != torch.float16:
        vl_input["pixel_values"] = vl_input["pixel_values"].to(torch.float16)

    assert (
        vl_input["pixel_values"].shape[0] <= 8
    ), "Batch size must be <= 8 because TensorRT engine was built with max_batch_size=8, you can try to adjust the max_batch_size in the build_engine.sh script and rebuild the engine."

    self.vit_engine.set_runtime_tensor_shape("pixel_values", vl_input["pixel_values"].shape)
    self.vit_engine.set_runtime_tensor_shape("position_ids", position_ids.shape)
    vit_embeds = self.vit_engine(vl_input["pixel_values"], position_ids)["vit_embeds"]

    self.llm_engine.set_runtime_tensor_shape("input_ids", vl_input["input_ids"].shape)
    self.llm_engine.set_runtime_tensor_shape("vit_embeds", vit_embeds.shape)
    self.llm_engine.set_runtime_tensor_shape("attention_mask", vl_input["attention_mask"].shape)
    embeddings = self.llm_engine(vl_input["input_ids"], vit_embeds, vl_input["attention_mask"])[
        "embeddings"
    ]

    return BatchFeature(
        data={
            "backbone_features": embeddings,
            "backbone_attention_mask": vl_input["attention_mask"],
        }
    )


def action_head_tensorrt_forward(self, backbone_output, action_input):
    # backbone_output = self.process_backbone_output(backbone_output)
    if backbone_output.backbone_features.dtype != torch.float16:
        backbone_output.backbone_features = backbone_output.backbone_features.to(torch.float16)
    self.vlln_vl_self_attention_engine.set_runtime_tensor_shape(
        "backbone_features", backbone_output.backbone_features.shape
    )
    backbone_output.backbone_features = self.vlln_vl_self_attention_engine(
        backbone_output.backbone_features
    )["output"]
    vl_embs = backbone_output.backbone_features
    embodiment_id = action_input.embodiment_id
    batch_size = vl_embs.shape[0]

    if action_input.state.dtype != torch.float16:
        action_input.state = action_input.state.to(torch.float16)

    if embodiment_id.dtype != torch.int64:
        embodiment_id = embodiment_id.to(torch.int64)

    if vl_embs.dtype != torch.float16:
        vl_embs = vl_embs.to(torch.float16)

    # Embed state with batch processing

    self.state_encoder_engine.set_runtime_tensor_shape("state", action_input.state.shape)
    self.state_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
    state_features = self.state_encoder_engine(action_input.state, embodiment_id)["output"]

    # Set initial actions as the sampled noise.
    device = vl_embs.device
    actions = torch.randn(
        size=(batch_size, self.config.action_horizon, self.config.action_dim),
        dtype=vl_embs.dtype,
        device=device,
    )

    # This attribute is used to ensure the same actions is used for both PyTorch and TensorRT inference
    if hasattr(self, "init_actions"):
        actions = self.init_actions.expand((batch_size, -1, -1))

    num_steps = self.num_inference_timesteps
    dt = 1.0 / num_steps

    for t in range(num_steps):
        t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
        t_discretized = int(t_cont * self.num_timestep_buckets)

        # Embed noised action trajectory with batch processing
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)

        self.action_encoder_engine.set_runtime_tensor_shape("actions", actions.shape)
        self.action_encoder_engine.set_runtime_tensor_shape(
            "timesteps_tensor", timesteps_tensor.shape
        )
        self.action_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
        action_features = self.action_encoder_engine(actions, timesteps_tensor, embodiment_id)[
            "output"
        ]

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0).to(torch.float16)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1).to(
            torch.float16
        )
        # Run model forward with batch processing
        if vl_embs.dtype != torch.float16:
            vl_embs = vl_embs.to(torch.float16)

        self.DiT_engine.set_runtime_tensor_shape("vl_embs", vl_embs.shape)
        self.DiT_engine.set_runtime_tensor_shape("sa_embs", sa_embs.shape)
        self.DiT_engine.set_runtime_tensor_shape("timesteps_tensor", timesteps_tensor.shape)
        model_output = self.DiT_engine(sa_embs, vl_embs, timesteps_tensor)["output"]

        self.action_decoder_engine.set_runtime_tensor_shape("model_output", model_output.shape)
        self.action_decoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
        pred = self.action_decoder_engine(model_output, embodiment_id)["output"]
        pred_velocity = pred[:, -self.action_horizon :]

        # Update actions using euler integration.
        actions = actions + dt * pred_velocity
    return BatchFeature(data={"action_pred": actions})



def full_model_tensorrt_forward(self, inputs):
    # get the device and dtype from trt engine
    device = torch.device("cuda")
    dtype = self.model_engine.in_meta[0][2]

    init_actions = torch.randn(
        size=(inputs["eagle_pixel_values"].shape[0], self.action_horizon, self.action_dim),
        dtype=dtype,
        device=device,
    )
    inputs["init_actions"] = init_actions

    def prepare_input(inputs, device, dtype):
        inputs = BatchFeature(data=inputs)

        def to_device_with_maybe_dtype(x):
            if torch.is_floating_point(x):
                return x.to(device, dtype=dtype)
            else:
                return x.to(device)

        inputs = tree.map_structure(to_device_with_maybe_dtype, inputs)
        return inputs

    inputs = prepare_input(inputs, device, dtype)

    self.model_engine.set_runtime_tensor_shape("pixel_values", inputs["eagle_pixel_values"].shape)
    self.model_engine.set_runtime_tensor_shape("input_ids", inputs["eagle_input_ids"].shape)
    self.model_engine.set_runtime_tensor_shape("attention_mask", inputs["eagle_attention_mask"].shape)
    self.model_engine.set_runtime_tensor_shape("state", inputs["state"].shape)
    self.model_engine.set_runtime_tensor_shape("embodiment_id", inputs["embodiment_id"].shape)
    self.model_engine.set_runtime_tensor_shape("init_actions", inputs["init_actions"].shape)

    engine_output = self.model_engine(inputs["eagle_pixel_values"], inputs["eagle_input_ids"], inputs["eagle_attention_mask"], inputs["state"], inputs["embodiment_id"], inputs["init_actions"])
    actions_trt = engine_output["output"]
    return actions_trt




def setup_tensorrt_engines(policy, trt_engine_path, full_model=False):
    """
    Setup TensorRT engines for GR00T model inference.

    Args:
        policy: GR00T policy model instance
    """

    if full_model:
        print("Setting up TensorRT full model engine")
        policy.action_horizon = policy.model.action_head.action_horizon
        policy.action_dim = policy.model.action_head.action_dim
        if hasattr(policy , "model"):
           del policy.model
        torch.cuda.empty_cache()
        policy.model_engine = trt.Engine(os.path.join(trt_engine_path, "full_model.engine"))
        policy._get_action_from_normalized_input = partial(full_model_tensorrt_forward, policy)
        return


    policy.model.backbone.num_patches = (
        policy.model.backbone.eagle_model.vision_model.vision_model.embeddings.num_patches
    )
    if hasattr(policy.model.backbone.eagle_model, "vision_model"):
        del policy.model.backbone.eagle_model.vision_model
    if hasattr(policy.model.backbone.eagle_model, "language_model"):
        del policy.model.backbone.eagle_model.language_model
    if hasattr(policy.model.action_head, "vlln"):
        del policy.model.action_head.vlln
    if hasattr(policy.model.action_head, "vl_self_attention"):
        del policy.model.action_head.vl_self_attention
    if hasattr(policy.model.action_head, "model"):
        del policy.model.action_head.model
    if hasattr(policy.model.action_head, "state_encoder"):
        del policy.model.action_head.state_encoder
    if hasattr(policy.model.action_head, "action_encoder"):
        del policy.model.action_head.action_encoder
    if hasattr(policy.model.action_head, "action_decoder"):
        del policy.model.action_head.action_decoder
    torch.cuda.empty_cache()

    # Setup backbone engines
    policy.model.backbone.vit_engine = trt.Engine(os.path.join(trt_engine_path, "vit.engine"))
    policy.model.backbone.llm_engine = trt.Engine(os.path.join(trt_engine_path, "llm.engine"))

    # Setup action head engines
    policy.model.action_head.vlln_vl_self_attention_engine = trt.Engine(
        os.path.join(trt_engine_path, "vlln_vl_self_attention.engine")
    )
    policy.model.action_head.action_encoder_engine = trt.Engine(
        os.path.join(trt_engine_path, "action_encoder.engine")
    )
    policy.model.action_head.action_decoder_engine = trt.Engine(
        os.path.join(trt_engine_path, "action_decoder.engine")
    )
    policy.model.action_head.DiT_engine = trt.Engine(os.path.join(trt_engine_path, "DiT.engine"))
    policy.model.action_head.state_encoder_engine = trt.Engine(
        os.path.join(trt_engine_path, "state_encoder.engine")
    )

    # Set TensorRT forward functions
    policy.model.backbone.forward = partial(eagle_tensorrt_forward, policy.model.backbone)
    policy.model.action_head.get_action = partial(
        action_head_tensorrt_forward, policy.model.action_head
    )
