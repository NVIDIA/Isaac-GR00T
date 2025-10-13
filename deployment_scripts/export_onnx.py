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

import argparse
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.utils.checkpoint as cp
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import (
    SiglipVisionEmbeddings,
    SiglipVisionTransformer,
)

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH, EagleBackbone
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values
from modelopt.torch._deploy.utils import get_onnx_bytes
import modelopt.torch.quantization as mtq


ONNX_MODEL_DTYPE = torch.float16


def get_input_info(policy, observations):
    is_batch = policy._check_state_is_batched(observations)
    if not is_batch:
        observations = unsqueeze_dict_values(observations)

    normalized_input = unsqueeze_dict_values
    # Apply transforms
    normalized_input = policy.apply_transforms(observations)

    return normalized_input["eagle_attention_mask"], normalized_input["state"]


def export_eagle2_vit(vision_model, output_dir):

    class SiglipVisionEmbeddingsOpt(SiglipVisionEmbeddings):
        def __init__(self, config):
            super().__init__(config)

        def forward(
            self,
            pixel_values: torch.FloatTensor,
            position_ids: torch.LongTensor,  # position_ids is now an input
            interpolate_pos_encoding=False,
        ) -> torch.Tensor:
            _, _, height, width = pixel_values.shape
            target_dtype = self.patch_embedding.weight.dtype
            patch_embeds = self.patch_embedding(
                pixel_values.to(dtype=target_dtype)
            )  # shape = [*, width, grid, grid]
            embeddings = patch_embeds.flatten(2).transpose(1, 2)

            if interpolate_pos_encoding:
                embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
            else:
                embeddings = embeddings + self.position_embedding(position_ids)
            return embeddings

    class SiglipVisionTransformerOpt(SiglipVisionTransformer):
        def __init__(self, config: SiglipVisionConfig):
            config._attn_implementation = "eager"
            super().__init__(config)
            self.embeddings = SiglipVisionEmbeddingsOpt(config)

        def forward(
            self,
            pixel_values,
            position_ids,  # Pass position_ids as input
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            interpolate_pos_encoding: Optional[bool] = False,
        ):
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )

            hidden_states = self.embeddings(
                pixel_values,
                position_ids=position_ids,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )

            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            last_hidden_state = encoder_outputs.last_hidden_state
            last_hidden_state = self.post_layernorm(last_hidden_state)

            return last_hidden_state

    model = SiglipVisionTransformerOpt(vision_model.config).to(torch.float16)
    model.load_state_dict(vision_model.state_dict())
    model.eval().cuda()

    pixel_values = torch.randn(
        (1, model.config.num_channels, model.config.image_size, model.config.image_size),
        dtype=torch.float16,
        device="cuda",
    )
    position_ids = torch.arange(model.embeddings.num_patches, device="cuda").expand((1, -1))

    os.makedirs(output_dir, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (pixel_values, position_ids),  # Include position_ids in ONNX export
            f"{output_dir}/eagle2/vit.onnx",
            input_names=["pixel_values", "position_ids"],  # Add position_ids to input names
            output_names=["vit_embeds"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "position_ids": {0: "batch_size"},
                "vit_embeds": {0: "batch_size"},
            },
        )


def export_eagle2_llm(backbone_model, backbone_config, output_dir, attention_mask):
    class EagleBackboneOpt(EagleBackbone):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Modify LlamamModel architecture for ONNX export
            config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
            config._attn_implementation = "eager"  # not use flash attention

            assert config.text_config.architectures[0] == "Qwen3ForCausalLM"
            self.eagle_model.language_model = Qwen3ForCausalLM(config.text_config)

            # # remove parts of the LLM
            while len(self.eagle_model.language_model.model.layers) > kwargs["select_layer"]:
                self.eagle_model.language_model.model.layers.pop(-1)

        def forward(self, input_ids, vit_embeds, attention_mask):
            if self.eagle_model.use_pixel_shuffle:
                h = w = int(vit_embeds.shape[1] ** 0.5)
                vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
                vit_embeds = self.pixel_shuffle(
                    vit_embeds, scale_factor=self.downsample_ratio
                )  # torch.Size([B, 1024, 1024]) -> torch.Size([B, 16, 16, 4096])
                vit_embeds = vit_embeds.reshape(
                    vit_embeds.shape[0], -1, vit_embeds.shape[-1]
                )  # torch.Size([B, 16, 16, 4096]) -> torch.Size([B, 256, 4096])

            if self.eagle_model.mlp_checkpoint and vit_embeds.requires_grad:
                vit_embeds = cp.checkpoint(self.eagle_model.mlp1, vit_embeds)
            else:
                vit_embeds = self.eagle_model.mlp1(vit_embeds)

            input_embeds = self.eagle_model.language_model.get_input_embeddings()(input_ids)

            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.eagle_model.image_token_index
            input_embeds[selected] = vit_embeds.reshape(-1, C)
            # try:
            #     input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            # except Exception as e:
            #     vit_embeds = vit_embeds.reshape(-1, C)
            #     print(
            #         f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
            #         f"vit_embeds.shape={vit_embeds.shape}"
            #     )
            #     n_token = selected.sum()
            #     input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

            input_embeds = input_embeds.reshape(B, N, C)

            outputs = self.eagle_model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            eagle_features = outputs.hidden_states[self.select_layer]
            eagle_features = self.eagle_linear(eagle_features)
            return eagle_features

    model = EagleBackboneOpt(**backbone_config).to(torch.float16)
    model.load_state_dict(backbone_model.state_dict())
    model.eval().cuda()

    input_ids = torch.randint(100, (1, attention_mask.shape[1]), dtype=torch.int64).cuda()
    input_ids[:, : model.eagle_model.num_image_token] = model.eagle_model.image_token_index
    vit_embeds = torch.randn(
        (
            1,
            model.eagle_model.vision_model.vision_model.embeddings.num_patches,
            model.eagle_model.vision_model.config.hidden_size,
        ),
        dtype=torch.float16,
    ).cuda()
    attention_mask = torch.ones((1, attention_mask.shape[1]), dtype=torch.int64).cuda()

    os.makedirs(output_dir, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (input_ids, vit_embeds, attention_mask),
            f"{output_dir}/eagle2/llm.onnx",
            input_names=["input_ids", "vit_embeds", "attention_mask"],
            output_names=["embeddings"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "vit_embeds": {0: "batch_size"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "embeddings": {0: "batch_size", 1: "sequence_length"},
            },
        )


class VLLN_VLSelfAttention(torch.nn.Module):
    def __init__(self, vlln, vl_self_attention):
        super().__init__()
        self.vlln = vlln
        self.vl_self_attention = vl_self_attention

    def forward(self, backbone_features):
        x = self.vlln(backbone_features)
        x = self.vl_self_attention(x)
        return x


def export_action_head(policy, ONNX_export_path, input_state, attention_mask):
    process_backbone_model = (
        VLLN_VLSelfAttention(
            policy.model.action_head.vlln, policy.model.action_head.vl_self_attention
        )
        .to(torch.float16)
        .cuda()
    )
    backbone_features = torch.randn(
        (1, attention_mask.shape[1], policy.model.action_head.config.backbone_embedding_dim),
        dtype=torch.float16,
    ).cuda()

    torch.onnx.export(
        process_backbone_model,
        (backbone_features),
        os.path.join(ONNX_export_path, "action_head/vlln_vl_self_attention.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["backbone_features"],
        output_names=["output"],
        dynamic_axes={
            "backbone_features": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        },
    )

    state_encoder = policy.model.action_head.state_encoder.to(torch.float16)

    state_tensor = torch.randn(
        (1, input_state.shape[1], input_state.shape[2]), dtype=torch.float16
    ).cuda()
    embodiment_id_tensor = torch.ones((1), dtype=torch.int64).cuda()

    torch.onnx.export(
        state_encoder,
        (state_tensor, embodiment_id_tensor),
        os.path.join(ONNX_export_path, "action_head/state_encoder.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["state", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "state": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    action_encoder = policy.model.action_head.action_encoder.to(torch.float16)
    actions_tensor = torch.randn(
        (
            1,
            policy.model.action_head.config.action_horizon,
            policy.model.action_head.config.action_dim,
        ),
        dtype=torch.float16,
    ).cuda()
    timesteps_tensor = torch.ones((1), dtype=torch.int64).cuda()

    torch.onnx.export(
        action_encoder,
        (actions_tensor, timesteps_tensor, embodiment_id_tensor),
        os.path.join(ONNX_export_path, "action_head/action_encoder.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["actions", "timesteps_tensor", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "actions": {0: "batch_size"},
            "timesteps_tensor": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    DiT = policy.model.action_head.model.to(torch.float16)
    sa_embs_tensor = torch.randn(
        (
            1,
            input_state.shape[1]
            + policy.model.action_head.config.action_horizon
            + policy.model.action_head.config.num_target_vision_tokens,
            policy.model.action_head.config.input_embedding_dim,
        ),
        dtype=torch.float16,
    ).cuda()
    vl_embs_tensor = torch.randn(
        (1, attention_mask.shape[1], policy.model.action_head.config.backbone_embedding_dim),
        dtype=torch.float16,
    ).cuda()

    torch.onnx.export(
        DiT,
        (sa_embs_tensor, vl_embs_tensor, timesteps_tensor),
        os.path.join(ONNX_export_path, "action_head/DiT.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["sa_embs", "vl_embs", "timesteps_tensor"],
        output_names=["output"],
        dynamic_axes={
            "sa_embs": {0: "batch_size"},
            "vl_embs": {0: "batch_size", 1: "sequence_length"},
            "timesteps_tensor": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    action_decoder = policy.model.action_head.action_decoder.to(torch.float16)
    model_output_tensor = torch.randn(
        (
            1,
            input_state.shape[1]
            + policy.model.action_head.config.action_horizon
            + policy.model.action_head.config.num_target_vision_tokens,
            policy.model.action_head.config.hidden_size,
        ),
        dtype=torch.float16,
    ).cuda()
    torch.onnx.export(
        action_decoder,
        (model_output_tensor, embodiment_id_tensor),
        os.path.join(ONNX_export_path, "action_head/action_decoder.onnx"),
        export_params=True,
        do_constant_folding=True,
        input_names=["model_output", "embodiment_id"],
        output_names=["output"],
        dynamic_axes={
            "model_output": {0: "batch_size"},
            "embodiment_id": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )


def prepare_sample_inputs(policy: Gr00tPolicy, step_data: Dict) -> Dict:
    """Prepare sample inputs from dataset for ONNX export"""

    # Apply transforms to get normalized inputs
    normalized_input = policy.apply_transforms(step_data)

    # Extract the inputs we need
    normalized_input["eagle_pixel_values"] = (
        normalized_input["eagle_pixel_values"].to(ONNX_MODEL_DTYPE).cuda()
    )
    normalized_input["eagle_input_ids"] = normalized_input["eagle_input_ids"].to(torch.int64).cuda()
    normalized_input["eagle_attention_mask"] = (
        normalized_input["eagle_attention_mask"].to(torch.int64).cuda()
    )
    normalized_input["state"] = normalized_input["state"].to(ONNX_MODEL_DTYPE).cuda()
    normalized_input["embodiment_id"] = normalized_input["embodiment_id"].to(torch.int64).cuda()
    pixel_values = normalized_input["eagle_pixel_values"].to(ONNX_MODEL_DTYPE).cuda()
    actions = torch.zeros(
        size=(
            pixel_values.shape[0],
            policy.model.action_head.config.action_horizon,
            policy.model.action_head.config.action_dim,
        ),
        dtype=ONNX_MODEL_DTYPE,
        device=pixel_values.device,
    )
    normalized_input["init_actions"] = actions

    return normalized_input


def export_full_model(
    policy,
    ONNX_export_path,
    dataset,
    num_calibration_samples,
    quantization,
):

    def onnx_forward(pixel_values, input_ids, attention_mask, state, embodiment_id, init_actions):
        from transformers.feature_extraction_utils import BatchFeature

        inputs_dict = {
            "eagle_pixel_values": pixel_values.to(ONNX_MODEL_DTYPE),
            "eagle_input_ids": input_ids.to(torch.int64),
            "eagle_attention_mask": attention_mask.to(torch.int64),
            "state": state.to(ONNX_MODEL_DTYPE),
            "embodiment_id": embodiment_id.to(torch.int64),
            "init_actions": init_actions,
        }
        inputs = BatchFeature(data=inputs_dict)

        backbone_outputs = policy.model.backbone(inputs)
        result = policy.model.action_head.get_action(backbone_outputs, inputs)
        output = result["action_pred"]

        return output

    full_model = policy.model
    full_model.eval()
    full_model = full_model.to(ONNX_MODEL_DTYPE).cuda()
    full_model.forward = onnx_forward
    full_model.backbone.eagle_model.config._attn_implementation = "eager"
    full_model.backbone.eagle_model.config.text_config._attn_implementation = "eager"
    full_model.backbone.eagle_model.config.vision_config._attn_implementation = "eager"

    if quantization == "fp8":
        quantization_config = mtq.FP8_DEFAULT_CFG
    else:
        quantization_config = None

    if quantization_config is not None:
        print(f"Quantizing full model to {quantization_config} ")
        quantize(full_model, dataset, policy, num_calibration_samples, quantization_config)

    step_data = dataset[0]
    step_data = unsqueeze_dict_values(step_data)  # Add batch dimension
    sample_inputs = prepare_sample_inputs(policy, step_data)

    # Create output directory
    os.makedirs(os.path.join(ONNX_export_path, "full_model"), exist_ok=True)

    # Extract sample inputs
    pixel_values = sample_inputs["eagle_pixel_values"]
    input_ids = sample_inputs["eagle_input_ids"]
    attention_mask = sample_inputs["eagle_attention_mask"]
    # Note: image_sizes not needed - forward_eagle uses .pop() to handle its absence
    state = sample_inputs["state"]
    embodiment_id = sample_inputs["embodiment_id"]

    # Create initial actions for ONNX export (this will be an input to the model)
    init_actions = torch.randn(
        size=(
            pixel_values.shape[0],
            policy.model.action_head.config.action_horizon,
            policy.model.action_head.config.action_dim,
        ),
        dtype=ONNX_MODEL_DTYPE,
        device=pixel_values.device,
    )

    # Ensure all floating point tensors are in the same dtype (float16)
    pixel_values = pixel_values.to(ONNX_MODEL_DTYPE)
    state = state.to(ONNX_MODEL_DTYPE)
    init_actions = init_actions.to(ONNX_MODEL_DTYPE)

    with torch.inference_mode():
        torch.onnx.export(
            full_model,
            (pixel_values, input_ids, attention_mask, state, embodiment_id, init_actions),
            os.path.join(ONNX_export_path, "full_model/full_model.onnx"),
            input_names=[
                "pixel_values",
                "input_ids",
                "attention_mask",
                "state",
                "embodiment_id",
                "init_actions",
            ],
            output_names=["output"],
            opset_version=20,
            do_constant_folding=True,  # Enable constant folding to optimize the graph
            export_params=True,  # Export model parameters
            verbose=False,  # Set to True for debugging
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "state": {0: "batch_size"},
                "embodiment_id": {0: "batch_size"},
                "init_actions": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )


def quantize(model_to_quantize, dataset, policy, num_calibration_samples, quantization_config):
    def forward_loop(model):
        print("Calibrating Full Model...")
        for i, step_data in enumerate(dataset):
            if i > num_calibration_samples:
                break

            step_data = unsqueeze_dict_values(step_data)  # Add batch dimension
            sample_inputs = prepare_sample_inputs(policy, step_data)

            # Extract sample inputs
            pixel_values = sample_inputs["eagle_pixel_values"]
            input_ids = sample_inputs["eagle_input_ids"]
            attention_mask = sample_inputs["eagle_attention_mask"]
            # Note: image_sizes not needed - forward_eagle uses .pop() to handle its absence
            state = sample_inputs["state"]
            embodiment_id = sample_inputs["embodiment_id"]

            actions = torch.zeros(
                size=(
                    pixel_values.shape[0],
                    policy.model.action_head.config.action_horizon,
                    policy.model.action_head.config.action_dim,
                ),
                dtype=ONNX_MODEL_DTYPE,
                device=pixel_values.device,
            )

            # Run forward pass with proper dtype handling
            model(pixel_values, input_ids, attention_mask, state, embodiment_id, actions)

        print("Full model calibration finished.")

    # Create a copy of the default FP8 configuration
    for layer_to_disable in ["*.embeddings.patch_embedding.*"]:
        quantization_config["quant_cfg"][layer_to_disable] = {"enable": False}

    mtq.quantize(model_to_quantize, quantization_config, forward_loop=forward_loop)
    mtq.print_quant_summary(model_to_quantize)


def run_groot_inference(
    dataset_path: str,
    model_path: str,
    onnx_model_path: str,
    full_model: str,
    quantization: str,
    device: str = "cuda",
    embodiment_tag: str = "gr1",
    data_config_id: str = "fourier_gr1_arms_only",
) -> Dict[str, float]:

    # load the policy
    data_config = DATA_CONFIG_MAP[data_config_id]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )
    modality_config = policy.modality_config
    # load the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=embodiment_tag,
    )

    step_data = dataset[0]
    # get the action
    predicted_action = policy.get_action(step_data)

    attention_mask, state = get_input_info(policy, step_data)

    # export onnx
    if full_model:
        os.makedirs(onnx_model_path, exist_ok=True)
        # quantization
        num_calibration_samples = 100
        export_full_model(
            policy,
            onnx_model_path,
            dataset,
            num_calibration_samples,
            quantization,
        )
    else:
        os.makedirs(os.path.join(onnx_model_path, "eagle2"), exist_ok=True)
        os.makedirs(os.path.join(onnx_model_path, "action_head"), exist_ok=True)
        export_eagle2_vit(
            policy.model.backbone.eagle_model.vision_model.vision_model, onnx_model_path
        )
        export_eagle2_llm(
            policy.model.backbone, policy.model.config.backbone_cfg, onnx_model_path, attention_mask
        )
        export_action_head(policy, onnx_model_path, state, attention_mask)

    return predicted_action


if __name__ == "__main__":
    # Make sure you have logged in to huggingface using `huggingface-cli login` with your nvidia email.
    parser = argparse.ArgumentParser(description="Run Groot Inference")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
        default=os.path.join(os.getcwd(), "demo_data/robot_sim.PickNPlace"),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model",
        default="nvidia/GR00T-N1.5-3B",
    )

    parser.add_argument(
        "--onnx_model_path",
        type=str,
        help="Path where the ONNX model will be stored",
        default=os.path.join(os.getcwd(), "gr00t_onnx"),
    )

    parser.add_argument(
        "--full_model",
        action="store_true",
        help="Export full model",
    )

    parser.add_argument(
        "--quantization",
        type=str,
        default="fp16", 
        help="Quantization type. Allowed values: fp16, fp8",
    )

    parser.add_argument(
        "--embodiment_tag",
        type=str,
        default="gr1",
        help="Embodiment tag",
    )

    parser.add_argument(
        "--data_config",
        type=str,
        default="fourier_gr1_arms_only",
        help="Data config",
    )

    args = parser.parse_args()

    print(f"Dataset path: {args.dataset_path}")
    print(f"Model path: {args.model_path}")
    print(f"ONNX model path: {args.onnx_model_path}")
    print(f"Full model: {args.full_model}")
    print(f"Quantization: {args.quantization}")
    print(f"Embodiment tag: {args.embodiment_tag}")
    print(f"Data config: {args.data_config}")
    predicted_action = run_groot_inference(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        onnx_model_path=args.onnx_model_path,
        full_model=args.full_model,
        quantization=args.quantization,
        embodiment_tag=args.embodiment_tag,
        data_config_id=args.data_config,
    )

    for key, value in predicted_action.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
