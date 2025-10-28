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
from functools import partial

import torch
from action_head_utils import action_head_pytorch_forward
from trt_model_forward import setup_tensorrt_engines

import gr00t
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy


def compare_predictions(pred_tensorrt, pred_torch):
    """
    Compare the similarity between TensorRT and PyTorch predictions

    Args:
        pred_tensorrt: TensorRT prediction results (numpy array)
        pred_torch: PyTorch prediction results (numpy array)
    """
    print("\n=== Prediction Comparison ===")

    # Ensure both predictions contain the same keys
    assert pred_tensorrt.keys() == pred_torch.keys(), "Prediction keys do not match"

    # Calculate max label width for alignment
    max_label_width = max(
        len("Cosine Similarity (PyTorch/TensorRT):"),
        len("L1 Mean/Max Distance (PyTorch/TensorRT):"),
        len("Max Output Values (PyTorch/TensorRT):"),
        len("Mean Output Values (PyTorch/TensorRT):"),
        len("Min Output Values (PyTorch/TensorRT):"),
    )

    for key in pred_tensorrt.keys():
        tensorrt_array = pred_tensorrt[key]
        torch_array = pred_torch[key]

        # Convert to PyTorch tensors
        tensorrt_tensor = torch.from_numpy(tensorrt_array).to(torch.float32)
        torch_tensor = torch.from_numpy(torch_array).to(torch.float32)

        # Ensure tensor shapes are the same
        assert (
            tensorrt_tensor.shape == torch_tensor.shape
        ), f"{key} shapes do not match: {tensorrt_tensor.shape} vs {torch_tensor.shape}"

        # Calculate cosine similarity
        flat_tensorrt = tensorrt_tensor.flatten()
        flat_torch = torch_tensor.flatten()

        # Manually calculate cosine similarity
        dot_product = torch.dot(flat_tensorrt, flat_torch)
        norm_tensorrt = torch.norm(flat_tensorrt)
        norm_torch = torch.norm(flat_torch)
        cos_sim = dot_product / (norm_tensorrt * norm_torch)

        # Calculate L1 distance
        l1_dist = torch.abs(flat_tensorrt - flat_torch)

        print(f"\n{key}:")
        print(f'{"Cosine Similarity (PyTorch/TensorRT):".ljust(max_label_width)} {cos_sim.item()}')
        print(
            f'{"L1 Mean/Max Distance (PyTorch/TensorRT):".ljust(max_label_width)} {l1_dist.mean().item():.4f}/{l1_dist.max().item():.4f}'
        )
        print(
            f'{"Max Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.max().item():.4f}/{tensorrt_tensor.max().item():.4f}'
        )
        print(
            f'{"Mean Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.mean().item():.4f}/{tensorrt_tensor.mean().item():.4f}'
        )
        print(
            f'{"Min Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.min().item():.4f}/{tensorrt_tensor.min().item():.4f}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GR00T inference")
    parser.add_argument(
        "--model-path", type=str, default="nvidia/GR00T-N1.5-3B", help="Path to the GR00T model"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the dataset (default: demo_data/robot_sim.PickNPlace)",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="fourier_gr1_arms_only",
        help="The name of the data config to use (e.g. fourier_gr1_arms_only) or a path to a custom data config file (e.g. 'module:ClassName')",
    )
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default="gr1",
        help="The embodiment tag for the model (e.g. gr1, g1, so100, etc.)",
    )
    parser.add_argument(
        "--inference-mode",
        type=str,
        choices=["pytorch", "tensorrt", "compare"],
        default="pytorch",
        help="Inference mode: 'pytorch' for PyTorch inference, 'tensorrt' for TensorRT inference, 'compare' for compare PyTorch and TensorRT outputs similarity",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        help="Number of denoising steps",
        default=4,
    )
    parser.add_argument(
        "--trt-engine-path",
        type=str,
        help="Path to the TensorRT engine",
        default="gr00t_engine",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        choices=["decord", "torchcodec"],
        help="Video backend to use for loading videos",
        default="decord",
    )
    parser.add_argument(
        "--vit-dtype",
        type=str,
        choices=["fp16", "fp8"],
        help="ViT model dtype (fp16, fp8)",
        default="fp8",
    )
    parser.add_argument(
        "--llm-dtype",
        type=str,
        choices=["fp16", "nvfp4", "fp8"],
        help="LLM model dtype (fp16, nvfp4, fp8)",
        default="nvfp4",
    )
    parser.add_argument(
        "--dit-dtype",
        type=str,
        choices=["fp16", "fp8"],
        help="DiT model dtype (fp16, fp8)",
        default="fp8",
    )
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
    DATASET_PATH = (
        args.dataset_path
        if args.dataset_path
        else os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")
    )
    EMBODIMENT_TAG = args.embodiment_tag

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data config
    from gr00t.experiment.data_config import load_data_config

    data_config = load_data_config(args.data_config)
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=MODEL_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=args.denoising_steps,
        device=device,
    )

    modality_config = policy.modality_config
    dataset = LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=modality_config,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=EMBODIMENT_TAG,
    )

    step_data = dataset[0]

    if args.inference_mode == "pytorch":
        predicted_action = policy.get_action(step_data)
        print("\n=== PyTorch Inference Results ===")
        for key, value in predicted_action.items():
            print(key, value.shape)

    elif args.inference_mode == "tensorrt":
        # Setup TensorRT engines
        setup_tensorrt_engines(
            policy, args.trt_engine_path, args.vit_dtype, args.llm_dtype, args.dit_dtype
        )

        predicted_action = policy.get_action(step_data)
        print("\n=== TensorRT Inference Results ===")
        for key, value in predicted_action.items():
            print(key, value.shape)

    else:
        # ensure PyTorch and TensorRT have the same init_actions
        if not hasattr(policy.model.action_head, "init_actions"):
            policy.model.action_head.init_actions = torch.randn(
                (1, policy.model.action_head.action_horizon, policy.model.action_head.action_dim),
                dtype=torch.float16,
                device=device,
            )
        # PyTorch inference
        policy.model.action_head.get_action = partial(
            action_head_pytorch_forward, policy.model.action_head
        )
        predicted_action_torch = policy.get_action(step_data)

        # Setup TensorRT engines and run inference
        setup_tensorrt_engines(
            policy, args.trt_engine_path, args.vit_dtype, args.llm_dtype, args.dit_dtype
        )
        predicted_action_tensorrt = policy.get_action(step_data)

        # Compare predictions
        compare_predictions(predicted_action_tensorrt, predicted_action_torch)
