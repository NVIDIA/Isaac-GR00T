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

"""
GR00T Inference Service

This script provides both ZMQ and HTTP server/client implementations for deploying GR00T models.
The HTTP server exposes a REST API for easy integration with web applications and other services.

1. Default is zmq server.

Run server: python scripts/inference_service.py --server
Run client: python scripts/inference_service.py --client

2. Run as Http Server:

Dependencies for `http_server` mode:
    => Server (runs GR00T model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

HTTP Server Usage:
    python scripts/inference_service.py --server --http-server --port 8000

HTTP Client Usage (assuming a server running on 0.0.0.0:8000):
    python scripts/inference_service.py --client --http-server --host 0.0.0.0 --port 8000

You can use bore to forward the port to your client: `159.223.171.199` is bore.pub.
    bore local 8000 --to 159.223.171.199

3. TensorRT Support:

For accelerated inference using TensorRT, first build the TensorRT engines using the deployment scripts,
then run the server with the --use-tensorrt flag:

TensorRT Server Usage:
    python scripts/inference_service.py --server --use-tensorrt --trt-engine-path gr00t_engine

TensorRT HTTP Server Usage:
    python scripts/inference_service.py --server --http-server --use-tensorrt --trt-engine-path gr00t_engine --port 8000

Note: TensorRT engines must be built before running with --use-tensorrt flag.
See deployment_scripts/README.md for instructions on building TensorRT engines.
"""

import os
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro
# Global attention implementation setting
# Read from environment variable first, otherwise default to "eager". 
# We use eager instead of flash_attention_2 since flash_attention_2 export is not supported by Torch-TensorRT.
ATTN_IMPLEMENTATION = os.environ.setdefault("ATTN_IMPLEMENTATION", "eager")

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy
from deployment_scripts.run_groot_torchtrt import compile_eagle_backbone, compile_action_head, get_dataset, get_input_info, get_torch_dtype
from deployment_scripts.action_head_utils import action_head_pytorch_forward

@dataclass
class ArgsConfig:
    """Command line arguments for the inference service."""

    model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path to the model checkpoint directory."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """The embodiment tag for the model."""

    data_config: str = "fourier_gr1_arms_waist"
    """
    The name of the data config to use, e.g. so100, fourier_gr1_arms_only, unitree_g1, etc.

    Or a path to a custom data config file. e.g. "module:ClassName" format.
    See gr00t/experiment/data_config.py for more details.
    """

    port: int = 5555
    """The port number for the server."""

    host: str = "localhost"
    """The host address for the server."""

    server: bool = False
    """Whether to run the server."""

    client: bool = False
    """Whether to run the client."""

    denoising_steps: int = 4
    """The number of denoising steps to use."""

    api_token: str = None
    """API token for authentication. If not provided, authentication is disabled."""

    http_server: bool = False
    """Whether to run it as HTTP server. Default is ZMQ server."""

    use_tensorrt: bool = False
    """Whether to use TensorRT for inference. Requires TensorRT engines to be built."""

    trt_engine_path: str = "gr00t_engine"
    """Path to the TensorRT engine directory. Only used when use_tensorrt is True."""

    vit_dtype: Literal["fp16", "fp8"] = "fp8"
    """ViT model dtype (fp16, fp8). Only used when use_tensorrt is True."""

    llm_dtype: Literal["fp16", "nvfp4", "fp8"] = "nvfp4"
    """LLM model dtype (fp16, nvfp4, fp8). Only used when use_tensorrt is True."""

    dit_dtype: Literal["fp16", "fp8"] = "fp8"
    """DiT model dtype (fp16, fp8). Only used when use_tensorrt is True."""

    use_torch_tensorrt: bool = False
    """Whether to use Torch-TensorRT for inference. Requires Torch-TensorRT to be installed."""

    dataset_path: str = os.path.join(os.getcwd(), "./demo_data/robot_sim.PickNPlace")
    """Path to the dataset. Only used when use_torch_tensorrt is True. Default is the path to the dataset in the repo."""

    precision: Literal["bf16", "fp16", "fp8"] = "bf16"
    """Precision for the model. Only used when use_torch_tensorrt is True."""

    device: Literal["cuda", "cpu"] = "cuda"
    """Device for the model. Only used when use_torch_tensorrt is True."""

    use_explicit_typing: bool = True
    """Whether to use explicit typing for the model. Only used when use_torch_tensorrt is True."""

    use_fp32_acc: bool = True
    """Whether to use fp32 accumulation for the model. Only used when use_torch_tensorrt is True."""


#####################################################################################


def _example_zmq_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    Example ZMQ client call to the server.
    """
    # Original ZMQ client mode
    # Create a policy wrapper
    policy_client = RobotInferenceClient(host=host, port=port, api_token=api_token)

    print("Available modality config available:")
    modality_configs = policy_client.get_modality_config()
    print(modality_configs.keys())

    time_start = time.time()
    action = policy_client.get_action(obs)
    print(f"Total time taken to get action from server: {time.time() - time_start} seconds")
    return action


def _example_http_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    Example HTTP client call to the server.
    """
    import json_numpy

    json_numpy.patch()
    import requests

    # Send request to HTTP server
    print("Testing HTTP server...")

    time_start = time.time()
    response = requests.post(f"http://{host}:{port}/act", json={"observation": obs})
    print(f"Total time taken to get action from HTTP server: {time.time() - time_start} seconds")

    if response.status_code == 200:
        action = response.json()
        return action
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return {}


def main(args: ArgsConfig):
    if args.server:
        # Create a policy
        # The `Gr00tPolicy` class is being used to create a policy object that encapsulates
        # the model path, transform name, embodiment tag, and denoising steps for the robot
        # inference system. This policy object is then utilized in the server mode to start
        # the Robot Inference Server for making predictions based on the specified model and
        # configuration.

        # we will use an existing data config to create the modality config and transform
        # if a new data config is specified, this expect user to
        # construct your own modality config and transform
        # see gr00t/utils/data.py for more details
        data_config = load_data_config(args.data_config)
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            compute_dtype=get_torch_dtype(args.precision),
            device=args.device
        )

        # Setup TensorRT if requested
        if args.use_tensorrt:
            print(f"Setting up TensorRT engines from: {args.trt_engine_path}")
            print(f"  ViT dtype: {args.vit_dtype}")
            print(f"  LLM dtype: {args.llm_dtype}")
            print(f"  DiT dtype: {args.dit_dtype}")
            from deployment_scripts.trt_model_forward import setup_tensorrt_engines

            setup_tensorrt_engines(
                policy, args.trt_engine_path, args.vit_dtype, args.llm_dtype, args.dit_dtype
            )
            print("TensorRT engines loaded successfully!")

        if args.use_torch_tensorrt:
            assert not args.use_tensorrt, "Cannot use both --use-tensorrt and --use_torch_tensorrt"
            
            import torch
            from functools import partial
            # Create argparse.Namespace with required arguments for Torch-TensorRT compilation
            import argparse
            trt_args = argparse.Namespace()
            trt_args.precision = args.precision
            trt_args.device = args.device
            trt_args.use_explicit_typing = args.use_explicit_typing
            trt_args.use_fp32_acc = args.use_fp32_acc
            trt_args.vit_dtype = getattr(args, 'vit_dtype', 'fp16')
            trt_args.llm_dtype = getattr(args, 'llm_dtype', 'fp16')
            trt_args.dit_dtype = getattr(args, 'dit_dtype', 'fp16')
            trt_args.use_onnx_vit = getattr(args, 'use_onnx_vit', False)
            trt_args.use_onnx_llm = getattr(args, 'use_onnx_llm', False)
            trt_args.debug = getattr(args, 'debug', False)
            trt_args.eval = getattr(args, 'eval', False)
            trt_args.benchmark = getattr(args, 'benchmark', False)
            trt_args.fn_name = getattr(args, 'fn_name', 'all')
            trt_args.dataset_path = getattr(args, 'dataset_path', None)
            trt_args.model_path = args.model_path
            trt_args.embodiment_tag = args.embodiment_tag
            trt_args.denoising_steps = args.denoising_steps
            trt_args.data_config = args.data_config
            trt_args.use_eagle_backbone_joint = getattr(args, 'use_eagle_backbone_joint', False)
            trt_args.disable_tf32 = getattr(args, 'disable_tf32', False)
            trt_args.use_cpp_runtime = getattr(args, 'use_cpp_runtime', False)
            trt_args.cpu_offload = getattr(args, 'cpu_offload', False)

            policy.model = policy.model.eval().to(get_torch_dtype(trt_args.precision)).to(trt_args.device)
            # Ensure attention implementation is set correctly (should already be set in EagleBackbone.__init__)
            policy.model.backbone.eagle_model.vision_model.config._attn_implementation = ATTN_IMPLEMENTATION
            policy.model.backbone.eagle_model.language_model.config._attn_implementation = ATTN_IMPLEMENTATION
            if not hasattr(policy.model.action_head, "init_actions"):
                policy.model.action_head.init_actions = torch.randn(
                    (1, policy.model.action_head.action_horizon, policy.model.action_head.action_dim),
                    dtype=get_torch_dtype(trt_args.precision),
                    device=args.device,
                )
            
            policy.model.action_head.get_action = partial(
                action_head_pytorch_forward, policy.model.action_head
            )
            
            step_data = get_dataset(trt_args)
            attention_mask, state = get_input_info(policy, step_data)
            trt_eagle_backbone = compile_eagle_backbone(policy.model.backbone.eagle_model, args=trt_args)
            policy.model.backbone.eagle_model = trt_eagle_backbone
            trt_action_head = compile_action_head(policy.model.action_head, trt_args, attention_mask=attention_mask, state=state)
            policy.model.action_head = trt_action_head

        # Start the server
        if args.http_server:
            from gr00t.eval.http_server import HTTPInferenceServer  # noqa: F401

            server = HTTPInferenceServer(
                policy, port=args.port, host=args.host, api_token=args.api_token
            )
            server.run()
        else:
            server = RobotInferenceServer(policy, port=args.port, api_token=args.api_token)
            server.run()

    # Here is mainly a testing code
    elif args.client:
        # In this mode, we will send a random observation to the server and get an action back
        # This is useful for testing the server and client connection

        # Making prediction...
        # - obs: video.ego_view: (1, 256, 256, 3)
        # - obs: state.left_arm: (1, 7)
        # - obs: state.right_arm: (1, 7)
        # - obs: state.left_hand: (1, 6)
        # - obs: state.right_hand: (1, 6)
        # - obs: state.waist: (1, 3)

        # - action: action.left_arm: (16, 7)
        # - action: action.right_arm: (16, 7)
        # - action: action.left_hand: (16, 6)
        # - action: action.right_hand: (16, 6)
        # - action: action.waist: (16, 3)
        obs = {
            "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
            "state.left_arm": np.random.rand(1, 7),
            "state.right_arm": np.random.rand(1, 7),
            "state.left_hand": np.random.rand(1, 6),
            "state.right_hand": np.random.rand(1, 6),
            "state.waist": np.random.rand(1, 3),
            "annotation.human.action.task_description": ["do your thing!"],
        }

        if args.http_server:
            action = _example_http_client_call(obs, args.host, args.port, args.api_token)
        else:
            action = _example_zmq_client_call(obs, args.host, args.port, args.api_token)

        for key, value in action.items():
            print(f"Action: {key}: {value.shape}")
    else:
        raise ValueError("Please specify either --server or --client")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
