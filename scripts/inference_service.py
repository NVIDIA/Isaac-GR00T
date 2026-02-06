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

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import load_data_config
from gr00t.data.dataset import ModalityConfig
from gr00t.model.policy import Gr00tPolicy, Gr00tInpaintingPolicy


@dataclass
class ArgsConfig:
    """Command line arguments for the inference service."""

    model_path: str = None
    """Path to the model checkpoint directory."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
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

    num_action_steps: int = 1
    """Number of action steps to use."""

    use_inpainting: bool = False
    """Whether to use inpainting-based real-time action chunking.
    When enabled, the policy fixes the first ``num_prefix_steps`` actions from
    the previous prediction and only denoises the remaining suffix."""

    n_action_execute_steps: int = 8
    """How many action steps to execute before re-planning (inpainting mode only)."""

    num_prefix_steps: int = 8
    """How many leading actions to clamp via inpainting (inpainting mode only)."""

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
        # data_config = load_data_config(args.data_config)
        # modality_config = data_config.modality_config()
        # modality_transform = data_config.transform()
        
        # 1.1 modality configs and transforms
        # data_config_cls = load_data_config(config.data_config)
        # modality_configs = data_config_cls.modality_config()


        video_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=["video.camera"],
        )

        state_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=['state.joint_state_position',
                        # 'state.joint_state_velocity',
                        # 'state.insertion_position',
                        # 'state.insertion_rotation'
                        ],
        )

        action_modality = ModalityConfig(
            delta_indices=list(range(args.num_action_steps)),
            modality_keys=["action.joint_position"],
        )

        language_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        )

        modality_config = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }



        # transforms = data_config_cls.transform()
        
        from gr00t.data.transform.base import ComposedModalityTransform
        from gr00t.data.transform import VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy
        from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
        from gr00t.data.transform.concat import ConcatTransform
        from gr00t.model.transforms import GR00TTransform


        # select the transforms you want to apply to the data
        modality_transform = ComposedModalityTransform(
            transforms=[
                # video transforms
                VideoToTensor(apply_to=video_modality.modality_keys, backend="torchvision"),
                VideoCrop(apply_to=video_modality.modality_keys, scale=0.95, backend="torchvision"),
                VideoResize(apply_to=video_modality.modality_keys, height=224, width=224,
                            interpolation="linear", backend="torchvision" ),
                VideoColorJitter(apply_to=video_modality.modality_keys, brightness=0.3,
                                contrast=0.4, saturation=0.5, hue=0.08, backend="torchvision"),
                VideoToNumpy(apply_to=video_modality.modality_keys),

                # state transforms
                StateActionToTensor(apply_to=state_modality.modality_keys),
                StateActionTransform(apply_to=state_modality.modality_keys, normalization_modes={
                    "state.joint_state_position": "min_max",
                    # "state.joint_state_velocity": "min_max",
                    # "state.insertion_position": "min_max",
                    # "state.insertion_rotation": "min_max",
                }),

                # action transforms
                StateActionToTensor(apply_to=action_modality.modality_keys),
                StateActionTransform(apply_to=action_modality.modality_keys, normalization_modes={
                    "action.joint_position": "min_max",
                }),

                # ConcatTransform
                ConcatTransform(
                    video_concat_order=video_modality.modality_keys,
                    state_concat_order=state_modality.modality_keys,
                    action_concat_order=action_modality.modality_keys,
                ),
                # model-specific transform
                GR00TTransform(
                    state_horizon=len(state_modality.delta_indices),
                    action_horizon=args.num_action_steps,
                    max_state_dim=64,
                    max_action_dim=6,
                ),
            ]
        )

        if args.use_inpainting:
            print(f"Using INPAINTING policy: "
                  f"n_action_execute_steps={args.n_action_execute_steps}, "
                  f"num_prefix_steps={args.num_prefix_steps}")
            policy = Gr00tInpaintingPolicy(
                model_path=args.model_path,
                modality_config=modality_config,
                modality_transform=modality_transform,
                embodiment_tag=args.embodiment_tag,
                denoising_steps=args.denoising_steps,
                n_action_steps=args.n_action_execute_steps,
                num_prefix_steps=args.num_prefix_steps,
            )
        else:
            policy = Gr00tPolicy(
                model_path=args.model_path,
                modality_config=modality_config,
                modality_transform=modality_transform,
                embodiment_tag=args.embodiment_tag,
                denoising_steps=args.denoising_steps,
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

        # Start the server
        if args.http_server:
            from gr00t.eval.http_server import HTTPInferenceServer  # noqa: F401

            server = HTTPInferenceServer(
                policy, port=args.port, host=args.host, api_token=args.api_token
            )
            server.run()
        else:
            server = RobotInferenceServer(policy, host='10.111.83.67', port=args.port, api_token=args.api_token)
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
            "video.camera": np.random.randint(0, 256, (1, 720, 1280, 3), dtype=np.uint8),
            "state.joint_state_position": np.random.rand(1, 6),
            # "state.right_arm": np.random.rand(1, 7),
            # "state.left_hand": np.random.rand(1, 6),
            # "state.right_hand": np.random.rand(1, 6),
            # "state.waist": np.random.rand(1, 3),
            "annotation.human.action.task_description": ["do your thing!"],
        }

        if args.http_server:
            action = _example_http_client_call(obs, args.host, args.port, args.api_token)
        else:
            for i in range(100):
                action = _example_zmq_client_call(obs, args.host, args.port, args.api_token)

        for key, value in action.items():
            print(f"Action: {key}: {value.shape}")
    else:
        raise ValueError("Please specify either --server or --client")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
