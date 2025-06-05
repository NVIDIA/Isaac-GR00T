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
import json
import numpy as np
import requests
from flask import Flask, request, jsonify
import torch
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.dataset import ModalityConfig

app = Flask(__name__)
policy = None

class HTTPRobotInferenceClient:
    def __init__(self, host: str = "localhost", port: int = 5555):
        self.base_url = f"http://{host}:{port}"
        
    def get_action(self, observations: dict) -> dict:
        # 将numpy数组转换为列表以便JSON序列化
        serialized_obs = {}
        for key, value in observations.items():
            if isinstance(value, np.ndarray):
                serialized_obs[key] = value.tolist()
            else:
                serialized_obs[key] = value
                
        response = requests.post(f"{self.base_url}/get_action", json=serialized_obs)
        response.raise_for_status()
        return response.json()
        
    def get_modality_config(self) -> dict:
        response = requests.get(f"{self.base_url}/get_modality_config")
        response.raise_for_status()
        return response.json()

@app.route('/get_action', methods=['POST'])
def get_action():
    if policy is None:
        return jsonify({"error": "Policy not initialized"}), 500
        
    try:
        observations = request.json
        # 将列表转换回numpy数组
        for key, value in observations.items():
            if isinstance(value, list):
                observations[key] = np.array(value)
                
        action = policy.get_action(observations)
        # 将numpy数组转换为列表以便JSON序列化
        serialized_action = {}
        for key, value in action.items():
            if isinstance(value, np.ndarray):
                serialized_action[key] = value.tolist()
            else:
                serialized_action[key] = value
                
        return jsonify(serialized_action)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_modality_config', methods=['GET'])
def get_modality_config():
    if policy is None:
        return jsonify({"error": "Policy not initialized"}), 500
        
    try:
        config = policy.get_modality_config()
        # 将ModalityConfig对象转换为可序列化的字典
        serialized_config = {}
        for key, value in config.items():
            if isinstance(value, ModalityConfig):
                serialized_config[key] = {
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                    "names": value.names
                }
            else:
                serialized_config[key] = value
        return jsonify(serialized_config)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="nvidia/GR00T-N1-2B",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="gr1",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        help="The name of the data config to use.",
        choices=list(DATA_CONFIG_MAP.keys()),
        default="gr1_arms_waist",
    )

    parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="localhost"
    )
    # server mode
    parser.add_argument("--server", action="store_true", help="Run the server.")
    # client mode
    parser.add_argument("--client", action="store_true", help="Run the client")
    parser.add_argument("--denoising_steps", type=int, help="Number of denoising steps.", default=4)
    args = parser.parse_args()

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
        data_config = DATA_CONFIG_MAP[args.data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
        )

        # Start the server
        app.run(host=args.host, port=args.port)

    elif args.client:
        # In this mode, we will send a random observation to the server and get an action back
        # This is useful for testing the server and client connection

        # Create a policy wrapper
        policy_client = HTTPRobotInferenceClient(host=args.host, port=args.port)

        print("Available modality config available:")
        modality_configs = policy_client.get_modality_config()
        print(modality_configs.keys())

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
        action = policy_client.get_action(obs)

        for key, value in action.items():
            print(f"Action: {key}: {value.shape}")

    else:
        raise ValueError("Please specify either --server or --client")
