# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
import torch

from gr00t.eval.robot import RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="GR00T Inference Server")

    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/Projects/Robotics/Foundation_Models/gr00t-n1.5/exps/n1_5_galbot_eef_pose_rot6d_bs100_gpu8/checkpoint-15000",
        help="Path to the GR00T model checkpoint",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="galbot_abs_eef_pose",
        help="Name of the data configuration to use, [galbot_abs_eef_pose, galbot_joint_space]",
    )
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment", help="Embodiment tag for the policy")
    parser.add_argument("--denoising_steps", type=int, default=4, help="Number of denoising steps for the policy")

    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind the server to")
    parser.add_argument("--port", type=int, default=5555, help="Port number to bind the server to")

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cuda or cpu)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate data config exists
    if args.data_config not in DATA_CONFIG_MAP:
        print(f"Error: Data config '{args.data_config}' not found in DATA_CONFIG_MAP")
        print(f"Available configs: {list(DATA_CONFIG_MAP.keys())}")
        sys.exit(1)

    print(f"Loading GR00T model from: {args.model_path}")
    print(f"Using data config: {args.data_config}")
    print(f"Embodiment tag: {args.embodiment_tag}")
    print(f"Device: {args.device}")
    print(f"Server will run on {args.host}:{args.port}")

    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy: BasePolicy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=args.device,
    )

    server = RobotInferenceServer(model=policy, host=args.host, port=args.port)
    print("Starting GR00T inference server...")
    server.run()


if __name__ == "__main__":
    main()
