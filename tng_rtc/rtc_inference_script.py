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

from dataclasses import dataclass
from typing import Literal

import tyro

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING

from .rtc_policy import RTCGr00tPolicy
from .rtc_inference_server import RTCInferenceServer
from .extended_data_config import DATA_CONFIG_MAP


# customized copy of the ArgsConfig dataclass in Isaac-GR00T/scripts/inference_service.py
# This dataclass in Isaac-GR00T/scripts/inference_service.py cannot be imported as it is not part of a python module
@dataclass
class ArgsConfig:
    """Command line arguments for the inference service."""

    model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path to the model checkpoint directory."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """The embodiment tag for the model."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "so100_dualcam"
    """The name of the data config to use."""

    port: int = 8000
    """The port number for the server."""

    host: str = "*"
    """The host address for the server."""

    denoising_steps: int = 4
    """The number of denoising steps to use."""

    api_token: str = None
    """API token for authentication. If not provided, authentication is disabled."""


#####################################################################################


def main(args: ArgsConfig):
    print(f"Available data configs: {list(DATA_CONFIG_MAP.keys())}")
    print(f"Using data config: {args.data_config}")
    data_config = DATA_CONFIG_MAP[args.data_config]
    print(f"Data config video keys: {data_config.video_keys}")
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    prediction_horizon = len(data_config.action_indices)

    # Use custom RTCGr00tPolicy for realtime action chunking
    policy = RTCGr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )

    # Start the server
    server = RTCInferenceServer(policy, host=args.host, port=args.port, api_token=args.api_token, pred_horiton=prediction_horizon)
    server.run()


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
