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

from typing import Any, Dict

import cv2
import numpy as np

from gr00t.data.dataset import ModalityConfig
from gr00t.eval.service import BaseInferenceClient, BaseInferenceServer
from gr00t.model.policy import BasePolicy


class RobotInferenceServer(BaseInferenceServer):
    """
    Server with three endpoints for real robot policies
    """

    def __init__(
        self,
        model,
        host: str = "*",
        port: int = 5555,
        api_token: str = None,
        encode_video: bool = False,
    ):
        """
        api token is used to authenticate the client.
        encode_video is used to encode the video to bytes to reduce image transfer bandwidth.
        """
        super().__init__(host, port, api_token)
        self.encode_video = encode_video
        self.model = model
        self.register_endpoint("get_action", self._get_action, arg_names=["observations", "config"])
        self.register_endpoint(
            "get_modality_config", model.get_modality_config, requires_input=False
        )

    @staticmethod
    def start_server(policy: BasePolicy, port: int, api_token: str = None):
        server = RobotInferenceServer(policy, port=port, api_token=api_token)
        server.run()

    def _get_action(self, observations: dict, config: dict) -> dict:
        if self.encode_video:
            observations = self._decode_video(observations)
        return self.model.get_action(observations, config)

    def _decode_video(self, observation: dict) -> dict:
        """
        TODO(YL): currently this encoding logic only works for non-batched video
        [history, H, W, C], Batched video [batch, history, H, W, C] is not supported yet.
        """
        for key, value in observation.items():
            if "video" in key:
                frames = []
                for frame in value:
                    if isinstance(frame, bytes):
                        frames.append(
                            cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
                        )
                    else:
                        frames.append(frame)
                observation[key] = np.array(frames)
        return observation


class RobotInferenceClient(BaseInferenceClient, BasePolicy):
    """
    Client for communicating with the RealRobotServer
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        api_token: str = None,
        encode_video: bool = False,
    ):
        """
        api token is used to authenticate the client.
        encode_video is used to encode the video to bytes to reduce image transfer bandwidth.
        """
        super().__init__(host=host, port=port, api_token=api_token)
        self.encode_video = encode_video

    def get_action(
        self, observations: Dict[str, Any], config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if self.encode_video:
            observations = self._encode_video(observations)
        return self.call_endpoint_with_args("get_action", observations=observations, config=config)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def _encode_video(self, observation: dict) -> dict:
        """
        TODO(YL): currently this encoding logic only works for non-batched video
        [history, H, W, C], Batched video [batch, history, H, W, C] is not supported yet.
        """
        for key, value in observation.items():
            if "video" in key:
                frames = []
                for frame in value:
                    frames.append(cv2.imencode(".jpg", frame)[1].tobytes())
                observation[key] = frames
        return observation
