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

import io
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict

import cv2
import msgpack
import numpy as np
import zmq

try:
    from gr00t.data.dataset import ModalityConfig
except ImportError:
    ModalityConfig = None


class MsgSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if ModalityConfig is not None and "__ModalityConfig_class__" in obj:
            obj = ModalityConfig(**json.loads(obj["as_json"]))
        if "__ndarray_class__" in obj:
            obj = np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        # Only check isinstance if ModalityConfig is a type
        if ModalityConfig is not None and isinstance(obj, ModalityConfig):
            return {"__ModalityConfig_class__": True, "as_json": obj.model_dump_json()}
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True
    arg_names: list[str] = None
    default_args: dict = None


class BaseInferenceServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(self, host: str = "*", port: int = 5555, api_token: str = None):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(
        self,
        name: str,
        handler: Callable,
        requires_input: bool = True,
        arg_names: list[str] = None,
        default_args: dict = None,
    ):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
            arg_names: List of argument names the handler expects. If None, all data will be passed as a single dict.
            default_args: Default values for arguments that might not be provided.

        Examples:
            # Endpoint with no arguments
            server.register_endpoint("ping", self._handle_ping, requires_input=False)

            # Endpoint with single dict argument (backward compatibility)
            server.register_endpoint("process_data", self._process_data)

            # Endpoint with specific named arguments
            server.register_endpoint("get_action", self._get_action, arg_names=["observations", "config"])

            # Endpoint with default values
            server.register_endpoint("process", self._process, arg_names=["data", "config"], default_args={"config": {}})
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input, arg_names, default_args)

    def _validate_token(self, request: dict) -> bool:
        """
        Validate the API token in the request.
        """
        if self.api_token is None:
            return True  # No token required
        return request.get("api_token") == self.api_token

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = MsgSerializer.from_bytes(message)

                # Validate token before processing request
                if not self._validate_token(request):
                    self.socket.send(
                        MsgSerializer.to_bytes({"error": "Unauthorized: Invalid API token"})
                    )
                    continue

                endpoint = request.get("endpoint", "get_action")

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]

                if handler.requires_input:
                    request_data = request.get("data", {})

                    if handler.arg_names is not None:
                        # Extract specific arguments by name
                        args = []
                        for arg_name in handler.arg_names:
                            if arg_name in request_data:
                                args.append(request_data[arg_name])
                            elif handler.default_args and arg_name in handler.default_args:
                                args.append(handler.default_args[arg_name])
                            else:
                                raise ValueError(f"Missing required argument: {arg_name}")
                        result = handler.handler(*args)
                    else:
                        # Pass all data as a single dict (backward compatibility)
                        result = handler.handler(request_data)
                else:
                    result = handler.handler()
                self.socket.send(MsgSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(MsgSerializer.to_bytes({"error": str(e)}))


class BaseInferenceClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        response = MsgSerializer.from_bytes(message)

        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def call_endpoint_with_args(self, endpoint: str, **kwargs) -> dict:
        """
        Call an endpoint on the server with named arguments.

        Args:
            endpoint: The name of the endpoint.
            **kwargs: Named arguments to pass to the endpoint.
        """
        request: dict = {"endpoint": endpoint, "data": kwargs}
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        response = MsgSerializer.from_bytes(message)

        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()


class ExternalRobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with the RealRobotServer
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        api_token: str = None,
        encode_video: bool = True,
    ):
        super().__init__(host, port, api_token)
        self.encode_video = encode_video

    def get_action(
        self, observations: Dict[str, Any], config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Get the action from the server.
        The exact definition of the observations is defined
        by the policy, which contains the modalities configuration.
        """
        observations = self._encode_video(observations)
        return self.call_endpoint_with_args("get_action", observations=observations, config=config)

    def _encode_video(self, observation: dict) -> dict:
        for key, value in observation.items():
            if "video" in key:
                frames = []
                for frame in value:
                    frames.append(cv2.imencode(".jpg", frame)[1].tobytes())
                observation[key] = frames
        return observation
