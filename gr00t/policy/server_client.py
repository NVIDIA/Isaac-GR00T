# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import functools
from typing import Any, Callable

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

from gr00t.data.types import ModalityConfig
from gr00t.data.utils import to_json_serializable

from .policy import BasePolicy


class MsgSerializer:
    """msgpack_numpy serializer with a hard ``allow_pickle=False`` boundary.

    Implementation note: msgpack_numpy's ``Packer``/``Unpacker`` wire any
    user-provided ``default``/``object_hook`` *behind* their own
    ``mnp.encode``/``mnp.decode`` (``functools.partial(encode, chain=user_fn)``).
    That means an object-dtype ndarray is serialised by ``mnp.encode`` (and a
    forged ``{nd: True, kind: 'O', ...}`` payload is fed to ``pickle.loads`` by
    ``mnp.decode``) *before* a hook installed via ``mnp.packb``/``mnp.unpackb``
    can intervene. We therefore drive ``msgpack`` directly and chain
    ``_safe_{encode,decode} → mnp.{encode,decode} → custom`` ourselves, so the
    refusal runs first.
    """

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        default = functools.partial(MsgSerializer._safe_encode, chain=MsgSerializer._encode_custom)
        return msgpack.packb(data, default=default)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        object_hook = functools.partial(
            MsgSerializer._safe_decode, chain=MsgSerializer._decode_custom
        )
        return msgpack.unpackb(data, object_hook=object_hook, raw=False)

    @staticmethod
    def _safe_encode(obj, chain=None):
        # Refuse object-dtype ndarrays before mnp.encode would emit a
        # ``{nd: True, kind: 'O', data: pickle.dumps(arr)}`` envelope, which
        # silently re-enables the arbitrary-code surface that the previous
        # ``np.save(..., allow_pickle=False)`` path explicitly forbade.
        if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
            raise TypeError(
                f"Refusing to encode object-dtype ndarray (shape={obj.shape}); "
                f"msgpack_numpy would invoke pickle. Convert to a concrete "
                f"numeric dtype before sending."
            )
        return mnp.encode(obj, chain=chain)

    @staticmethod
    def _safe_decode(obj, chain=None):
        # Refuse object-dtype ndarray payloads before mnp.decode would call
        # ``pickle.loads`` on attacker-controlled bytes. Check both bytes- and
        # str-encoded keys, and accept any truthy ``nd`` value (not just
        # boolean ``True``) so a forged ``{nd: 1, kind: 'O', ...}`` payload
        # can't sidestep the guard via msgpack's int-vs-bool type codes.
        # msgpack_numpy 0.4.8's own check is ``obj[b'nd'] is True``, so the
        # described variants don't actually reach pickle.loads today, but
        # MsgSerializer enforces the contract at this boundary regardless of
        # mnp's wire / identity-check conventions.
        if isinstance(obj, dict):
            nd_val = obj.get(b"nd", obj.get("nd"))
            kind_val = obj.get(b"kind", obj.get("kind"))
            if nd_val and kind_val in (b"O", "O"):
                raise ValueError(
                    "Refusing to decode object-dtype ndarray payload (pickle-bearing); "
                    "the allow_pickle=False contract is enforced by MsgSerializer."
                )
        return mnp.decode(obj, chain=chain)

    @staticmethod
    def _encode_custom(obj):
        if isinstance(obj, ModalityConfig):
            return {"__ModalityConfig__": True, "as_json": to_json_serializable(obj)}
        return obj

    @staticmethod
    def _decode_custom(obj):
        if not isinstance(obj, dict):
            return obj
        # If the ModalityConfig marker is present but 'as_json' is missing,
        # raise instead of returning a half-broken dict.
        if "__ModalityConfig__" in obj or b"__ModalityConfig__" in obj:
            key = next((k for k in ("as_json", b"as_json") if k in obj), None)
            if key is None:
                raise ValueError(
                    f"Malformed ModalityConfig payload: marker present but "
                    f"'as_json' missing. keys={sorted(repr(k) for k in obj.keys())}"
                )
            return ModalityConfig(**obj[key])
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class PolicyServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(
        self,
        policy: BasePolicy,
        host: str = "*",
        port: int = 5555,
        api_token: str = None,
    ):
        self.policy = policy
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)
        self.register_endpoint("get_action", self.policy.get_action)
        self.register_endpoint("reset", self.policy.reset)
        self.register_endpoint(
            "get_modality_config",
            getattr(self.policy, "get_modality_config", lambda: {}),
            requires_input=False,
        )

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

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

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
                result = (
                    handler.handler(**request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(MsgSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(MsgSerializer.to_bytes({"error": str(e)}))

    @staticmethod
    def start_server(policy: BasePolicy, port: int, host: str = "*", api_token: str = None):
        server = PolicyServer(policy, host=host, port=port, api_token=api_token)
        server.run()


class PolicyClient(BasePolicy):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
        strict: bool = False,
    ):
        super().__init__(strict=strict)
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
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
    ) -> Any:
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

        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            # Timeout — REQ socket is now in an invalid state (waiting for a
            # reply that will never arrive).  Recreate it so the next call can
            # send again, then re-raise so the caller knows this request failed.
            self._init_socket()
            raise
        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure we are running the correct policy server.")
        response = MsgSerializer.from_bytes(message)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)  # Convert list (from msgpack) to tuple of (action, info)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def check_observation(self, observation: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_observation is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."
        )

    def check_action(self, action: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_action is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."
        )
