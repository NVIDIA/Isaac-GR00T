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
from collections.abc import Callable
import dataclasses
from dataclasses import dataclass
import enum
import io
import json
from typing import Any, cast

import msgpack
import numpy as np
import zmq

from groot.core.data import types
from groot.core.policy import base


# --- Inlined from jpeg_codec.py ---


def _is_image_array(arr: np.ndarray) -> bool:
    """Check if array looks like a single image: uint8, 3D, last dim in {1, 3}."""
    return arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[-1] in (1, 3)


def _is_batched_image_array(arr: np.ndarray) -> int | None:
    """Return number of leading batch dims if array is a batched image, else None."""
    if arr.dtype != np.uint8:
        return None
    if arr.ndim == 4 and arr.shape[-1] in (1, 3):
        return 1
    if arr.ndim == 5 and arr.shape[-1] in (1, 3):
        return 2
    return None


class JpegCodec:
    """JPEG codec for uint8 image frames, RGB on both sides.

    Contract (both encode input and decode output):
    - Shape: ``(H, W, 3)`` for color frames, ``(H, W, 1)`` or ``(H, W)`` for
      grayscale (encode), ``(H, W, 3)`` / ``(H, W, 1)`` for decode.
    - dtype: ``np.uint8``.
    - Color order: **RGB** (not BGR). This matches the training-time
      convention. Upstream callers that work with OpenCV/camera-native BGR
      data MUST convert before encoding.

    Uses PIL, which is RGB-native on both encode and decode — no channel
    swaps needed.
    """

    def encode(self, frame: np.ndarray, quality: int = 95) -> bytes:
        from PIL import Image

        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[:, :, 0]
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def decode(self, data: bytes) -> np.ndarray:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        return arr


# --- End inlined jpeg_codec ---


class _EnumSafeEncoder(json.JSONEncoder):
    """JSON encoder that converts Enum values to their .value for serialization."""

    def default(self, o):
        if isinstance(o, enum.Enum):
            return o.value
        return super().default(o)


def _to_msgpack_safe(obj: Any) -> Any:
    """Convert config-style Python objects to msgpack-safe containers.

    The policy server boundary should not expose live dataclass instances. Keeping
    the wire format as plain dict/list/scalar data makes client-side history
    config parsing robust to schema drift between policy and robotics packages.
    Numpy arrays are intentionally left alone so ``MsgSerializer`` can encode
    them through the ndarray path.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            field.name: _to_msgpack_safe(getattr(obj, field.name))
            for field in dataclasses.fields(obj)
        }
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, dict):
        return {
            _to_msgpack_safe(key): _to_msgpack_safe(value)
            for key, value in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_to_msgpack_safe(value) for value in obj]
    if isinstance(obj, set):
        return [_to_msgpack_safe(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class MsgSerializer:
    """Serialize/deserialize dicts containing numpy arrays and ModalityConfig.

    When ``jpeg_enabled`` is True, uint8 image arrays (single frame ``(H, W, C)``,
    temporal ``(T, H, W, C)``, or batched ``(B, T, H, W, C)`` with ``C in (1, 3)``)
    are JPEG-compressed instead of using ``np.save``. Non-image arrays are
    unaffected.

    **Color convention: RGB on both sides.**
    Encoding expects RGB uint8 input; decoding produces RGB uint8 output.
    Callers that start from BGR (OpenCV / camera drivers) MUST convert to RGB
    before handing arrays to the client. See ``JpegCodec`` for details.

    Args:
        jpeg_enabled: When True, uint8 image arrays are JPEG-compressed.
        jpeg_quality: JPEG quality level (1-100).
    """

    def __init__(self, jpeg_enabled: bool = True, jpeg_quality: int = 95):
        if not 1 <= jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality must be between 1 and 100, got {jpeg_quality}")
        self.jpeg_enabled = jpeg_enabled
        self.jpeg_quality = jpeg_quality
        self._codec: JpegCodec | None = None

    def _get_codec(self) -> JpegCodec:
        if self._codec is None:
            self._codec = JpegCodec()
        return self._codec

    def to_bytes(self, data: Any) -> bytes:
        return cast(bytes, msgpack.packb(data, default=self._encode_custom_classes))

    def from_bytes(self, data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=self._decode_custom_classes)

    def _decode_custom_classes(self, obj):
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            # Filter unknown fields + fall back to the raw dict on mismatch so
            # a schema drift between server and client does not crash the
            # client. Downstream code that only reads ``delta_indices`` /
            # ``modality_keys`` still works when a dict is returned.
            data = obj["as_json"]
            if isinstance(data, (str, bytes)):
                data = json.loads(data)
            known_fields = {f.name for f in dataclasses.fields(types.ModalityConfig)}
            filtered = {k: v for k, v in data.items() if k in known_fields}
            try:
                return types.ModalityConfig(**filtered)
            except Exception:  # noqa: BLE001 - fall back to dict on any mismatch
                return data
        if "__ndarray_jpeg__" in obj:
            codec = self._get_codec()
            frames = [codec.decode(b) for b in obj["as_jpeg"]]
            stacked = np.stack(frames)
            target_shape = obj["jpeg_shape"]
            return stacked.reshape(target_shape)
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    def _encode_custom_classes(self, obj):
        if isinstance(obj, types.ModalityConfig):
            return {
                "__ModalityConfig_class__": True,
                "as_json": json.dumps(dataclasses.asdict(obj), cls=_EnumSafeEncoder),
            }
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return _to_msgpack_safe(obj)
        if isinstance(obj, np.ndarray):
            if self.jpeg_enabled:
                batch_dims = _is_batched_image_array(obj)
                if _is_image_array(obj):
                    codec = self._get_codec()
                    jpeg_data = [codec.encode(obj, self.jpeg_quality)]
                    return {
                        "__ndarray_jpeg__": True,
                        "as_jpeg": jpeg_data,
                        "jpeg_shape": list(obj.shape),
                    }
                elif batch_dims is not None:
                    codec = self._get_codec()
                    original_shape = obj.shape
                    # Flatten leading batch dims: (B1,..,Bn,H,W,C) -> (N,H,W,C)
                    flat = obj.reshape(-1, *obj.shape[-3:])
                    jpeg_data = [
                        codec.encode(flat[i], self.jpeg_quality) for i in range(flat.shape[0])
                    ]
                    return {
                        "__ndarray_jpeg__": True,
                        "as_jpeg": jpeg_data,
                        "jpeg_shape": list(original_shape),
                    }
            # Default: np.save path
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        if isinstance(obj, enum.Enum):
            return obj.value
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
        policy: base.BasePolicy,
        host: str = "*",
        port: int = 5555,
        api_token: str | None = None,
    ):
        self.policy = policy
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token
        self.serializer = MsgSerializer(jpeg_enabled=False)

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)
        self.register_endpoint("get_action", self.policy.get_action)
        self.register_endpoint("reset", self.policy.reset)
        self.register_endpoint(
            "get_modality_config",
            self._handle_get_modality_config,
            requires_input=False,
        )

    def _handle_get_modality_config(self) -> dict[str, Any]:
        get_modality_config = getattr(self.policy, "get_modality_config", None)
        if get_modality_config is None:
            return {}
        return cast(dict[str, Any], _to_msgpack_safe(get_modality_config()))

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
                request = self.serializer.from_bytes(message)

                # Validate token before processing request
                if not self._validate_token(request):
                    self.socket.send(
                        self.serializer.to_bytes({"error": "Unauthorized: Invalid API token"})
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
                self.socket.send(self.serializer.to_bytes(result))
            except Exception as e:  # noqa: BLE001
                print(f"Error in server: {e}")
                import traceback
                tb = traceback.format_exc()
                print(tb)
                self.socket.send(self.serializer.to_bytes({"error": f"{e}\n{tb}"}))

    @staticmethod
    def start_server(policy: base.BasePolicy, port: int, api_token: str | None = None):
        server = PolicyServer(policy, port=port, api_token=api_token)
        server.run()


class PolicyClient(base.BasePolicy):
    """ZMQ policy client that talks to :class:`PolicyServer`.

    The client is both a transport (``call_endpoint``, ``ping``, ``close``,
    context manager) and a :class:`base.BasePolicy` so it can be dropped into
    wrappers like ``Gr00tRTCPolicyWrapper`` / ``Gr00tSimPolicyWrapper``.

    Subclasses (e.g. UMI's ``Gr00tPolicyClient``) override
    ``get_action`` to add client-side preprocessing; the default
    ``_get_action`` simply forwards the observation to the server.
    """

    def __init__(
        self,
        address: str = "tcp://localhost:5555",
        timeout_ms: int = 15000,
        api_token: str | None = None,
        verbose: bool = False,
        strict: bool = False,
        jpeg_enabled: bool = True,
        jpeg_quality: int = 95,
    ):
        """Initialize the policy client.

        Args:
            address: Server address (e.g. ``tcp://localhost:5555`` or
                ``ipc:///tmp/policy.ipc``).
            timeout_ms: Send/receive timeout in milliseconds. ``0`` disables.
            api_token: Optional API token for authentication.
            verbose: Enable verbose logging.
            strict: Whether to enforce ``check_observation`` / ``check_action``.
                The default client has no schema, so defaults to ``False``.
            jpeg_enabled: Whether outgoing uint8 image arrays are
                JPEG-compressed in the serializer.
            jpeg_quality: JPEG quality level (1-100).
        """
        super().__init__(strict=strict)
        self.context = zmq.Context()
        self.address = address
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.verbose = verbose
        self.serializer = MsgSerializer(
            jpeg_enabled=jpeg_enabled,
            jpeg_quality=jpeg_quality,
        )
        self._init_socket()

    def _init_socket(self):
        """Initialize the ZMQ REQ socket with current settings."""
        self.socket = self.context.socket(zmq.REQ)
        if self.timeout_ms > 0:
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.address)

    def _reset_socket(self):
        """Recreate the socket after an error.

        After a ``RCVTIMEO`` timeout on a REQ socket the send/recv state
        machine is broken and the socket must be recreated.
        """
        try:
            self.socket.close()
        except Exception:  # noqa: BLE001
            pass
        self._init_socket()

    def _log(self, message: str):
        """Log a message when ``verbose=True`` (used by subclasses)."""
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")

    def ping(self) -> bool:
        """Ping the server; return ``True`` iff it responds."""
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except (zmq.error.ZMQError, TimeoutError, RuntimeError):
            self._reset_socket()
            return False

    def kill_server(self):
        """Send a kill signal to the server."""
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> Any:
        """Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.

        Raises:
            TimeoutError: If the request times out.
            RuntimeError: If the server returns an error.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        self._log(f"Sending request to endpoint: {endpoint}")

        try:
            self.socket.send(self.serializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again as exc:
            raise TimeoutError(f"Request to endpoint '{endpoint}' timed out") from exc

        if message == b"ERROR":
            raise RuntimeError(
                "Server error. Make sure the correct policy server is running."
            )
        response = self.serializer.from_bytes(message)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")

        self._log(f"Received response from endpoint: {endpoint}")
        return response

    def close(self):
        """Close the socket and release the ZMQ context."""
        self._log("Closing client")
        try:
            self.socket.close()
            self.context.term()
        except Exception as e:  # noqa: BLE001
            self._log(f"Error during close: {e}")

    def __del__(self):
        """Best-effort cleanup on destruction."""
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)  # Convert list (from msgpack) to tuple of (action, info)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self) -> dict[str, types.ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def check_observation(self, observation: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_observation is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."  # noqa: E501
        )

    def check_action(self, action: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_action is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."  # noqa: E501
        )
