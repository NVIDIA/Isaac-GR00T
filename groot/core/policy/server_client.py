# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, wait
import contextlib
import dataclasses
from dataclasses import dataclass
import enum
import io
import json
import math
import time
from typing import Any, cast

import msgpack
import numpy as np
import zmq

from groot.core.data import types
from groot.core.policy import base

RPC_TIMING_INFO_KEY = "_rpc_timing"
JPEG_VIDEO_VERSION = 1
JPEG_VIDEO_MARKER = "__camera_jpeg_video__"


@dataclass(frozen=True)
class JpegVideo:
    """Time-major JPEGs, with the source's decoding convention per frame.

    ``rgb`` is a standard device JPEG (OpenCV decode needs BGR -> RGB).
    ``opencv_legacy`` is the camera publisher's historical RGB-array encode
    without a channel swap (OpenCV decode already gives the intended RGB).
    These bytes must not be reused after any pixel transformation.
    """

    frames: tuple[bytes, ...]
    encodings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frames, tuple) or not isinstance(self.encodings, tuple):
            raise TypeError("JPEG frames and encodings must be immutable tuples")
        if not self.frames or len(self.frames) != len(self.encodings):
            raise ValueError("JPEG frames and encodings must have equal nonzero lengths")
        for frame, encoding in zip(self.frames, self.encodings, strict=True):
            if not isinstance(frame, bytes) or not frame.startswith(b"\xff\xd8"):
                raise ValueError("camera frame must contain JPEG bytes")
            if encoding not in ("rgb", "opencv_legacy"):
                raise ValueError(f"unsupported camera JPEG encoding: {encoding!r}")

    def decode_frame(self, index: int) -> np.ndarray:
        """Decode an RGB frame, possibly as a view, for stack_decoded_frames."""
        import cv2
        from PIL import Image

        frame = self.frames[index]
        # Retain Pillow's image-size safeguards before OpenCV allocates pixels.
        with Image.open(io.BytesIO(frame)) as image:
            if image.format != "JPEG" or image.mode not in ("RGB", "L"):
                raise ValueError("camera JPEG must be RGB or grayscale")
        decoded = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("camera JPEG could not be decoded")
        if self.encodings[index] == "rgb":
            # Batch assembly materializes RGB once; retain the decoder's buffer here.
            decoded = decoded[..., ::-1]
        return decoded

    def stack_decoded_frames(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        """Assemble decode_frame results using one contiguous RGB batch allocation."""
        import cv2

        frames = list(frames)
        if len(frames) != len(self.frames):
            raise ValueError("decoded frame count does not match JPEG history")
        shape = frames[0].shape
        if any(frame.shape != shape for frame in frames):
            raise ValueError("all decoded frames must have the same shape")
        batch = np.empty((len(frames), *shape), dtype=np.uint8)
        for index, frame in enumerate(frames):
            if self.encodings[index] == "rgb":
                # Convert the contiguous BGR view into the final slice without an
                # intermediate RGB allocation or a negative-stride NumPy copy.
                cv2.cvtColor(frame[..., ::-1], cv2.COLOR_BGR2RGB, dst=batch[index])
            else:
                batch[index] = frame
        return batch

    def decode(self) -> np.ndarray:
        """Materialize RGB uint8 (T,H,W,3), rejecting inconsistent frame sizes."""
        return self.stack_decoded_frames(
            self.decode_frame(index) for index in range(len(self.frames))
        )


class PolicyServerUnhealthyError(RuntimeError):
    """The server has latched a permanent failure and must be restarted."""


def _attach_server_rpc_timing(
    result: Any,
    *,
    request_id: int | None,
    server_processing_ms: float,
    server_request_deserialize_ms: float | None = None,
) -> Any:
    """Attach transport metadata to a get_action result without mutating policy info."""
    if not isinstance(result, list | tuple) or len(result) != 2:
        return result

    action, info = result
    if not isinstance(info, dict):
        return result

    timed_info = dict(info)
    rpc_timing = timed_info.get(RPC_TIMING_INFO_KEY, {})
    rpc_timing = dict(rpc_timing) if isinstance(rpc_timing, dict) else {}
    rpc_timing["server_processing_ms"] = server_processing_ms
    if server_request_deserialize_ms is not None:
        rpc_timing["server_request_deserialize_ms"] = server_request_deserialize_ms
    if request_id is not None:
        rpc_timing["request_id"] = request_id
    timed_info[RPC_TIMING_INFO_KEY] = rpc_timing

    if isinstance(result, tuple):
        return action, timed_info
    return [action, timed_info]


def _get_response_rpc_timing(response: Any) -> dict[str, Any]:
    """Return RPC metadata from a get_action response, if the server supplied it."""
    if not isinstance(response, list | tuple) or len(response) != 2:
        return {}
    info = response[1]
    if not isinstance(info, dict):
        return {}
    rpc_timing = info.get(RPC_TIMING_INFO_KEY)
    return dict(rpc_timing) if isinstance(rpc_timing, dict) else {}


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

    Pillow is the reference codec. OpenCV accelerates RGB/grayscale JPEGs with
    explicit color conversion and unchanged JPEG options. Other image formats
    and modes retain the Pillow path.
    """

    def encode(self, frame: np.ndarray, quality: int = 95, *, use_opencv: bool = False) -> bytes:
        from PIL import Image

        if (
            use_opencv
            and _is_image_array(frame)
            and frame.shape[0] > 0
            and frame.shape[1] > 0
            and isinstance(quality, int)
            and 1 <= quality <= 100
        ):
            try:
                import cv2
            except ImportError:
                pass
            else:
                try:
                    image = (
                        frame[:, :, 0]
                        if frame.shape[-1] == 1
                        else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    )
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        image,
                        [
                            cv2.IMWRITE_JPEG_QUALITY,
                            quality,
                            cv2.IMWRITE_JPEG_PROGRESSIVE,
                            0,
                            cv2.IMWRITE_JPEG_OPTIMIZE,
                            0,
                        ],
                    )
                    if ok:
                        return encoded.tobytes()
                except cv2.error:
                    pass
                # Optional codec missing/failing: preserve Pillow's output or
                # original exception. Never change quality to make encoding work.
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[:, :, 0]
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def decode(self, data: bytes) -> np.ndarray:
        from PIL import Image

        # Inspect with Pillow first to retain format/mode detection and its
        # decompression-bomb checks. IMREAD_UNCHANGED also ignores EXIF rotation,
        # just like the legacy decoder. CMYK, non-JPEG and truncated data fall back.
        with Image.open(io.BytesIO(data)) as img:
            if img.format == "JPEG" and img.mode in ("RGB", "L") and data.endswith(b"\xff\xd9"):
                try:
                    import cv2
                except ImportError:
                    pass
                else:
                    try:
                        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
                        channels = 3 if img.mode == "RGB" else 1
                        shape = (
                            (img.height, img.width, 3) if channels == 3 else (img.height, img.width)
                        )
                        if (
                            decoded is not None
                            and decoded.dtype == np.uint8
                            and decoded.shape == shape
                        ):
                            return (
                                cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                                if channels == 3
                                else decoded[:, :, np.newaxis]
                            )
                    except cv2.error:
                        pass
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
        return {_to_msgpack_safe(key): _to_msgpack_safe(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
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

    Explicit ``JpegVideo`` inputs bypass encoding and carry their source color
    convention. They decode to RGB uint8 ``(T, H, W, 3)`` before policy handling;
    clients must negotiate ``get_transport_capabilities`` before sending them.

    **Color convention: RGB on both sides.**
    Encoding expects RGB uint8 input; decoding produces RGB uint8 output.
    Callers that start from BGR (OpenCV / camera drivers) MUST convert to RGB
    before handing arrays to the client. See ``JpegCodec`` for details.

    Args:
        jpeg_enabled: When True, uint8 image arrays are JPEG-compressed.
        jpeg_quality: JPEG quality level (1-100).
        jpeg_decode_workers: Maximum workers for independent JPEG history frames.
            One preserves serial decoding.
        jpeg_encode_workers: Maximum workers for independent JPEG history frames.
            One preserves serial Pillow encoding. Multiple workers use OpenCV
            without changing JPEG quality, frame order or the wire format.
    """

    def __init__(
        self,
        jpeg_enabled: bool = True,
        jpeg_quality: int = 95,
        jpeg_decode_workers: int = 4,
        jpeg_encode_workers: int = 4,
    ):
        if not 1 <= jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality must be between 1 and 100, got {jpeg_quality}")
        self.jpeg_enabled = jpeg_enabled
        self.jpeg_quality = jpeg_quality
        if not isinstance(jpeg_decode_workers, int) or jpeg_decode_workers < 1:
            raise ValueError("jpeg_decode_workers must be a positive integer")
        self.jpeg_decode_workers = jpeg_decode_workers
        if not isinstance(jpeg_encode_workers, int) or jpeg_encode_workers < 1:
            raise ValueError("jpeg_encode_workers must be a positive integer")
        self.jpeg_encode_workers = jpeg_encode_workers
        self._codec: JpegCodec | None = None
        self._decode_executor: ThreadPoolExecutor | None = None
        self._encode_executor: ThreadPoolExecutor | None = None

    def close(self) -> None:
        """Release codec workers after the last sequential RPC."""
        if self._encode_executor is not None:
            self._encode_executor.shutdown(wait=True)
            self._encode_executor = None
        if self._decode_executor is not None:
            self._decode_executor.shutdown(wait=True)
            self._decode_executor = None

    def _get_codec(self) -> JpegCodec:
        if self._codec is None:
            self._codec = JpegCodec()
        return self._codec

    def to_bytes(self, data: Any) -> bytes:
        jpeg_batches = self._encode_jpeg_batches(data) if self.jpeg_enabled else {}

        def encode(obj):
            if isinstance(obj, np.ndarray) and id(obj) in jpeg_batches:
                return jpeg_batches[id(obj)]
            return self._encode_custom_classes(obj)

        return cast(bytes, msgpack.packb(data, default=encode))

    def _encode_jpeg_batches(self, data: Any) -> dict[int, dict[str, Any]]:
        if self.jpeg_encode_workers == 1:
            return {}
        arrays = {}
        pending = [data]
        seen = set()
        # Discover histories across cameras before waiting on any one of them.
        # Keep this request-local; prompt changes and reused arrays must never
        # pick up encoded data from an earlier RPC.
        while pending:
            obj = pending.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if isinstance(obj, np.ndarray):
                if _is_batched_image_array(obj) is not None and math.prod(obj.shape[:-3]) > 1:
                    arrays[id(obj)] = obj
            elif isinstance(obj, types.ModalityConfig):
                continue  # This object's existing wire representation is JSON.
            elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                pending.extend(getattr(obj, field.name) for field in dataclasses.fields(obj))
            elif isinstance(obj, enum.Enum):
                pending.append(obj.value)
            elif isinstance(obj, dict):
                pending.extend(obj.values())
            elif isinstance(obj, list | tuple):
                pending.extend(obj)
        if len(arrays) < 2:
            return {}

        batches = [obj.reshape(-1, *obj.shape[-3:]) for obj in arrays.values()]
        encoded = iter(self._encode_frames([frame for batch in batches for frame in batch]))
        return {
            key: {
                "__ndarray_jpeg__": True,
                "as_jpeg": [next(encoded) for _ in batch],
                "jpeg_shape": list(obj.shape),
            }
            for (key, obj), batch in zip(arrays.items(), batches, strict=False)
        }

    def _encode_frames(self, frames) -> list[bytes]:
        if self._encode_executor is None:
            self._encode_executor = ThreadPoolExecutor(
                max_workers=self.jpeg_encode_workers,
                thread_name_prefix="policy-jpeg-encode",
            )
        futures = [self._encode_executor.submit(self._encode_frame, frame) for frame in frames]
        try:
            return [future.result() for future in futures]
        except BaseException:
            # A failed call must not leave workers reading caller-owned images.
            for future in futures:
                future.cancel()
            wait(futures)
            raise

    def from_bytes(self, data: bytes) -> Any:
        if self.jpeg_decode_workers == 1:
            return msgpack.unpackb(data, object_hook=self._decode_custom_classes)
        batches = {}
        ready = {}

        def restore(obj):
            root = [obj]
            containers = [root]
            while containers:
                container = containers.pop()
                children = (
                    container.items() if isinstance(container, dict) else enumerate(container)
                )
                for key, value in children:
                    if isinstance(value, dict) and id(value) in batches:
                        if id(value) not in ready:
                            ready[id(value)] = self._decode_custom_classes(value)
                        container[key] = ready[id(value)]
                    elif isinstance(value, dict | list):
                        containers.append(value)
            return root[0]

        def decode(obj):
            if "__ModalityConfig_class__" in obj or "__ndarray_class__" in obj:
                # These constructors must see decoded children, as with the
                # original bottom-up msgpack hook (including legacy config dicts).
                return self._decode_custom_classes(restore(obj))
            if "__ndarray_jpeg__" in obj and len(obj["as_jpeg"]) > 1:
                batches[id(obj)] = obj
                return obj
            return self._decode_custom_classes(obj)

        unpacked = msgpack.unpackb(data, object_hook=decode)
        pending = [obj for key, obj in batches.items() if key not in ready]
        if pending:
            executor = self._get_decode_executor()
            decoded = executor.map(
                self._get_codec().decode,
                [frame for obj in pending for frame in obj["as_jpeg"]],
            )
            try:
                # Consume in wire order and release each frame batch after stacking.
                # Validate even overwritten dict entries, not just surviving leaves.
                for obj in pending:
                    ready[id(obj)] = np.stack([next(decoded) for _ in obj["as_jpeg"]]).reshape(
                        obj["jpeg_shape"]
                    )
            except BaseException:
                decoded.close()
                executor.shutdown(wait=True, cancel_futures=True)
                self._decode_executor = None
                raise
            finally:
                decoded.close()
        return restore(unpacked) if batches else unpacked

    def _get_decode_executor(self) -> ThreadPoolExecutor:
        if self._decode_executor is None:
            self._decode_executor = ThreadPoolExecutor(
                max_workers=self.jpeg_decode_workers,
                thread_name_prefix="policy-jpeg-decode",
            )
        return self._decode_executor

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        return self._get_codec().encode(frame, self.jpeg_quality, use_opencv=True)

    def _decode_custom_classes(self, obj):
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            # Filter unknown fields + fall back to the raw dict on mismatch so
            # a schema drift between server and client does not crash the
            # client. Downstream code that only reads ``delta_indices`` /
            # ``modality_keys`` still works when a dict is returned.
            data = obj["as_json"]
            if isinstance(data, str | bytes):
                data = json.loads(data)
            known_fields = {f.name for f in dataclasses.fields(types.ModalityConfig)}
            filtered = {k: v for k, v in data.items() if k in known_fields}
            try:
                return types.ModalityConfig(**filtered)
            except Exception:  # noqa: BLE001 - fall back to dict on any mismatch
                return data
        if JPEG_VIDEO_MARKER in obj:
            if (
                type(obj[JPEG_VIDEO_MARKER]) is not int
                or obj[JPEG_VIDEO_MARKER] != JPEG_VIDEO_VERSION
            ):
                raise ValueError("unsupported camera JPEG video version")
            video = JpegVideo(tuple(obj["frames"]), tuple(obj["encodings"]))
            if self.jpeg_decode_workers == 1 or len(video.frames) == 1:
                return video.decode()
            executor = self._get_decode_executor()
            decoded = executor.map(video.decode_frame, range(len(video.frames)))
            try:
                return video.stack_decoded_frames(decoded)
            except BaseException:
                decoded.close()
                executor.shutdown(wait=True, cancel_futures=True)
                self._decode_executor = None
                raise
            finally:
                decoded.close()
        if "__ndarray_jpeg__" in obj:
            codec = self._get_codec()
            encoded_frames = obj["as_jpeg"]
            if self.jpeg_decode_workers == 1 or len(encoded_frames) <= 1:
                frames = [codec.decode(frame) for frame in encoded_frames]
            else:
                # Pillow decoding releases the GIL. Ordered map preserves B/T order;
                # each call owns its Image/BytesIO, with no shared decoder state.
                frames = list(self._get_decode_executor().map(codec.decode, encoded_frames))
            stacked = np.stack(frames)
            target_shape = obj["jpeg_shape"]
            return stacked.reshape(target_shape)
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    def _encode_custom_classes(self, obj):
        if isinstance(obj, JpegVideo):
            return {
                JPEG_VIDEO_MARKER: JPEG_VIDEO_VERSION,
                "frames": obj.frames,
                "encodings": obj.encodings,
            }
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
                    if self.jpeg_encode_workers == 1 or len(flat) <= 1:
                        jpeg_data = [codec.encode(frame, self.jpeg_quality) for frame in flat]
                    else:
                        jpeg_data = self._encode_frames(flat)
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
    """An inference server that spin up a ZeroMQ socket and listen for incoming requests.

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
        self.register_endpoint(
            "get_transport_capabilities",
            self._handle_get_transport_capabilities,
            requires_input=False,
        )
        self.register_endpoint("kill", self._kill_server, requires_input=False)
        self.register_endpoint("get_action", self._handle_get_action)
        self.register_endpoint("warmup", self.policy.get_action)
        self.register_endpoint("reset", self.policy.reset)
        self.register_endpoint(
            "get_modality_config",
            self._handle_get_modality_config,
            requires_input=False,
        )
        self.register_endpoint(
            "get_runtime_contract",
            self._handle_get_runtime_contract,
            requires_input=False,
        )
        self.register_endpoint(
            "get_readiness",
            self._handle_get_readiness,
            requires_input=False,
        )

    @staticmethod
    def _handle_get_transport_capabilities() -> dict[str, int]:
        return {"camera_jpeg_video": JPEG_VIDEO_VERSION}

    def _handle_get_modality_config(self) -> dict[str, Any]:
        get_modality_config = getattr(self.policy, "get_modality_config", None)
        if get_modality_config is None:
            return {}
        return cast(dict[str, Any], _to_msgpack_safe(get_modality_config()))

    def _handle_get_runtime_contract(self) -> dict[str, Any]:
        """Return benchmark backend state through transparent policy wrappers."""
        policy: Any = self.policy
        visited: set[int] = set()
        while policy is not None and id(policy) not in visited:
            visited.add(id(policy))
            contract = getattr(policy, "benchmark_runtime_contract", None)
            if callable(contract):
                contract = contract()
            if isinstance(contract, dict):
                return cast(dict[str, Any], _to_msgpack_safe(contract))
            policy = getattr(policy, "policy", None)
        return {}

    def _handle_get_readiness(self) -> dict[str, Any]:
        """Return action-serving readiness through transparent wrappers."""
        policy: Any = self.policy
        visited: set[int] = set()
        while policy is not None and id(policy) not in visited:
            visited.add(id(policy))
            readiness = getattr(policy, "server_readiness", None)
            readiness = readiness() if callable(readiness) else readiness
            if isinstance(readiness, dict):
                return cast(dict[str, Any], _to_msgpack_safe(readiness))
            policy = getattr(policy, "policy", None)
        return {
            "schema_version": "policy-server-readiness-v1",
            "status": "ready",
            "healthy": True,
            "ready": True,
            "message": "policy is ready",
        }

    def _handle_get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Refuse to return a dispatchable action until serving is ready."""
        readiness = self._handle_get_readiness()
        if readiness.get("healthy") is not True:
            raise PolicyServerUnhealthyError(
                "policy server is unhealthy and must be restarted before serving actions: "
                f"{readiness.get('message', readiness)}"
            )
        if readiness.get("ready") is not True:
            raise RuntimeError(
                "policy server requires an explicit real-observation warmup before serving "
                "actions; call PolicyClient.warmup_and_require_ready() before arming"
            )
        return self.policy.get_action(observation, options)

    def _kill_server(self):
        """Kill the server."""
        self.running = False

    def _handle_ping(self) -> dict:
        """Report liveness plus action-serving health/readiness."""
        readiness = self._handle_get_readiness()
        return {
            **readiness,
            "status": "ok" if readiness.get("ready") else readiness.get("status", "warming"),
        }

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _validate_token(self, request: dict) -> bool:
        """Validate the API token in the request."""
        if self.api_token is None:
            return True  # No token required
        return request.get("api_token") == self.api_token

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is listening on {addr}; check get_readiness before arming")  # noqa: T201
        try:
            while self.running:
                try:
                    message = self.socket.recv()
                    deserialize_start = time.perf_counter()
                    request = self.serializer.from_bytes(message)
                    server_request_deserialize_ms = (time.perf_counter() - deserialize_start) * 1000

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
                    handler_start = time.perf_counter()
                    result = (
                        handler.handler(**request.get("data", {}))
                        if handler.requires_input
                        else handler.handler()
                    )
                    server_processing_ms = (time.perf_counter() - handler_start) * 1000
                    if endpoint == "get_action":
                        result = _attach_server_rpc_timing(
                            result,
                            request_id=request.get("request_id"),
                            server_processing_ms=server_processing_ms,
                            server_request_deserialize_ms=server_request_deserialize_ms,
                        )
                    self.socket.send(self.serializer.to_bytes(result))
                except Exception as e:  # noqa: BLE001
                    print(f"Error in server: {e}")  # noqa: T201
                    import traceback

                    tb = traceback.format_exc()
                    print(tb)  # noqa: T201
                    self.socket.send(
                        self.serializer.to_bytes(
                            {
                                "error": f"{e}\n{tb}",
                                "error_type": type(e).__name__,
                            }
                        )
                    )
        finally:
            self.serializer.close()

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
        self._next_request_id = 0
        self.last_rpc_timing: dict[str, Any] = {}
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
        with contextlib.suppress(Exception):
            self.socket.close()
        self._init_socket()

    def _log(self, message: str):
        """Log a message when ``verbose=True`` (used by subclasses)."""
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")  # noqa: T201

    def ping(self) -> bool:
        """Ping the server; return ``True`` iff it responds and is healthy."""
        try:
            response = self.call_endpoint("ping", requires_input=False)
            return not (isinstance(response, dict) and response.get("healthy") is False)
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
        request_id = self._next_request_id
        self._next_request_id += 1
        request: dict = {"endpoint": endpoint, "request_id": request_id}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        self._log(f"Sending request to endpoint: {endpoint}")

        rpc_start = time.perf_counter()
        serialize_start = time.perf_counter()
        request_bytes = self.serializer.to_bytes(request)
        request_serialize_ms = (time.perf_counter() - serialize_start) * 1000

        send_recv_start = time.perf_counter()
        try:
            self.socket.send(request_bytes)
            message = self.socket.recv()
        except zmq.error.Again as exc:
            self.last_rpc_timing = {
                "request_id": request_id,
                "request_serialize_ms": request_serialize_ms,
                "rpc_total_ms": (time.perf_counter() - rpc_start) * 1000,
                "timed_out": True,
            }
            self._reset_socket()
            raise TimeoutError(f"Request to endpoint '{endpoint}' timed out") from exc
        send_recv_ms = (time.perf_counter() - send_recv_start) * 1000

        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure the correct policy server is running.")
        deserialize_start = time.perf_counter()
        response = self.serializer.from_bytes(message)
        response_deserialize_ms = (time.perf_counter() - deserialize_start) * 1000
        rpc_total_ms = (time.perf_counter() - rpc_start) * 1000

        server_rpc_timing = _get_response_rpc_timing(response)
        echoed_request_id = server_rpc_timing.get("request_id")
        if echoed_request_id is not None and echoed_request_id != request_id:
            self._reset_socket()
            raise RuntimeError(
                f"RPC response ID mismatch: sent {request_id}, received {echoed_request_id}"
            )

        server_processing_ms = server_rpc_timing.get("server_processing_ms")
        server_request_deserialize_ms = server_rpc_timing.get("server_request_deserialize_ms")
        transport_overhead_ms = None
        if isinstance(server_processing_ms, int | float):
            # Includes network/ZMQ time and server-side request/response serialization.
            # Keep the signed nesting residual. Clamping would make invalid
            # client/server timing order look like a valid zero-overhead RPC.
            transport_overhead_ms = send_recv_ms - server_processing_ms
        self.last_rpc_timing = {
            "request_id": request_id,
            "request_serialize_ms": request_serialize_ms,
            "send_recv_ms": send_recv_ms,
            "response_deserialize_ms": response_deserialize_ms,
            "client_serde_ms": request_serialize_ms + response_deserialize_ms,
            "rpc_total_ms": rpc_total_ms,
            "server_processing_ms": server_processing_ms,
            "server_request_deserialize_ms": server_request_deserialize_ms,
            "transport_overhead_ms": transport_overhead_ms,
            "request_bytes": len(request_bytes),
            "response_bytes": len(message),
            "request_id_verified": echoed_request_id is not None,
            "timed_out": False,
        }

        if isinstance(response, dict) and "error" in response:
            message = f"Server error: {response['error']}"
            if response.get("error_type") == PolicyServerUnhealthyError.__name__:
                raise PolicyServerUnhealthyError(message)
            raise RuntimeError(message)

        self._log(f"Received response from endpoint: {endpoint}")
        return response

    def close(self):
        """Close the socket and release the ZMQ context."""
        self._log("Closing client")
        try:
            self.serializer.close()
            self.socket.close()
            self.context.term()
        except Exception as e:  # noqa: BLE001
            self._log(f"Error during close: {e}")

    def __del__(self):
        """Best-effort cleanup on destruction."""
        with contextlib.suppress(Exception):
            self.close()

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

    def get_runtime_contract(self) -> dict[str, Any]:
        """Return the server's benchmark backend state, if it exposes one."""
        response = self.call_endpoint("get_runtime_contract", requires_input=False)
        return dict(response) if isinstance(response, dict) else {}

    def get_readiness(self) -> dict[str, Any]:
        """Return whether the policy is ready to serve actions safely."""
        response = self.call_endpoint("get_readiness", requires_input=False)
        return dict(response) if isinstance(response, dict) else {}

    def warmup(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run one non-dispatched policy request for graph capture qualification."""
        response = self.call_endpoint(
            "warmup",
            {"observation": observation, "options": options},
        )
        return tuple(response)

    def warmup_and_require_ready(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Require readiness; force a discarded request when the task changes."""
        readiness = self.get_readiness()
        if readiness.get("healthy") is not True:
            raise PolicyServerUnhealthyError(
                "policy server is unhealthy before warmup: "
                f"{readiness.get('message', readiness)}"
            )
        if force or readiness.get("ready") is not True:
            try:
                self.warmup(observation, options)
            except (TimeoutError, RuntimeError, zmq.error.ZMQError) as error:
                try:
                    readiness = self.get_readiness()
                except (TimeoutError, RuntimeError, zmq.error.ZMQError) as readiness_error:
                    raise error from readiness_error
                if readiness.get("healthy") is not True:
                    raise PolicyServerUnhealthyError(
                        "policy server became unhealthy during warmup: "
                        f"{readiness.get('message', readiness)}"
                    ) from error
                raise
            readiness = self.get_readiness()
        if readiness.get("healthy") is not True:
            raise PolicyServerUnhealthyError(
                "policy server became unhealthy during warmup: "
                f"{readiness.get('message', readiness)}"
            )
        if readiness.get("ready") is not True:
            raise RuntimeError(
                "policy server did not become ready after warmup: "
                f"{readiness.get('message', readiness)}"
            )
        return readiness

    def check_observation(self, observation: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_observation is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."  # noqa: E501
        )

    def check_action(self, action: dict[str, Any]) -> None:
        raise NotImplementedError(
            "check_action is not implemented. Please use `strict=False` to disable strict mode or implement this method in the subclass."  # noqa: E501
        )
