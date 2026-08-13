#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Serve a GR00T N2 checkpoint through the Genie Sim websocket protocol."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import io
import logging
from pathlib import Path
import random
import socket
import traceback
from typing import Any

import numpy as np
from PIL import Image
import tyro

from groot.core.data import schema

logger = logging.getLogger(__name__)

TARGET_VIDEO_SIZE = (256, 256)
IMAGE_ALIASES = {
    "top_head": "top_head",
    "top_head_pad_res256_freq10": "top_head",
    "hand_left": "hand_left",
    "hand_left_pad_res256_freq10": "hand_left",
    "hand_right": "hand_right",
    "hand_right_pad_res256_freq10": "hand_right",
}
STATE_SLICES = {
    "left_arm_joint_position": slice(0, 7),
    "right_arm_joint_position": slice(7, 14),
    "left_effector_position": slice(14, 15),
    "right_effector_position": slice(15, 16),
    "head_position": slice(16, 18),
    "waist_pitch": slice(18, 19),
    "waist_lift": slice(19, 20),
}


def _normalize_embodiment_tag(tag: str) -> schema.EmbodimentTag:
    try:
        return schema.EmbodimentTag[tag]
    except KeyError:
        return schema.EmbodimentTag(tag)


def _ensure_ndarray(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, dict):
        raise TypeError(f"Expected an array-like value, got dict keys: {list(value)}")
    arr = value if isinstance(value, np.ndarray) else np.asarray(value)
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr


def _decode_maybe_b64_ndarray(value: Any) -> np.ndarray:
    if isinstance(value, str):
        return np.load(io.BytesIO(base64.b64decode(value)), allow_pickle=False)
    return _ensure_ndarray(value)


def _prepare_frame_hwc_u8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    elif frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _ensure_video_thwc_u8(value: Any) -> np.ndarray:
    arr = _ensure_ndarray(value)
    if arr.ndim in (2, 3):
        return _prepare_frame_hwc_u8(arr)[np.newaxis, ...]
    if arr.ndim != 4:
        raise ValueError(f"Video must be HWC, THWC, or TCHW; got {arr.shape}")
    if arr.shape[-1] in (3, 4):
        video = arr[..., :3]
    elif arr.shape[1] == 3:
        video = np.transpose(arr, (0, 2, 3, 1))
    else:
        raise ValueError(f"Unrecognized video layout {arr.shape}")
    if video.dtype != np.uint8:
        video = np.clip(video, 0, 255).astype(np.uint8)
    return video


def _resize_keep_aspect(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    resized = Image.fromarray(frame).resize((new_w, new_h), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.uint8)
    pad_h = target_h - arr.shape[0]
    pad_w = target_w - arr.shape[1]
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    return np.pad(
        arr,
        ((pad_top, pad_h - pad_top), (pad_left, pad_w - pad_left), (0, 0)),
        mode="constant",
        constant_values=0,
    ).astype(np.uint8)


def _match_horizon(arr: np.ndarray, horizon: int) -> np.ndarray:
    if arr.shape[0] == horizon:
        return arr
    if arr.shape[0] > horizon:
        return arr[-horizon:]
    if arr.shape[0] == 0:
        raise ValueError("Cannot expand an empty temporal sequence")
    pad = np.repeat(arr[-1:], horizon - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def _images_to_video_payload(
    images: dict[str, Any],
    video_keys: list[str],
    horizon: int,
) -> dict[str, np.ndarray]:
    videos = {}
    for video_key in video_keys:
        source_key = IMAGE_ALIASES.get(video_key, video_key)
        if source_key not in images:
            raise ValueError(f"Payload missing images.{source_key} for policy key {video_key}")
        raw = _decode_maybe_b64_ndarray(images[source_key])
        video = _ensure_video_thwc_u8(raw)
        video = np.stack(
            [
                _resize_keep_aspect(frame, TARGET_VIDEO_SIZE[0], TARGET_VIDEO_SIZE[1])
                for frame in video
            ],
            axis=0,
        )
        videos[video_key] = _match_horizon(video, horizon)
    return videos


def _state_dict_to_arrays(
    state: dict[str, Any],
    state_keys: list[str],
    horizon: int,
) -> dict[str, np.ndarray]:
    arrays = {}
    for key in state_keys:
        if key not in state:
            arrays[key] = np.zeros((horizon, 1), dtype=np.float32)
            continue
        arr = _ensure_ndarray(state[key], dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(f"state.{key} must be D or T,D; got {arr.shape}")
        arrays[key] = _match_horizon(arr, horizon).astype(np.float32)
    return arrays


def _flat_state_to_arrays(
    flat_state: list[Any] | np.ndarray,
    state_keys: list[str],
    horizon: int,
) -> dict[str, np.ndarray]:
    arr = _ensure_ndarray(flat_state, dtype=np.float32).reshape(-1)
    arrays = {}
    for key in state_keys:
        state_slice = STATE_SLICES.get(key)
        if state_slice is None:
            value = np.zeros((1, 1), dtype=np.float32)
        elif arr.size >= state_slice.stop:
            value = arr[state_slice].reshape(1, -1)
        elif state_slice.start < 16:
            raise ValueError(f"Flat Genie Sim state needs at least {state_slice.stop} dims")
        else:
            value = np.zeros((1, state_slice.stop - state_slice.start), dtype=np.float32)
        arrays[key] = np.repeat(value.astype(np.float32), horizon, axis=0)
    return arrays


def payload_to_observation(policy: Any, payload: dict[str, Any]) -> dict[str, Any]:
    modality_configs = policy.get_modality_config()
    if "images" not in payload or "state" not in payload:
        raise ValueError("Payload must contain images and state")

    video_horizon = len(modality_configs["video"].delta_indices)
    state_horizon = len(modality_configs["state"].delta_indices)
    language_horizon = len(modality_configs["language"].delta_indices)
    language_key = modality_configs["language"].modality_keys[0]
    prompt = payload.get("prompt") or payload.get("task_name") or ""
    if isinstance(prompt, bytes | bytearray):
        prompt = prompt.decode("utf-8", errors="replace")

    video_keys = list(modality_configs["video"].modality_keys)
    state_keys = list(modality_configs["state"].modality_keys)
    videos = _images_to_video_payload(payload["images"], video_keys, video_horizon)
    raw_state = payload["state"]
    states = (
        _state_dict_to_arrays(raw_state, state_keys, state_horizon)
        if isinstance(raw_state, dict)
        else _flat_state_to_arrays(raw_state, state_keys, state_horizon)
    )

    return {
        "video": {key: value[np.newaxis, ...] for key, value in videos.items()},
        "state": {key: value[np.newaxis, ...] for key, value in states.items()},
        "language": {language_key: [[str(prompt)] * language_horizon]},
    }


def _action_part_2d(action: dict[str, np.ndarray], key: str, default_cols: int) -> np.ndarray:
    value = action.get(key)
    if value is None:
        value = action.get(f"action.{key}")
    if value is None:
        return np.zeros((1, default_cols), dtype=np.float32)
    arr = _ensure_ndarray(value, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"action.{key} expected T,D or B,T,D, got {arr.shape}")
    return arr.astype(np.float32)


def action_to_wire(action: dict[str, np.ndarray], horizon: int | None = None) -> dict[str, Any]:
    parts = [
        _action_part_2d(action, "left_arm_joint_position", 7),
        _action_part_2d(action, "right_arm_joint_position", 7),
        _action_part_2d(action, "left_effector_position", 1),
        _action_part_2d(action, "right_effector_position", 1),
    ]
    timesteps = max(part.shape[0] for part in parts)
    padded = [
        (
            np.pad(part, ((0, timesteps - part.shape[0]), (0, 0)), constant_values=0)
            if part.shape[0] < timesteps
            else part
        )
        for part in parts
    ]
    if horizon is not None:
        timesteps = min(timesteps, max(0, int(horizon)))
        padded = [part[:timesteps] for part in padded]
    commands = np.concatenate(padded, axis=-1)
    return {"actions": [command.tolist() for command in commands]}


def _make_pack_unpack():
    import msgpack

    def _decode_genie_numpy_object(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if b"__ndarray__" in value:
            return np.ndarray(
                buffer=value[b"data"],
                dtype=np.dtype(value[b"dtype"]),
                shape=value[b"shape"],
            )
        if b"__npgeneric__" in value:
            return np.dtype(value[b"dtype"]).type(value[b"data"])
        return value

    def _bytes_keys_to_str(value: Any) -> Any:
        if isinstance(value, dict):
            out = {}
            for raw_key, item in value.items():
                key = (
                    raw_key.decode("utf-8", errors="replace")
                    if isinstance(raw_key, bytes | bytearray)
                    else raw_key
                )
                out[key] = _bytes_keys_to_str(item)
            return out
        if isinstance(value, list):
            return [_bytes_keys_to_str(item) for item in value]
        return value

    try:
        import msgpack_numpy as msgpack_np

        msgpack_np.patch()

        def pack(value: Any) -> bytes:
            return msgpack.packb(value, default=msgpack_np.encode)

        def decode_object(value: Any) -> Any:
            decoded = _decode_genie_numpy_object(value)
            return msgpack_np.decode(value) if decoded is value else decoded

        def unpack(value: bytes) -> Any:
            return _bytes_keys_to_str(
                msgpack.unpackb(
                    value,
                    raw=True,
                    strict_map_key=False,
                    object_hook=decode_object,
                )
            )

    except ImportError:

        def encode(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            raise TypeError(type(value))

        def pack(value: Any) -> bytes:
            return msgpack.packb(value, default=encode)

        def unpack(value: bytes) -> Any:
            return _bytes_keys_to_str(
                msgpack.unpackb(
                    value,
                    raw=True,
                    strict_map_key=False,
                    object_hook=_decode_genie_numpy_object,
                )
            )

    return pack, unpack


def _is_connection_closed(exc: Exception) -> bool:
    return exc.__class__.__name__.startswith("ConnectionClosed")


async def _handler(
    ws: Any,
    policy: Any,
    pack: Any,
    unpack: Any,
    action_horizon: int | None,
) -> None:
    logger.info("Connection from %s opened", ws.remote_address)
    modality_configs = policy.get_modality_config()
    metadata = {
        "embodiment": policy.embodiment_tag.value,
        "video_keys": modality_configs["video"].modality_keys,
        "state_keys": modality_configs["state"].modality_keys,
        "action_keys": modality_configs["action"].modality_keys,
    }
    await ws.send(pack(metadata))

    while True:
        try:
            try:
                raw = await ws.recv()
            except Exception as exc:
                if _is_connection_closed(exc):
                    logger.info("Connection from %s closed", ws.remote_address)
                    return
                raise
            payload = unpack(raw)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected dict payload, got {type(payload)}")
            observation = payload_to_observation(policy, payload)
            action, _info = policy.get_action(observation)
            await ws.send(pack(action_to_wire(action, horizon=action_horizon)))
        except Exception:
            logger.exception("Inference error")
            try:
                await ws.send(
                    pack({"error": "inference_failed", "traceback": traceback.format_exc()})
                )
            finally:
                await ws.close(1011, "Internal error")


@dataclass
class ServeArgs:
    model_path: str | Path
    embodiment_tag: str = "agibot"
    device: str = "cuda:0"
    host: str = "0.0.0.0"
    port: int = 8000
    strict: bool = True
    num_inference_timesteps: int = 4
    action_horizon: int | None = None
    seed: int | None = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seed dependencies that use NumPy global state.

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args: ServeArgs) -> None:
    try:
        import websockets.asyncio.server as ws_server
    except ImportError as exc:
        raise ImportError(
            "websockets is required; install it with `uv pip install websockets`."
        ) from exc

    from groot.core.policy.n2_policy import Gr00tN2Policy

    try:
        from groot.vla.omni.utils.dreamcosmos_model_deps import ensure_dreamcosmos_model_deps
    except ModuleNotFoundError:
        ensure_dreamcosmos_model_deps = None

    if args.seed is not None:
        _seed_everything(args.seed)
    if ensure_dreamcosmos_model_deps is not None:
        ensure_dreamcosmos_model_deps(str(args.model_path))

    tag = _normalize_embodiment_tag(args.embodiment_tag)
    policy = Gr00tN2Policy(
        embodiment_tag=tag,
        model_path=str(args.model_path),
        device=args.device,
        strict=args.strict,
        num_inference_timesteps=args.num_inference_timesteps,
    )

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "127.0.0.1"
    logger.info(
        "Genie Sim N2 websocket server: host=%s ip=%s bind=%s:%s embodiment=%s",
        hostname,
        local_ip,
        args.host,
        args.port,
        tag.value,
    )

    pack, unpack = _make_pack_unpack()

    async def run() -> None:
        async with ws_server.serve(
            lambda ws: _handler(ws, policy, pack, unpack, args.action_horizon),
            args.host,
            args.port,
            compression=None,
            max_size=2**28,
        ) as server:
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(ServeArgs))
