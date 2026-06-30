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

import math
from typing import List, Optional, Tuple

import numpy as np


_TORCHCODEC_INSTALL_HINT = (
    "torchcodec is required for video decoding. Install it via the platform "
    "deployment script (scripts/deployment/{dgpu,orin,thor,spark}/install_deps.sh) "
    "or `uv pip install torchcodec`."
)

_DEFAULT_DECODER_KWARGS = {
    "device": "cpu",
    "dimension_order": "NHWC",
    "num_ffmpeg_threads": 0,
}


def _get_video_decoder_cls():
    try:
        from torchcodec.decoders import VideoDecoder
    except (ImportError, OSError, RuntimeError) as exc:
        raise ImportError(_TORCHCODEC_INSTALL_HINT) from exc
    return VideoDecoder


def _build_decoder(video_path: str, decoder_kwargs: Optional[dict]):
    video_decoder_cls = _get_video_decoder_cls()
    kwargs = {**_DEFAULT_DECODER_KWARGS, **(decoder_kwargs or {})}
    return video_decoder_cls(video_path, **kwargs)


def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    decoder_kwargs: Optional[dict] = None,
) -> np.ndarray:
    decoder = _build_decoder(video_path, decoder_kwargs)
    return decoder.get_frames_at(indices=indices).data.numpy()


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    decoder_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Get frames from a video at specified timestamps.

    Args:
        video_path (str): Path to the video file.
        timestamps (list[float] | np.ndarray): Timestamps to retrieve frames for, in seconds.

    Returns:
        np.ndarray: Frames at the specified timestamps.
    """
    decoder = _build_decoder(video_path, decoder_kwargs)

    # https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.VideoStreamMetadata.html#torchcodec.decoders.VideoStreamMetadata
    fps = decoder.metadata.average_fps
    interval = 1 / fps
    timestamps = np.array(timestamps).astype(np.float64)

    # Correct float precision issues in timestamps
    # E.g. for 5fps video: [1.0, 1.20000005, 1.39999998] -> [1.0, 1.2, 1.4]
    # Without this, torchcodec will read the delayed frame (e.g. 1.39999998 -> 1.2)
    # Round to nearest frame interval to prevent torchcodec from reading wrong frames.
    # Allow max 1% error from expected interval.
    closest_timestamps = np.round(timestamps / interval) * interval
    timestamp_errors = np.abs(closest_timestamps - timestamps) / interval
    invalid_mask = timestamp_errors >= 0.01
    if np.any(invalid_mask):
        invalid_indices = np.where(invalid_mask)[0]
        invalid_timestamps = timestamps[invalid_indices]
        raise ValueError(
            f"Try to read invalid timestamps {invalid_timestamps} from video {video_path} (FPS: {fps})"
        )

    timestamps = closest_timestamps
    return decoder.get_frames_played_at(seconds=timestamps).data.numpy()


def get_all_frames(
    video_path: str,
    decoder_kwargs: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Get all frames from a video.

    Returns:
        tuple[np.ndarray, np.ndarray]: Frames and timestamps.
    """
    decoder = _build_decoder(video_path, decoder_kwargs)
    frames = decoder.get_frames_at(indices=range(len(decoder)))
    return frames.data.numpy(), frames.pts_seconds.numpy()


def get_accumulate_timestamp_idxs(
    timestamps: List[float],
    start_time: float,
    dt: float,
    eps: float = 1e-5,
    next_global_idx: Optional[int] = 0,
    allow_negative=False,
) -> Tuple[List[int], List[int], int]:
    """
    For each dt window, choose the first timestamp in the window.
    Assumes timestamps sorted. One timestamp might be chosen multiple times due to dropped frames.
    next_global_idx should start at 0 normally, and then use the returned next_global_idx.
    However, when overwiting previous values are desired, set last_global_idx to None.

    Returns:
    local_idxs: which index in the given timestamps array to chose from
    global_idxs: the global index of each chosen timestamp
    next_global_idx: used for next call.
    """
    local_idxs = list()
    global_idxs = list()
    for local_idx, ts in enumerate(timestamps):
        # add eps * dt to timestamps so that when ts == start_time + k * dt
        # is always recorded as kth element (avoiding floating point errors)
        global_idx = math.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats
    return local_idxs, global_idxs, next_global_idx
