import json
import math
import subprocess
from typing import List, Optional, Tuple

import av
import cv2
import numpy as np
import torchvision


# Import decord with graceful fallback
try:
    import decord

    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import torchcodec

    TORCHCODEC_AVAILABLE = True
except (ImportError, RuntimeError):
    TORCHCODEC_AVAILABLE = False


def _get_video_info_ffmpeg(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,duration,r_frame_rate",
        "-of",
        "json",
        video_path,
    ]

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8")
        probe_data = json.loads(output)
        stream = probe_data["streams"][0]

        # Parse frame rate (comes as fraction like "15/1")
        if "/" in stream["r_frame_rate"]:
            num, den = map(int, stream["r_frame_rate"].split("/"))
            fps = num / den
        else:
            fps = float(stream["r_frame_rate"])

        # Get frame count and duration
        nb_frames = int(stream.get("nb_frames", 0))
        duration = float(stream.get("duration", 0))

        # If nb_frames is not available, estimate from duration and fps
        if nb_frames == 0 and duration > 0:
            nb_frames = int(duration * fps)

        return {
            "nb_frames": nb_frames,
            "fps": fps,
            "duration": duration,
        }
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to get video info for {video_path}: {e}")


def _extract_frames_ffmpeg(video_path: str, frame_indices: list[int]) -> np.ndarray:
    """Extract specific frames using ffmpeg."""
    frames = []

    for idx in frame_indices:
        # Use ffmpeg to extract a specific frame
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"select=eq(n\\,{idx})",
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-",
        ]

        try:
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)

            # Check if output is empty (frame doesn't exist)
            if len(output) == 0:
                raise subprocess.CalledProcessError(1, cmd)

            # Get frame dimensions by probing first
            if len(frames) == 0:
                info_cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "json",
                    video_path,
                ]
                info_output = subprocess.check_output(info_cmd).decode("utf-8")
                info_data = json.loads(info_output)
                width = info_data["streams"][0]["width"]
                height = info_data["streams"][0]["height"]

            # Decode raw RGB data
            frame_data = np.frombuffer(output, dtype=np.uint8)
            frame = frame_data.reshape((height, width, 3))
            frames.append(frame)

        except subprocess.CalledProcessError:
            # Frame might not exist, create a black frame
            if len(frames) > 0:
                frames.append(np.zeros_like(frames[0]))
            else:
                # Default fallback frame
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

    return np.array(frames)


def _extract_frames_at_timestamps_ffmpeg(video_path: str, timestamps: list[float]) -> np.ndarray:
    """Extract frames at specific timestamps using ffmpeg."""
    frames = []

    for timestamp in timestamps:
        cmd = [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            video_path,
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-",
        ]

        try:
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)

            # Check if output is empty (timestamp doesn't exist)
            if len(output) == 0:
                raise subprocess.CalledProcessError(1, cmd)

            # Get frame dimensions
            if len(frames) == 0:
                info_cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "json",
                    video_path,
                ]
                info_output = subprocess.check_output(info_cmd).decode("utf-8")
                info_data = json.loads(info_output)
                width = info_data["streams"][0]["width"]
                height = info_data["streams"][0]["height"]

            # Decode raw RGB data
            frame_data = np.frombuffer(output, dtype=np.uint8)
            frame = frame_data.reshape((height, width, 3))
            frames.append(frame)

        except subprocess.CalledProcessError:
            # Timestamp might be out of bounds, use last frame or black frame
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

    return np.array(frames)


def _extract_all_frames_ffmpeg(video_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract all frames and their timestamps using ffmpeg."""
    # Get video info
    info = _get_video_info_ffmpeg(video_path)
    fps = info["fps"]

    # Extract all frames
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-f",
        "image2pipe",
        "-pix_fmt",
        "rgb24",
        "-vcodec",
        "rawvideo",
        "-",
    ]

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)

        # Get frame dimensions
        info_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            video_path,
        ]
        info_output = subprocess.check_output(info_cmd).decode("utf-8")
        info_data = json.loads(info_output)
        width = info_data["streams"][0]["width"]
        height = info_data["streams"][0]["height"]

        # Decode all frames
        frame_data = np.frombuffer(output, dtype=np.uint8)
        total_pixels = len(frame_data) // 3
        actual_frames = total_pixels // (width * height)

        frames = frame_data[: actual_frames * width * height * 3].reshape(
            (actual_frames, height, width, 3)
        )

        # Generate timestamps
        timestamps = np.arange(actual_frames) / fps

        return frames, timestamps

    except subprocess.CalledProcessError as e:
        raise ValueError(f"Failed to extract frames from {video_path}: {e}")


def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    video_backend: str = "ffmpeg",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available. Install it with: pip install decord")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        return decoder.get_frames_at(indices=indices).data.numpy()
    elif video_backend == "ffmpeg":
        return _extract_frames_ffmpeg(video_path, list(indices))
    elif video_backend == "opencv":
        frames = []
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    else:
        raise NotImplementedError


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "ffmpeg",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    """Get frames from a video at specified timestamps.

    Args:
        video_path (str): Path to the video file.
        timestamps (list[int] | np.ndarray): Timestamps to retrieve frames for, in seconds.
        video_backend (str, optional): Video backend to use. Defaults to "ffmpeg".

    Returns:
        np.ndarray: Frames at the specified timestamps.
    """
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available. Install it with: pip install decord")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        num_frames = len(vr)
        # Retrieve the timestamps for each frame in the video
        frame_ts: np.ndarray = vr.get_frame_timestamp(range(num_frames))
        # Map each requested timestamp to the closest frame index
        # Only take the first element of the frame_ts array which corresponds to start_seconds
        indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )

        # https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.VideoStreamMetadata.html#torchcodec.decoders.VideoStreamMetadata
        fps = decoder.metadata.average_fps
        interval = 1 / fps
        timestamps = np.array(timestamps).astype(np.float64)

        # Correct float precision issues in timestamps
        # E.g. for 5fps video: [1.0, 1.20000005, 1.39999998] -> [1.0, 1.2, 1.4]
        # Without this, the torchcodec will read the delayed frame (e.g. 1.39999998 -> 1.2)
        # Round to nearest frame interval to prevent torchcodec from reading wrong frames
        # Allow max 1% error from expected interval
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
    elif video_backend == "ffmpeg":
        return _extract_frames_at_timestamps_ffmpeg(video_path, list(timestamps))
    elif video_backend == "opencv":
        # Open the video file
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        # Retrieve the total number of frames
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate timestamps for each frame
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ts = np.arange(num_frames) / fps
        frame_ts = frame_ts[:, np.newaxis]  # Reshape to (num_frames, 1) for broadcasting
        # Map each requested timestamp to the closest frame index
        indices = np.abs(frame_ts - timestamps).argmin(axis=0)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames

    elif video_backend == "torchvision_av":
        # set backend
        torchvision.set_video_backend("pyav")

        # set a video stream reader
        # TODO(rcadene): also load audio stream at the same time
        reader = torchvision.io.VideoReader(video_path, "video")

        # set the first and last requested timestamps
        # Note: previous timestamps are usually loaded, since we need to access the previous key frame
        first_ts = timestamps[0]
        last_ts = timestamps[-1]

        # access closest key frame of the first requested frame
        # Note: closest key frame timestamp is usally smaller than `first_ts` (e.g. key frame can be the first frame of the video)
        # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
        reader.seek(first_ts, keyframes_only=True)

        # load all frames until last requested frame
        loaded_frames = []
        loaded_ts = []
        for frame in reader:
            current_ts = frame["pts"]
            loaded_frames.append(frame["data"])
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break

        reader.container.close()
        reader = None
        frames = np.array(loaded_frames)
        return frames.transpose(0, 2, 3, 1)

    else:
        raise NotImplementedError


def get_all_frames(
    video_path: str,
    video_backend: str = "ffmpeg",
    video_backend_kwargs: dict = {},
) -> tuple[np.ndarray, np.ndarray]:
    """Get all frames from a video.

    Returns:
        tuple[np.ndarray, np.ndarray]: Frames and timestamps.
    """
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available. Install it with: pip install decord")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(range(len(vr))).asnumpy()
        return frames, vr.get_frame_timestamp(range(len(vr)))[:, 0]
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        frames = decoder.get_frames_at(indices=range(len(decoder)))
        return frames.data.numpy(), frames.pts_seconds.numpy()
    elif video_backend == "ffmpeg":
        return _extract_all_frames_ffmpeg(video_path)
    elif video_backend == "pyav":
        container = av.open(video_path)
        stream = container.streams.video[0]
        assert stream.time_base is not None
        frames = []
        timestamps = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            timestamps.append(frame.pts * stream.time_base)
        container.close()
        return np.stack(frames), np.array(timestamps)

    else:
        raise NotImplementedError


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
