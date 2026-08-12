# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Video encoding/decoding utilities for PackDSStep.

This module centralizes all video encoding and decoding logic for PackDS,
providing functions to convert between decoded (numpy array) and encoded
(JPEG bytes or multi-frame video blob) video frames within PackDSStep objects.

Public API:
    encode_steps(steps, ...) — return new list of steps with encoded video
    decode_steps(steps) — return new list of steps with decoded video
    PackDSStepBatch.encode_video(...) — return new batch with encoded video
    PackDSStepBatch.decode_video() — return new batch with decoded video
"""

from __future__ import annotations

import abc
import copy
from pathlib import Path
import subprocess
import tempfile
import threading
from typing import TYPE_CHECKING

import numpy as np

import groot.core.data.packds.constants as constants
import groot.core.data.packds.utils as utils

if TYPE_CHECKING:
    from groot.core.data.packds import step as packds_step


# =============================================================================
# Video Codec Abstraction
# =============================================================================


class VideoCodec(abc.ABC):
    """Abstract codec for video blob encode/decode."""

    @abc.abstractmethod
    def encode_frames(self, frames: list[np.ndarray], crf: int = 18) -> bytes:
        """Encode RGB frames into a video blob."""
        ...

    @abc.abstractmethod
    def decode_blob(self, blob: bytes) -> list[np.ndarray]:
        """Decode a video blob into RGB frames."""
        ...


_PYAV_CODEC_CONFIG: dict[str, dict] = {
    "h264": {
        "encoder": constants.H264_ENCODER,
        "container": "mp4",
        "base_opts": {"preset": constants.H264_PRESET},
        "quality_param": "crf",
    },
    "av1": {
        "encoder": constants.AV1_ENCODER,
        "container": "mp4",
        "base_opts": {"preset": constants.AV1_PRESET},
        "quality_param": "crf",
    },
}


class PyAVVideoCodec(VideoCodec):
    """Software video codec using PyAV. Stateless, handles any resolution.

    Args:
        codec: Video codec — ``"h264"`` (default) or ``"av1"``.

    Raises:
        ValueError: If *codec* is not a supported codec name.
    """

    def __init__(self, codec: str = "h264") -> None:
        if codec not in _PYAV_CODEC_CONFIG:
            raise ValueError(f"Unsupported codec {codec!r}, choose from {list(_PYAV_CODEC_CONFIG)}")
        self._cfg = _PYAV_CODEC_CONFIG[codec]

    def encode_frames(self, frames: list[np.ndarray], crf: int = 18) -> bytes:
        import io

        import av

        cfg = self._cfg
        h, w = frames[0].shape[:2]
        buf = io.BytesIO()
        output = av.open(buf, mode="w", format=cfg["container"])
        stream = output.add_stream(cfg["encoder"], rate=30)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {**cfg["base_opts"], cfg["quality_param"]: str(crf)}

        for frame_data in frames:
            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)

        for packet in stream.encode():
            output.mux(packet)
        output.close()
        return buf.getvalue()

    def decode_blob(self, blob: bytes) -> list[np.ndarray]:
        import io

        import av

        container = av.open(io.BytesIO(blob))
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()
        return frames


_thread_local = threading.local()


def get_video_codec(codec: str = "h264") -> VideoCodec:
    """Get cached PyAV-backed video codec for the current thread.

    PackDS stores video blobs as MP4 containers, so encoding and decoding now
    always use the PyAV implementation for both ``"h264"`` and ``"av1"``.

    Args:
        codec: Video codec — ``"h264"`` (default) or ``"av1"``.

    Returns:
        A cached :class:`PyAVVideoCodec` instance.
    """
    attr = f"_video_codec_{codec}"
    if not hasattr(_thread_local, attr):
        setattr(_thread_local, attr, PyAVVideoCodec(codec=codec))
    return getattr(_thread_local, attr)


def _encode_step_video_jpeg(
    step: packds_step.PackDSStep, quality: int = 95
) -> packds_step.PackDSStep:
    """Encode decoded_video to encoded_video (JPEG bytes) in a PackDSStep.

    If the step already has encoded_video and no decoded_video, returns unchanged.
    If the step has both, encoded_video is overwritten with newly encoded data.

    Args:
        step: PackDSStep with decoded_video to encode
        quality: JPEG quality (0-100)

    Returns:
        The same step object with encoded_video populated
    """
    if step.decoded_video is None:
        return step

    if step.encoded_video is None:
        step.encoded_video = {}

    for key, frame in step.decoded_video.items():
        step.encoded_video[key] = utils.encode_frame_to_jpeg(frame, quality=quality)

    return step


def _decode_step_video(
    step: packds_step.PackDSStep, cache: dict | None = None
) -> packds_step.PackDSStep:
    """Decode encoded video to decoded_video (numpy arrays) in a PackDSStep.

    Supports both JPEG and multi-frame video blob encoded video. JPEG is preferred
    when both are present.

    Args:
        step: PackDSStep with encoded_video or encoded_video_blob to decode.
        cache: Dict for caching decoded video blob frames across steps sharing
            the same blob. Keyed by blob hash. A new cache is created
            automatically if not provided.

    Returns:
        The same step object with decoded_video populated.
    """
    if cache is None:
        cache = {}

    # JPEG path (preferred when non-None values exist)
    if step.encoded_video:
        jpeg_keys = {k for k, v in step.encoded_video.items() if v is not None}
        if jpeg_keys:
            if step.decoded_video is None:
                step.decoded_video = {}
            for key in jpeg_keys:
                step.decoded_video[key] = utils.decode_jpeg_to_frame(step.encoded_video[key])
            # If all keys are covered by JPEG, we're done
            blob_keys = set(step.encoded_video_blob.keys()) if step.encoded_video_blob else set()
            remaining = blob_keys - jpeg_keys
            if not remaining:
                return step

    # Video blob path (skip keys already decoded via JPEG)
    if step.encoded_video_blob:
        if step.decoded_video is None:
            step.decoded_video = {}
        for key, video_blob in step.encoded_video_blob.items():
            if video_blob is None:
                continue
            if key in step.decoded_video:
                continue
            # Use cache to avoid re-decoding the same blob
            blob_key = hash(video_blob)
            if blob_key not in cache:
                cache[blob_key] = get_video_codec().decode_blob(video_blob)
            all_frames = cache[blob_key]
            frame_idx = step.video_blob_frame_index[key] if step.video_blob_frame_index else 0
            step.decoded_video[key] = all_frames[frame_idx]
        return step

    return step


def _decode_steps_video(steps: list[packds_step.PackDSStep]) -> list[packds_step.PackDSStep]:
    """Decode video for a list of steps, handling both JPEG and video blob formats.

    For video blob steps, automatically manages a decode cache so each blob is
    decoded only once.

    Args:
        steps: List of PackDSStep with encoded video data.

    Returns:
        The same list of steps with decoded_video populated.
    """
    cache: dict = {}
    for step in steps:
        _decode_step_video(step, cache=cache)
    return steps


# =============================================================================
# Immutable API — encode_steps / decode_steps
# =============================================================================


def encode_steps(
    steps: list[packds_step.PackDSStep],
    *,
    encode_jpeg: bool = False,
    encode_video_blob: bool = True,
    group_size: int = constants.DEFAULT_VIDEO_BLOB_GROUP_SIZE,
    crf: int = constants.DEFAULT_VIDEO_BLOB_CRF,
    jpeg_quality: int = 95,
    codec: str = "h264",
) -> list[packds_step.PackDSStep]:
    """Return a new list of steps with encoded video.

    Creates shallow copies of steps and applies encoding to the copies,
    leaving the originals unchanged.

    Args:
        steps: List of PackDSStep with decoded_video to encode.
        encode_jpeg: If True, encode as per-frame JPEG.
        encode_video_blob: If True, encode as multi-frame video blobs.
        group_size: Number of frames per video blob group.
        crf: CRF quality parameter (0=lossless, higher=worse).
        jpeg_quality: JPEG quality (0-100).
        codec: Video blob codec — ``"h264"`` (default) or ``"av1"``.

    Returns:
        A new list of PackDSStep with encoded video populated.

    Raises:
        ValueError: If both encode_jpeg and encode_video_blob are False.
    """
    if not encode_jpeg and not encode_video_blob:
        raise ValueError("At least one of encode_jpeg or encode_video_blob must be True")

    new_steps = [copy.copy(s) for s in steps]
    # Deep-copy mutable video fields so originals are not mutated
    for s in new_steps:
        if s.encoded_video is not None:
            s.encoded_video = dict(s.encoded_video)
        if s.decoded_video is not None:
            s.decoded_video = dict(s.decoded_video)
        if s.encoded_video_blob is not None:
            s.encoded_video_blob = dict(s.encoded_video_blob)
        if s.video_blob_frame_index is not None:
            s.video_blob_frame_index = dict(s.video_blob_frame_index)

    if encode_video_blob:
        # Transcode JPEG→video_blob: decode JPEG to populate decoded_video
        # for JPEG keys that don't yet have a corresponding video blob.
        for s in new_steps:
            if s.encoded_video and not s.decoded_video:
                # Per-key check: does any JPEG key lack a video blob?
                needs_transcode = not s.encoded_video_blob or any(
                    k not in s.encoded_video_blob for k in s.encoded_video
                )
                if needs_transcode:
                    _decode_step_video(s)
                    if not encode_jpeg:
                        s.encoded_video = None

        # Find decoded_video keys that still need blob encoding.
        keys_needing_blob: set[str] = set()
        steps_with_decoded = [s for s in new_steps if s.decoded_video]
        for s in steps_with_decoded:
            for k in s.decoded_video:
                if not s.encoded_video_blob or k not in s.encoded_video_blob:
                    keys_needing_blob.add(k)

        if keys_needing_blob and steps_with_decoded:
            _encode_steps_video_blob(
                steps_with_decoded,
                group_size=group_size,
                crf=crf,
                video_keys=sorted(keys_needing_blob),
                codec=codec,
            )

    if encode_jpeg:
        for s in new_steps:
            if s.decoded_video and not s.encoded_video:
                _encode_step_video_jpeg(s, quality=jpeg_quality)

    return new_steps


def decode_steps(
    steps: list[packds_step.PackDSStep],
) -> list[packds_step.PackDSStep]:
    """Return a new list of steps with decoded video.

    Creates shallow copies of steps and populates decoded_video on the
    copies, leaving the originals unchanged.

    Args:
        steps: List of PackDSStep with encoded_video or encoded_video_blob.

    Returns:
        A new list of PackDSStep with decoded_video populated.
    """
    new_steps = [copy.copy(s) for s in steps]
    for s in new_steps:
        # Ensure decoded_video is a new dict so originals are not mutated
        s.decoded_video = dict(s.decoded_video) if s.decoded_video else None
    _decode_steps_video(new_steps)
    return new_steps


def _decode_batch_video(batch: packds_step.PackDSStepBatch) -> packds_step.PackDSStepBatch:
    """Decode video data in a PackDSStepBatch in-place.

    Populates ``batch.decoded_video`` for all video keys found in either
    ``batch.video`` (JPEG) or ``batch.video_blob`` (H.264). JPEG is preferred
    when both exist for a key.

    Args:
        batch: PackDSStepBatch with encoded video data.

    Returns:
        The same batch with decoded_video populated.
    """
    n = len(batch)

    # Collect all video keys from both JPEG and video blob
    all_keys = set(batch.video.keys()) | set(batch.video_blob.keys())

    for key in all_keys:
        # JPEG path (preferred when both exist)
        if key in batch.video:
            frames = []
            for i in range(n):
                jpeg_bytes = batch.video[key][i]
                if jpeg_bytes is not None:
                    frames.append(utils.decode_jpeg_to_frame(jpeg_bytes))
                else:
                    frames.append(None)
            batch.decoded_video[key] = frames
            continue

        # Video blob path
        if key in batch.video_blob:
            frames = [None] * n
            # Group blobs by identity to decode each unique blob only once
            cache: dict[int, list] = {}
            for i in range(n):
                blob = batch.video_blob[key][i]
                if blob is None:
                    # Forward-fill: the leader row is at i - frame_index
                    frame_idx = int(batch.video_blob_frame_index[key][i])
                    leader_idx = i - frame_idx
                    if 0 <= leader_idx < n:
                        blob = batch.video_blob[key][leader_idx]
                if blob is None:
                    continue
                blob_key = hash(blob)
                if blob_key not in cache:
                    cache[blob_key] = get_video_codec().decode_blob(blob)
                all_decoded = cache[blob_key]
                frame_idx = int(batch.video_blob_frame_index[key][i])
                frames[i] = all_decoded[frame_idx]
            batch.decoded_video[key] = frames

    return batch


def _encode_steps_video_blob(
    steps: list[packds_step.PackDSStep],
    group_size: int = constants.DEFAULT_VIDEO_BLOB_GROUP_SIZE,
    crf: int = constants.DEFAULT_VIDEO_BLOB_CRF,
    video_keys: list[str] | None = None,
    codec: str = "h264",
) -> list[packds_step.PackDSStep]:
    """Encode video frames across steps into multi-frame compressed blobs.

    Groups N adjacent frames into a single video blob using inter-frame
    compression. The blob is stored only on the first step of each group
    (None for others) to avoid data duplication in LanceDB.

    Args:
        steps: List of PackDSStep with decoded_video populated.
        group_size: Number of frames per group.
        crf: CRF quality parameter (0=lossless, higher=worse). Default 18.
        video_keys: Video keys to encode. If None, uses all keys from first step.
        codec: Video codec — ``"h264"`` (default) or ``"av1"``.

    Returns:
        The same list of steps with encoded_video_blob and video_blob_frame_index set,
        and encoded_video cleared for the converted keys.

    Raises:
        ValueError: If a step is missing decoded_video for a required key.
    """
    if not steps:
        return steps

    if video_keys is None:
        video_keys = steps[0].video_keys

    # Split steps into per-episode runs so groups never span episode boundaries.
    # Steps within a run share the same episode_id.
    episode_runs: list[list[int]] = []  # list of index ranges
    run_start = 0
    for idx in range(1, len(steps)):
        if steps[idx].episode_id != steps[run_start].episode_id:
            episode_runs.append(list(range(run_start, idx)))
            run_start = idx
    episode_runs.append(list(range(run_start, len(steps))))

    for key in video_keys:
        for run_indices in episode_runs:
            # Collect frames for this episode run
            run_frames = []
            for idx in run_indices:
                step = steps[idx]
                if step.decoded_video and key in step.decoded_video:
                    run_frames.append(step.decoded_video[key])
                else:
                    raise ValueError(
                        f"Step {step.step_index} missing decoded_video for key '{key}'"
                    )

            # Group frames and encode within this episode
            for group_start in range(0, len(run_frames), group_size):
                group_end = min(group_start + group_size, len(run_frames))
                group_frames = run_frames[group_start:group_end]
                video_blob = get_video_codec(codec).encode_frames(group_frames, crf=crf)

                for i in range(group_start, group_end):
                    step = steps[run_indices[i]]
                    if step.encoded_video_blob is None:
                        step.encoded_video_blob = {}
                    if step.video_blob_frame_index is None:
                        step.video_blob_frame_index = {}

                    # Only first step in group gets the blob
                    frame_in_group = i - group_start
                    step.encoded_video_blob[key] = video_blob if frame_in_group == 0 else None
                    step.video_blob_frame_index[key] = frame_in_group

    return steps


# =============================================================================
# Video export pipeline
# =============================================================================

_EXPORT_CODEC_MAP: dict[str, tuple[str, dict[str, str]]] = {
    "h264": (constants.H264_ENCODER, {"crf": str(constants.DEFAULT_VIDEO_BLOB_CRF), "preset": constants.H264_PRESET}),
    "av1": (constants.AV1_ENCODER, {"crf": str(constants.AV1_CRF), "preset": constants.AV1_PRESET}),
}


def export_jpeg_frames(
    jpeg_blobs: list[bytes],
    out_path: str | Path,
    *,
    target_codec: str = "h264",
    fps: float = 30.0,
) -> Path:
    """Encode a sequence of JPEG frames into a video file.

    Decodes each JPEG blob to an RGB frame and encodes into the target
    video codec via PyAV.

    Args:
        jpeg_blobs: Ordered list of JPEG-encoded frame bytes.
        out_path: Destination file path (e.g. ``"output.mp4"``).
        target_codec: Desired output codec — ``"h264"`` or ``"av1"``.
        fps: Frame rate for the output video.

    Returns:
        The resolved *out_path* as a ``Path``.

    Raises:
        RuntimeError: If no frames are provided.
    """
    import fractions

    import av

    if not jpeg_blobs:
        raise RuntimeError("No JPEG frames to export")

    out = Path(out_path)
    codec_name, codec_opts = _EXPORT_CODEC_MAP[target_codec]
    rate = fractions.Fraction(fps).limit_denominator(10000)

    first_frame = utils.decode_jpeg_to_frame(jpeg_blobs[0])
    h, w = first_frame.shape[:2]

    output = av.open(str(out), mode="w")
    ostream = output.add_stream(codec_name, rate=rate)
    ostream.width = w
    ostream.height = h
    ostream.pix_fmt = "yuv420p"
    ostream.time_base = rate
    ostream.options = codec_opts

    for i, jpeg in enumerate(jpeg_blobs):
        rgb = utils.decode_jpeg_to_frame(jpeg)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = i
        frame.time_base = rate
        for pkt in ostream.encode(frame):
            output.mux(pkt)
    for pkt in ostream.encode():
        output.mux(pkt)
    output.close()

    return out


def _ffmpeg_remux(ffmpeg_input: list[str], out_path: str | Path) -> None:
    """Remux via ffmpeg ``-c copy`` (no decode/re-encode).

    Args:
        ffmpeg_input: Input args for ffmpeg (e.g. ``["-f", "h264", "-i", "raw.h264"]``).
        out_path: Destination file path.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero code.
    """
    result = subprocess.run(
        ["ffmpeg", "-y", *ffmpeg_input, "-c", "copy", str(out_path)],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-500:]
        raise RuntimeError(f"ffmpeg remux failed (exit {result.returncode}): {stderr}")


def _write_concat_list(blobs: list[bytes], tmp: Path, ext: str) -> Path:
    """Write blobs as individual files and return the path to an ffmpeg concat list.

    Args:
        blobs: Ordered list of video blob bytes.
        tmp: Temporary directory to write files into.
        ext: File extension for blob files (e.g. ``".mp4"``, ``".h264"``).

    Returns:
        Path to the concat list file.
    """
    concat_path = tmp / "concat.txt"
    with open(concat_path, "w") as cl:
        for i, blob in enumerate(blobs):
            bp = tmp / f"{i:06d}{ext}"
            bp.write_bytes(blob)
            cl.write(f"file '{bp}'\n")
    return concat_path


def _pyav_transcode(
    raw_path: str | Path,
    out_path: str | Path,
    target_codec: str,
    fps: float,
    input_format: str | None,
) -> None:
    """Transcode a raw stream file to a different codec using PyAV.

    PyAV bundles its own ffmpeg with libsvtav1, so av1 encoding always
    works regardless of the system ffmpeg build.

    Args:
        raw_path: Path to the input video to transcode.
        out_path: Destination video file path.
        target_codec: Target codec — ``"h264"`` or ``"av1"``.
        fps: Frame rate.
        input_format: PyAV format hint (e.g. ``"h264"``), or None for auto.
    """
    import fractions

    import av

    codec_name, codec_opts = _EXPORT_CODEC_MAP[target_codec]
    rate = fractions.Fraction(fps).limit_denominator(10000)

    kwargs: dict = {}
    if input_format:
        kwargs["format"] = input_format

    inp = av.open(str(raw_path), **kwargs)
    output = av.open(str(out_path), mode="w")
    in_stream = inp.streams.video[0]
    ostream = output.add_stream(codec_name, rate=rate)
    ostream.width = in_stream.codec_context.width
    ostream.height = in_stream.codec_context.height
    ostream.pix_fmt = "yuv420p"
    ostream.time_base = rate
    ostream.options = codec_opts

    for i, frame in enumerate(inp.decode(video=0)):
        frame.pts = i
        frame.dts = None
        frame.time_base = rate
        for pkt in ostream.encode(frame):
            output.mux(pkt)
    for pkt in ostream.encode():
        output.mux(pkt)
    output.close()
    inp.close()


def _normalize_export_codec_name(codec_name: str | None) -> str | None:
    if codec_name is None:
        return None
    if codec_name in {"h264", "libx264", "mpeg4"}:
        return "h264"
    if codec_name in {"av1", "libsvtav1", "av01", "libdav1d"}:
        return "av1"
    return None


def probe_video_blob_codec(blob: bytes) -> str | None:
    """Inspect one MP4 video blob and return ``"h264"``/``"av1"`` when recognized."""
    import io

    import av

    with av.open(io.BytesIO(blob)) as container:
        stream = container.streams.video[0]
        names = [
            getattr(getattr(stream.codec_context, "codec", None), "name", None),
            getattr(stream.codec_context, "name", None),
        ]
        for name in names:
            normalized = _normalize_export_codec_name(name)
            if normalized is not None:
                return normalized
    return None


def _pyav_transcode_blobs(
    blobs: list[bytes],
    out_path: str | Path,
    target_codec: str,
    fps: float,
) -> None:
    import fractions
    import io

    import av

    codec_name, codec_opts = _EXPORT_CODEC_MAP[target_codec]
    rate = fractions.Fraction(fps).limit_denominator(10000)

    output = None
    ostream = None
    frame_idx = 0
    try:
        for blob in blobs:
            with av.open(io.BytesIO(blob)) as inp:
                for frame in inp.decode(video=0):
                    if output is None:
                        output = av.open(str(out_path), mode="w")
                        ostream = output.add_stream(codec_name, rate=rate)
                        ostream.width = frame.width
                        ostream.height = frame.height
                        ostream.pix_fmt = "yuv420p"
                        ostream.time_base = rate
                        ostream.options = codec_opts
                    assert ostream is not None
                    frame.pts = frame_idx
                    frame.dts = None
                    frame.time_base = rate
                    for pkt in ostream.encode(frame):
                        output.mux(pkt)
                    frame_idx += 1
        if output is None or ostream is None:
            raise RuntimeError("No frames were decoded from video blobs during export")
        for pkt in ostream.encode():
            output.mux(pkt)
    finally:
        if output is not None:
            output.close()


def export_video_blobs(
    blobs: list[bytes],
    out_path: str | Path,
    *,
    stored_codec: str | None = None,
    stored_codecs: list[str | None] | None = None,
    target_codec: str = "h264",
    fps: float = 30.0,
) -> Path:
    """Write ordered video blobs to a single video file.

    All blobs are MP4 containers. When every blob already matches the target
    codec, export uses ffmpeg concat/remux without decoding. Otherwise, blobs
    are decoded and re-encoded via PyAV, which also supports mixed-codec input.

    Args:
        blobs: Ordered list of video blob bytes (group leaders only,
            already filtered and sorted by the caller).
        out_path: Destination file path (e.g. ``"output.mp4"``).
        stored_codec: Optional uniform codec hint for all blobs.
        stored_codecs: Optional per-blob codecs. When omitted, codecs are
            probed from the blobs.
        target_codec: Desired output codec — ``"h264"`` or ``"av1"``.
        fps: Frame rate for the output video.

    Returns:
        The resolved *out_path* as a ``Path``.

    Raises:
        RuntimeError: If remux or transcoding fails.
    """
    out = Path(out_path)
    if stored_codecs is None:
        if stored_codec is not None:
            stored_codecs = [stored_codec] * len(blobs)
        else:
            stored_codecs = [probe_video_blob_codec(blob) for blob in blobs]

    unique_codecs = {codec for codec in stored_codecs if codec is not None}
    can_fast_remux = len(unique_codecs) == 1 and next(iter(unique_codecs), None) == target_codec

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        if can_fast_remux:
            concat_path = _write_concat_list(blobs, tmp, ".mp4")
            ffmpeg_input = ["-f", "concat", "-safe", "0", "-i", str(concat_path)]
            _ffmpeg_remux(ffmpeg_input, out)
        else:
            _pyav_transcode_blobs(blobs, out, target_codec, fps)

    return out
