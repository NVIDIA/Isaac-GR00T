# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utility functions for PackDS format.

This module provides encode/decode helpers, format detection functions,
and standalone conversion functions for PackDS data interchange.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
import torchvision.io

from groot.core.data.packds import constants

if TYPE_CHECKING:
    from groot.core.data.packds import step
    from groot.core.data.packds.reader import PackDSReader

#: Union type accepted by :func:`PackDS.write_steps` and :func:`into_table`.
IntoSteps = Union[list["step.PackDSStep"], "step.PackDSStepBatch", pa.Table, pa.RecordBatch]


class SchemaMismatchError(Exception):
    """Raised when steps contain columns not present in the provided schema."""


# Check CUDA availability once at module load
_CUDA_AVAILABLE = torch.cuda.is_available()


def is_remote(path: str) -> bool:
    """Check if path is a remote URI (s3://, gs://, etc.)."""
    return "://" in path


def gear_creds_to_lance_storage_options(creds: dict[str, str]) -> dict[str, str]:
    """Convert gear S3 credentials to LanceDB storage_options format.

    Args:
        creds: Dict from ``gear.common.get_s3_credentials_from_file()``
            with keys: aws_access_key_id, aws_secret_access_key, region,
            endpoint_url.

    Returns:
        Dict with keys: aws_access_key_id, aws_secret_access_key,
        aws_region, aws_endpoint, allow_http (str "true"/"false").
    """
    endpoint = creds.get("endpoint_url", "")
    allow_http = "true" if endpoint.startswith("http://") else "false"

    return {
        "aws_access_key_id": creds["aws_access_key_id"],
        "aws_secret_access_key": creds["aws_secret_access_key"],
        "aws_region": creds.get("region", "us-east-1"),
        "aws_endpoint": endpoint,
        "allow_http": allow_http,
    }


def strip_s3_scheme(uri: str) -> str:
    """Remove the ``s3://`` scheme from a URI, returning ``bucket/key``."""
    if uri.startswith("s3://"):
        return uri[len("s3://"):]
    return uri


def make_arrow_s3fs(storage_options: dict[str, str]) -> "pa.fs.S3FileSystem":
    """Create a PyArrow S3FileSystem from Lance-style storage options."""
    import pyarrow.fs as pafs

    endpoint = storage_options.get("aws_endpoint") or None
    scheme = "http" if storage_options.get("allow_http") == "true" else "https"
    if endpoint and "://" in endpoint:
        endpoint = endpoint.split("://", 1)[1]

    return pafs.S3FileSystem(
        access_key=storage_options.get("aws_access_key_id"),
        secret_key=storage_options.get("aws_secret_access_key"),
        region=storage_options.get("aws_region") or None,
        endpoint_override=endpoint,
        scheme=scheme,
    )


def storage_options_to_s3_env(storage_options: dict[str, str]) -> dict[str, str]:
    """Convert Lance-style storage options to AWS env vars for s5cmd/aws-cli."""
    import os

    env = dict(os.environ)
    env["AWS_ACCESS_KEY_ID"] = storage_options.get("aws_access_key_id", "")
    env["AWS_SECRET_ACCESS_KEY"] = storage_options.get("aws_secret_access_key", "")
    env["AWS_DEFAULT_REGION"] = storage_options.get("aws_region", "us-east-1")
    endpoint = storage_options.get("aws_endpoint", "")
    if endpoint:
        env["AWS_ENDPOINT_URL"] = endpoint
        env["S3_ENDPOINT_URL"] = endpoint
    return env


def validate_episode_id_for_write(episode_id: str) -> str:
    """Validate a PackDS episode_id for safe storage and filtering."""
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("PackDS requires a non-empty string episode_id")
    if any(ch in episode_id for ch in ("'", '"', "\\")):
        raise ValueError(
            "episode_id must not contain single-quote, double-quote, or backslash: "
            f"{episode_id!r}"
        )
    return episode_id


def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 95) -> bytes:
    """Encode a frame to JPEG bytes using torchvision.

    Args:
        frame: RGB image as numpy array (H, W, C)
        quality: JPEG quality (0-100)

    Returns:
        JPEG-encoded bytes
    """
    # Convert to torch tensor (C, H, W) format
    tensor = torch.from_numpy(frame).permute(2, 0, 1).contiguous()
    # Encode to JPEG
    jpeg_tensor = torchvision.io.encode_jpeg(tensor, quality=quality)
    return jpeg_tensor.numpy().tobytes()


def _encode_frames_to_jpeg_batch(
    frames: np.ndarray | list[np.ndarray],
    quality: int = 95,
    use_cuda: bool | None = None,
    batch_size: int = 240,
) -> list[bytes]:
    """Encode multiple frames to JPEG bytes using torchvision with optional CUDA acceleration.

    This function batches JPEG encoding for better performance, especially when
    CUDA is available. On GPU, encoding is significantly faster than CPU.

    To avoid GPU memory issues with large videos, frames are processed in chunks
    of `batch_size` frames at a time.

    Args:
        frames: Either a numpy array of shape (N, H, W, C) or a list of (H, W, C) arrays
        quality: JPEG quality (0-100)
        use_cuda: Whether to use CUDA for encoding. If None, auto-detect availability.
        batch_size: Maximum number of frames to process at once (default 240)

    Returns:
        List of JPEG-encoded bytes, one per frame
    """

    def _batched(iterable, n):
        """Batch an iterable into chunks of size n."""
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) == n:
                yield batch
                batch = []
        if batch:
            yield batch

    if use_cuda is None:
        use_cuda = _CUDA_AVAILABLE

    # Convert to list if numpy array
    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        frame_list = [frames[i] for i in range(frames.shape[0])]
    else:
        frame_list = list(frames)

    if len(frame_list) == 0:
        return []

    jpeg_bytes_list = []

    # Process frames in batches to avoid GPU memory issues
    for frame_batch in _batched(frame_list, batch_size):
        # Convert batch frames to tensors in (C, H, W) format
        tensors = [torch.from_numpy(frame).permute(2, 0, 1).contiguous() for frame in frame_batch]

        # Stack into batch tensor (N, C, H, W)
        batch_tensor = torch.stack(tensors)

        # Move to CUDA if available and requested
        if use_cuda:
            batch_tensor = batch_tensor.cuda()

        # Encode each frame in the batch
        for i in range(batch_tensor.shape[0]):
            jpeg_tensor = torchvision.io.encode_jpeg(batch_tensor[i], quality=quality)
            # Move back to CPU if needed and convert to bytes
            if use_cuda:
                jpeg_tensor = jpeg_tensor.cpu()
            jpeg_bytes_list.append(jpeg_tensor.numpy().tobytes())

        # Free GPU memory after each batch
        if use_cuda:
            del batch_tensor
            torch.cuda.empty_cache()

    return jpeg_bytes_list


def decode_jpeg_to_frame(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to numpy array using torchvision.

    Args:
        jpeg_bytes: JPEG-encoded bytes

    Returns:
        RGB image as numpy array (H, W, C)
    """
    # Create a writable tensor from the bytes (single copy via bytearray)
    jpeg_tensor = torch.frombuffer(bytearray(jpeg_bytes), dtype=torch.uint8)
    tensor = torchvision.io.decode_jpeg(jpeg_tensor)
    # tensor is (C, H, W), convert to (H, W, C) numpy
    return tensor.permute(1, 2, 0).numpy()


def _encode_frames_to_video_blob(
    frames: list[np.ndarray], crf: int = constants.DEFAULT_VIDEO_BLOB_CRF
) -> bytes:
    """Encode multiple RGB frames into a single compressed video blob.

    Delegates to the thread-local :class:`~.video.VideoCodec`, which uses the
    PyAV-backed PackDS blob codec implementation.

    Args:
        frames: List of (H, W, C) uint8 numpy arrays.
        crf: H.264 Constant Rate Factor (0=lossless, 51=worst). Default 18.

    Returns:
        H.264-encoded video bytes containing all frames.
    """
    from groot.core.data.packds import video

    return video.get_video_codec().encode_frames(frames, crf=crf)


def _decode_video_blob_to_frames(video_blob_bytes: bytes) -> list[np.ndarray]:
    """Decode all frames from a compressed video blob.

    Delegates to the thread-local :class:`~.video.VideoCodec`, which uses the
    PyAV-backed PackDS blob codec implementation.

    Args:
        video_blob_bytes: Video blob bytes in an MP4 container.

    Returns:
        List of (H, W, C) uint8 numpy arrays.
    """
    from groot.core.data.packds import video

    return video.get_video_codec().decode_blob(video_blob_bytes)


def extract_scalar(v: np.ndarray | float | str | None) -> int | float | str | None:
    """Extract scalar value from scalar, 0-d array, or 1-d array.

    Handles various input types commonly encountered when extracting values from
    observation/action dictionaries in robotics environments.

    Args:
        v: Input value - can be None, scalar, 0-d numpy array, or 1-d numpy array

    Returns:
        Extracted scalar value, or None for null-like/empty inputs
    """
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            # Object array - extract and check for None
            if v.ndim == 0:
                return v.item()  # May be None
            elif v.size > 0:
                return v.flat[0]  # May be None
            else:
                return None
        # Numeric array - extract value
        elif v.ndim == 0:
            return v.item()
        elif v.size > 0:
            return v.flat[0]
        else:
            return None
    return v  # Already a scalar (int, float, str, etc.)


def numpy_shape_to_fixed_size_list_type(shape: tuple[int, ...]) -> pa.DataType:
    """Build a nested fixed-size-list Arrow type for a tensor shape."""
    if not shape:
        raise ValueError("Tensor shape must have at least one dimension")
    type_: pa.DataType = pa.float32()
    for size in reversed(shape):
        type_ = pa.list_(type_, int(size))
    return type_


def is_nested_fixed_size_list_tensor_type(type_: pa.DataType) -> bool:
    """Return True when *type_* is a nested fixed-size-list float tensor type."""
    if not pa.types.is_fixed_size_list(type_):
        return False
    value_type = type_.value_type
    while pa.types.is_fixed_size_list(value_type):
        value_type = value_type.value_type
    return pa.types.is_floating(value_type)


def nested_fixed_size_list_shape(type_: pa.DataType) -> tuple[int, ...]:
    """Extract a tensor shape from a nested fixed-size-list type."""
    shape: list[int] = []
    current = type_
    while pa.types.is_fixed_size_list(current):
        shape.append(current.list_size)
        current = current.value_type
    if not shape or not pa.types.is_floating(current):
        raise ValueError(f"Not a nested fixed-size-list tensor type: {type_}")
    return tuple(shape)


def numpy_to_fixed_size_list_array(arr: np.ndarray, type_: pa.DataType | None = None) -> pa.Array:
    """Convert an ``(N, *shape)`` numpy array to nested fixed-size-list Arrow values.

    Uses zero-copy flattening + ``FixedSizeListArray.from_arrays`` to avoid
    the O(N * product(shape)) Python-list intermediary that ``pa.array(arr.tolist())``
    would create.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"Expected batched tensor array with ndim >= 2, got {arr.shape}")
    # Flatten to 1-D and wrap in nested FixedSizeListArrays from innermost out.
    flat = pa.array(arr.reshape(-1), type=pa.float32())
    result: pa.Array = flat
    for size in reversed(arr.shape[1:]):
        result = pa.FixedSizeListArray.from_arrays(result, size)
    return result


def to_table(
    episode_id: str,
    num_steps: int,
    *,
    steps_per_second: float | None = None,
    step_time_ns: np.ndarray | None = None,
    state: dict[str, np.ndarray] | None = None,
    action: dict[str, np.ndarray] | None = None,
    annotations: dict[str, list] | None = None,
    video: dict[str, list[bytes]] | None = None,
    has_valid_action: np.ndarray | None = None,
) -> pa.Table:
    """Build a PackDS ``pa.Table`` directly from numpy arrays.

    This is the fast path for converting structured episode data to PackDS
    format without going through :class:`PackDSStep` objects.  State and
    action arrays should be ``(num_steps, *shape)`` numpy arrays.

    Provide exactly one of *steps_per_second* (uniform timestamps) or
    *step_time_ns* (explicit nanosecond timestamps per step).

    Example::

        table = to_table(
            episode_id="ep_000",
            num_steps=T,
            steps_per_second=30.0,
            state={"joint_pos": joint_pos, "dof": dof},
            action={"gripper": gripper_cmd},
        )
        ds.write_steps(table)
    """
    if (steps_per_second is None) == (step_time_ns is None):
        raise ValueError("Provide exactly one of steps_per_second or step_time_ns")

    step_indices = np.arange(num_steps, dtype=np.int64)
    if step_time_ns is not None:
        assert len(step_time_ns) == num_steps
        time_ns = np.asarray(step_time_ns, dtype=np.int64)
    else:
        ns_per_step = round(1e9 / steps_per_second)
        time_ns = step_indices * ns_per_step

    columns: dict[str, pa.Array] = {
        "episode_id": pa.array([episode_id] * num_steps, type=pa.utf8()),
        "step_index": pa.array(step_indices),
        "step_time_ns": pa.array(time_ns),
        "original_step_time_ns": pa.array(time_ns),
        "has_valid_action": pa.array(
            has_valid_action if has_valid_action is not None
            else np.ones(num_steps, dtype=bool)
        ),
    }

    if state:
        for key, arr in sorted(state.items()):
            arr = np.asarray(arr, dtype=np.float32)
            col = key_to_column(constants.STATE_PREFIX, key)
            assert arr.shape[0] == num_steps, f"{col} has {arr.shape[0]} rows, expected {num_steps}"
            columns[col] = numpy_to_fixed_size_list_array(arr)

    if action:
        for key, arr in sorted(action.items()):
            arr = np.asarray(arr, dtype=np.float32)
            col = key_to_column(constants.ACTION_PREFIX, key)
            assert arr.shape[0] == num_steps, f"{col} has {arr.shape[0]} rows, expected {num_steps}"
            columns[col] = numpy_to_fixed_size_list_array(arr)

    if video:
        for key, frames in sorted(video.items()):
            col = key_to_column(constants.VIDEO_PREFIX, key)
            assert len(frames) == num_steps, f"{col} has {len(frames)} frames, expected {num_steps}"
            columns[col] = pa.array(frames, type=pa.binary())

    if annotations:
        for key, values in sorted(annotations.items()):
            col = key_to_column("annotation", key)
            assert len(values) == num_steps, f"{col} has {len(values)} values, expected {num_steps}"
            columns[col] = pa.array(values)

    return pa.table(columns)


def fixed_size_list_array_to_numpy(col: pa.ChunkedArray | pa.Array, type_: pa.DataType) -> np.ndarray:
    """Convert nested fixed-size-list tensor values back to a numpy array.

    Uses zero-copy buffer access where possible: flattens nested
    fixed-size-list arrays to their underlying float buffer and reshapes.
    Falls back to ``to_pylist()`` for non-contiguous or non-float data.
    """
    shape = nested_fixed_size_list_shape(type_)
    if isinstance(col, pa.ChunkedArray):
        arrays = [fixed_size_list_array_to_numpy(chunk, type_) for chunk in col.chunks]
        return np.concatenate(arrays, axis=0) if arrays else np.empty((0, *shape), dtype=np.float32)
    n = len(col)
    if n == 0:
        return np.empty((0, *shape), dtype=np.float32)
    # Fast path: flatten to the innermost values buffer and reshape
    flat = col.flatten()
    while pa.types.is_fixed_size_list(flat.type):
        flat = flat.flatten()
    try:
        arr = flat.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        return arr.reshape((n, *shape))
    except Exception:
        # Fallback for non-numeric or complex types
        values = col.to_pylist()
        return np.asarray(values, dtype=np.float32).reshape((n, *shape))


def infer_schema(
    steps: list[step.PackDSStep],
    metadata: dict[str, str] | None = None,
) -> pa.Schema:
    """Infer a PyArrow schema from a list of PackDSStep objects.

    Uses the first step in *steps* for shape inference (video resolution,
    state/action dimensionality, annotation types).

    Args:
        steps: Non-empty list of PackDSStep objects.
        metadata: Optional dict of key→value pairs to embed as schema
            metadata (e.g. ``packds:*`` configuration keys).

    Returns:
        PyArrow schema with nested fixed-size-list state/action tensors,
        binary with field metadata for video, and typed annotation columns.

    Raises:
        ValueError: If an annotation has an unsupported type.
    """
    step_ = steps[0]

    fields: list[pa.Field] = [
        pa.field("episode_id", pa.utf8()),
        pa.field("step_index", pa.int64()),
        pa.field("step_time_ns", pa.int64()),
        pa.field("original_step_time_ns", pa.int64()),
        pa.field("has_valid_action", pa.bool_()),
    ]

    # Video fields with shape metadata
    video_blob_keys = set(step_.encoded_video_blob.keys()) if step_.encoded_video_blob else set()
    if step_.encoded_video:
        for key in sorted(step_.encoded_video.keys()):
            video_metadata = {}
            if step_.decoded_video and key in step_.decoded_video:
                frame = step_.decoded_video[key]
                h, w, c = frame.shape
                video_metadata["shape"] = json.dumps([h, w, c])
                video_metadata["dim_names"] = json.dumps(["height", "width", "channel"])
            col = key_to_column(constants.VIDEO_PREFIX, key)
            fields.append(pa.field(col, pa.binary(), metadata=video_metadata or None))
    elif step_.decoded_video:
        for key in sorted(step_.decoded_video.keys()):
            if key in video_blob_keys:
                continue
            frame = step_.decoded_video[key]
            h, w, c = frame.shape
            video_metadata = {
                "shape": json.dumps([h, w, c]),
                "dim_names": json.dumps(["height", "width", "channel"]),
            }
            col = key_to_column(constants.VIDEO_PREFIX, key)
            fields.append(pa.field(col, pa.binary(), metadata=video_metadata))

    # Video blob fields (multi-frame video blobs)
    if step_.encoded_video_blob:
        for key in sorted(step_.encoded_video_blob.keys()):
            video_blob_metadata = {}
            if step_.decoded_video and key in step_.decoded_video:
                frame = step_.decoded_video[key]
                h, w, c = frame.shape
                video_blob_metadata["shape"] = json.dumps([h, w, c])
                video_blob_metadata["dim_names"] = json.dumps(["height", "width", "channel"])
            col = key_to_column(constants.VIDEO_BLOB_PREFIX, key)
            fields.append(pa.field(col, pa.binary(), metadata=video_blob_metadata or None))
            idx_col = key_to_column(constants.VIDEO_BLOB_FRAME_INDEX_PREFIX, key)
            fields.append(pa.field(idx_col, pa.int32()))

    # State fields with nested fixed-size-list tensors
    for key in sorted(step_.state.keys()):
        arr = step_.state[key]
        shape = tuple(arr.shape)
        col = key_to_column(constants.STATE_PREFIX, key)
        fields.append(pa.field(col, numpy_shape_to_fixed_size_list_type(shape)))

    # Action fields with nested fixed-size-list tensors
    for key in sorted(step_.action.keys()):
        arr = step_.action[key]
        shape = tuple(arr.shape)
        col = key_to_column(constants.ACTION_PREFIX, key)
        fields.append(pa.field(col, numpy_shape_to_fixed_size_list_type(shape)))

    # Annotation fields – union keys across ALL steps so sparse annotations
    # aren't silently dropped when only present on later steps in the batch.
    all_annotation_keys: dict[str, type] = {}
    for s in steps:
        for key, val in s.annotations.items():
            if key not in all_annotation_keys and val is not None:
                all_annotation_keys[key] = type(val)

    override = getattr(step_, "schema_override", {})
    override_annotation_keys = {
        column_to_key(col)[1]: col for col in override if col.startswith("annotation:")
    }

    for key in sorted(set(all_annotation_keys.keys()) | set(override_annotation_keys.keys())):
        col = key_to_column("annotation", key)
        if col in override:
            fields.append(pa.field(col, override[col]))
        else:
            val_type = all_annotation_keys[key]
            if val_type is bool:
                fields.append(pa.field(col, pa.bool_()))
            elif val_type is int:
                fields.append(pa.field(col, pa.int64()))
            elif val_type is float:
                fields.append(pa.field(col, pa.float64()))
            elif val_type is str:
                fields.append(pa.field(col, pa.utf8()))
            else:
                raise ValueError(f"Unsupported annotation type for key '{key}': {val_type}")

    schema = pa.schema(fields)
    if metadata:
        schema = schema.with_metadata(metadata)
    return schema


def into_table(
    data: IntoSteps,
    *,
    schema: pa.Schema | None = None,
    episode_id_validator: Callable[[str], str] | None = None,
    encode_jpeg: bool = False,
    encode_video_blob: bool = False,
    group_size: int = constants.DEFAULT_VIDEO_BLOB_GROUP_SIZE,
    crf: int = constants.DEFAULT_VIDEO_BLOB_CRF,
    jpeg_quality: int = constants.DEFAULT_JPEG_QUALITY,
    codec: str = constants.DEFAULT_VIDEO_BLOB_CODEC,
    schema_metadata: dict[str, str] | None = None,
) -> pa.Table:
    """Normalize any :data:`IntoSteps` value to a :class:`pa.Table`.

    This is the universal converter used by :meth:`PackDS.write_steps`.
    It does **not** resolve episode identifiers — that is the caller's
    responsibility when writing ``list[PackDSStep]``.

    For ``list[PackDSStep]`` inputs, video may be pre-encoded already, or
    this function can encode it when the write-path encoding kwargs are provided.

    Args:
        data: Steps in any accepted form.
        schema: Optional PyArrow schema to use. If *None* and *data* is a
            list of steps, a schema is inferred via :func:`infer_schema`.

    Returns:
        A ``pa.Table`` ready for LanceDB ingestion.

    Raises:
        TypeError: If *data* is not an accepted type.
    """
    if isinstance(data, pa.Table):
        return data

    if isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data], schema=data.schema)

    # Deferred imports to avoid circular dependency at module scope
    from groot.core.data.packds import step

    if isinstance(data, step.PackDSStepBatch):
        return _step_batch_to_table(data, schema=schema)

    if isinstance(data, list):
        if not data:
            if schema is not None:
                return pa.table([], schema=schema)
            return pa.table([])
        if isinstance(data[0], step.PackDSStep):
            if episode_id_validator is not None or encode_jpeg or encode_video_blob:
                validator = episode_id_validator or validate_episode_id_for_write
                return _prepare_steps(
                    data,
                    validate_episode_id_for_write=validator,
                    encode_jpeg=encode_jpeg,
                    encode_video_blob=encode_video_blob,
                    group_size=group_size,
                    crf=crf,
                    jpeg_quality=jpeg_quality,
                    codec=codec,
                    schema=schema,
                    schema_metadata=schema_metadata,
                )
            return _steps_list_to_table(data, schema=schema)

    raise TypeError(f"into_table() does not accept {type(data).__name__}")


def from_table(table: pa.Table) -> step.PackDSStepBatch:
    """Convert an Arrow table to a :class:`PackDSStepBatch`.

    The inverse of :func:`to_table`.  State and action columns are
    converted to numpy arrays with their natural ``(T, *shape)`` form.

    Example::

        tables = ds.read_episodes(["ep_000", "ep_001"])
        batch = from_table(tables["ep_000"])
        print(batch.state["pose_aa"].shape)  # (T, 30, 3)
    """
    return _pa_table_to_step_batch(table)


def into_step_batch(
    data: step.PackDSStepBatch | pa.Table | pa.RecordBatch,
) -> step.PackDSStepBatch:
    """Convert tabular data to a :class:`PackDSStepBatch`.

    Prefer :func:`from_table` for ``pa.Table`` input.  This function
    additionally accepts ``PackDSStepBatch`` (passthrough) and
    ``pa.RecordBatch``.
    """
    from groot.core.data.packds import step

    if isinstance(data, step.PackDSStepBatch):
        return data

    if isinstance(data, pa.RecordBatch):
        data = pa.Table.from_batches([data], schema=data.schema)

    if not isinstance(data, pa.Table):
        raise TypeError(f"into_step_batch() does not accept {type(data).__name__}")

    return _pa_table_to_step_batch(data)


# ---- private helpers --------------------------------------------------------


def _pa_table_to_step_batch(table: pa.Table) -> step.PackDSStepBatch:
    """Convert a PyArrow Table to a PackDSStepBatch.

    Routes each column by its prefix to the appropriate batch field.
    """
    from groot.core.data.packds import step

    n = table.num_rows

    # Core fields with defaults
    episode_index: np.ndarray | None = None
    step_index = np.zeros(n, dtype=np.int64)
    step_time_ns = np.zeros(n, dtype=np.int64)
    original_step_time_ns = np.zeros(n, dtype=np.int64)
    has_valid_action = np.ones(n, dtype=bool)
    episode_id: list[str] = [""] * n

    video_: dict[str, list[bytes]] = {}
    state: dict[str, np.ndarray] = {}
    action: dict[str, np.ndarray] = {}
    annotations: dict[str, list] = {}
    video_blob: dict[str, list[bytes | None]] = {}
    video_blob_frame_index: dict[str, np.ndarray] = {}

    for col_name in table.column_names:
        col = table.column(col_name)
        field = table.schema.field(col_name)

        if col_name == "episode_index":
            episode_index = col.to_numpy().copy()
        elif col_name == "step_index":
            step_index = col.to_numpy().copy()
        elif col_name == "step_time_ns":
            step_time_ns = col.to_numpy().copy()
        elif col_name == "original_step_time_ns":
            original_step_time_ns = col.to_numpy().copy()
        elif col_name == "has_valid_action":
            has_valid_action = np.array(col.to_pylist(), dtype=bool)
        elif col_name == "episode_id":
            episode_id = col.to_pylist()
        elif col_name.startswith(constants.VIDEO_PREFIX + ":"):
            _, key = column_to_key(col_name)
            video_[key] = col.to_pylist()
        elif col_name.startswith(constants.VIDEO_BLOB_PREFIX + ":"):
            _, key = column_to_key(col_name)
            video_blob[key] = col.to_pylist()
        elif col_name.startswith(constants.VIDEO_BLOB_FRAME_INDEX_PREFIX + ":"):
            _, key = column_to_key(col_name)
            video_blob_frame_index[key] = np.array(col.to_pylist(), dtype=np.int32)
        elif col_name.startswith(constants.STATE_PREFIX + ":"):
            _, key = column_to_key(col_name)
            if is_nested_fixed_size_list_tensor_type(field.type):
                state[key] = fixed_size_list_array_to_numpy(col, field.type)
            else:
                vals = col.to_pylist()
                arrs = [np.array(v, dtype=np.float32) for v in vals if v is not None]
                state[key] = np.stack(arrs) if arrs else np.empty((0,), dtype=np.float32)
        elif col_name.startswith(constants.ACTION_PREFIX + ":"):
            _, key = column_to_key(col_name)
            if is_nested_fixed_size_list_tensor_type(field.type):
                action[key] = fixed_size_list_array_to_numpy(col, field.type)
            else:
                vals = col.to_pylist()
                arrs = [np.array(v, dtype=np.float32) for v in vals if v is not None]
                action[key] = np.stack(arrs) if arrs else np.empty((0,), dtype=np.float32)
        elif col_name.startswith("annotation:"):
            _, key = column_to_key(col_name)
            annotations[key] = col.to_pylist()

    return step.PackDSStepBatch(
        episode_index=episode_index,
        step_index=step_index,
        step_time_ns=step_time_ns,
        original_step_time_ns=original_step_time_ns,
        has_valid_action=has_valid_action,
        episode_id=episode_id,
        video=video_,
        state=state,
        action=action,
        annotations=annotations,
        video_blob=video_blob,
        video_blob_frame_index=video_blob_frame_index,
    )


def _step_batch_to_table(
    batch: step.PackDSStepBatch,
    schema: pa.Schema | None = None,
) -> pa.Table:
    """Convert a PackDSStepBatch to a pa.Table."""
    arrays: dict[str, pa.Array] = {}

    if batch.episode_index is not None:
        arrays["episode_index"] = pa.array(batch.episode_index.tolist(), type=pa.int64())
    arrays["episode_id"] = pa.array(batch.episode_id, type=pa.utf8())
    arrays["step_index"] = pa.array(batch.step_index.tolist(), type=pa.int64())
    arrays["step_time_ns"] = pa.array(batch.step_time_ns.tolist(), type=pa.int64())
    arrays["original_step_time_ns"] = pa.array(
        batch.original_step_time_ns.tolist(), type=pa.int64()
    )
    arrays["has_valid_action"] = pa.array(batch.has_valid_action.tolist(), type=pa.bool_())

    for key, vals in batch.video.items():
        col = key_to_column(constants.VIDEO_PREFIX, key)
        arrays[col] = pa.array(vals, type=pa.binary())

    for key, vals in batch.video_blob.items():
        col = key_to_column(constants.VIDEO_BLOB_PREFIX, key)
        arrays[col] = pa.array(vals, type=pa.binary())

    for key, arr in batch.video_blob_frame_index.items():
        col = key_to_column(constants.VIDEO_BLOB_FRAME_INDEX_PREFIX, key)
        arrays[col] = pa.array(arr.tolist(), type=pa.int32())

    for key, arr in batch.state.items():
        col = key_to_column(constants.STATE_PREFIX, key)
        field_type = schema.field(col).type if schema is not None and col in schema.names else None
        arrays[col] = numpy_to_fixed_size_list_array(arr, field_type)

    for key, arr in batch.action.items():
        col = key_to_column(constants.ACTION_PREFIX, key)
        field_type = schema.field(col).type if schema is not None and col in schema.names else None
        arrays[col] = numpy_to_fixed_size_list_array(arr, field_type)

    for key, vals in batch.annotations.items():
        col = key_to_column("annotation", key)
        arrays[col] = pa.array(vals)

    if schema is not None:
        # Build table following schema field order, using schema types
        ordered_arrays = []
        for field in schema:
            if field.name in arrays:
                ordered_arrays.append(arrays[field.name])
            else:
                # Create null array for missing columns
                ordered_arrays.append(pa.nulls(len(batch), type=field.type))
        return pa.table(ordered_arrays, schema=schema)

    # No schema provided — build from the arrays we have
    names = list(arrays.keys())
    return pa.table(list(arrays.values()), names=names)


def _steps_list_to_table(
    steps: list[step.PackDSStep],
    *,
    schema: pa.Schema | None = None,
) -> pa.Table:
    """Convert a list of PackDSStep to a pa.Table.

    Does **not** encode video — callers must use ``encode_steps()``
    first. It writes the canonical episode identity columns
    ``episode_id`` and ``step_index``.
    """
    # Verify all video is encoded — callers must use encode_steps() first
    for step_ in steps:
        if step_.decoded_video and not step_.encoded_video and not step_.encoded_video_blob:
            raise ValueError(
                "Steps have decoded_video but no encoded form. "
                "Call encode_steps() before into_table()."
            )

    # Infer schema if needed
    if schema is None:
        schema = infer_schema(steps)

    arrays: list[pa.Array] = []

    for field in schema:
        col = field.name

        if col == "episode_index":
            arrays.append(
                pa.array(
                    [step_.episode_index for step_ in steps],
                    type=pa.int64(),
                )
            )
        elif col == "episode_id":
            arrays.append(pa.array([step_.episode_id for step_ in steps], type=pa.utf8()))
        elif col == "step_index":
            arrays.append(pa.array([step_.step_index for step_ in steps], type=pa.int64()))
        elif col == "step_time_ns":
            arrays.append(pa.array([step_.step_time_ns for step_ in steps], type=pa.int64()))
        elif col == "original_step_time_ns":
            arrays.append(
                pa.array([step_.original_step_time_ns for step_ in steps], type=pa.int64())
            )
        elif col == "has_valid_action":
            arrays.append(pa.array([step_.has_valid_action for step_ in steps], type=pa.bool_()))
        elif col.startswith(constants.VIDEO_PREFIX + ":"):
            _, key = column_to_key(col)
            vals = []
            for step_ in steps:
                if step_.encoded_video and key in step_.encoded_video:
                    vals.append(step_.encoded_video[key])
                else:
                    vals.append(None)
            arrays.append(pa.array(vals, type=pa.binary()))
        elif col.startswith(constants.VIDEO_BLOB_PREFIX + ":"):
            _, key = column_to_key(col)
            vals = []
            for step_ in steps:
                if step_.encoded_video_blob and key in step_.encoded_video_blob:
                    vals.append(step_.encoded_video_blob[key])
                else:
                    vals.append(None)
            arrays.append(pa.array(vals, type=pa.binary()))
        elif col.startswith(constants.VIDEO_BLOB_FRAME_INDEX_PREFIX + ":"):
            _, key = column_to_key(col)
            vals = []
            for step_ in steps:
                if step_.video_blob_frame_index and key in step_.video_blob_frame_index:
                    vals.append(step_.video_blob_frame_index[key])
                else:
                    vals.append(None)
            arrays.append(pa.array(vals, type=pa.int32()))
        elif col.startswith(constants.STATE_PREFIX + ":"):
            _, key = column_to_key(col)
            arrs = []
            for step_ in steps:
                arr = step_.state.get(key)
                if arr is not None:
                    arrs.append(arr.astype(np.float32))
                else:
                    shape = nested_fixed_size_list_shape(field.type)
                    arrs.append(np.zeros(shape, dtype=np.float32))
            stacked = np.stack(arrs, axis=0)
            arrays.append(numpy_to_fixed_size_list_array(stacked, field.type))
        elif col.startswith(constants.ACTION_PREFIX + ":"):
            _, key = column_to_key(col)
            arrs = []
            for step_ in steps:
                arr = step_.action.get(key)
                if arr is not None:
                    arrs.append(arr.astype(np.float32))
                else:
                    shape = nested_fixed_size_list_shape(field.type)
                    arrs.append(np.zeros(shape, dtype=np.float32))
            stacked = np.stack(arrs, axis=0)
            arrays.append(numpy_to_fixed_size_list_array(stacked, field.type))
        elif col.startswith("annotation:"):
            _, ann_key = column_to_key(col)
            vals = [step_.annotations.get(ann_key) for step_ in steps]
            arrays.append(pa.array(vals, type=field.type))

    # Detect annotation keys present in steps but missing from the provided
    # schema.  Raise instead of silently dropping them.
    if schema is not None:
        schema_ann_keys = {
            column_to_key(f.name)[1] for f in schema if f.name.startswith("annotation:")
        }
        extra_keys: set[str] = set()
        for step_ in steps:
            for key in step_.annotations:
                if key not in schema_ann_keys:
                    extra_keys.add(key)
        if extra_keys:
            raise SchemaMismatchError(
                f"Steps contain annotation keys not in schema: {sorted(extra_keys)}. "
                f"Re-infer the schema to include them."
            )

    return pa.table(arrays, schema=schema)


def _get_relative_action_pairs(
    embodiment_tag: str | None,
) -> tuple[dict[str, tuple[str, str]], list[int], list[int]]:
    """Get action→(state, action_type) pairs and delta indices for relative stats."""
    empty: tuple[dict[str, tuple[str, str]], list[int], list[int]] = ({}, [], [])
    if not embodiment_tag:
        return empty
    try:
        from groot.core.training.configs.data import embodiment_configs
        from groot.core.data import types

        emb_config = embodiment_configs.MODALITY_CONFIGS.get(embodiment_tag, {})
        action_config = emb_config.get("action") if emb_config else None
        state_config = emb_config.get("state") if emb_config else None
        if not action_config or not action_config.action_configs:
            return empty

        pairs: dict[str, tuple[str, str]] = {}
        for action_key, act_cfg in zip(
            action_config.modality_keys, action_config.action_configs, strict=True
        ):
            if act_cfg.rep == types.ActionRepresentation.RELATIVE:
                state_key = act_cfg.state_key or action_key
                pairs[key_to_column(constants.ACTION_PREFIX, action_key)] = (
                    key_to_column(constants.STATE_PREFIX, state_key),
                    act_cfg.type.value,
                )
        if not pairs:
            return empty

        action_delta_indices = list(action_config.delta_indices)
        state_delta_indices = list(state_config.delta_indices) if state_config else [0]
        return pairs, action_delta_indices, state_delta_indices
    except (ImportError, KeyError, AttributeError):
        return empty


def _compute_and_set_stats_to_meta(
    packds_reader: PackDSReader,
    schema_meta: dict[str, str],
    *,
    embodiment_tag: str | None,
) -> None:
    """Compute column and relative statistics and add them to schema metadata.

    Both the column-stats and relative-stats paths stream the dataset
    in row-batch chunks (see ``stats.compute_column_statistics`` and
    ``stats.compute_relative_statistics``).  Peak memory is one batch's
    worth of state + action columns regardless of total dataset size,
    so we don't fall over on large datasets the way a whole-column
    materialisation does.
    """
    from groot.core.data.packds import stats

    stat_columns = [
        key_to_column(constants.STATE_PREFIX, k) for k in packds_reader.state_keys
    ] + [
        key_to_column(constants.ACTION_PREFIX, k) for k in packds_reader.action_keys
    ]
    column_stats = stats.compute_column_statistics(
        packds_reader, columns=stat_columns,
    )
    if column_stats:
        schema_meta["packds:stats"] = json.dumps(stats.remap_column_stats(column_stats))

    relative_pairs_with_types, action_delta_indices, state_delta_indices = _get_relative_action_pairs(
        embodiment_tag
    )
    if relative_pairs_with_types:
        relative_stats = stats.compute_relative_statistics(
            packds_reader,
            relative_pairs=relative_pairs_with_types,
            action_delta_indices=action_delta_indices,
            state_delta_indices=state_delta_indices,
        )
        if relative_stats:
            schema_meta["packds:relative_stats"] = json.dumps(
                {k.removeprefix("action:"): v for k, v in relative_stats.items()}
            )



def _prepare_steps(
    steps: list[step.PackDSStep],
    *,
    validate_episode_id_for_write: Callable[[str], str],
    encode_jpeg: bool,
    encode_video_blob: bool,
    group_size: int,
    crf: int,
    jpeg_quality: int,
    codec: str,
    schema: pa.Schema | None,
    schema_metadata: dict[str, str] | None,
) -> pa.Table:
    """Validate episode ids, encode video, and normalize steps to a table."""
    from groot.core.data.packds import video

    for step_ in steps:
        validate_episode_id_for_write(step_.episode_id)

    steps = video.encode_steps(
        steps,
        encode_jpeg=encode_jpeg,
        encode_video_blob=encode_video_blob,
        group_size=group_size,
        crf=crf,
        jpeg_quality=jpeg_quality,
        codec=codec,
    )

    if schema is None:
        schema = infer_schema(steps, metadata=schema_metadata)

    try:
        return into_table(steps, schema=schema)
    except SchemaMismatchError:
        batch_schema = infer_schema(steps)
        return into_table(steps, schema=batch_schema)



def key_to_column(prefix: str, key: str) -> str:
    """Convert a user-facing modality key to a PackDS column name.

    Lance v3 rejects ``.`` in top-level Arrow field names, interpreting
    them as struct field paths.  PackDS already uses ``:`` as a namespace
    separator (``state:joint_pos``), so we extend that convention: any
    ``.`` in the user key is escaped to ``:`` before the prefix is
    prepended.  :func:`column_to_key` reverses this.

    To keep the round-trip unambiguous, keys containing ``:`` (would
    collide with the separator/escaping) or ``..`` (no valid field name;
    reserves ``::`` in column space) are rejected.

    Examples::

        key_to_column("state", "joint_pos")       # "state:joint_pos"
        key_to_column("state", "left_arm.eef")     # "state:left_arm:eef"
        key_to_column("state", "left_arm.eef.pos") # "state:left_arm:eef:pos"

    Args:
        prefix: Modality prefix (e.g. ``"state"``, ``"action"``,
            ``"video"``, ``"annotation"``).
        key: User-facing modality key. May contain ``.`` but must not
            contain ``:`` or ``..``.

    Returns:
        Lance-safe column name.

    Raises:
        ValueError: If *key* contains ``:`` or ``..``.
    """
    if ":" in key:
        raise ValueError(
            f"Key must not contain ':' (reserved as separator): {key!r}"
        )
    if ".." in key:
        raise ValueError(
            f"Key must not contain '..' (ambiguous): {key!r}"
        )
    escaped = key.replace(".", ":")
    return f"{prefix}:{escaped}"


def column_to_key(column_name: str) -> tuple[str, str]:
    """Convert a PackDS column name back to its prefix and user-facing key.

    Reverses :func:`key_to_column`: splits on the first ``:`` to recover
    the prefix, then replaces every remaining ``:`` with ``.`` to restore
    the original key.

    This is needed because Lance v3 rejects ``.`` in top-level field
    names, so ``.`` is escaped to ``:`` at write time.  See
    :func:`key_to_column` for the full rationale.

    Examples::

        column_to_key("state:joint_pos")        # ("state", "joint_pos")
        column_to_key("state:left_arm:eef")      # ("state", "left_arm.eef")
        column_to_key("state:left_arm:eef:pos")  # ("state", "left_arm.eef.pos")

    Args:
        column_name: A PackDS column name containing at least one ``:``.

    Returns:
        ``(prefix, key)`` where *key* has ``.`` restored.

    Raises:
        ValueError: If *column_name* contains no ``:``.
    """
    if ":" not in column_name:
        raise ValueError(f"Column name must contain ':' separator: {column_name}")
    prefix, escaped_key = column_name.split(":", 1)
    key = escaped_key.replace(":", ".")
    return prefix, key


# Backwards-compatible aliases
_key_to_column = key_to_column
_column_to_key = column_to_key
