# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# PackDS: High-Performance Dataset Format for VLA Training

PackDS is a high-performance dataset format with a backend-agnostic read API and a
[LanceDB](https://lancedb.github.io/lancedb/) backend for writing and advanced storage
operations.
It provides efficient storage of video frames, state vectors, and action vectors with
native support for random access and parallel loading.

## Key Features

- **Columnar Storage**: Built on LanceDB for efficient columnar storage with native
  support for variable-length arrays
- **JPEG Video Encoding**: Video frames stored as JPEG bytes (default quality 95) for
  compact storage
- **Video Blob Encoding**: Groups of adjacent frames compressed as H.264 or AV1 video
  segments for even higher compression via inter-frame prediction
- **Typed Tensor Storage**: State and action vectors stored as
  nested Arrow `fixed_size_list` columns — shape-preserving and interoperable
- **Random Access**: O(1) access to any row by index
- **Episode Iteration**: Efficient sequential access within episodes
- **Batch Operations**: Read multiple rows efficiently with `read_batch()`
- **Columnar Interchange**: `write_steps()` accepts `list[PackDSStep]`,
  `PackDSStepBatch`, `pa.Table`, or `pa.RecordBatch` — no unnecessary row↔columnar
  conversions
- **Async Writes**: Background batching for high-throughput data collection
- **Schema Metadata**: All metadata stored in the LanceDB schema (no external files)
- **Schema Validation**: Automatic validation on open (read/append mode)

## File Structure

A PackDS dataset is a directory with the `.packds` extension. It contains one or more
named table directories:

```text
{table_name}.{format}
```

In version 6, the primary logical table is `steps`, with an optional
`episodes` companion table for precomputed per-episode metadata. Supported
formats are `lance` and `parquet`, so a dataset may contain entries like:

```text
my_dataset.packds/
├── steps.lance/
├── episodes.lance/   # optional companion for episode metadata
└── steps.parquet/
```

If multiple table formats are present, `PackDSReader(..., prefer=[...])` chooses which one
to read. `prefer` takes canonical format names such as `"lance"` and `"parquet"`.
Without an explicit preference, opening a multi-format dataset is an error. A format not
listed in `prefer` is selected only when it is the sole format present.

`steps.lance/` is the canonical PackDS Lance-backed table directory. PackDS does not
support a separate `steps.lancedb` path or any legacy-vs-modern distinction in the on-disk
format.

All metadata (version, stats, source info) is stored in the selected table schema metadata
with the `packds:` prefix. There is no separate `meta/` directory.

## Quick Start

### Reading a Dataset

```python
from groot.core.data.packds import PackDS, decode_step_video

# Open existing dataset
ds = PackDS("/path/to/dataset.packds")

# Inspect
print(f"Total steps: {ds.num_steps}")
print(f"Episodes: {ds.num_episodes}")
print(f"Video keys: {ds.video_keys}")
print(f"State keys: {ds.state_keys}")
print(f"Action keys: {ds.action_keys}")

# Read all steps for an episode
episode_steps = ds.read_episode_steps(0)
step = decode_step_video(episode_steps[0])
print(step.decoded_video["cam_left"].shape)  # (H, W, 3)
print(step.state["joint_pos"])  # numpy array
print(step.action["joint_pos"])  # numpy array

# Read arbitrary rows by index (columnar)
batch = ds.read_batch([0, 1, 2, 3, 4])
print(batch.state["joint_pos"].shape)  # (5, dim)

ds.close()
```

`PackDSReader` can also resolve different source types before choosing a table format:

- local paths like `/data/run.packds`
- direct remote URIs like `s3://bucket/run.packds`
- `dc://...` DeltaCAT locators are currently not supported on this branch

### Creating a Dataset

```python
from groot.core.data.packds import PackDS, PackDSStep, encode_step_video
import numpy as np

with PackDS("/path/to/new.packds", mode="w", steps_per_second=30.0) as ds:
    for episode_index in range(2):
        for step_index in range(100):
            t_ns = round(step_index * int(1e9) / 30.0)
            step = PackDSStep(
                episode_id=f"ep_{episode_index:06d}",
                episode_index=episode_index,
                step_index=step_index,
                step_time_ns=t_ns,
                original_step_time_ns=t_ns,
                annotations={"task": "pick up the red cup"},
                decoded_video={"cam_left": np.zeros((480, 640, 3), dtype=np.uint8)},
                state={"joint_pos": np.random.randn(7).astype(np.float32)},
                action={"joint_pos": np.random.randn(7).astype(np.float32)},
                has_valid_action=step_index < 99,  # last step has no valid action
            )
            step = encode_step_video(step)
            ds.write_steps([step])
    # close() auto-finalizes (local only): flush -> stats -> episodes
    # table -> compact.  See "Write Modes" below for remote paths.
```

### Context Manager

```python
with PackDS("/path/to/dataset.packds") as ds:
    step = ds.read_episode_steps(0)[0]
```

## Write Modes

PackDS supports two write paths.  A third (remote + finalize) is
intentionally rejected.

### 1. Local write + finalize (the conversion path)

Use this whenever you want a fully-finalized dataset: stats computed
into schema metadata, episodes companion populated, fragments
compacted.

```python
# Write directly to local disk, then optionally upload.
with PackDS("/tmp/output.packds", mode="w") as ds:
    ds.write_steps(...)
# close() auto-finalizes: flush -> stats -> episodes table -> compact

# Or use the conversion pipeline, which stages a local Lance dir under
# $TMPDIR/packds_staging/... and bulk-uploads with s5cmd once finalize
# is done.
from groot.vla.omni.data.conversion.convert_to_packds import convert_to_packds
convert_to_packds(input_path=..., output_path="s3://bucket/out.packds", config=...)
```

For remote outputs the conversion orchestrator does the staging
automatically — see `groot.core.data.packds.staged_upload`.  Targeting
a remote URI directly with `PackDS(..., mode="w")` and the default
`finalize_on_close=True` raises (see mode 3 below).

### 2. Remote write, no finalize (streaming / online learning)

Use this for long-running ingest where each episode is finalized
individually and you never want the close-time `_finalize` pass to
run.  The episodes companion is still kept up to date — the stub
row goes in on the first `write_steps` for each new id, and either
`finalize_episode(eid)` or `close()` clears the `incomplete` flag.
Just no stats, no global compaction.

```python
with PackDS("s3://bucket/streaming.packds", mode="w",
            finalize_on_close=False) as ds:
    for episode_id, steps in stream:
        ds.write_steps(steps)
        ds.finalize_episode(episode_id)  # clears incomplete=True per episode
# close() flushes any remaining dirty episodes to incomplete=False —
# no stats compute, no fragment optimize.
```

### 3. Remote write + finalize — NOT SUPPORTED

`PackDSLanceDB._finalize` raises if `self.path` is a remote URI.
Reading back columns for stats and compacting fragments through an
object-store API is dominated by per-request latency; bulk-uploading
a locally-finalized dataset is consistently faster.

```python
# This raises:
with PackDS("s3://bucket/foo.packds", mode="w") as ds:  # default finalize_on_close=True
    ds.write_steps(...)
# RuntimeError: Cannot finalize a remote PackDS dataset (...).
#   - Set finalize_on_close=False to skip finalization (the episodes
#     companion is still written per-episode), or
#   - Write to a local path, finalize there, then bulk-upload via
#     groot.core.data.packds.staged_upload.
```

If you find yourself wanting this combination, the right answer is
almost always mode 1 with the conversion-pipeline staging — that's
the same logical workflow, just an order of magnitude faster.

### Async Writes for Data Collection

Writes are always non-blocking. Steps are buffered and written as fragments
in a background thread. Buffered writes flush only when ``min_batch_size`` is
reached or when you call ``flush()`` / ``close()``:

```python
with PackDS("/path/to/dataset.packds", mode="w", min_batch_size=30) as ds:
    for step in generate_steps():
        ds.write_steps([step])
```

### Writing Columnar Data Directly

`write_steps()` accepts columnar data to avoid unnecessary row↔columnar conversions:

```python
# Read a batch from one dataset, write directly to another
with PackDS("source.packds") as src, PackDS("dest.packds", mode="w") as dst:
    batch = src.read_batch(list(range(100)))  # PackDSStepBatch
    dst.write_steps(batch)  # no conversion needed

# Also accepts pa.Table and pa.RecordBatch
import pyarrow as pa
table = src.steps_table.to_lance().to_table()
dst.write_steps(table)
```

## API Reference

### PackDS

```python
class PackDS:
    def __init__(
        self,
        path: Path | str,
        mode: Literal["r", "w", "a"] = "r",
        min_batch_size: int = 30,
        jpeg_quality: int = 95,
        steps_per_second: float = 30.0,
        source_format: str = "packds",
        source_paths: list[str] | None = None,
        processing: dict[str, Any] | None = None,
        embodiment_tag: str | None = None,
        validate: bool = True,
    ): ...

    # --- Public fields (set at init, updated from schema on read/append) ---
    path: str
    jpeg_quality: int
    steps_per_second: float
    source_format: str
    source_paths: list[str]
    processing: dict[str, Any] | None
    steps_table: lancedb.table.Table | None   # Direct LanceDB table handle
    schema: pa.Schema | None                   # PyArrow schema of the table

    # --- Computed properties ---
    stats: dict[str, Any]               # Normalization statistics from schema metadata
    relative_stats: dict[str, Any]      # Relative action statistics
    num_episodes: int
    num_steps: int
    video_keys: list[str]
    state_keys: list[str]
    action_keys: list[str]

    # --- Write ---
    def write_steps(self, data: IntoSteps) -> None: ...
    def flush(self) -> None: ...

    # --- Read ---
    def read_batch(self, indices: list[int], columns: list[str] | None = None) -> PackDSStepBatch: ...
    def read_episode_steps(self, episode: int | str) -> list[PackDSStep]: ...
    def get_episode_indices(self) -> list[int]: ...
    def get_episode_ids(self) -> list[str]: ...
    def get_episodes_metadata(self) -> list[dict[str, Any]]: ...

    # --- Lifecycle ---
    def validate(self, check_all_rows: bool = False, sample_size: int = 100) -> tuple[bool, list[str]]: ...
    def optimize(self) -> None: ...
    def close(self) -> None: ...
```

**`IntoSteps`** is a union type:
`list[PackDSStep] | PackDSStepBatch | pa.Table | pa.RecordBatch`.

### PackDSStep

Dataclass representing a single timestep.

```python
@dataclass(kw_only=True)
class PackDSStep:
    episode_id: str                    # String episode ID (required)
    episode_index: int | None = None   # Integer index (auto-assigned if None)
    step_index: int = 0                  # Step index within episode
    has_valid_action: bool = True      # Whether action data is valid
    annotations: dict[str, str | int | float | bool]  # Typed annotations
    encoded_video: dict[str, bytes] | None   # JPEG-encoded video frames
    decoded_video: dict[str, np.ndarray] | None  # Decoded video (H, W, C)
    state: dict[str, np.ndarray]       # State vectors (key -> (dim,) array)
    action: dict[str, np.ndarray]      # Action vectors (key -> (dim,) array)
```

Common annotations:

- `task`: str - Task description
- `sub_task`: str - Sub-task description
- `action_source`: str - Action source (e.g., "human", "model", "other")
- `original_frame_index`: int - Original frame index before filtering

### PackDSStepBatch

Columnar batch returned by `read_batch()`. Supports indexing and iteration to extract
individual `PackDSStep` objects.

```python
@dataclass
class PackDSStepBatch:
    episode_index: np.ndarray          # (N,) int64
    step_index: np.ndarray               # (N,) int64
    has_valid_action: np.ndarray       # (N,) bool
    episode_id: list[str]              # N strings
    video: dict[str, list[bytes]]      # key -> N JPEG byte strings
    state: dict[str, np.ndarray]       # key -> (N, *shape) float32
    action: dict[str, np.ndarray]      # key -> (N, *shape) float32
    annotations: dict[str, list]       # key -> N values
```

### Standalone Conversion Functions

```python
from groot.core.data.packds import infer_schema, into_table, into_step_batch

# Infer schema from steps
schema = infer_schema(steps, metadata={"packds:version": "4"})

# Convert any IntoSteps to pa.Table
table = into_table(steps)
table = into_table(batch, schema=existing_schema)

# Convert pa.Table or pa.RecordBatch to PackDSStepBatch
batch = into_step_batch(table)
```

### Video Encoding/Decoding

```python
from groot.core.data.packds import encode_step_video, decode_step_video

step = encode_step_video(step, quality=95)  # decoded_video -> encoded_video
step = decode_step_video(step)              # encoded_video -> decoded_video
```

## Schema Metadata

All dataset metadata is stored in the LanceDB schema metadata with the `packds:` prefix:

| Key | Type | Description |
| --- | --- | --- |
| `packds:version` | str | Format version (currently "6") |
| `packds:steps_per_second` | str | Control frequency in Hz |
| `packds:jpeg_quality` | str | JPEG encoding quality |
| `packds:source_format` | str | Origin format (e.g., "xdof", "lerobot") |
| `packds:source_paths` | JSON | Original data paths |
| `packds:processing` | JSON | Conversion processing params |
| `packds:stats` | JSON | Normalization statistics (mean, std, min, max, q01, q99) |
| `packds:relative_stats` | JSON | Relative action statistics |

## LanceDB Column Layout

| Column | Type | Description |
| --- | --- | --- |
| `episode_index` | int64 | Integer episode index |
| `episode_id` | string | String episode identifier (required) |
| `step_index` | int64 | Step index within episode |
| `has_valid_action` | bool | Whether action is valid |
| `video:{key}` | binary | JPEG-encoded frame (shape in field metadata) |
| `video_blob:{key}` | binary (nullable) | Compressed video segment (H.264/AV1) |
| `video_blob_frame_index:{key}` | int32 | Frame index within its blob (0-based) |
| `state:{key}` | nested `fixed_size_list(float32, ...)` | State tensor with preserved shape |
| `action:{key}` | nested `fixed_size_list(float32, ...)` | Action tensor with preserved shape |
| `annotation:{key}` | varies (bool/int64/float64/utf8) | Typed annotation |

## Video Color Format

All video data in PackDS uses **RGB channel order**. This applies to:

- `decoded_video` arrays in `PackDSStep` — `(H, W, 3)` uint8, channels are R, G, B
- JPEG encoding/decoding (baseline JPEG in RGB)
- Video blob encoding/decoding (PyAV converts to/from `rgb24`)
- MuJoCo's `Renderer.render()` also returns RGB, so simulation frames can be
  stored directly without color conversion

## Multiprocessing

LanceDB uses threading/locking that doesn't work well with `fork`. When using PackDS
with PyTorch DataLoader or Ray, use `spawn`:

```python
# PyTorch DataLoader
DataLoader(dataset, num_workers=4, multiprocessing_context="spawn")
```

PackDS supports pickling via `__getstate__`/`__setstate__` — connections are excluded
and eagerly restored on unpickle.

## Storage Efficiency

PackDS achieves compact storage through:

1. **JPEG video encoding**: Frames compressed to ~50-150KB each depending on resolution
   and quality (vs ~900KB raw at 640x480)
2. **Columnar storage**: LanceDB efficiently stores repetitive patterns
3. **Fixed-shape tensors**: Zero-copy reads for state/action vectors, no pickle overhead

Typical storage: ~300-500KB per step (with 3 camera views at quality 95).

## Useful Commands

### List available PackDS datasets on S3

```bash
uv run gear data ls 'GrootDatasets/yam_packds_v1/'
```

### Download a dataset

```bash
uv run gear data download 'GrootDatasets/yam_packds_v1/<dataset_name>.packds/**/*' <destination>/
```
