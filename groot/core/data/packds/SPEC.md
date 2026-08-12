# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# PackDS Format Specification (Version 5)

This document specifies the PackDS on-disk format independently of any particular
implementation or storage backend. A conforming implementation MAY use LanceDB, Parquet,
Zarr, SQLite, or any columnar store that can faithfully represent the schema described below.

## 1. File Layout

A PackDS dataset is a directory whose name ends with `.packds`.

A dataset contains one or more **table directories**. Table directories are named:

```text
{table_name}.{format}
```

`table_name` is the logical table name. In version 5, the only valid table name is:

- `steps`

So a dataset may contain entries like:

```text
my_dataset.packds/
├── steps.lance/
└── steps.parquet/
```

A reader selects exactly one representation of the `steps` table.

### 1.1 Supported table formats

The supported `format` values are:

- `lance`
- `parquet`

Format-specific layout rules:

- `steps.lance/` MUST contain a managed Lance dataset directory structure with
  `data/`, `_transactions/`, and `_versions/`.
- `steps.parquet/` MUST contain one or more `<uuid>.parquet` files and no other files.

### 1.2 Multiple representations

If multiple `steps` table representations are present in the same `.packds` directory,
readers MUST raise an error unless an explicit preference order is supplied by the caller
(e.g. `prefer=["lance", "parquet"]`).

Selection semantics are:

1. try the preferred representations in the given order
2. if none of those preferred representations are present, then:
   - if exactly one format is present in the dataset, readers MAY load that sole format,
     even though it was not listed in `prefer`
   - if more than one format is present in the dataset, readers MUST raise an error

In other words, a format not listed in `prefer` may be selected only when it is the only
format present in the dataset.

### 1.3 Lance-backed representation

`steps.lance/` is the canonical PackDS Lance-backed table directory. PackDS does not
support a separate `steps.lancedb/` path or any legacy-vs-modern distinction in the
on-disk format.

All data and metadata live inside the selected table representation — there are no
required sidecar files outside of it.

## 2. Row Model

Each row represents a single **step** (one timestep of a robot trajectory). Steps are
grouped into **episodes** (contiguous sequences of steps from a single trajectory).

## 3. Column Schema

Every column belongs to one of five namespaces. Namespaced columns use a **`prefix:key`**
naming convention with a colon separator.

### 3.1 Core Columns (no prefix)

These columns are required on every row.

| Column | Type | Description |
| --- | --- | --- |
| `episode_id` | utf8 | Primary episode identifier (non-empty string). |
| `step_index` | int64 | Step position within the episode, 0-based. |
| `step_time_ns` | int64 | Step timestamp in nanoseconds. |
| `original_step_time_ns` | int64 | Original timestamp before any filtering or resampling. |
| `has_valid_action` | bool | Whether the action columns on this row contain valid data. Typically `false` on the last step of an episode. |

### 3.2 Video Columns — `video:{key}`

| Column | Type | Field Metadata |
| --- | --- | --- |
| `video:{key}` | binary (nullable) | `{"shape": [H, W, C], "dim_names": ["height", "width", "channel"]}` (JSON) |

Each value is a **JPEG-encoded** frame (baseline JPEG, quality configurable, default 95).
The decoded frame is an `(H, W, C)` uint8 array in RGB channel order.

The shape metadata is stored on the Arrow/Parquet **field** (not on individual values) and
describes the decoded frame dimensions. All frames under the same key MUST share the same
shape.

### 3.3 Video Blob Columns — `video_blob:{key}` + `video_blob_frame_index:{key}`

Video blobs are an optional, more compact encoding that groups N adjacent frames into a
single compressed video segment using inter-frame prediction.

| Column | Type | Description |
| --- | --- | --- |
| `video_blob:{key}` | binary (nullable) | Compressed video segment containing N frames. |
| `video_blob_frame_index:{key}` | int32 | Index of this row's frame within its blob (0-based). |

The `video_blob:{key}` field carries the same shape metadata as `video:{key}`.

**Blob grouping rules:**

- Adjacent steps are grouped into blobs of a fixed **group size** (default 16).
- Groups MUST NOT span episode boundaries.
- Within a group, the first row (the **leader**) stores the blob bytes. All other rows in
  the group store null. Readers reconstruct non-leader frames by decoding the leader's
  blob and indexing by `video_blob_frame_index`.

**Supported codecs:** H.264, AV1.

**Valid containers:** MP4 (ISO BMFF) only.

A dataset records its blob codec in schema metadata (`packds:video_blob_codec`).

### 3.4 State & Action Columns — `state:{key}`, `action:{key}`

| Column | Type | Description |
| --- | --- | --- |
| `state:{key}` | nested `fixed_size_list(float32, shape)` | Observation tensor. |
| `action:{key}` | nested `fixed_size_list(float32, shape)` | Action tensor. |

Tensor columns are stored as nested Arrow fixed-size lists. The shape is part of the type
(e.g., `(6,) -> fixed_size_list(float32, 6)` and `(3, 3) -> fixed_size_list(fixed_size_list(float32, 3), 3)`).
All values are **float32**.

A dataset MAY have zero state or zero action columns (e.g., video-only datasets).

### 3.5 Annotation Columns — `annotation:{key}`

Annotations are per-step metadata with typed values.

| Python Type | Arrow Type |
| --- | --- |
| `bool` | bool |
| `int` | int64 |
| `float` | float64 |
| `str` | utf8 |

No other types are allowed. Annotation columns are nullable (a step need not carry every
annotation key present in the schema).

**Common annotation keys** (by convention, not required):

| Key | Type | Meaning |
| --- | --- | --- |
| `task` | utf8 | Natural-language task description. |
| `sub_task` | utf8 | Sub-task description. |
| `action_source` | utf8 | Origin of actions: `"human"`, `"model"`, etc. |
| `original_frame_index` | int64 | Frame index in the source recording before filtering. |

## 4. Schema Metadata

Dataset-level metadata is stored in the table schema where possible, not in a sidecar file. Each key
uses the **`packds:`** prefix.

| Key | Value Type | Required | Description |
| --- | --- | --- | --- |
| `packds:version` | str | yes | Format version. Currently `"5"`. |
| `packds:steps_per_second` | str (float) | yes | Control frequency in Hz (e.g., `"30.0"`). |
| `packds:jpeg_quality` | str (int) | no | JPEG encoding quality (1–100). |
| `packds:video_blob_codec` | str | no | Blob codec: `"h264"` or `"av1"`. |
| `packds:video_blob_crf` | str (int) | no | CRF value used for blob encoding. |
| `packds:video_blob_group_size` | str (int) | no | Frames per blob group. |
| `packds:source_format` | str | no | Origin format (e.g., `"lerobot"`, `"xdof"`, `"packds"`). |
| `packds:source_paths` | JSON array of str | no | Original data paths. |
| `packds:processing` | JSON object | no | Arbitrary conversion/processing parameters. |
| `packds:stats` | JSON object | no | Normalization statistics (see §5). |
| `packds:relative_stats` | JSON object | no | Relative action statistics (see §5). |

All metadata values are strings (JSON-encoded where noted). Readers MUST ignore unknown
`packds:` keys for forward compatibility.

## 5. Normalization Statistics

Statistics are computed over **state** and **action** columns only and stored in
`packds:stats` as a JSON object. Video and annotation columns are excluded.

### 5.1 Key Naming

Column names are mapped to stat keys by replacing the prefix:

| Column | Stat Key |
| --- | --- |
| `state:{key}` | `observation.{key}` |
| `action:{key}` | `action.{key}` |

### 5.2 Per-Key Statistics

Each stat key maps to an object with six fields, each a list of floats with length equal
to the flattened dimension of the tensor:

```json
{
  "observation.joint_pos": {
    "mean": [0.1, 0.2, ...],
    "std":  [1.0, 1.1, ...],
    "min":  [-3.0, -2.5, ...],
    "max":  [3.0, 2.5, ...],
    "q01":  [-2.8, -2.3, ...],
    "q99":  [2.8, 2.3, ...]
  }
}
```

- `mean`, `std`: sample mean and standard deviation.
- `min`, `max`: element-wise extremes.
- `q01`, `q99`: 1st and 99th percentiles.

### 5.3 Relative Statistics

`packds:relative_stats` contains statistics over **action − state** deltas across an
action horizon, used for relative action normalization. Same six fields per key, but
values are shaped as `(horizon, dim)` nested lists.

## 6. Episode Invariants

A conforming dataset MUST satisfy:

1. **Unique (episode_id, step_index) pairs.** No two rows share the same episode and step
   index.
2. **Sorted order.** Rows SHOULD be sorted by `(episode_id, step_index)` for efficient
   episode scans, but readers MUST NOT assume this.

## 7. Schema Inference

When creating a dataset from a sequence of steps:

1. **Core columns** are always present with fixed types.
2. **Video columns** are inferred from the first step that carries decoded video. Shape
   metadata is derived from the frame dimensions.
3. **State/action columns** are inferred from the first step. Shape is derived from the
   numpy array shape.
4. **Annotation columns** are the union of all annotation keys across all steps in the
   write batch. The type is inferred from the first non-null value for each key.
5. Steps MAY declare additional annotation columns (with no values in the current batch)
   via a **schema override** mechanism, ensuring sparse annotations are not dropped.

Once a schema is established, subsequent writes MUST match it. Extra or missing columns
(outside of nullable annotation columns) are schema errors.

## 8. Columnar Interchange

Write operations accept four input forms without unnecessary row↔columnar conversion:

| Input | Description |
| --- | --- |
| `list[Step]` | List of row-oriented step objects. Converted to columnar internally. |
| `StepBatch` | Columnar batch (dict of arrays). Converted to table. |
| `Table` | Backend-native columnar table. Written directly. |
| `RecordBatch` | Single-batch columnar data. Wrapped into table. |

This design avoids materializing row objects when data is already columnar (e.g., copying
between datasets).

## 9. Indices

For efficient filtering, implementations SHOULD create scalar indices on:

- `episode_id`
- `step_index`
- `step_time_ns`
- `original_step_time_ns`

## 10. Multiprocessing

The underlying storage (LanceDB, Lance) uses threading and file locking that is
incompatible with `fork`. Processes that read PackDS datasets MUST use **spawn** (not
fork) for multiprocessing. PackDS objects support pickling — storage connections are
excluded from pickle state and lazily re-established on unpickle.

## 11. Versioning

The format version is an integer stored as a string in `packds:version`. Breaking changes
to the column layout, metadata keys, or encoding conventions increment this version.
Readers SHOULD reject datasets whose version they do not support.

Current version: **5**.
