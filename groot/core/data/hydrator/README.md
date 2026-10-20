# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Data Hydrator

The Data Hydrator is a module for loading training data from remote storage (S3) into
DataLoader workers. It provides modality-aware file access, asynchronous prefetching, an
LRU cache with eviction, and a pluggable storage backend.
The public API (`Hydrator`, `HydratorClient`, `DatasetFactory`) is unchanged from the
caller's perspective.

## Architecture

```text
+=============================================================================================+
|                                  MAIN PROCESS (Train Rank)                                  |
|                                                                                             |
|  +------------------------+                                                                 |
|  | DatasetFactory.build() |                                                                 |
|  +-----------+------------+                                                                 |
|              |                                                                              |
|              v                 +-------------------------------------------------------+    |
|  +------------------------+    |                 IPC SERVER (Background Thread)        |    |
|  |        Hydrator        |    |                                                       |    |
|  +------------------------+    |  +-------------+                     +-------------+  |    |
|  | - config               |    |  | loop():     | <--- get(req) ----- | Mgr.Queue() |  |    |
|  | - cache_mgr            |    |  |  getattr()  |                     | (Requests)  |  |    |
|  +-----------+------------+    |  |  (*args)    |                     +-------------+  |    |
|              |                 |  |             |                     +-------------+  |    |
|              v                 |  |             | --- put(resp) ----> | Mgr.dict()  |  |    |
|  +------------------------+    |  +------+------+                     | (Responses) |  |    |
|  |      CacheManager      |<---+---------+                            +-------------+  |    |
|  +------------------------+    |         |                                             |    |
|  | - metadata             |    |         v                                             |    |
|  | - LRU etc.             |    |  +------------------+               +-------------+  |    |
|  +------------------------+    |  | WorkerPool       | -- schedules-> | Ray Workers |  |    |
|                                |  | (single pool)    |               | StorageIO   |  |    |
|                                |  | task: (kind,     |               | per-profile |  |    |
|                                |  |  path, data,     |               | lazy backend|  |    |
|                                |  |  profile)        |               +-------------+  |    |
|                                |  +------------------+                                |    |
|                                +-------------------------------------------------------+    |
|                                            |                                                |
|   get_ipc_client() ---> creates HydratorClient(Queue, Dict) passes to workers               |
+============================================|================================================+
                                             |
                                     (Queue & Dict refs)
                                             |
+============================================|================================================+
|                             FORKED / SPAWNED WORKER PROCESSES                               |
|                                                                                             |
|  +---------------------------------------------------------------------------------------+  |
|  |                                  HydratorClient                                       |  |
|  |                                                                                       |  |
|  |  1. get_file(path)                                                                    |  |
|  |  2. Queue.put((request_id, "get_file", (path,), {}))                                  |  |
|  |  3. while request_id not in response_dict:                                            |  |
|  |         sleep(0.01)                                                                   |  |
|  |  4. result = response_dict.pop(request_id)                                            |  |
|  +-----------------------------------------^---------------------------------------------+  |
|                                            |                                                |
|  +-----------------------------------------+---------------------------------------------+  |
|  |  DataLoader Workers (dataset getitem)   |   ProcessPoolExecutor (generate_stats)      |  |
|  +---------------------------------------------------------------------------------------+  |
+=============================================================================================+
```

## Why IPC (workaround for Ray + fork)

Only the **main process** owns the `Hydrator` and its Ray-backed `WorkerPool`. Worker
processes never touch `Hydrator` or Ray directly.

**Ray and fork don't mix.** The download/upload path goes through
`WorkerPool → Ray WorkerGroup` (Ray actors).
Ray's runtime is not safe to use across `fork()`: the child inherits duplicated or
invalid Ray state (drivers, GCS connections, in-memory handles), which causes hangs,
crashes, or undefined behavior.

**Who runs in workers:**

- **DataLoader workers** — PyTorch creates them via `fork` or `spawn`. Dataset
  `__getitem__` calls `get_file()` inside these workers.
- **Stats generation** — `_generate_all_stats()` uses a `ProcessPoolExecutor`; each
  worker runs `_generate_stats_for_path_worker(..., hydrator_client)` and needs hydrator
  operations without holding the real `Hydrator`.

**What we do instead:** The main process runs a single `Hydrator` and an **IPC server
thread** that reads from a `multiprocessing.Manager().Queue()` and writes results into a
`Manager().dict()`. Workers receive only a `HydratorClient` holding proxy references to
that queue and dict.
Any call that needs the hydrator (e.g. `get_file`, `put_file`, `register_files`) is sent
as `(request_id, method_name, args, kwargs)`; the server thread calls
`getattr(hydrator, method_name)(*args, **kwargs)` and stores the result.
Workers never hold the `Hydrator` or any Ray state.

## Module layout

```text
hydrator/
  service.py              — Hydrator: main entry point (IPC server, cache, backend)
  client.py               — HydratorClient: IPC client for worker processes
  client_protocol.py      — Protocol / interface shared by Hydrator and HydratorClient
  config.py               — HydratorConfig dataclass
  types.py                — FileState, FileMetadata, FileInfo, CacheStats, Modality
  exceptions.py           — HydratorException, HydratorFileNotFoundError, HydratorTimeoutError
  modality.py             — ModalityRouter, AccessMode (DOWNLOAD / DIRECT_READ / STREAMING)
  data_source_resolver.py — DataSourceResolver: resolve local paths to S3/PDX URIs and
                            select the correct credentials profile per URI
                            (see data_source_resolver.md for design doc)
  cache/
    manager.py            — CacheManager: file lifecycle, LRU eviction, download scheduling
    eviction_policy.py
  workers/
    pool.py               — WorkerPool: single Ray-backed task queue; profile travels in task
    worker.py             — StorageIOWorker Ray actor: lazy per-profile backend cache
  backends/
    base.py               — StorageBackend ABC
    factory.py            — create_backend()
    s3_boto3.py           — boto3 S3 backend
  utils/
    io.py                 — normalize_path, parse_s3_uri, glob_paths, read_file_content,
                            read_parquet_path, consume
    aws.py                — get_boto3_client_config, make_s3_client, read_s3_file_as_text,
                            get_pyarrow_s3_fs, get_s3_storage_options
  e2e_tests/              — end-to-end tests (require live S3 / Ray)
```

## Usage

### Basic (main process)

```python
from groot.core.data.hydrator import Hydrator, HydratorConfig
from groot.core.data.hydrator.types import Modality

config = HydratorConfig(max_workers=4, max_cache_size_gb=50.0)
hydrator = Hydrator(job_name="train_job", worker_id=0, config=config)

# Optional: prefetch videos asynchronously
video_paths = ["s3://bucket/episode0/video.mp4", "s3://bucket/episode1/video.mp4"]
hydrator.register_files(video_paths, modality=Modality.VIDEO)

# Blocks until ready; auto-registers if not prefetched
local_path = hydrator.get_file("s3://bucket/episode0/video.mp4")

# Upload
hydrator.put_file("s3://bucket/results/stats.json", content='{"ok": true}')

# Release when an episode is done (makes files eligible for LRU eviction)
hydrator.release_files(video_paths)

hydrator.shutdown()
```

### In DataLoader workers (via IPC)

The `DatasetFactory` creates the `Hydrator` in the main process and passes a
`HydratorClient` to each dataset worker.
You do not instantiate `HydratorClient` directly in normal usage.

```python
# In main process (DatasetFactory does this):
client = hydrator.get_ipc_client()

# In worker process (HydratorClient has the same interface):
local_path = client.get_file("s3://bucket/episode0/video.mp4")
df = client.hydrate("s3://bucket/meta/stats.parquet")
paths = client.glob("s3://bucket/data", "**/*.parquet")
```

### Path utilities

```python
# Join paths safely for both local and S3
path = Hydrator.join_path("s3://bucket/dataset", "meta/info.json")
# -> "s3://bucket/dataset/meta/info.json"

# Glob works for both local and S3
parquet_files = hydrator.glob("s3://bucket/data", "**/*.parquet")

# Load .json, .jsonl, or .parquet from local or S3
# Always use hydrator.hydrate() (or client.hydrate() in workers), never consume() directly
data = hydrator.hydrate("s3://bucket/meta/stats.parquet")  # -> pd.DataFrame
```

## Modality routing

| Modality | Extension | Access mode | Behavior |
| --- | --- | --- | --- |
| `VIDEO` | `.mp4`, `.mkv`, … | `DOWNLOAD` | Downloaded to local cache; `get_file` returns local path |
| `PARQUET` | `.parquet` | `DIRECT_READ` | Read directly from S3 via pyarrow; `get_file` returns S3 URI |
| `AUTO` | (any) | Detected from extension | As above |

## Configuration

Key `HydratorConfig` fields:

| Field | Default | Description |
| --- | --- | --- |
| `max_workers` | `4` | Concurrent Ray download workers |
| `max_cache_size_gb` | `50.0` | LRU eviction triggers above this |
| `tmp_root` | `None` (uses `./hydrate_cache`) | Local cache root |
| `download_timeout` | `300.0` s | Per-file download timeout |
| `retry_attempts` | `3` | Retries on download failure |
| `eviction_buffer` | `0.9` | Evict until size < `max_cache_size_gb * 0.9` |
| `cleanup_on_shutdown` | `True` | Delete cache on shutdown |
| `backend` | `"boto3"` | Storage backend (`"boto3"`, `"s5cmd"`, `"lustre"`) |
| `credentials_profile` | `None` (auto-resolved) | Profile in `~/.gear/data_credentials`; see below |

## Per-URI credential routing

The hydrator supports two credential tiers simultaneously — AWS S3 (GB200 clusters) and
PDX/Ceph swiftstack — without requiring multiple worker pools or manual profile
selection.

### How it works

1. `DataSourceResolver.resolve(path)` returns `(uri, profile)`: the resolved S3 URI
   **and** the credentials profile (`"aws"` or `"default"`) that successfully reached
   that bucket.
2. The resolver caches the bucket → profile mapping in `_bucket_profile_map`.
3. `Hydrator._get_winning_profile(path)` looks up that map via
   `resolver.get_profile_for_uri(path)`; falls back to `config.credentials_profile` for
   unknown buckets (e.g. direct `s3://` URIs that bypass resolution).
4. Every operation that touches S3 passes the winning profile all the way down:
   - `hydrate(path)` → `io.consume(path, profile=profile)`
   - `glob(base, pattern)` → `io.glob_paths(base, pattern, profile=profile)`
   - `put_file(path, content)` → `CacheManager.submit_upload(..., profile=profile)`
   - `register_files(paths)` → `CacheManager.register_file(..., profile=profile)`
5. Each task enqueued to `WorkerPool` carries the profile as the 4th element of the task
   tuple: `(kind, path, content, profile)`.
6. `StorageIOWorker` lazily creates one backend per profile
   (`_backends: dict[str | None, StorageBackend]`). An explicit profile overrides
   `credentials_profile` in `backend_config`; `None` uses the config as-is.

### Credential config default

`HydratorConfig.credentials_profile` defaults to `None`. At Hydrator init time, if it is
`None`, `DataSourceResolver._get_active_profile()` probes available credentials and
resolves it to `"aws"` or `"default"` automatically.
An explicit value skips this probe.

`credentials_profile` is now a **fallback** for paths whose bucket has not been seen by
the resolver (e.g. a raw `s3://` URI passed before any `resolve()` call).
For paths that went through `resolve()`, the per-bucket profile stored in
`_bucket_profile_map` takes precedence.

### `HydratorClient` credential routing

`HydratorClient.hydrate()` and `.glob()` also consult
`data_source_resolver.default_resolver().get_profile_for_uri(path)` before falling back
to `_credentials_profile`. This ensures direct S3 reads in worker processes use the same
per-bucket profile selection as the main process.

## Exceptions

| Exception | When raised |
| --- | --- |
| `HydratorFileNotFoundError` | S3 object does not exist (404 / `NoSuchKey`) |
| `HydratorTimeoutError` | Download or IPC call did not complete within timeout |
| `HydratorException` | Base class; other download or backend failures |

## Lifecycle (DatasetFactory)

1. `DatasetFactory._create_hydrator()` — creates `Hydrator`; IPC server starts
   automatically in `__init__`.
2. `DatasetFactory.build()` — passes `hydrator.get_ipc_client()` into each dataset.
3. Training loop — `dataset.__getitem__` calls `client.get_file(...)` via IPC.
4. `DatasetFactory.stop()` — calls `hydrator.shutdown()`, which stops the IPC server
   thread then shuts down `CacheManager` and (optionally) cleans the cache.

## S3 credentials

Credentials are read from `~/.gear/data_credentials` (gear credential file), or from the
`GEAR_DATA_CREDENTIALS` environment variable (base64-encoded JSON, injected by osmo into
Ray actor runtime envs).

The default profile is auto-resolved by `DataSourceResolver._get_active_profile()` on
first use. The per-URI winning profile is selected by `_get_winning_profile(path)` at
operation time and travels with every task.

Credential flow: `DataSourceResolver (bucket→profile map)` →
`Hydrator._get_winning_profile` → `WorkerPool task tuple` →
`StorageIOWorker._get_backend(profile)` → `StorageBackend`.

## Tests

Unit tests live in `tests/groot/core/data/hydrator/`. End-to-end tests (require live S3
and Ray) live in `tests/groot/core/data/hydrator/e2e_tests/`; their shared helpers live in
`groot/core/data/hydrator/e2e_tests/`.

```bash
# Unit tests
uv run pytest -s tests/groot/core/data/hydrator/

# Worker unit tests (no Ray required)
uv run pytest -s tests/groot/core/data/hydrator/workers/

# E2E (requires S3 access and a running Ray cluster)
uv run pytest -s tests/groot/core/data/hydrator/e2e_tests/test_cache.py
```
