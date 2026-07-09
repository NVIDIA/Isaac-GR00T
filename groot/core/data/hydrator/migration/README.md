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
# hydrator/migration

Tools for provisioning S3 buckets and copying local dataset trees to object storage on
GB200 GPU clusters. All tools are designed for datasets with **billions of files** and
are built on the xenna Pipeline for distributed execution.

* * *

## data_source_resolver.py (DataSourceResolver)

Path resolution is handled by `DataSourceResolver` in
`groot/core/data/hydrator/data_source_resolver.py`. See
[`data_source_resolver.md`](../data_source_resolver.md) for the full design doc.

In brief: given a local Lustre path, `DataSourceResolver.resolve()` tries each
configured tier (AWS S3, then PDX/Ceph) in order, falling back to the original local
path if no remote copy is found.
Paths already starting with `s3://` are passed through unchanged.

### Bucket naming

Bucket names are derived deterministically from the local path and region:

```text
path_to_bucket_name("/mnt/aws-lfs-02/GrootDatasets", "us-west-2")
→ mnt-aws-lfs-02-grootdatasets-us-west-2-nvidia-gear
```

Segments are lowercased, underscores replaced with hyphens, joined with `-`, and
suffixed with `-{region}-nvidia-gear`.

Static mapping values are used as-is — write the complete `s3://` URI:

```yaml
# path_mapping.yaml
mappings:
  /mnt/aws-lfs-02/GrootDatasets: s3://my-bucket-us-west-2-nvidia-gear/GrootDatasets
  /mnt/aws-lfs-02/shared/datasets: s3://GrootDatasets
```

### Region auto-detection

The region is detected at runtime from `WORKFLOW_ID` + `osmo task list`, then mapped via
`POOL_TO_REGION` in `data_source_resolver.py`. All GB200 pools map to `us-west-2`.

* * *

## provision_object_store.py

Idempotently provisions an S3 bucket for a local path on a target GB200 cluster pool.

### Bucket naming

Path segments are joined with `-`, lowercased, and suffixed with the AWS region and
`-nvidia-gear`. **Underscores within path segments are also replaced with hyphens** —
standard AWS S3 does not allow underscores in bucket names.

```text
/mnt/aws-lfs-02/GrootDatasets  +  groot-gb200-02
→  mnt-aws-lfs-02-grootdatasets-us-west-2-nvidia-gear

/A_B/C/D  +  us-east-1
→  a-b-c-d-us-east-1-nvidia-gear
```

### Supported pools

`POOL_TO_REGION` covers all GB200 pools, all mapping to `us-west-2`:

- `groot-gb200-01`, `groot-gb200-02`
- `groot-gb200-ci-01`, `groot-gb200-ci-02`
- `groot-gb200-nvlink-02`

### Usage

```bash
./scripts/provision_object_store.sh --path /mnt/aws-lfs-02/GrootDatasets --pool groot-gb200-02
./scripts/provision_object_store.sh --path /mnt/aws-lfs-02/GrootDatasets --region us-west-2
```

`--pool` and `--region` are mutually exclusive.
If the pool is not in `POOL_TO_REGION` the script exits immediately with a list of known
pools.

### resolve_bucket

`resolve_bucket()` maps a local path to an existing bucket, returning a
`(bucket_name, key_prefix)` tuple.
`key_prefix` is empty for an exact match, or the relative sub-path for a parent match
(used as the S3 key prefix inside the bucket).

| Case | Behaviour |
| --- | --- |
| Exact match — path `/A/B`, bucket `a-b-us-west-2-nvidia-gear` exists | Return `(bucket, "")` immediately |
| Parent match — path `/A/B/C/D`, bucket `a-b-c-us-west-2-nvidia-gear` exists | Return `(bucket, "D")` — reuse closest parent automatically |
| Child match — path `/A/B`, buckets under it exist | Warn and exit — use a more specific path |
| No match | Print provisioning command and exit |

Credentials are resolved via `get_boto3_client_config()` (gear `"aws"` profile), so this
works both locally and on remote Ray workers where only `GEAR_DATA_CREDENTIALS` is
available.

* * *

## copy_to_object_store.py

Distributed dataset copier that parallelizes `aws s3 sync` at the directory level using
a two-stage **xenna Pipeline**.

### Architecture

```text
input: [WalkRequest per path]
         │
    ┌────▼──────────────────────────────────────┐
    │  Stage 1: WalkWorker  (--walk-workers N)  │
    │  Walks local tree to --depth levels        │
    │  Returns list[SyncUnit]  →  fan-out        │
    └────┬──────────────────────────────────────┘
         │  bounded queue  (backpressure)
    ┌────▼──────────────────────────────────────┐
    │  Stage 2: SyncWorker  (--workers M)       │
    │  Runs: aws s3 sync                         │
    │        --region <region>                   │
    │        --endpoint-url https://s3.<r>.amazonaws.com │
    └────┬──────────────────────────────────────┘
         │
    result dicts (one per synced directory)
```

The two stages run concurrently — uploads begin as soon as the first walk completes
while remaining paths are still being walked.
Backpressure from the bounded inter-stage queue prevents the walk stage from running
unboundedly ahead.

### Key design decisions

- **Directory-level sync units** — `walk_to_depth(path, bucket, depth)` descends
  `--depth` levels using `os.scandir` (no full file listing).
  Each subdirectory becomes one `aws s3 sync` call.
  Shallower trees fall back to the deepest reachable directories.
  Symlinks are excluded.
- **Fan-out via Pipeline** — `WalkWorker.process()` returns `list[SyncUnit]`; the
  Pipeline automatically flattens these one-by-one into stage 2's queue.
- **No race conditions** — S3 is a flat key-value store, so syncing disjoint prefixes in
  parallel is safe. `_assert_disjoint()` is available as a pre-flight utility.
- **Explicit region and endpoint** — every `aws s3 sync` call includes `--region` and
  `--endpoint-url https://s3.<region>.amazonaws.com`, derived from the pool/region
  argument. This prevents `PermanentRedirect` errors caused by the AWS CLI defaulting to
  the wrong regional endpoint.
- **Credential isolation** — `SyncWorker.setup()` reads the gear `--profile` credentials
  via `get_boto3_client_config()` and overrides `AWS_ACCESS_KEY_ID` /
  `AWS_SECRET_ACCESS_KEY` in the subprocess environment, preventing swiftstack or other
  ambient credentials from leaking into `aws s3 sync`.
- **Proxy bypass** — `NO_PROXY`/`no_proxy` are set to `.amazonaws.com,...` in every
  worker subprocess so uploads go directly to S3 rather than through the cluster proxy.
- **Parent bucket support** — when no exact bucket exists for a path, `resolve_bucket()`
  finds the closest provisioned ancestor and uses the relative sub-path as an S3 key
  prefix automatically.
- **Multiple paths** — `--paths` or `--paths-file` accepts one or more local
  directories; each resolves its own bucket via `resolve_bucket`.
- **Multi-profile credentials** — `--profile` selects the gear credential profile
  (default `aws`; use e.g. `msc` for swiftstack).

### Rate control

Two-level:

- **`--workers M`** — number of concurrent `aws s3 sync` processes (stage 2 workers)
- **`--s3-concurrency N`** — `AWS_MAX_CONCURRENT_REQUESTS` per worker (file-level
  parallelism within each sync)
- **`--rate-limit-mbps`** — optional per-worker cap via `AWS_S3_TARGET_BANDWIDTH`

Total theoretical peak bandwidth ≈ `workers × s3_concurrency × per-connection speed`,
capped by `workers × rate_limit_mbps` when set.

### Usage

```bash
# Single path
python -m groot.core.data.hydrator.migration.copy_to_object_store \
    --paths /mnt/aws-lfs-02/GrootDatasets --pool groot-gb200-02

# Multiple paths (walked concurrently with uploads)
python -m groot.core.data.hydrator.migration.copy_to_object_store \
    --paths /mnt/aws-lfs-02/DatasetA /mnt/aws-lfs-02/DatasetB \
    --pool groot-gb200-02 \
    --depth 2 --walk-workers 4 --workers 16 \
    --s3-concurrency 32 --rate-limit-mbps 500

# From a file listing
python -m groot.core.data.hydrator.migration.copy_to_object_store \
    --paths-file datasets.txt --pool groot-gb200-02

# Dry run (no files transferred)
python -m groot.core.data.hydrator.migration.copy_to_object_store \
    --paths /mnt/data --pool groot-gb200-02 --dry-run

# Swiftstack / MSC profile
python -m groot.core.data.hydrator.migration.copy_to_object_store \
    --paths /mnt/data --region us-east-1 --profile msc
```

### Prerequisites

- `aws` CLI v2 on `PATH` (installed on cluster nodes via the GB200 cluster template)
- The target bucket must already exist — run `provision_object_store.sh` first
- A Ray cluster must be reachable, or one will be started locally

* * *

## launch_copy_to_object_store.sh

Submits a `copy_to_object_store` job to an osmo Ray cluster via
`scripts/launch_ray_job.py`. Handles repo packaging, credential passthrough, and log
streaming.

- Expands `--paths-file FILE` **locally** into explicit `--paths path1 path2 ...` before
  submission so the remote job receives concrete paths only (the file does not exist on
  the cluster).
- All `--paths`, `--depth`, `--workers`, `--profile`, etc.
  flags are forwarded as-is to the remote Python process.

### Usage

```bash
./groot/core/data/hydrator/scripts/launch_copy_to_object_store.sh \
    <workflow_id> <job_name> \
    (--paths PATH ... | --paths-file FILE) \
    (--pool POOL | --region REGION) \
    [--depth N] [--walk-workers N] [--workers N] \
    [--s3-concurrency N] [--rate-limit-mbps N] \
    [--profile PROFILE] \
    [--delete] [--dry-run]
```

Example (4-node GB200 cluster, 10 datasets):

```bash
./groot/core/data/hydrator/scripts/launch_copy_to_object_store.sh \
    lustre_s3_migration_uuid_7e2c-1 sync_datasets_2026_03_17 \
    --paths-file ~/Downloads/datasets.txt \
    --pool groot-gb200-02 \
    --depth 2 --workers 64 --s3-concurrency 32
```

Stream logs after submission:

```bash
gear ray tail <workflow_id> <submission_id>
```
