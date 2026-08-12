# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# DataSourceResolver — Design Doc

`groot/core/data/hydrator/data_source_resolver.py`

## Purpose

Training configs reference datasets by their Lustre paths (e.g.
`/mnt/aws-lfs-02/GrootDatasets/my_dataset`). When training runs on a GB200 cluster where
datasets have been migrated to S3, those paths must be transparently rewritten to the
correct `s3://` URI before they reach the dataset constructors and hydrator.

`DataSourceResolver` is responsible for this rewrite.
It is the single place where all path resolution logic lives.

## Resolution order

```text
# Step 0 — whitelist gate (local paths only)
if whitelist is non-empty and path does not start with any entry:
    return (original_path, None)       # stay on LustreFS, no tier tried

# Steps 1–N — tier loop
for tier in tiers (in order):          # e.g. aws → pdx
    1) Static mappings on this tier
    2) Auto-derive on this tier (only if tier has a region)
    if tier produced a reachable s3:// URI:
        return (uri, tier.profile)    # first winning tier — stop
return local path unchanged            # no tier succeeded
```

## Whitelist

`path_mapping.yaml` may contain a `whitelist:` list of local path prefixes.
When present and non-empty, **only paths that start with one of those prefixes proceed
to tier resolution**. All other local paths are returned unchanged (LustreFS fallback),
regardless of whether credentials or mappings are available.

```yaml
whitelist:
  - /mnt/aws-lfs-02/shared/datasets/agibot-custom-converted-1104-diverse-fullres
  - /mnt/aws-lfs-02/shared/datasets/gr1-diverse-2025
```

This is the primary operator control for limiting which datasets are read from S3/PDX.
Add a dataset to the whitelist only after it has been fully migrated.
An absent or empty `whitelist:` imposes no restriction (all paths eligible).

**Explicit `s3://` URIs are never gated by the whitelist.** The whitelist only applies
when the config still uses a **local** path and the resolver may rewrite it to remote.
If the user passes `s3://…` directly, that choice is respected: `resolve()` and
`get_profile_for_uri()` probe credentials for that URI without consulting `whitelist`.
(This is intentional: whitelist controls *local→remote* promotion, not blocking access
to buckets when the caller already named S3.)

The whitelist is a **prefix list**, not a glob.
`/mnt/aws-lfs-02/GrootDatasets/agibot-custom` matches every path that starts with that
string followed by `/` (or exactly equals it), so a shared prefix covers multiple
datasets without listing each one.

## Two-level resolution (tiers × steps per tier)

After the whitelist gate, resolution is **not** a single flat list of steps.
Think **outer loop over tiers**, **inner sequence per tier**:

**Per tier, in order:**

1. **Static mapping** — prefix→`s3://` entries (YAML + `TierConfig.mappings`, merged for
   *this* tier; longest local-path prefix wins).
   For the match, build the candidate URI and run **`HeadBucket`** with **this tier’s**
   credentials and endpoint style.
   - Accessible → return `(uri, profile)` for this tier.
   - If the prefix **matched** but **`HeadBucket` failed**, the loop **breaks** (no
     further static rows are tried for this path on this tier), then execution continues
     to **auto-derive below** on the **same** tier if `region` is set.
     (The in-code comment *“skip auto-derive”* next to `break` is misleading —
     auto-derive still runs.)

2. **Auto-derive** — only if `TierConfig.region` is set for this tier (after pool
   auto-fill for tier 0). Derives bucket names from the local path, checks exact bucket
   then ancestors (see [Auto-derive](#auto-derive)). If nothing matches, this tier
   returns `None` and the **next** tier runs.

**Cross-tier rule:** the **first** tier whose `resolve()` returns a non-`None`
`(uri, profile)` wins.
If tier A’s static mapping fails `HeadBucket` but tier A’s auto-derive finds a bucket,
tier A wins before tier B runs.
If tier A returns `None` entirely, tier B runs the same static→auto sequence.

**`s3://` input** (user-supplied URI, not a local path): the URI string is returned
unchanged. The **whitelist is not consulted** (see [Whitelist](#whitelist)). With
resolution enabled, the resolver runs **`HeadBucket` on the URI’s bucket for each tier
in order** (`_probe_s3_profile`) to pick `profile` and fill `_bucket_profile_map`. If
**no** tier can access the bucket, `resolve()` raises `ValueError`. With
`disable_path_mapping: true` (resolver disabled), `s3://` input returns `(uri, None)`
and does not probe.

**LustreFS fallback** — for a **local** path, if every tier returns `None`, `resolve()`
returns `(original_path, None)`.

## Tiers

Two tiers are configured by default (in `default_resolver()`):

| Tier | Profile | Region | `endpoint_style` | Purpose |
| --- | --- | --- | --- | --- |
| `aws` | `"aws"` | auto-detected (tier 0) | `regional` | Primary: AWS S3 on GB200 clusters |
| `pdx` | `"default"` | none | `profile` | Fallback: PDX/Ceph (profile `endpoint_url`) |

Each tier uses its own credentials (`TierConfig.profile`) and boto3 client
(`aws_utils.make_s3_client` with `TierConfig.endpoint_style`). If a tier’s credentials
are missing (e.g. `FileNotFoundError`), that tier’s `resolve` / client creation fails
softly where caught and the **next** tier is tried.

## Bucket naming

```text
path_to_bucket_name(local_path, region)
```

Segments of `local_path` are split on `/`, lowercased, underscores replaced with
hyphens, joined with `-`, then suffixed with `-{region}-nvidia-gear`.

```text
/mnt/aws-lfs-02/GrootDatasets  +  us-west-2
→ mnt-aws-lfs-02-grootdatasets-us-west-2-nvidia-gear

/A_B/C/D  +  us-east-1
→ a-b-c-d-us-east-1-nvidia-gear
```

The suffix function is `_bucket_name_suffix(region)` = `f"-{region}-nvidia-gear"`.

## Static mappings

Loaded from `path_mapping.yaml` at the repo default (or
`DataSourceConfig.mapping_yaml`).

**Merged into every tier** (same key→value table for each), so each tier evaluates the
same static prefixes with **its own** credentials and endpoint (AWS regional vs PDX/Ceph
profile endpoint). Per-tier `TierConfig.mappings` entries override YAML keys where both
define a key (`{**yaml_mappings, **tier_cfg.mappings}`).

```yaml
# path_mapping.yaml
mappings:
  /mnt/aws-lfs-02/GrootDatasets: s3://my-bucket/GrootDatasets
disable_path_mapping: false
```

- Longest local-path prefix wins (greedy matching).
- Values are used **as-is** — write the complete `s3://` URI including the exact bucket
  name. No region suffix is appended automatically for static mappings.
- If the mapped **bucket name** exists only on PDX (e.g. alias `GrootDatasets`), the AWS
  tier will typically log `HeadBucket` 404 before the PDX tier succeeds — expected, not
  a sign that the wrong “profile order” ran.
- Only add a static mapping **after** the dataset has been fully migrated when you rely
  on that mapping; auto-derive handles partial migrations when no static row matches.
- `disable_path_mapping: true` disables resolution (local paths unchanged; `s3://`
  inputs get `profile=None` without probing).

## Auto-derive

When no static mapping matches, and the tier has a region, the resolver tries to find a
bucket automatically:

1. Compute the exact bucket name for the full path: `path_to_bucket_name(path, region)`.
2. If the bucket exists → return `s3://{bucket}`.
3. Walk up ancestor paths (longest first).
   For each ancestor bucket that exists, check whether the relative sub-path prefix has
   been synced (`list_objects_v2 MaxKeys=1`).
   - Prefix synced → return `s3://{ancestor_bucket}/{key_prefix}`.
   - Prefix not synced → return `None` (skip tier; data not yet available here).
4. No ancestor bucket found → return `None`.

## Region auto-detection

When `TierConfig.region` is `None` for the first tier, the region is auto-detected at
load time:

1. If `WORKFLOW_ID` is set, call `osmo task list` to get the osmo pool name.
2. Else read `CLUSTER_TYPE` (Sky jobs set this in `sky_ray_job.yaml`, e.g. `gb300`).
3. Look up the pool / cluster name in `pool_to_region` in `path_mapping.yaml`.

On GB300 Sky jobs, `CLUSTER_TYPE=gb300` maps to `us-east-2` so tier-0 uses
`https://s3.us-east-2.amazonaws.com` for `HeadBucket` instead of a global PDX
`S3_ENDPOINT_URL`.

If neither env is set, osmo is unavailable, or the name is unknown in YAML, region
detection returns `None` and auto-derive is skipped for that tier.

## Credential handling and endpoints

Each tier builds its own boto3 S3 client via `_Tier._make_s3_client`, which delegates to
`aws_utils.make_s3_client(profile, region, endpoint_style=...)` from `utils/aws.py`.

Credential resolution order (in `aws.py`):

1. `GEAR_DATA_CREDENTIALS` env var (base64-encoded multi-profile JSON, injected by
   osmo).
2. `~/.gear/data_credentials` file on disk.

`endpoint_style`:

- **`regional`** — when `region` is set, forces `https://s3.{region}.amazonaws.com`
  (overrides profile `endpoint_url`). Used for the default AWS tier.
- **`profile`** — keeps profile `endpoint_url` for PDX/Ceph; region may still be set for
  naming only.

If credential loading fails for a tier (e.g. `FileNotFoundError` when no credentials
file exists and the env var is not set), that tier is skipped where the code catches the
error and the next tier is tried.

## Lazy loading and caching

The resolver loads config lazily on the first `resolve()` call:

- YAML is read once.
- Osmo is queried once for region detection.
- S3 clients are created lazily per tier on first use.
- Auto-derive results are cached per `(tier, path)` so repeated resolves of the same
  path do not make additional S3 API calls.

Call `resolver.reload()` to clear all state (useful in tests).

## `resolve()` return value — `(uri, profile)` tuple

`resolve()` always returns a `(uri, profile)` tuple:

```python
# Local path resolved to AWS S3
uri, profile = resolver.resolve("/mnt/aws-lfs-02/GrootDatasets/my_dataset")
# uri     -> "s3://my-bucket/GrootDatasets/my_dataset"
# profile -> "aws"   (the tier that won)

# User-supplied s3:// URI — probed, not rewritten
uri, profile = resolver.resolve("s3://some-bucket/data/video.mp4")
# uri     -> "s3://some-bucket/data/video.mp4"  (unchanged)
# profile -> "aws" or "default"  (first tier whose HeadBucket succeeds)
# raises ValueError if no tier can access the bucket
```

- For a **local path**, `profile` is the winning tier's `TierConfig.profile`; `None` on
  LustreFS fallback.
- For an **`s3://` input** with resolution enabled, `profile` is discovered by probing
  each tier via `HeadBucket` (`_probe_s3_profile`). `ValueError` is raised if no tier
  can access the bucket.
  With resolution disabled, `profile` is `None`.
- The resolver caches bucket → profile in `_bucket_profile_map`; subsequent calls for
  the same bucket are O(1) without re-probing.

## `get_profile_for_uri(uri)` — per-URI credential lookup

```python
# Warm cache (bucket seen previously via resolve() or a prior get_profile_for_uri call)
profile = resolver.get_profile_for_uri("s3://my-bucket/path/file.mp4")
# -> "aws"  (O(1) map lookup)

# Cold cache — probes tiers via HeadBucket on first access
profile = resolver.get_profile_for_uri("s3://new-bucket/path/file.mp4")
# -> "default"  (first tier that can HeadBucket the bucket)
# raises ValueError if no tier can access the bucket

profile = resolver.get_profile_for_uri("/local/path")
# -> None  (non-S3 URI)
```

On a **cache miss** for an `s3://` URI, `get_profile_for_uri` loads the resolver and
calls `_probe_s3_profile` — so callers do not need to call `resolve()` first for the
bucket to be known. The **whitelist does not apply** (same rule as `resolve()` for
explicit `s3://`: only local→remote rewriting is gated).
Used by `Hydrator._get_winning_profile()` and `HydratorClient.hydrate/glob`.

## How profile flows to S3 operations

```text
DataSourceResolver.resolve(path)
  → (uri, profile)  [cached in _bucket_profile_map]

Hydrator._get_winning_profile(path)
  → resolver.get_profile_for_uri(path) or config.credentials_profile

Hydrator.hydrate / glob / put_file / register_files
  → pass profile= to io.consume / io.glob_paths / CacheManager

CacheManager.register_file / submit_upload
  → WorkerPool.enqueue_download/upload(path, profile=profile)
  → task tuple: (kind, path, content, profile)

StorageIOWorker.process(task)
  → _get_backend(profile)
  → lazily creates / reuses StorageBackend with merged credentials_profile
  → backend.download_file / upload_file
```

A single `WorkerPool` serves all profiles.
No per-profile pools are created.
`StorageIOWorker` maintains `_backends: dict[str | None, StorageBackend]`:

- `None` → uses `backend_config` as-is (default `credentials_profile`).
- `"aws"` or `"default"` → creates a backend with `credentials_profile` overridden.
- Same profile key → cached; second call reuses the existing backend.

## Public API

```python
from groot.core.data.hydrator import data_source_resolver

# Module-level utilities used by provisioning tools
data_source_resolver.POOL_TO_REGION          # dict[str, str]
data_source_resolver.path_to_bucket_name(path, region)  # -> str
data_source_resolver.bucket_exists(s3_client, bucket)   # -> bool (re-raises unexpected errors)
data_source_resolver.ancestor_paths(path)               # -> list[str]

# Singleton resolver (used by Hydrator.resolve_path)
resolver = data_source_resolver.default_resolver()
uri, profile = resolver.resolve("/mnt/aws-lfs-02/GrootDatasets/my_dataset")

# Per-URI profile lookup — probes tiers on cache miss (resolve() not required first)
profile = resolver.get_profile_for_uri("s3://my-bucket/path/file.mp4")

# Reset singleton (useful in tests)
data_source_resolver.reset_default_resolver()
```

## Key classes

### `DataSourceResolver`

Top-level resolver. Holds the list of `_Tier` objects and the disabled flag.

| Method | Description |
| --- | --- |
| `resolve(path)` | Return `(uri, profile)`; local paths use tier loop; `s3://` probes tiers for `profile` (or `None` if disabled) |
| `get_profile_for_uri(uri)` | Map lookup, else `_probe_s3_profile`; `None` for non-S3; may raise `ValueError` |
| `reload()` | Clear all cached state; next resolve re-reads YAML and re-detects region |
| `_load()` | Build `_Tier` objects, merge YAML, fill in auto-detected region |
| `_load_yaml_mappings(yaml_path)` | Parse `path_mapping.yaml` → `(mappings, disable_flag)` |
| `_detect_region_from_pool()` | Query osmo for pool → region |

### `_Tier`

One resolution tier (e.g. AWS or PDX).

| Method | Description |
| --- | --- |
| `resolve(path)` | Return `s3://` URI or `None` |
| `_get_s3_client()` | Lazily build boto3 client |
| `_bucket_accessible(s3_uri)` | HeadBucket check; catch-all (never raises) |
| `_make_s3_client(region, profile, endpoint_style)` | Delegates to `aws_utils.make_s3_client` |
| `_apply_region_suffix(s3_uri, region)` | Append `-{region}-nvidia-gear` to bucket |
| `_bucket_exists(s3_client, bucket)` | Wraps `bucket_exists`; catch-all |
| `_prefix_exists(s3_client, bucket, prefix)` | `list_objects_v2` presence check |
| `_derive_s3_path(local_path, region, client)` | Auto-derive: exact + ancestor walk |

### `DataSourceConfig` / `TierConfig`

Plain dataclasses passed to `DataSourceResolver`.

```python
DataSourceConfig(
    tiers=[
        TierConfig(name="aws", profile="aws", region=None,
                   endpoint_style="regional",
                   mappings={"/mnt/lfs": "s3://explicit-bucket"}),
        TierConfig(name="pdx", profile="default",
                   endpoint_style="profile"),   # keep profile's custom endpoint_url
    ],
    mapping_yaml="/path/to/path_mapping.yaml",  # None = use repo-root default
    disable_resolution=False,
)
```

## Integration with Hydrator

`Hydrator.resolve_path(path)` calls `default_resolver().resolve(path)`. The returned
`(uri, profile)` pair is used both to rewrite the path and to prime
`_bucket_profile_map` so subsequent per-file operations automatically use the right
credentials.

`DatasetFactory` calls `Hydrator.resolve_path` for each dataset path before constructing
datasets. Paths that resolve to `s3://` trigger the hydrator to be auto-enabled.

## Tests

Unit tests are in `tests/groot/core/data/hydrator/test_data_source_resolver.py`. All S3
calls and credential lookups are mocked — no real AWS access required.

```bash
uv run pytest -s tests/groot/core/data/hydrator/test_data_source_resolver.py
```
