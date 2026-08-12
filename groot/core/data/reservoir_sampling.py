# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Reservoir sampling and streaming merge for quantile (q01/q99) and Welford stats.

Used by relative stats computation to either:
- use_reservoir_sampling=True: stream with Algorithm R for approximate q01/q99 (low memory).
- use_reservoir_sampling=False: accumulate all data and compute exact quantiles (higher memory).
"""

import numpy as np

DEFAULT_RESERVOIR_SIZE = 50_000


def merge_batch_stats_with_reservoir(
    global_n: int,
    global_mean: np.ndarray | None,
    global_M2: np.ndarray | None,
    global_min: np.ndarray | None,
    global_max: np.ndarray | None,
    reservoir: np.ndarray | None,
    batch_chunks: list[np.ndarray],
    reservoir_size: int = DEFAULT_RESERVOIR_SIZE,
) -> tuple[
    int,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Merge batch statistics (Welford + min/max) and update reservoir sample for later q01/q99.

    Algorithm R: uniform random sample of the stream for approximate quantiles.
    """
    if not batch_chunks:
        return global_n, global_mean, global_M2, global_min, global_max, reservoir

    batch_arr = np.stack(batch_chunks, axis=0).astype(np.float64)
    n_batch = batch_arr.shape[0]

    # 1. Reservoir sampling (Algorithm R)
    if reservoir is None:
        reservoir = np.zeros(
            (reservoir_size,) + batch_arr.shape[1:],
            dtype=np.float64,
        )

    if global_n < reservoir_size:
        space_left = reservoir_size - global_n
        take = min(n_batch, space_left)
        reservoir[global_n : global_n + take] = batch_arr[:take]

        if take < n_batch:
            remaining_batch = batch_arr[take:]
            B_rem = remaining_batch.shape[0]
            j = np.random.randint(
                0,
                global_n + take + np.arange(1, B_rem + 1),
                size=B_rem,
            )
            mask = j < reservoir_size
            for res_idx, batch_idx in zip(j[mask], np.arange(B_rem)[mask]):
                reservoir[res_idx] = remaining_batch[batch_idx]
    else:
        stream_starts = global_n + np.arange(1, n_batch + 1)
        j = np.random.randint(0, stream_starts, size=n_batch)
        mask = j < reservoir_size
        for res_idx, batch_idx in zip(j[mask], np.arange(n_batch)[mask]):
            reservoir[res_idx] = batch_arr[batch_idx]

    # 2. Welford stats & min/max
    batch_min = np.min(batch_arr, axis=0)
    batch_max = np.max(batch_arr, axis=0)
    batch_mean = np.mean(batch_arr, axis=0)
    batch_var = np.var(batch_arr, axis=0)
    batch_M2 = batch_var * n_batch

    if global_n == 0:
        return (
            n_batch,
            batch_mean,
            batch_M2,
            batch_min,
            batch_max,
            reservoir,
        )

    assert (
        global_mean is not None
        and global_M2 is not None
        and global_min is not None
        and global_max is not None
    ), "global state must be set when global_n > 0"
    n_new = global_n + n_batch
    delta = batch_mean - global_mean
    new_mean = global_mean + delta * n_batch / n_new
    new_M2 = global_M2 + batch_M2 + (delta**2) * global_n * n_batch / n_new
    new_min = np.minimum(global_min, batch_min)
    new_max = np.maximum(global_max, batch_max)
    return (n_new, new_mean, new_M2, new_min, new_max, reservoir)


def merge_batch_stats_exact(
    global_n: int,
    global_mean: np.ndarray | None,
    global_M2: np.ndarray | None,
    global_min: np.ndarray | None,
    global_max: np.ndarray | None,
    batch_chunks: list[np.ndarray],
) -> tuple[
    int,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Merge batch statistics (Welford + min/max) only. No reservoir; use for exact path."""
    if not batch_chunks:
        return global_n, global_mean, global_M2, global_min, global_max

    batch_arr = np.stack(batch_chunks, axis=0).astype(np.float64)
    n_batch = batch_arr.shape[0]
    batch_min = np.min(batch_arr, axis=0)
    batch_max = np.max(batch_arr, axis=0)
    batch_mean = np.mean(batch_arr, axis=0)
    batch_var = np.var(batch_arr, axis=0)
    batch_M2 = batch_var * n_batch

    if global_n == 0:
        return (n_batch, batch_mean, batch_M2, batch_min, batch_max)

    assert (
        global_mean is not None
        and global_M2 is not None
        and global_min is not None
        and global_max is not None
    ), "global state must be set when global_n > 0"
    n_new = global_n + n_batch
    delta = batch_mean - global_mean
    new_mean = global_mean + delta * n_batch / n_new
    new_M2 = global_M2 + batch_M2 + (delta**2) * global_n * n_batch / n_new
    new_min = np.minimum(global_min, batch_min)
    new_max = np.maximum(global_max, batch_max)
    return (n_new, new_mean, new_M2, new_min, new_max)


def quantiles_from_reservoir(reservoir: np.ndarray, n_valid: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute q01 and q99 from a reservoir sample (first n_valid rows)."""
    sample = reservoir[:n_valid]
    q01 = np.quantile(sample, 0.01, axis=0).astype(np.float64)
    q99 = np.quantile(sample, 0.99, axis=0).astype(np.float64)
    return q01, q99


def exact_stats_from_chunks(all_chunks: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Compute exact mean, std, min, max, q01, q99 from stacked chunks.

    Chunks are expected to be a list of arrays each of shape (horizon, dim)
    (e.g. one relative action chunk per trajectory). They are stacked along
    axis=0 to form (N, horizon, dim); stats are computed over axis=0, yielding
    outputs of shape (horizon, dim) for per-horizon-step statistics.
    """
    arr = np.stack(all_chunks, axis=0).astype(np.float64)
    return {
        "mean": np.mean(arr, axis=0).astype(np.float64),
        "std": np.std(arr, axis=0).astype(np.float64),
        "min": np.min(arr, axis=0).astype(np.float64),
        "max": np.max(arr, axis=0).astype(np.float64),
        "q01": np.quantile(arr, 0.01, axis=0).astype(np.float64),
        "q99": np.quantile(arr, 0.99, axis=0).astype(np.float64),
    }
