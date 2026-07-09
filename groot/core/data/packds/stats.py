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
"""Statistics computation for PackDS datasets.

Provides:
- compute_column_statistics: compute stats from a PackDS dataset
- compute_relative_statistics: compute relative (action - state) stats from a PackDS dataset

Both functions read the dataset in fixed-size batches so peak memory is
bounded by ``batch_size`` rows regardless of dataset size.  Mean / std /
min / max are exact via running ``sum`` + ``sum-of-squares`` totals; q01
and q99 are approximated by size-weighted averaging of per-batch
``np.percentile`` calls — within a few percent of the exact percentiles
on the multi-million-row datasets we feed into action normalisation.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Iterator

import numpy as np
import pyarrow as pa

from groot.core.data.packds import constants
from groot.core.data.packds.utils import (
    column_to_key,
    from_table,
    key_to_column,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from groot.core.data.packds.dataset import PackDSReader
    from groot.core.data.types import ActionType


class _StreamingStats:
    """Running mean/std/min/max + size-weighted batched quantiles.

    Accepts samples in successive ``update(data)`` calls without holding
    history.  Mean and std are exact (computed from running ``sum`` and
    ``sum-of-squares``); min and max are exact running reductions.  q01
    and q99 are approximated by a size-weighted average of per-batch
    ``np.percentile`` calls — accurate enough for action-normalisation
    use without the O(n log n) cost of sorting the full distribution.
    """

    __slots__ = ("_n", "_sum", "_sumsq", "_min", "_max", "_q01_acc", "_q99_acc")

    def __init__(self) -> None:
        # Lazily sized on first ``update`` so callers don't have to
        # pre-declare the dimension.
        self._n = 0
        self._sum: np.ndarray | None = None
        self._sumsq: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._q01_acc: np.ndarray | None = None
        self._q99_acc: np.ndarray | None = None

    def update(self, data: np.ndarray) -> None:
        """Add a batch of samples.  *data* has shape ``(batch_rows, dim)``."""
        if data.ndim != 2 or data.shape[0] == 0:
            return
        if data.dtype != np.float64:
            data = data.astype(np.float64)
        if self._sum is None:
            dim = data.shape[1]
            self._sum = np.zeros(dim, dtype=np.float64)
            self._sumsq = np.zeros(dim, dtype=np.float64)
            self._min = np.full(dim, np.inf, dtype=np.float64)
            self._max = np.full(dim, -np.inf, dtype=np.float64)
            self._q01_acc = np.zeros(dim, dtype=np.float64)
            self._q99_acc = np.zeros(dim, dtype=np.float64)
        w = data.shape[0]
        self._n += w
        self._sum += data.sum(axis=0)
        self._sumsq += np.square(data).sum(axis=0)
        self._min = np.minimum(self._min, data.min(axis=0))
        self._max = np.maximum(self._max, data.max(axis=0))
        # ``np.percentile`` here runs on a bounded-size batch, so cost
        # is O(batch * log batch) per update — independent of total n.
        self._q01_acc += np.percentile(data, 1, axis=0) * w
        self._q99_acc += np.percentile(data, 99, axis=0) * w

    def get_stats(self) -> dict[str, list[float]]:
        if self._n == 0 or self._sum is None:
            return {"mean": [], "std": [], "min": [], "max": [], "q01": [], "q99": []}
        n = self._n
        mean = self._sum / n
        # max(0, …) guards against tiny negative variance from floating
        # point rounding when all samples in a column are identical.
        var = np.maximum(self._sumsq / n - mean ** 2, 0.0)
        std = np.sqrt(var)
        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": self._min.tolist(),
            "max": self._max.tolist(),
            "q01": (self._q01_acc / n).tolist(),
            "q99": (self._q99_acc / n).tolist(),
        }


def remap_column_stats(
    column_stats: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, list[float]]]:
    """Remap raw column stats keys to the observation./action. convention.

    Converts ``state:foo`` → ``observation.foo`` and ``action:foo`` → ``action.foo``.
    Columns that don't match either prefix are dropped.  Uses
    :func:`~groot.core.data.packds.utils.column_to_key` to unescape ``:``
    back to ``.`` in key portions.

    Args:
        column_stats: Dict mapping column names (e.g. ``state:joint_pos``)
            to their stat dicts.

    Returns:
        Remapped dict with ``observation.`` / ``action.`` prefixed keys.
    """
    remapped: dict[str, dict[str, list[float]]] = {}
    for col_name, col_stats in column_stats.items():
        if col_name.startswith("state:"):
            _, key = column_to_key(col_name)
            stats_key = "observation." + key
        elif col_name.startswith("action:"):
            _, key = column_to_key(col_name)
            stats_key = "action." + key
        else:
            continue
        remapped[stats_key] = col_stats
    return remapped


def _iter_shuffled_index_batches(
    total_rows: int,
    batch_size: int,
    *,
    shuffle_seed: int | None,
) -> Iterator[list[int]]:
    """Yield sorted-within-batch index lists covering ``[0, total_rows)``.

    Cross-batch order is shuffled (so each batch is a representative
    sample of the dataset) but indices *within* a batch are sorted
    ascending — Lance scans its fragments in dataset order and
    ``np.percentile`` is order-invariant inside a batch, so we get
    unbiased quantile estimates without sacrificing read locality.

    ``shuffle_seed=None`` reproduces the pre-shuffle behaviour (plain
    sequential index ranges).
    """
    if total_rows <= 0 or batch_size <= 0:
        return
    if shuffle_seed is None:
        for start in range(0, total_rows, batch_size):
            yield list(range(start, min(start + batch_size, total_rows)))
        return
    rng = np.random.default_rng(shuffle_seed)
    permuted = rng.permutation(total_rows)
    for start in range(0, total_rows, batch_size):
        yield np.sort(permuted[start:start + batch_size]).tolist()


def compute_column_statistics(
    packds: PackDSReader,
    columns: list[str] | None = None,
    batch_size: int = 10_000_000,
    shuffle_seed: int | None = 0,
) -> dict[str, dict[str, list[float]]]:
    """Compute statistics (mean, std, min, max, q01, q99) for dataset columns.

    Streams the dataset in ``batch_size`` row chunks and feeds each
    batch to a :class:`_StreamingStats` accumulator per column.  Peak
    memory is one batch's worth of state + action columns regardless of
    total dataset size.

    Quantile (q01, q99) estimation accuracy depends strongly on each
    batch being a representative sample of the global distribution.
    Sequential reads bunch episodes together — a single batch may
    cover only one or two episodes, biasing per-batch quantiles
    severely.  Reading rows in a fixed pseudo-random permutation
    instead spreads each episode's rows across many batches, so the
    size-weighted quantile average converges to the true global
    quantile.  Every row is still read exactly once.

    Index batching lives in :func:`_iter_shuffled_index_batches`,
    which sorts within each batch so Lance scans fragments in
    dataset order.  Quantile contributions are order-invariant
    inside a batch, so within-batch sort is statistically identical
    to a fully shuffled read.

    Args:
        packds: A PackDS dataset instance.
        columns: Column names to compute stats for. If None, computes for all
            columns starting with ``state:`` or ``action:``.
        batch_size: Rows per batch for memory efficiency.
        shuffle_seed: RNG seed for the row-permutation read order.  Pass
            ``None`` to read sequentially (legacy behaviour).  Default
            ``0`` makes the result deterministic across runs while still
            producing the same shuffle every call.

    Returns:
        Dict mapping column names to stat dicts with keys: mean, std, min, max,
        q01, q99.
    """
    total_rows = packds.num_steps
    if total_rows == 0:
        return {}

    # Determine columns to process
    if columns is None:
        columns = [
            key_to_column(constants.STATE_PREFIX, k) for k in packds.state_keys
        ] + [
            key_to_column(constants.ACTION_PREFIX, k) for k in packds.action_keys
        ]
    if not columns:
        return {}

    trackers: dict[str, _StreamingStats] = {}
    num_batches = (total_rows + batch_size - 1) // batch_size
    logger.info(
        "stats: streaming %d column(s) across %d step(s) in %d batch(es) of %d (%s order)",
        len(columns), total_rows, num_batches, batch_size,
        "sequential" if shuffle_seed is None else "shuffled",
    )
    stats_start = time.perf_counter()
    for batch_idx, batch_indices in enumerate(_iter_shuffled_index_batches(
        total_rows, batch_size, shuffle_seed=shuffle_seed,
    )):
        batch = from_table(packds.read_raw_indices(batch_indices, columns=columns))
        # Process state and action columns from batch separately
        # (state and action can share key names like "gripper", "x", etc.)
        for prefix, group in [("state", batch.state), ("action", batch.action)]:
            for key, arr in group.items():
                col = f"{prefix}:{key}"
                if col not in columns:
                    continue
                data = arr.reshape(arr.shape[0], -1)
                if data.shape[0] == 0:
                    continue
                if col not in trackers:
                    trackers[col] = _StreamingStats()
                trackers[col].update(data)
        # At default ``batch_size=10_000_000`` we emit a handful of
        # log lines per pass — no throttling needed.
        logger.info(
            "stats: column-stats batch %d/%d (%d steps this batch, %.1fs elapsed)",
            batch_idx + 1, num_batches, len(batch_indices),
            time.perf_counter() - stats_start,
        )

    return {col: tracker.get_stats() for col, tracker in trackers.items()}


def _relative_samples(
    ep_table: pa.Table,
    action_col: str,
    state_col: str,
    action_delta_indices: list[int],
    state_delta_indices: list[int],
    action_type: "ActionType",
) -> np.ndarray | None:
    """Compute ``(num_samples, horizon * dim)`` relative-action samples for one episode.

    Returns ``None`` when the episode has no usable samples (too short
    for the action horizon, or the requested columns are missing /
    empty).  Inputs are converted to numpy via :func:`from_table` so
    we never round-trip per-row dicts.
    """
    from groot.core.data.stats import compute_relative_action

    n = ep_table.num_rows
    max_action_offset = max(action_delta_indices)
    usable = n - max_action_offset
    if usable <= 0:
        return None
    state_ref_offset = state_delta_indices[-1]

    # ``from_table`` materialises state/action columns to numpy arrays
    # with their natural ``(T, dim)`` shape, so per-step indexing is
    # just a slice.
    batch = from_table(ep_table)
    _, action_key = column_to_key(action_col)
    _, state_key = column_to_key(state_col)
    action_arr = batch.action.get(action_key)
    state_arr = batch.state.get(state_key)
    if action_arr is None or state_arr is None or len(action_arr) == 0:
        return None
    action_arr = np.asarray(action_arr, dtype=np.float32).reshape(n, -1)
    state_arr = np.asarray(state_arr, dtype=np.float32).reshape(n, -1)

    delta_arr = np.asarray(action_delta_indices, dtype=np.int64)
    samples = []
    for i in range(usable):
        ref_state = state_arr[state_ref_offset + i]
        actions = action_arr[delta_arr + i]  # (horizon, dim)
        rel = compute_relative_action(actions, ref_state, action_type)
        samples.append(rel.flatten())
    if not samples:
        return None
    return np.stack(samples, axis=0)


def compute_relative_statistics(
    packds: PackDSReader,
    relative_pairs: dict[str, tuple[str, str]],
    action_delta_indices: list[int],
    state_delta_indices: list[int],
    batch_rows: int = 10_000_000,
    shuffle_seed: int | None = 0,
) -> dict[str, dict[str, list[float]]]:
    """Compute horizon-aware relative (action − state) statistics.

    Uses the same domain-aware math as LeRobot (via
    :func:`~groot.core.data.stats.compute_relative_action`) so that
    joint-space and (future) EEF actions are handled identically.

    Episodes are bundled into chunks whose total step count is roughly
    ``batch_rows``, then each chunk is read in one
    ``packds.read_episodes`` call (single ``isin`` filter).  This
    matches the column-stats I/O pattern (one ~10 M-row read per
    iteration) instead of paying per-call overhead per tiny episode.
    Episode-level shuffle still distributes per-batch quantile
    contributions across the dataset.

    Args:
        packds: A PackDS dataset instance.
        relative_pairs: Mapping ``{action_col: (state_col, action_type_str)}``
            where *action_type_str* is the ``.value`` of
            :class:`~groot.core.data.types.ActionType`.
        action_delta_indices: Action horizon offsets, e.g. ``[0, 1, ..., 15]``.
        state_delta_indices: State offsets, e.g. ``[0]``.
        batch_rows: Target step count per chunked read.  A chunk may
            slightly exceed this if a single episode is larger than
            ``batch_rows``.
        shuffle_seed: RNG seed for the episode-shuffle order.  Pass
            ``None`` to iterate episodes in their natural (catalog)
            order.  Default ``0`` is deterministic across runs.

    Returns:
        Dict mapping action column names to stat dicts with keys:
        mean, std, min, max, q01, q99. Values are lists of length
        ``horizon * dim``.
    """
    if not relative_pairs:
        return {}
    if packds.num_steps == 0:
        return {}

    from groot.core.data.types import ActionType

    eids = list(packds.get_episode_ids())
    if not eids:
        return {}
    lens = packds.get_episode_lengths(eids)
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(eids)

    # Greedy chunking — keep total rows per chunk near batch_rows.  A
    # single oversized episode still gets its own chunk; we never
    # split an episode across chunks because the horizon math needs
    # contiguous step rows.
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_n = 0
    for eid in eids:
        n = lens[eid]
        if cur and cur_n + n > batch_rows:
            chunks.append(cur)
            cur, cur_n = [], 0
        cur.append(eid)
        cur_n += n
    if cur:
        chunks.append(cur)

    action_types = {col: ActionType(at) for col, (_, at) in relative_pairs.items()}
    trackers: dict[str, _StreamingStats] = {}
    rel_start = time.perf_counter()
    logger.info(
        "stats: streaming relative-stats across %d episode(s) in %d chunk(s) "
        "(target %d row(s)/chunk, %s order)",
        len(eids), len(chunks), batch_rows,
        "sequential" if shuffle_seed is None else "shuffled",
    )
    for chunk_idx, chunk in enumerate(chunks):
        ep_tables = packds.read_episodes(chunk)
        chunk_rows = sum(t.num_rows for t in ep_tables.values())
        for ep_table in ep_tables.values():
            for action_col, (state_col, _) in relative_pairs.items():
                samples = _relative_samples(
                    ep_table, action_col, state_col,
                    action_delta_indices, state_delta_indices,
                    action_types[action_col],
                )
                if samples is None:
                    continue
                if action_col not in trackers:
                    trackers[action_col] = _StreamingStats()
                trackers[action_col].update(samples)
        logger.info(
            "stats: relative-stats chunk %d/%d (%d episode(s), %d row(s), %.1fs elapsed)",
            chunk_idx + 1, len(chunks), len(chunk), chunk_rows,
            time.perf_counter() - rel_start,
        )

    # Reshape from flat (horizon*dim,) to nested (horizon, dim) lists
    # to match LeRobot convention expected by downstream normalizers.
    horizon = len(action_delta_indices)
    result = {}
    for col, tracker in trackers.items():
        raw = tracker.get_stats()
        if not raw["mean"]:
            continue
        dim = len(raw["mean"]) // horizon
        result[col] = {k: np.array(v).reshape(horizon, dim).tolist() for k, v in raw.items()}
    return result

