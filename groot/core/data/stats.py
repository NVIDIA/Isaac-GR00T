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

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from groot.core.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from groot.core.data.hydrator.client_protocol import HydratorClientProtocol
from groot.core.data.hydrator.service import Hydrator
from groot.core.data.reservoir_sampling import (
    DEFAULT_RESERVOIR_SIZE,
    exact_stats_from_chunks,
    merge_batch_stats_with_reservoir,
    quantiles_from_reservoir,
)
from groot.core.data.state_action.action_chunking import (
    JointActionChunk,
)
from groot.core.data.state_action.pose import JointPose
from groot.core.data.types import (
    ActionRepresentation,
    ActionType,
    EmbodimentTag,
    ModalityConfig,
)
from groot.core.data.utils import to_json_serializable
from groot.core.training.configs.data.embodiment_configs import MODALITY_CONFIGS
from groot.core.utils.state_action.rotation import compute_se3_relative

LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"


def compute_relative_action(
    actions: np.ndarray,
    ref_state: np.ndarray,
    action_type: ActionType,
) -> np.ndarray:
    """Compute relative actions w.r.t. a reference state.

    This is the shared implementation used by both LeRobot and PackDS stats
    computation so that the math is identical regardless of dataset format.

    Args:
        actions: Action chunk, shape ``(horizon, dim)``.
        ref_state: Reference state, shape ``(dim,)``.
        action_type: Determines the subtraction semantics (joint-space
            difference for NON_EEF, SE(3) relative transform for EEF).

    Returns:
        Relative actions, shape ``(horizon, dim)``.
    """
    if action_type == ActionType.NON_EEF:
        reference_frame = JointPose(ref_state)
        traj = JointActionChunk([JointPose(m) for m in actions]).relative_chunking(
            reference_frame=reference_frame
        )
        return np.stack([p.joints for p in traj.poses], dtype=np.float32)
    elif action_type == ActionType.EEF:
        assert len(ref_state) == 9  # xyz + rot6d
        assert actions.shape[1] == 9  # xyz + rot6d
        return compute_se3_relative(actions, ref_state)
    else:
        raise ValueError(f"Unknown ActionType: {action_type}")


LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_REL_STATS_FILENAME = "meta/relative_stats.json"

STAT_KEYS = ["mean", "std", "min", "max", "q01", "q99"]


def _normalize_stats_dict(data: dict) -> dict:
    """If stats are stored as {stat_name: {feature: val}}, reshape to {feature: {stat_name: val}}."""
    if not data or not set(data.keys()).issubset(STAT_KEYS):
        return data
    first = next(iter(data.values()))
    if not isinstance(first, dict):
        return data
    features = list(first.keys())
    return {f: {s: data[s][f] for s in data if f in data.get(s, {})} for f in features}


def _check_stats_validity(
    base_path: str, features: list[str], hydrator: HydratorClientProtocol
) -> tuple[bool, str]:
    """Return (True, reason) if valid, (False, reason) if missing/invalid. Reason is for main-process logging."""
    stats_uri = Hydrator.join_path(base_path, LE_ROBOT_STATS_FILENAME)
    try:
        data = hydrator.hydrate(stats_uri)
        raw = data[0] if data else {}
        stats = _normalize_stats_dict(raw) if isinstance(raw, dict) else {}
    except Exception as e:  # noqa: BLE001
        return (
            False,
            f"meta/stats.json NOT FOUND or failed to load at {stats_uri!r}: {e}",
        )
    required_stat_keys = ["mean", "std", "min", "max", "q01", "q99"]
    stats_keys = list(stats.keys())
    for feature in features:
        if feature not in stats:
            return (
                False,
                f"required feature {feature!r} not in stats.json "
                f"(stats keys: {stats_keys!r}; required from info.json: {features!r})",
            )
        if not isinstance(stats[feature], dict):
            return (False, f"stats[{feature!r}] is not a dict")
        for stat in required_stat_keys:
            if stat not in stats[feature]:
                return (
                    False,
                    f"feature {feature!r} missing stat {stat!r} (has: {list(stats[feature].keys())!r})",
                )
    return (True, f"valid at {stats_uri!r}, features {features!r} ok")


def _generate_stats(base_path: str, hydrator: HydratorClientProtocol) -> str:
    """Generate meta/stats.json using hydrator for all I/O. Returns reason string for main-process logging."""
    info_uri = Hydrator.join_path(base_path, LE_ROBOT_INFO_FILENAME)
    try:
        info_list = hydrator.hydrate(info_uri)
        first = info_list[0] if info_list else None
        le_features = first.get("features", {}) if isinstance(first, dict) else {}
    except Exception as e:
        raise FileNotFoundError(f"Dataset info not found: {info_uri}") from e
    if not le_features or not isinstance(le_features, dict):
        raise FileNotFoundError(f"Dataset info has no features: {info_uri}")
    lowdim_features = [f for f in le_features if "float" in le_features[f]["dtype"]]
    valid, reason = _check_stats_validity(base_path, lowdim_features, hydrator)
    if valid:
        return f"skipped: {reason}"
    parquet_files = hydrator.glob(base_path, LE_ROBOT_DATA_FILENAME)
    stats = calculate_dataset_statistics(parquet_files, lowdim_features, base_path=base_path)
    stats_path = Hydrator.join_path(base_path, LE_ROBOT_STATS_FILENAME)
    if stats_path.startswith("s3://"):
        hydrator.put_file(stats_path, json.dumps(stats, indent=4).encode("utf-8"))
        return f"generated: {reason} -> wrote to {stats_path}"
    write_path = hydrator.get_file(stats_path) or stats_path
    with open(write_path, "w") as f:
        json.dump(stats, f, indent=4)
    return f"generated: {reason} -> wrote to {write_path}"


def calculate_dataset_statistics(
    parquet_paths: list[Path] | list[str],
    features: list[str] | None = None,
    base_path: str | None = None,
) -> dict[str, dict[str, float]]:
    """Calculate the dataset statistics of all columns for a list of parquet files.

    Args:
        parquet_paths: List of paths (Path or S3 URI strings) to parquet files.
        features: List of feature names to compute statistics for. If None, all columns.
        base_path: Optional dataset root (e.g. s3://bucket/key). If S3, reads whole data/ in one go.

    Returns:
        Dictionary mapping feature names to statistical values (mean, std, min, max, q01, q99).
    """
    use_s3 = (
        parquet_paths
        and isinstance(parquet_paths[0], str)
        and str(parquet_paths[0]).startswith("s3://")
    )

    if use_s3 and base_path and str(base_path).startswith("s3://"):
        import pyarrow.dataset as ds

        prefix = str(base_path).rstrip("/") + "/data"
        arrow_table = ds.dataset(prefix, format="parquet").to_table()
        all_low_dim_data = arrow_table.to_pandas()
    else:
        all_low_dim_data_list = []
        for parquet_path in tqdm(
            sorted(parquet_paths),
            desc="Collecting all parquet files...",
        ):
            all_low_dim_data_list.append(pd.read_parquet(parquet_path))
        all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    if features is None:
        features = list(all_low_dim_data.columns)
    for le_modality in features:
        print(f"Computing statistics for {le_modality}...")  # noqa: T201
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


def check_stats_validity(dataset_path: Path | str, features: list[str]):
    stats_path = Path(dataset_path) / LE_ROBOT_STATS_FILENAME
    if not stats_path.exists():
        return False
    with open(stats_path) as f:
        stats = json.load(f)
    for feature in features:
        if feature not in stats:
            return False
        if not isinstance(stats[feature], dict):
            return False
        for stat in ["mean", "std", "min", "max", "q01", "q99"]:
            if stat not in stats[feature]:
                return False
    return True


def generate_stats(
    dataset_path: Path | str,
    hydrator: HydratorClientProtocol | None = None,
    auto_generate: bool = True,
) -> str | None:
    """Generate meta/stats.json.

    When hydrator is provided, supports S3 paths. Returns a logging reason or None.
    """
    if hydrator is not None:
        base_path = Hydrator.normalize_path(str(dataset_path))
        print(f"[stats.py] Generating stats for {base_path}")  # noqa: T201
        return _generate_stats(base_path, hydrator)
    dataset_path = Path(dataset_path)
    print(f"Generating stats for {dataset_path!s}")  # noqa: T201
    lowdim_features = []
    with open(dataset_path / LE_ROBOT_INFO_FILENAME) as f:
        le_features = json.load(f)["features"]
    for feature in le_features:
        if "float" in le_features[feature]["dtype"]:
            lowdim_features.append(feature)
    if check_stats_validity(dataset_path, lowdim_features):
        return None

    if not auto_generate:
        stats_path = dataset_path / LE_ROBOT_STATS_FILENAME
        raise ValueError(
            f"Stats file missing or invalid: {stats_path}. "
            f"Set data.auto_generate_stats=True or regenerate with: "
            f"python -m groot.core.data.stats {dataset_path}"
        )

    parquet_files = list(dataset_path.glob(LE_ROBOT_DATA_FILENAME))
    stats = calculate_dataset_statistics(parquet_files, lowdim_features)
    stats_path = dataset_path / LE_ROBOT_STATS_FILENAME
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    return None


class RelativeActionLoader:
    def __init__(
        self,
        dataset_path: Path | str,
        embodiment_tag: EmbodimentTag,
        action_key: str,
        hydrator: HydratorClientProtocol | None = None,
        max_action_chunk_size: int | None = None,
    ):
        self.dataset_path = Path(dataset_path) if hydrator is None else dataset_path
        self.modality_configs: dict[str, ModalityConfig] = {}
        self.action_key = action_key
        # Check action config
        assert action_key in MODALITY_CONFIGS[embodiment_tag.value]["action"].modality_keys
        idx = MODALITY_CONFIGS[embodiment_tag.value]["action"].modality_keys.index(action_key)
        action_configs = MODALITY_CONFIGS[embodiment_tag.value]["action"].action_configs
        assert action_configs is not None, MODALITY_CONFIGS[embodiment_tag.value]["action"]
        self.action_config = action_configs[idx]
        # Action horizon for the stats: default to the embodiment's configured
        # delta_indices; override to range(max_action_chunk_size) to compute stats
        # over a longer (or shorter) horizon than training uses. The consumer
        # truncates stats rows to the action length (see _align_stats_bounds), so a
        # longer stats horizon is a safe superset of any shorter training horizon.
        if max_action_chunk_size is not None:
            action_delta_indices = list(range(max_action_chunk_size))
        else:
            action_delta_indices = MODALITY_CONFIGS[embodiment_tag.value]["action"].delta_indices
        self.modality_configs["action"] = ModalityConfig(
            delta_indices=action_delta_indices,
            modality_keys=[action_key],
        )
        # Check state config (state and action must have same dim for relative_chunking).
        state_key = self.action_config.state_key or action_key
        assert state_key in MODALITY_CONFIGS[embodiment_tag.value]["state"].modality_keys
        self.modality_configs["state"] = ModalityConfig(
            delta_indices=MODALITY_CONFIGS[embodiment_tag.value]["state"].delta_indices,
            modality_keys=[state_key],
        )
        # Check state-action consistency
        assert (
            self.modality_configs["state"].delta_indices[-1]
            == self.modality_configs["action"].delta_indices[0]
        )
        self.loader = LeRobotEpisodeLoader(
            str(dataset_path), self.modality_configs, hydrator=hydrator
        )

    def load_relative_actions(self, trajectory_id: int) -> list[np.ndarray]:
        trajectory_df = self.loader[trajectory_id]

        action_key_col = f"action.{self.action_key}"
        # pl.Array(pl.Float32, width) → 2D ndarray (episode_length, width).
        action_data = trajectory_df[action_key_col].to_numpy()
        trajectories = []
        usable_length = trajectory_df.height - self.modality_configs["action"].delta_indices[-1]
        action_delta_indices = np.array(self.modality_configs["action"].delta_indices)

        # Compute relative from state and absolute action.
        state_key = (
            f"state.{self.action_config.state_key}"
            if self.action_config.state_key is not None
            else f"state.{self.action_key}"
        )
        state_data = trajectory_df[state_key].to_numpy()
        for i in range(usable_length):
            state_ind = self.modality_configs["state"].delta_indices[-1] + i
            action_inds = action_delta_indices + i
            last_state = np.array(state_data[state_ind])
            actions = np.stack(action_data[action_inds])
            trajectories.append(
                compute_relative_action(actions, last_state, self.action_config.type)
            )
        return trajectories

    def __len__(self) -> int:
        return len(self.loader)


def calculate_stats_for_key(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    group_key: str,
    max_episodes: int = -1,
    use_reservoir_sampling: bool = True,
    hydrator: HydratorClientProtocol | None = None,
    max_action_chunk_size: int | None = None,
) -> dict:
    """Compute relative action stats for one key.

    Args:
        dataset_path: Path to LeRobot-style dataset.
        embodiment_tag: Embodiment tag for modality config.
        group_key: Action key (e.g. joint position) to compute stats for.
        max_episodes: Cap number of episodes (-1 = no cap).
        use_reservoir_sampling: If True, use reservoir sampling for q01/q99 (streaming, lower memory).
            If False, accumulate all chunks and compute exact quantiles (higher memory).
        hydrator: Optional Hydrator for S3/local paths. If None, uses local Path only.
        max_action_chunk_size: Override the stats horizon (number of action-offset
            rows) instead of using the embodiment's configured delta_indices. None
            uses the configured horizon.

    Returns:
        Dict with keys mean, std, min, max, q01, q99 (numpy arrays as float64).
    """
    loader = RelativeActionLoader(
        dataset_path,
        embodiment_tag,
        group_key,
        hydrator=hydrator,
        max_action_chunk_size=max_action_chunk_size,
    )
    batch_size_episodes = 50
    path_str = str(dataset_path)
    if use_reservoir_sampling:
        return _calculate_stats_for_key_reservoir(
            loader,
            group_key,
            max_episodes,
            batch_size_episodes,
            dataset_path=path_str,
            embodiment_tag=embodiment_tag,
            hydrator=hydrator,
            max_action_chunk_size=max_action_chunk_size,
        )
    return _calculate_stats_for_key_exact(
        loader,
        group_key,
        max_episodes,
        dataset_path=path_str,
        embodiment_tag=embodiment_tag,
        hydrator=hydrator,
        max_action_chunk_size=max_action_chunk_size,
    )


def _load_episode_relative_actions(args: tuple) -> list[np.ndarray]:
    """Load and compute relative actions for a single episode. Designed for ray.util.multiprocessing.Pool."""
    dataset_path, embodiment_tag, group_key, episode_id, hydrator, max_action_chunk_size = args
    loader = RelativeActionLoader(
        dataset_path,
        embodiment_tag,
        group_key,
        hydrator=hydrator,
        max_action_chunk_size=max_action_chunk_size,
    )
    return loader.load_relative_actions(episode_id)


def _calculate_stats_for_key_reservoir(
    loader: RelativeActionLoader,
    group_key: str,
    max_episodes: int,
    batch_size_episodes: int,
    dataset_path: str | None = None,
    embodiment_tag: EmbodimentTag | None = None,
    hydrator: HydratorClientProtocol | None = None,
    max_action_chunk_size: int | None = None,
) -> dict:
    """Streaming stats with reservoir sampling for q01/q99."""
    num_episodes = len(loader) if max_episodes == -1 else min(len(loader), max_episodes)

    # Collect all episode results in parallel via Ray multiprocessing pool
    import ray
    from ray.util.multiprocessing import Pool as RayPool
    from ray.util.multiprocessing.pool import PoolTaskError

    if not ray.is_initialized():
        from groot.core.utils.distributed.ray_init import ray_init_maybe_new_cluster

        ray_init_maybe_new_cluster()

    pool_args = [
        (dataset_path, embodiment_tag, group_key, ep_id, hydrator, max_action_chunk_size)
        for ep_id in range(num_episodes)
    ]
    with RayPool() as pool:
        all_episode_chunks = list(
            tqdm(
                pool.imap(
                    _load_episode_relative_actions,
                    pool_args,
                ),
                total=num_episodes,
                desc=f"Loading trajectories for key {group_key}",
            )
        )

    for result in all_episode_chunks:
        if isinstance(result, PoolTaskError):
            raise result.underlying

    # Aggregate with reservoir sampling
    global_n = 0
    global_mean = None
    global_M2 = None
    global_min = None
    global_max = None
    reservoir = None
    batch_chunks: list[np.ndarray] = []
    episodes_in_batch = 0

    for episode_chunks in all_episode_chunks:
        batch_chunks.extend(episode_chunks)
        episodes_in_batch += 1

        if episodes_in_batch >= batch_size_episodes:
            (
                global_n,
                global_mean,
                global_M2,
                global_min,
                global_max,
                reservoir,
            ) = merge_batch_stats_with_reservoir(
                global_n,
                global_mean,
                global_M2,
                global_min,
                global_max,
                reservoir,
                batch_chunks,
                reservoir_size=DEFAULT_RESERVOIR_SIZE,
            )
            batch_chunks = []
            episodes_in_batch = 0

    if batch_chunks:
        (
            global_n,
            global_mean,
            global_M2,
            global_min,
            global_max,
            reservoir,
        ) = merge_batch_stats_with_reservoir(
            global_n,
            global_mean,
            global_M2,
            global_min,
            global_max,
            reservoir,
            batch_chunks,
            reservoir_size=DEFAULT_RESERVOIR_SIZE,
        )

    if global_n == 0:
        raise ValueError(f"No trajectory data for key {group_key}")
    assert (
        global_mean is not None
        and global_M2 is not None
        and global_min is not None
        and global_max is not None
        and reservoir is not None
    ), "streaming stats state must be set when global_n > 0"

    n_valid = min(global_n, DEFAULT_RESERVOIR_SIZE)
    q01, q99 = quantiles_from_reservoir(reservoir, n_valid)
    std = np.sqrt(global_M2 / global_n)
    return {
        "max": global_max.astype(np.float64),
        "min": global_min.astype(np.float64),
        "q01": q01,
        "q99": q99,
        "mean": global_mean.astype(np.float64),
        "std": std.astype(np.float64),
    }


def _calculate_stats_for_key_exact(
    loader: RelativeActionLoader,
    group_key: str,
    max_episodes: int,
    dataset_path: str | None = None,
    embodiment_tag: EmbodimentTag | None = None,
    hydrator: HydratorClientProtocol | None = None,
    max_action_chunk_size: int | None = None,
) -> dict:
    """Exact stats by accumulating all chunks then computing mean, std, min, max, q01, q99."""
    num_episodes = len(loader) if max_episodes == -1 else min(len(loader), max_episodes)

    import ray
    from ray.util.multiprocessing import Pool as RayPool
    from ray.util.multiprocessing.pool import PoolTaskError

    if not ray.is_initialized():
        from groot.core.utils.distributed.ray_init import ray_init_maybe_new_cluster

        ray_init_maybe_new_cluster()

    pool_args = [
        (dataset_path, embodiment_tag, group_key, ep_id, hydrator, max_action_chunk_size)
        for ep_id in range(num_episodes)
    ]
    with RayPool() as pool:
        all_episode_chunks = list(
            tqdm(
                pool.imap(
                    _load_episode_relative_actions,
                    pool_args,
                ),
                total=num_episodes,
                desc=f"Loading trajectories for key {group_key}",
            )
        )

    for result in all_episode_chunks:
        if isinstance(result, PoolTaskError):
            raise result.underlying

    all_chunks = [chunk for episode_chunks in all_episode_chunks for chunk in episode_chunks]
    if not all_chunks:
        raise ValueError(f"No trajectory data for key {group_key}")
    return exact_stats_from_chunks(all_chunks)


def generate_rel_stats(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    use_reservoir_sampling: bool = True,
    force: bool = False,
    hydrator: HydratorClientProtocol | None = None,
    auto_generate: bool = True,
    max_action_chunk_size: int | None = None,
) -> str | None:
    """Generate relative action stats. Returns reason str for main-process logging, or None when no action_configs.

    max_action_chunk_size overrides the stats horizon (number of action-offset rows)
    instead of using the embodiment's configured delta_indices; None uses the
    configured horizon.
    """
    path_str = str(dataset_path)
    action_config = MODALITY_CONFIGS[embodiment_tag.value]["action"]
    if action_config.action_configs is None:
        return None
    action_keys = [
        key
        for key, ac in zip(action_config.modality_keys, action_config.action_configs, strict=False)
        if ac.rep == ActionRepresentation.RELATIVE
    ]
    if not action_keys:
        return None  # No relative actions for this embodiment
    if hydrator is not None:
        from groot.core.data.hydrator.service import Hydrator as H

        rel_stats_uri = H.join_path(path_str, LE_ROBOT_REL_STATS_FILENAME)
        if not rel_stats_uri.startswith("s3://"):
            # path_str resolved to a local path (dataset not yet synced to S3);
            # fall through to the local-write code below.
            hydrator = None
        else:
            if force:
                stats = {}
            else:
                try:
                    data = hydrator.hydrate(rel_stats_uri)
                    first = data[0] if data else None
                    stats = first if isinstance(first, dict) else {}
                except Exception:  # noqa: BLE001
                    stats = {}
            keys_to_compute = [k for k in sorted(action_keys) if k not in stats]
            if not keys_to_compute:
                return f"skipped: meta/relative_stats.json all keys present (have {list(stats.keys())!r})"
            for action_key in keys_to_compute:
                result = calculate_stats_for_key(
                    path_str,
                    embodiment_tag,
                    action_key,
                    use_reservoir_sampling=use_reservoir_sampling,
                    hydrator=hydrator,
                    max_action_chunk_size=max_action_chunk_size,
                )
                stats[action_key] = to_json_serializable(result)
            hydrator.put_file(rel_stats_uri, json.dumps(stats, indent=4).encode("utf-8"))
            return f"generated: computed keys {keys_to_compute!r} -> wrote to {rel_stats_uri!r}"
    dataset_path = Path(dataset_path)
    stats_path = dataset_path / LE_ROBOT_REL_STATS_FILENAME
    if force:
        stats = {}
    elif stats_path.exists():
        with open(stats_path) as f:
            content = f.read()
        if not content.strip():
            raise ValueError(
                f"Stats file is empty (0 bytes): {stats_path}. "
                f"Delete it and re-run to regenerate."
            )
        try:
            stats = json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"Corrupt stats file {stats_path}: {e}. Delete it and re-run to regenerate."
            ) from e
    else:
        stats = {}
    keys_to_compute = [k for k in sorted(action_keys) if k not in stats]
    if keys_to_compute and not auto_generate:
        raise ValueError(
            f"Relative stats missing keys {keys_to_compute} in {stats_path}. "
            f"Set data.auto_generate_stats=True or regenerate with: "
            f"python -m groot.core.data.stats {dataset_path} {embodiment_tag.value}"
        )
    for action_key in keys_to_compute:
        result = calculate_stats_for_key(
            path_str,
            embodiment_tag,
            action_key,
            use_reservoir_sampling=use_reservoir_sampling,
            hydrator=None,
            max_action_chunk_size=max_action_chunk_size,
        )
        stats[action_key] = to_json_serializable(result)
    if keys_to_compute:
        content = json.dumps(to_json_serializable(dict(stats)), indent=4)
        with open(stats_path, "w") as f:
            f.write(content)
    return None


def main(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    use_reservoir_sampling: bool = True,
    force_rel_stats: bool = False,
    max_action_chunk_size: int | None = None,
):
    """Generate stats and relative stats for a LeRobot-style dataset."""
    generate_stats(dataset_path)
    generate_rel_stats(
        dataset_path,
        embodiment_tag,
        use_reservoir_sampling=use_reservoir_sampling,
        force=force_rel_stats,
        max_action_chunk_size=max_action_chunk_size,
    )


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
