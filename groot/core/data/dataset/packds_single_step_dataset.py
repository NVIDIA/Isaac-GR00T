# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""ShardedDataset backed by PackDS (LanceDB) for VLA training.

Loads pre-converted data from a PackDS database where video frames are stored
as JPEG bytes, and state/action data as fixed-shape float32 tensors. This
avoids video decoding at training time and enables fast random access via
Lance columnar storage.

Sharding is performed at load time: episodes are shuffled, split by
``episode_sampling_rate``, and distributed across shards of ``shard_size``
timesteps using greedy balancing. Action horizons are gathered from
consecutive rows at read time.

See ``groot/vla/data/packds/README.md`` for the PackDS format specification.
"""

from pathlib import Path
from typing import Any

import numpy as np

from groot.core.data.packds import (
    PackDS,
)
from groot.core.data.types import (
    EmbodimentTag,
    MessageType,
    ModalityConfig,
    ShardedDataset,
    VLAStepData,
)


class PackSingleStepDataset(ShardedDataset):
    """ShardedDataset that reads from a PackDS database.

    Args:
        packds_path: Path to the ``.packds`` directory.
        embodiment_tag: Robot embodiment (used for relative stats lookup).
        modality_configs: ``{"video": ..., "state": ..., "action": ...}``
            ModalityConfig per modality. Action ``delta_indices`` determines the
            action horizon gathered at read time.
        shard_size: Target timesteps per shard.
        episode_sampling_rate: Fraction of episodes to use (1.0 = all).
        seed: Random seed for reproducible sharding.
    """

    def __init__(
        self,
        packds_path: str | Path,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        shard_size: int = 1024,
        episode_sampling_rate: float = 1.0,
        seed: int = 42,
    ):
        super().__init__(packds_path)
        self.packds_path = str(packds_path)
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.shard_size = shard_size
        self.episode_sampling_rate = episode_sampling_rate
        self.seed = seed
        self.processor = None
        self.rng = np.random.default_rng(seed)

        # Compute action horizon from config
        action_delta_indices = modality_configs["action"].delta_indices
        self.action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1

        # Open dataset (PackDS handles pickle/unpickle via __getstate__/__setstate__)
        # validate=False: stats may live in DeltaCAT properties rather than
        # Lance schema metadata; _recompute_stats() handles this at read time.
        self._packds: PackDS = PackDS(
            self.packds_path,
            embodiment_tag=str(embodiment_tag) if embodiment_tag else None,
            validate=False,
        )
        self.stats = self._packds.stats
        self.total_timesteps = self._packds.num_steps
        self.total_episodes = self._packds.num_episodes

        # Create shards
        self._create_shards()

    def _build_episode_index(self):
        """Build episode metadata and (episode_id, step_index) -> row_index mapping.

        This queries the table once to build:
        - episode_lengths: dict mapping episode_id -> effective episode length
        - row_index_map: dict mapping (episode_id, step_index) -> row_index

        Effective episode length is computed as: total_frames - action_horizon + 1
        This matches ShardedSingleStepDataset's get_effective_episode_length().

        Returns:
            Tuple of (episode_ids, episode_lengths, row_index_map) where
            episode_ids is a sorted list and episode_lengths maps episode_id
            to its effective length.
        """
        # Query episode/step indices for sharding.
        table = self._packds.read_raw_indices(
            list(range(self._packds.num_steps)), columns=[]
        )
        ep_id_list = table.column("episode_id").to_pylist()
        step_index_list = table.column("step_index").to_pylist()

        # Build row index mapping and total episode frame counts
        row_index_map = {}
        episode_total_frames: dict[str, int] = {}  # episode_id -> total frames

        for row_idx, (ep_id, step_index) in enumerate(zip(ep_id_list, step_index_list)):
            episode_id = str(ep_id)
            row_index_map[(episode_id, step_index)] = row_idx

            # Track total episode length (all frames, including last frame without valid action)
            if episode_id not in episode_total_frames:
                episode_total_frames[episode_id] = 0
            episode_total_frames[episode_id] = max(
                episode_total_frames[episode_id], step_index + 1
            )

        # Compute effective lengths: total_frames - action_horizon + 1
        # This matches ShardedSingleStepDataset.get_effective_episode_length()
        episode_ids = sorted(episode_total_frames.keys())
        episode_lengths: dict[str, int] = {}
        for episode_id in episode_ids:
            total_frames = episode_total_frames[episode_id]
            episode_lengths[episode_id] = max(0, total_frames - self.action_horizon + 1)

        return episode_ids, episode_lengths, row_index_map

    def _create_shards(self):
        """Create shard assignments matching ShardedSingleStepDataset algorithm.

        Uses the same sharding strategy as ShardedSingleStepDataset:
        1. Shuffle episode order for randomization
        2. Split each episode into multiple sub-sequences based on sampling rate
        3. Distribute sub-sequences across shards to balance shard sizes
        4. Use greedy assignment to minimize shard size variance

        This ensures PackSingleStepDataset produces the same shard contents
        as ShardedSingleStepDataset when using the same seed.
        """
        # Build episode index and row mapping
        episode_ids, episode_lengths, row_index_map = self._build_episode_index()
        self._row_index_map = row_index_map

        num_episodes = len(episode_ids)
        if num_episodes == 0:
            self.shard_lengths = np.array([], dtype=int)
            return

        # Shuffle episode order via permuted integer indices into the episode_ids list
        shuffled_order = self.rng.permutation(num_episodes)
        num_splits = int(1 / self.episode_sampling_rate)

        # Calculate total timesteps and required number of shards
        total_steps = sum(episode_lengths[episode_ids[i]] for i in shuffled_order)
        num_shards = int(np.ceil(total_steps / self.shard_size))

        # Initialize shard containers
        sharded_episodes = [[] for _ in range(num_shards)]
        shard_lengths = np.zeros(num_shards, dtype=int)

        # Distribute episode sub-sequences across shards (same algorithm as ShardedSingleStepDataset)
        for i in shuffled_order:
            episode_id = episode_ids[i]
            effective_length = episode_lengths[episode_id]
            if effective_length == 0:
                continue

            # Split episode timesteps into multiple sub-sequences
            step_indices = np.arange(effective_length)
            self.rng.shuffle(step_indices)

            for j in range(num_splits):
                split_step_indices = step_indices[j::num_splits]
                if len(split_step_indices) == 0:
                    continue
                # Assign to shard with minimum current length (greedy balancing)
                shard_index = np.argmin(shard_lengths)
                sharded_episodes[shard_index].append((episode_id, split_step_indices))
                shard_lengths[shard_index] += len(split_step_indices)

        # Store (episode_id, step_index) pairs for action horizon gathering at read time
        # We need episode/step info to gather consecutive rows for action horizons
        self.shard_samples = []  # List of lists of (episode_id, step_index) tuples
        for shard_eps in sharded_episodes:
            samples = []
            for episode_id, step_indices in shard_eps:
                for step_index in step_indices:
                    samples.append((episode_id, step_index))
            self.shard_samples.append(samples)

        self.shard_lengths = np.array([len(samples) for samples in self.shard_samples])

        print(f"Created {num_shards} shards for dataset {self.packds_path}")
        print(
            f"Total steps: {total_steps}, average shard length: {total_steps / num_shards:.1f}, "
            f"shard length std: {np.std(self.shard_lengths):.1f}"
        )

    def _read_columns(self) -> list[str]:
        """Build the column projection for ``read_raw_indices`` calls.

        Returns the minimal set of steps-table columns that
        :meth:`_row_to_vla_step_data` consumes for one shard's worth of
        samples: core columns (episode_id, step indices, has_valid_action),
        the per-modality state/action columns, the configured video keys
        (both JPEG and video-blob variants — only the ones present in the
        schema actually round-trip), and the annotation columns the
        processor reads. Anything not in this list is skipped at the Lance
        layer, so unused video resolutions never leave S3.
        """
        video_keys = self.modality_configs["video"].modality_keys
        state_keys = self.modality_configs["state"].modality_keys
        action_keys = self.modality_configs["action"].modality_keys

        # Project against the dataset schema so we only ask for columns
        # that exist — older PackDS variants don't always carry every
        # state/action column the embodiment_tag declares (e.g. legacy
        # YAM datasets predate ``state:left_wrist_eef`` etc.). Asking
        # Lance for a non-existent column raises; silently drop those
        # and let the dataloader's existing missing-key handling cover
        # the consequences.
        available = set(self._packds.schema.names)

        columns: list[str] = []
        # Core columns the loader / from_table always need.
        for col in (
            "episode_id",
            "episode_index",
            "step_index",
            "step_time_ns",
            "original_step_time_ns",
            "has_valid_action",
        ):
            if col in available:
                columns.append(col)

        # Per-modality projections. ``video:`` is the JPEG path,
        # ``video_blob:``/``video_blob_frame_index:`` is the H.264 path.
        # Both prefixes are checked so JPEG-only and video-blob-only
        # datasets work without a write-time switch.
        for k in video_keys:
            for prefix in ("video", "video_blob", "video_blob_frame_index"):
                col = f"{prefix}:{k}"
                if col in available:
                    columns.append(col)
        for k in state_keys:
            col = f"state:{k}"
            if col in available:
                columns.append(col)
        for k in action_keys:
            col = f"action:{k}"
            if col in available:
                columns.append(col)

        # Annotation columns ``_row_to_vla_step_data`` reads (currently
        # only ``task``; pull the others through too so processors that
        # opt into them don't trigger an extra S3 round-trip).
        for col in (
            "annotation:task",
            "annotation:sub_task",
            "annotation:action_source",
        ):
            if col in available:
                columns.append(col)

        return columns

    def __len__(self) -> int:
        """Return the number of shards."""
        return len(self.shard_lengths)

    def get_shard_length(self, idx: int) -> int:
        """Get the length of the shard at index idx."""
        return self.shard_lengths[idx]

    def _row_to_vla_step_data(
        self,
        batch: Any,
        pos: int,
        horizon_positions: list[int],
    ) -> VLAStepData:
        """Convert batch data at a position to VLAStepData.

        Args:
            batch: PackDSStepBatch from read_batch.
            pos: Position index within the batch arrays for this sample's
                base data (state, video, text).
            horizon_positions: List of position indices for action horizon gathering.

        Returns:
            VLAStepData with decoded images, states, actions, and text
        """
        video_keys = self.modality_configs["video"].modality_keys
        state_keys = self.modality_configs["state"].modality_keys
        action_keys = self.modality_configs["action"].modality_keys

        # Read pre-decoded video frames (batch must have been decoded already)
        images = {}
        for video_key in video_keys:
            frame = batch.decoded_video[video_key][pos]
            images[video_key] = [frame]  # List for temporal stacking

        # State data - already numpy from columnar read
        states = {}
        for state_key in state_keys:
            state_arr = batch.state[state_key][pos]
            states[state_key] = np.asarray(state_arr, dtype=np.float32).reshape(1, -1)

        # Gather actions from horizon positions
        actions = {}
        for action_key in action_keys:
            action_arr = batch.action[action_key]
            action_steps = [
                np.asarray(action_arr[hp], dtype=np.float32).flatten() for hp in horizon_positions
            ]
            actions[action_key] = np.stack(action_steps)  # Shape: [horizon, dim]

        task_list = batch.annotations.get("task", [])
        text = task_list[pos] if pos < len(task_list) and task_list[pos] else ""

        return VLAStepData(
            images=images,
            states=states,
            actions=actions,
            text=text,
            embodiment=self.embodiment_tag,
        )

    def get_shard(self, idx: int) -> list:
        """Load all timesteps in a shard with action horizon gathering.

        For each training sample, gathers consecutive rows within the same episode
        to construct the full action horizon. This matches LeRoBot's approach where
        single-step actions are stored and horizons are gathered at read time.

        Uses columnar reads for efficient bulk data access.

        Args:
            idx: Shard index

        Returns:
            List of processed datapoints ready for training
        """
        action_horizon = self.action_horizon
        shard_samples = self.shard_samples[idx]  # List of (episode_id, step_index) tuples

        # Collect all row indices needed (including horizon rows)
        all_row_indices = set()
        sample_to_horizon_indices = []  # For each sample, list of row indices for action horizon

        for episode_id, step_index in shard_samples:
            horizon_row_indices = []
            for delta in range(action_horizon):
                row_idx = self._row_index_map.get((episode_id, step_index + delta))
                if row_idx is not None:
                    horizon_row_indices.append(row_idx)
                    all_row_indices.add(row_idx)
            sample_to_horizon_indices.append(horizon_row_indices)

        # Query all needed rows at once using columnar read
        all_row_indices_list = sorted(all_row_indices)
        if not all_row_indices_list:
            return []

        # Build position map: row_idx -> position in result arrays
        row_idx_to_pos = {idx: pos for pos, idx in enumerate(all_row_indices_list)}

        # Read all needed data at once and decode video.
        # Restrict ``columns=`` to the keys this dataloader actually consumes
        # (per modality_configs). This avoids fetching every camera variant
        # the dataset stored — important for PackDS datasets that hold both
        # native and resized video blobs (e.g. ``top_camera-images-rgb`` AND
        # ``top_camera-images-rgb_320_240``); without the filter we'd pull
        # both off S3 and decode both, even when training only reads the
        # 320x240 variant.
        from groot.core.data.packds import from_table
        from groot.core.data.packds import deltacat_sql_loader as _dsl

        columns = self._read_columns()
        _timing = _dsl._shard_timing_enabled()
        if _timing:
            import time as _t

            _dsl.timing_snapshot_reset()
            _t0 = _t.perf_counter()
        batch = from_table(
            self._packds.read_raw_indices(all_row_indices_list, columns=columns)
        )
        if _timing:
            _t1 = _t.perf_counter()
        batch = batch.decode_video()
        if _timing:
            _t2 = _t.perf_counter()
            _snap = _dsl.timing_snapshot_reset()
            print(
                f"[shard-timing] rows={len(all_row_indices_list)} "
                f"read={_t1 - _t0:.1f}s decode_video_wall={_t2 - _t1:.1f}s "
                f"fetches={_snap['n_fetch']} fetch_sum={_snap['fetch_s']:.1f}s "
                f"decode_sum={_snap['decode_s']:.1f}s "
                f"fetch_MB={_snap['fetch_bytes'] / 1024**2:.0f}",
                flush=True,
            )

        # Convert to processed datapoints
        datapoints = []
        for (episode_id, step_index), horizon_indices in zip(
            shard_samples, sample_to_horizon_indices
        ):
            if not horizon_indices:
                continue

            # Get base position (for state, video, text)
            base_row_idx = horizon_indices[0]
            base_pos = row_idx_to_pos.get(base_row_idx)
            if base_pos is None:
                continue

            # Get action horizon positions
            horizon_positions = [row_idx_to_pos.get(idx) for idx in horizon_indices]
            horizon_positions = [p for p in horizon_positions if p is not None]

            if len(horizon_positions) < action_horizon:
                continue

            vla_step_data = self._row_to_vla_step_data(batch, base_pos, horizon_positions)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            datapoints.append(self.processor(messages))

        return datapoints

    def _recompute_stats(self) -> None:
        """Recompute all statistics from the table.

        Always recomputes so that shape and stat keys (e.g. q01/q99) match
        the current code, regardless of what was persisted on disk.
        """
        from groot.core.data.packds.stats import compute_column_statistics, remap_column_stats

        column_stats = compute_column_statistics(self._packds)
        if column_stats:
            self.stats = remap_column_stats(column_stats)

    def get_dataset_statistics(self) -> dict[str, Any]:
        """Return dataset statistics for normalization.

        Returns:
            Nested dictionary: {modality: {joint_group: {stat_type: values}}}
        """
        # Recompute all stats from the table so that shape and keys
        # (e.g. q01/q99) always match the current code.
        self._recompute_stats()

        if not self.stats:
            raise ValueError(
                "No statistics available. Ensure the dataset has stats in schema metadata."
            )

        # Mapping from modality name to stats key prefix
        prefix_mapping = {"state": "observation", "action": "action"}
        dataset_statistics: dict[str, Any] = {}

        for modality in ["state", "action"]:
            dataset_statistics[modality] = {}
            for joint_key in self.modality_configs[modality].modality_keys:
                # Try new format first: per-joint stats like "observation.ee_pose_obs_left"
                per_joint_stats_key = f"{prefix_mapping[modality]}.{joint_key}"

                if per_joint_stats_key in self.stats:
                    # Per-joint stats like "observation.ee_pose_obs_left"
                    dataset_statistics[modality][joint_key] = self.stats[per_joint_stats_key]
                else:
                    # Stats not available for this joint key
                    dataset_statistics[modality][joint_key] = {}

        if self._packds.relative_stats:
            dataset_statistics["relative_action"] = {}
            for action_key in self.modality_configs["action"].modality_keys:
                if action_key in self._packds.relative_stats:
                    dataset_statistics["relative_action"][action_key] = self._packds.relative_stats[
                        action_key
                    ]

        return dataset_statistics
