import pyarrow as pa
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import io
import lance
from typing import Dict, Any, List

from gr00t.data.types import ModalityConfig, VLAStepData, MessageType
from gr00t.data.interfaces import ShardedDataset
from gr00t.data.embodiment_tags import EmbodimentTag


class ShardedLanceDataset(ShardedDataset):
    """
    Single-step dataset that streams manipulation data from Lance.
    """
    def __init__(
        self,
        dataset_path: dict[str, str],
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        shard_size: int = 2**10,
        episode_sampling_rate: float = 0.1,
        seed: int = 42,
        allow_padding: bool = False,
    ):
        # We store the main dataset path to pass to super class, though not strictly required
        super().__init__(dataset_path.get("MAIN_DATASET", ""))
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.shard_size = shard_size
        self.episode_sampling_rate = episode_sampling_rate
        self.seed = seed
        self.allow_padding = allow_padding
        self.rng = np.random.default_rng(seed)
        self.processor = None

        main_path = dataset_path.get("MAIN_DATASET")
        curated_path = dataset_path.get("CURATED_DATASET")
        if not main_path or not curated_path:
            raise ValueError("dataset_paths must be a dict containing 'MAIN_DATASET' and 'CURATED_DATASET'")

        self.main_ds = lance.dataset(main_path)
        self.curated_ds = lance.dataset(curated_path)

        self.action_delta_indices = modality_configs["action"].delta_indices
        self.action_horizon = max(self.action_delta_indices) - min(self.action_delta_indices) + 1

        self.shard_dataset()

    def shard_dataset(self):
        """
        Shards the curated rows into smaller balanced lists for data loading.
        Filters for parallel gripper / manipulation data based on the schema.
        """
        # Fetch curated rows (table) filtered down to relevant rows
        # The schema provides: episode_uuid, chunk_in_episode, is_parallel_gripper, etc.
        import pyarrow.compute as pc

        scanner = self.curated_ds.scanner(
            columns=["episode_uuid", "chunk_in_episode", "instruction", "is_parallel_gripper"],
        )
        table = scanner.to_table()

        if "is_parallel_gripper" in table.schema.names:
            table = table.filter(pc.field("is_parallel_gripper") == True)

        total_rows = table.num_rows
        if total_rows == 0:
            print("Warning: no manipulation rows found in Lance dataset")
            self.sharded_rows = []
            self.shard_lengths = []
            return

        df = table.to_pandas()
        indices = np.arange(total_rows)
        self.rng.shuffle(indices)

        # Subsample based on episode_sampling_rate
        # For simplicity, we sample randomly across the chunks.
        # Alternatively we can sample episodes. Here we just sample rows.
        num_sampled = int(total_rows * self.episode_sampling_rate)
        if num_sampled == 0 and total_rows > 0:
            num_sampled = 1

        sampled_indices = indices[:num_sampled]

        num_shards = int(np.ceil(len(sampled_indices) / self.shard_size))

        self.sharded_rows = [[] for _ in range(num_shards)]
        self.shard_lengths = np.zeros(num_shards, dtype=int)

        for i, idx in enumerate(sampled_indices):
            shard_idx = i % num_shards
            row = df.iloc[idx]
            self.sharded_rows[shard_idx].append(row)
            self.shard_lengths[shard_idx] += 1

        # We don't fetch all data here to avoid OOM, but we can index the main dataset faster.

        print(f"Generated {num_shards} shards for Lance dataset")
        print(f"Total steps: {len(sampled_indices)}")

    def __len__(self) -> int:
        return len(self.shard_lengths)

    def get_shard_length(self, idx: int) -> int:
        return self.shard_lengths[idx]

    def get_shard(self, idx: int) -> list:
        rows = self.sharded_rows[idx]

        if len(rows) == 0:
            return []

        if self.processor is None:
            raise ValueError("Processor must be set before getting datapoints")

        import pyarrow.compute as pc

        # Collect uuids and chunks to load them all at once
        ep_uuids = [row["episode_uuid"] for row in rows]
        chunks = [row["chunk_in_episode"] for row in rows]
        instructions = [row.get("instruction", "") for row in rows]

        # Load columns
        cols_to_load = ["episode_uuid", "chunk_in_episode"]
        for key in self.modality_configs.get("video", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys:
            if key == "image" or key == "primary_image_key": cols_to_load.append("obs/camera/left_image_256")
            elif key == "wrist_image": cols_to_load.append("obs/camera/wrist_left_image_256")
            else: cols_to_load.append(f"obs/camera/{key}_image_256")

        for key in self.modality_configs.get("state", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys:
            cols_to_load.append(f"obs/positions/{key}")

        for key in self.modality_configs.get("action", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys:
            cols_to_load.append(f"action/q_target/{key}")

        # We can do an IN filter to scan once for the whole shard
        # In lance, we can filter using `is_in`.
        # However, to be perfectly safe, since a shard might be 1000 items, we can just use Lance's `take` if we had row IDs.
        # Since we don't have row IDs of MAIN_DATASET here, we filter via episode_uuid.

        import pyarrow as pa
        uuid_arr = pa.array(ep_uuids, type=pa.binary(16))

        scanner = self.main_ds.scanner(
            columns=cols_to_load,
            filter=pc.is_in(pc.field("episode_uuid"), value_set=uuid_arr)
        )
        table = scanner.to_table()
        if table.num_rows == 0:
            return []

        main_df = table.to_pandas()

        # Map them
        datapoints = []
        for i, row in enumerate(rows):
            ep_uuid = ep_uuids[i]
            chunk = chunks[i]
            instr = instructions[i]

            # Find in main_df
            match = main_df[(main_df["episode_uuid"] == ep_uuid) & (main_df["chunk_in_episode"] == chunk)]
            if len(match) == 0:
                continue

            main_row = match.iloc[0]
            dp = self.get_datapoint(main_row, instr)
            if dp is not None:
                datapoints.append(dp)

        return datapoints

    def get_datapoint(self, main_row, instruction) -> dict | None:

        video_data = {}
        for key in self.modality_configs.get("video", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys:
            # e.g., image -> obs/camera/left_image_256
            if key == "image" or key == "primary_image_key":
                img_col = "obs/camera/left_image_256"
            elif key == "wrist_image":
                img_col = "obs/camera/wrist_left_image_256"
            else:
                img_col = f"obs/camera/{key}_image_256"

            if img_col in main_row:
                img_bytes = main_row[img_col]
                if pd.notna(img_bytes) and img_bytes is not None:
                    # decode image
                    img = Image.open(io.BytesIO(img_bytes))
                    video_data[key] = [np.array(img)]

        # Usually chunk_len is inferred from action horizon or default chunk len like 50. Let's infer chunk len from delta indices if possible, or just deduce it from array size.
        chunk_len = len(self.action_delta_indices) if len(self.action_delta_indices) > 0 else 50

        states = {}
        for key in self.modality_configs.get("state", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys:
            col = f"obs/positions/{key}"
            if col in main_row:
                arr = np.array(main_row[col], dtype=np.float32)
                if len(arr) % chunk_len == 0 and len(arr) > chunk_len:
                    arr = arr.reshape(chunk_len, -1)
                elif len(arr) % 50 == 0 and len(arr) > 50:
                    arr = arr.reshape(50, -1)
                states[key] = arr

        actions = {}
        for key in self.modality_configs.get("action", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys:
            col = f"action/q_target/{key}"
            if col in main_row:
                arr = np.array(main_row[col], dtype=np.float32)
                if len(arr) % chunk_len == 0 and len(arr) > chunk_len:
                    arr = arr.reshape(chunk_len, -1)
                elif len(arr) % 50 == 0 and len(arr) > 50:
                    arr = arr.reshape(50, -1)
                actions[key] = arr

        # create step data
        vla_step_data = VLAStepData(
            images=video_data,
            states=states,
            actions=actions,
            text=instruction,
            embodiment=self.embodiment_tag,
            masks=None,
        )

        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        return self.processor(messages)


    def get_dataset_statistics(self) -> dict:
        """
        Compute normalization stats on a rolling basis by streaming chunks.
        We sample a subset of rows to calculate statistics efficiently to avoid OOM.
        """
        if hasattr(self, "_cached_stats"):
            return self._cached_stats

        import pyarrow.compute as pc
        from collections import defaultdict

        print("Computing statistics from Lance dataset chunks...")

        # Determine columns to query for stats
        state_keys = self.modality_configs.get("state", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys
        action_keys = self.modality_configs.get("action", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys

        if not state_keys and not action_keys:
            self._cached_stats = {}
            return self._cached_stats

        cols_to_load = []
        for key in state_keys:
            cols_to_load.append(f"obs/positions/{key}")
        for key in action_keys:
            cols_to_load.append(f"action/q_target/{key}")

        # Optional: Instead of scanning everything, we limit to the first N rows from curated
        # We can just use the sampled shards we already have in self.sharded_rows
        # Flatten all sampled rows
        all_sampled_rows = []
        for rows in self.sharded_rows:
            all_sampled_rows.extend(rows)

        # For stats, computing over say 5,000 to 10,000 chunks is usually enough
        max_stat_samples = 5000
        if len(all_sampled_rows) > max_stat_samples:
            sampled_for_stats = self.rng.choice(all_sampled_rows, max_stat_samples, replace=False)
        else:
            sampled_for_stats = all_sampled_rows

        if len(sampled_for_stats) == 0:
            return {}

        ep_uuids = [row["episode_uuid"] for row in sampled_for_stats]
        chunks = [row["chunk_in_episode"] for row in sampled_for_stats]

        import pyarrow as pa
        uuid_arr = pa.array(ep_uuids, type=pa.binary(16))

        # Add episode_uuid and chunk_in_episode so we can filter locally if needed
        scan_cols = cols_to_load + ["episode_uuid", "chunk_in_episode"]
        scanner = self.main_ds.scanner(
            columns=scan_cols,
            filter=pc.is_in(pc.field("episode_uuid"), value_set=uuid_arr)
        )

        # Read in batches
        all_state_data = defaultdict(list)
        all_action_data = defaultdict(list)

        for batch in scanner.to_batches():
            df = batch.to_pandas()
            for key in state_keys:
                col = f"obs/positions/{key}"
                if col in df.columns:
                    # Explode list columns into flat arrays
                    for arr in df[col]:
                        if arr is not None:
                            all_state_data[key].append(np.array(arr, dtype=np.float32))

            for key in action_keys:
                col = f"action/q_target/{key}"
                if col in df.columns:
                    for arr in df[col]:
                        if arr is not None:
                            all_action_data[key].append(np.array(arr, dtype=np.float32))

        # Calculate statistics
        stats = {"state": defaultdict(dict), "action": defaultdict(dict)}

        for key, arr_list in all_state_data.items():
            if not arr_list: continue
            concat_data = np.concatenate(arr_list, axis=0) # (N*chunk_len, dim) or flat
            if concat_data.ndim == 1:
                # Based on previous logic, these flat arrays are (chunk_len * dim)
                chunk_len = len(self.action_delta_indices) if len(self.action_delta_indices) > 0 else 50
                if len(concat_data) % chunk_len == 0 and len(concat_data) > chunk_len:
                    concat_data = concat_data.reshape(-1, concat_data.shape[0] // chunk_len) # Actually we want (-1, dim)
                    # For stats, we just need the feature dim
                    # The array is [t0_d0, t0_d1, t1_d0, t1_d1...]
                    dim = len(concat_data) // chunk_len # This is an approximation if we concat multiple
                else:
                    # Let's just compute stats assuming it's flat N*dim or NxDim
                    pass

            # Since the array from Lance is a list of lists e.g. [1.0, 2.0, ...] representing time x dim
            # To compute proper stats, we can just reshape it to (-1, joint_dim).
            # We don't strictly know joint_dim here unless we infer it.
            # If the original array length for one chunk is chunk_len * joint_dim,
            # then joint_dim = len(arr) // chunk_len

            first_valid = next(arr for arr in arr_list if len(arr) > 0)
            chunk_len = len(self.action_delta_indices) if len(self.action_delta_indices) > 0 else 50
            if len(first_valid) % chunk_len == 0:
                joint_dim = len(first_valid) // chunk_len
                concat_data = concat_data.reshape(-1, joint_dim)
            elif len(first_valid) % 50 == 0:
                joint_dim = len(first_valid) // 50
                concat_data = concat_data.reshape(-1, joint_dim)
            else:
                # Fallback, just treat as 1D or leave it as is if it's already 2D
                if concat_data.ndim == 1:
                    concat_data = concat_data.reshape(-1, 1)

            stats["state"][key]["mean"] = np.mean(concat_data, axis=0).tolist()
            stats["state"][key]["std"] = np.std(concat_data, axis=0).tolist()
            stats["state"][key]["min"] = np.min(concat_data, axis=0).tolist()
            stats["state"][key]["max"] = np.max(concat_data, axis=0).tolist()
            stats["state"][key]["q01"] = np.quantile(concat_data, 0.01, axis=0).tolist()
            stats["state"][key]["q99"] = np.quantile(concat_data, 0.99, axis=0).tolist()

        for key, arr_list in all_action_data.items():
            if not arr_list: continue
            concat_data = np.concatenate(arr_list, axis=0)

            first_valid = next(arr for arr in arr_list if len(arr) > 0)
            chunk_len = len(self.action_delta_indices) if len(self.action_delta_indices) > 0 else 50
            if len(first_valid) % chunk_len == 0:
                joint_dim = len(first_valid) // chunk_len
                concat_data = concat_data.reshape(-1, joint_dim)
            elif len(first_valid) % 50 == 0:
                joint_dim = len(first_valid) // 50
                concat_data = concat_data.reshape(-1, joint_dim)
            else:
                if concat_data.ndim == 1:
                    concat_data = concat_data.reshape(-1, 1)

            stats["action"][key]["mean"] = np.mean(concat_data, axis=0).tolist()
            stats["action"][key]["std"] = np.std(concat_data, axis=0).tolist()
            stats["action"][key]["min"] = np.min(concat_data, axis=0).tolist()
            stats["action"][key]["max"] = np.max(concat_data, axis=0).tolist()
            stats["action"][key]["q01"] = np.quantile(concat_data, 0.01, axis=0).tolist()
            stats["action"][key]["q99"] = np.quantile(concat_data, 0.99, axis=0).tolist()

        # Convert inner defaultdicts to regular dicts
        stats["state"] = dict(stats["state"])
        stats["action"] = dict(stats["action"])

        self._cached_stats = dict(stats)
        return self._cached_stats
