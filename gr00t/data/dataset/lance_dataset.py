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

        from gr00t.data.dataset.g1_fk import G1ForwardKinematics
        try:
            self.g1_fk = G1ForwardKinematics()
        except Exception as e:
            print(f"Failed to initialize G1 FK (URDF missing?): {e}")
            self.g1_fk = None

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

        datapoints = []
        for i, row in enumerate(rows):
            ep_uuid = ep_uuids[i]
            chunk = chunks[i]
            instr = instructions[i]

            if isinstance(ep_uuid, bytes):
                ep_uuid_val = ep_uuid
            else:
                ep_uuid_val = ep_uuid

            scanner = self.main_ds.scanner(
                columns=cols_to_load,
                filter=(pc.field("episode_uuid") == ep_uuid_val) & (pc.field("chunk_in_episode") == chunk),
            )
            table = scanner.to_table()
            if table.num_rows > 0:
                main_row = table.to_pandas().iloc[0]
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
        state_parts = []
        left_arm_state = None
        right_arm_state = None

        if "obs/positions/core" in main_row:
            arr = np.array(main_row["obs/positions/core"], dtype=np.float32)
            if len(arr) % chunk_len == 0 and len(arr) > chunk_len:
                arr = arr.reshape(chunk_len, -1)
            elif len(arr) % 50 == 0 and len(arr) > 50:
                arr = arr.reshape(50, -1)
            # If it's a humanoid core (e.g. 29 motors), skip first 15 (legs/waist)
            if arr.shape[-1] >= 29:
                arms = arr[..., 15:]
                # Arms are 14 joints (7 left, 7 right)
                left_arm_state = arms[..., :7]
                right_arm_state = arms[..., 7:14]
                arr = arms
            state_parts.append(arr)

        if self.g1_fk is not None and left_arm_state is not None and right_arm_state is not None:
            if left_arm_state.ndim == 2:
                left_eef, right_eef = self.g1_fk.compute_eef_9d_batch(left_arm_state, right_arm_state)
            else:
                left_eef_single, right_eef_single = self.g1_fk.compute_eef_9d(left_arm_state, right_arm_state)
                left_eef = np.expand_dims(left_eef_single, 0)
                right_eef = np.expand_dims(right_eef_single, 0)
            states["left_eef_9d"] = left_eef
            states["right_eef_9d"] = right_eef

        for hand in ["left_hand", "right_hand"]:
            col = f"obs/positions/{hand}"
            if col in main_row:
                arr = np.array(main_row[col], dtype=np.float32)
                if len(arr) % chunk_len == 0 and len(arr) > chunk_len:
                    arr = arr.reshape(chunk_len, -1)
                elif len(arr) % 50 == 0 and len(arr) > 50:
                    arr = arr.reshape(50, -1)
                state_parts.append(arr)

        if len(state_parts) > 0:
            # We assume modality config expects a single combined state or specific keys
            # Let's check what the modality config asked for
            state_keys = self.modality_configs.get("state", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys
            if len(state_keys) == 1:
                # Concatenate everything into the single state key
                states[state_keys[0]] = np.concatenate(state_parts, axis=-1)
            else:
                # If they mapped them explicitly, we'll try to put it back.
                # Usually for unified manipulation we just concat them.
                for key in ["core", "left_hand", "right_hand"]:
                    if key == "state" or key == "joint_positions":
                        states[key] = np.concatenate(state_parts, axis=-1)

        actions = {}
        action_parts = []
        left_arm_action = None
        right_arm_action = None

        if "action/q_target/core" in main_row:
            arr = np.array(main_row["action/q_target/core"], dtype=np.float32)
            if len(arr) % chunk_len == 0 and len(arr) > chunk_len:
                arr = arr.reshape(chunk_len, -1)
            elif len(arr) % 50 == 0 and len(arr) > 50:
                arr = arr.reshape(50, -1)
            if arr.shape[-1] >= 29:
                arms = arr[..., 15:]
                left_arm_action = arms[..., :7]
                right_arm_action = arms[..., 7:14]
                arr = arms
            action_parts.append(arr)

        if self.g1_fk is not None and left_arm_action is not None and right_arm_action is not None:
            if left_arm_action.ndim == 2:
                left_eef, right_eef = self.g1_fk.compute_eef_9d_batch(left_arm_action, right_arm_action)
            else:
                left_eef_single, right_eef_single = self.g1_fk.compute_eef_9d(left_arm_action, right_arm_action)
                left_eef = np.expand_dims(left_eef_single, 0)
                right_eef = np.expand_dims(right_eef_single, 0)
            actions["left_eef_9d"] = left_eef
            actions["right_eef_9d"] = right_eef

        for hand in ["left_hand", "right_hand"]:
            col = f"action/q_target/{hand}"
            if col in main_row:
                arr = np.array(main_row[col], dtype=np.float32)
                if len(arr) % chunk_len == 0 and len(arr) > chunk_len:
                    arr = arr.reshape(chunk_len, -1)
                elif len(arr) % 50 == 0 and len(arr) > 50:
                    arr = arr.reshape(50, -1)
                action_parts.append(arr)

        if len(action_parts) > 0:
            action_keys = self.modality_configs.get("action", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys
            if len(action_keys) == 1:
                actions[action_keys[0]] = np.concatenate(action_parts, axis=-1)
            else:
                for key in ["core", "left_hand", "right_hand"]:
                    if key == "action" or key == "joint_velocities" or key == "joint_positions":
                        actions[key] = np.concatenate(action_parts, axis=-1)

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
        for key in ["core", "left_hand", "right_hand"]:
            cols_to_load.append(f"obs/positions/{key}")
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

        all_state_data = defaultdict(list)
        all_action_data = defaultdict(list)

        # Read each sampled row iteratively to avoid loading entire episodes into memory (OOM)
        for row in sampled_for_stats:
            ep_uuid = row["episode_uuid"]
            chunk = row["chunk_in_episode"]

            if isinstance(ep_uuid, bytes):
                ep_uuid_val = ep_uuid
            else:
                ep_uuid_val = ep_uuid

            scanner = self.main_ds.scanner(
                columns=cols_to_load,
                filter=(pc.field("episode_uuid") == ep_uuid_val) & (pc.field("chunk_in_episode") == chunk)
            )
            table = scanner.to_table()
            if table.num_rows > 0:
                df = table.to_pandas()
                for key in ["core", "left_hand", "right_hand"]:
                    col = f"obs/positions/{key}"
                    if col in df.columns:
                        arr = df[col].iloc[0]
                        if arr is not None:
                            all_state_data[key].append(np.array(arr, dtype=np.float32))

                for key in ["core", "left_hand", "right_hand"]:
                    col = f"action/q_target/{key}"
                    if col in df.columns:
                        arr = df[col].iloc[0]
                        if arr is not None:
                            all_action_data[key].append(np.array(arr, dtype=np.float32))

        # Calculate statistics
        stats = {"state": defaultdict(dict), "action": defaultdict(dict)}

        # We need to apply the exact same slicing to the statistics
        state_parts_all = []
        action_parts_all = []

        # Process states
        if "core" in all_state_data:
            arr_list = all_state_data["core"]
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

            if concat_data.shape[-1] >= 29:
                concat_data = concat_data[..., 15:]
            state_parts_all.append(concat_data)

        for hand in ["left_hand", "right_hand"]:
            if hand in all_state_data:
                arr_list = all_state_data[hand]
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
                state_parts_all.append(concat_data)

        if state_parts_all:
            final_state = np.concatenate(state_parts_all, axis=-1)
            for key in state_keys:
                stats["state"][key]["mean"] = np.mean(final_state, axis=0).tolist()
                stats["state"][key]["std"] = np.std(final_state, axis=0).tolist()
                stats["state"][key]["min"] = np.min(final_state, axis=0).tolist()
                stats["state"][key]["max"] = np.max(final_state, axis=0).tolist()
                stats["state"][key]["q01"] = np.quantile(final_state, 0.01, axis=0).tolist()
                stats["state"][key]["q99"] = np.quantile(final_state, 0.99, axis=0).tolist()

        # Process actions
        if "core" in all_action_data:
            arr_list = all_action_data["core"]
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

            if concat_data.shape[-1] >= 29:
                concat_data = concat_data[..., 15:]
            action_parts_all.append(concat_data)

        for hand in ["left_hand", "right_hand"]:
            if hand in all_action_data:
                arr_list = all_action_data[hand]
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
                action_parts_all.append(concat_data)

        if action_parts_all:
            final_action = np.concatenate(action_parts_all, axis=-1)
            for key in action_keys:
                stats["action"][key]["mean"] = np.mean(final_action, axis=0).tolist()
                stats["action"][key]["std"] = np.std(final_action, axis=0).tolist()
                stats["action"][key]["min"] = np.min(final_action, axis=0).tolist()
                stats["action"][key]["max"] = np.max(final_action, axis=0).tolist()
                stats["action"][key]["q01"] = np.quantile(final_action, 0.01, axis=0).tolist()
                stats["action"][key]["q99"] = np.quantile(final_action, 0.99, axis=0).tolist()

        # Convert inner defaultdicts to regular dicts
        stats["state"] = dict(stats["state"])
        stats["action"] = dict(stats["action"])

        self._cached_stats = dict(stats)
        return self._cached_stats
