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


    def _coerce_uuid_to_bytes(self, raw_uuid) -> bytes:
        if isinstance(raw_uuid, bytes):
            return raw_uuid
        elif isinstance(raw_uuid, bytearray):
            return bytes(raw_uuid)
        elif isinstance(raw_uuid, dict) and 'bytes' in raw_uuid:
            return bytes(raw_uuid['bytes'])
        elif isinstance(raw_uuid, np.ndarray):
            return raw_uuid.tobytes()
        elif isinstance(raw_uuid, str):
            import uuid
            return uuid.UUID(raw_uuid).bytes
        elif isinstance(raw_uuid, list):
            return bytes(raw_uuid)
        else:
            raise TypeError(f"Cannot coerce UUID of type {type(raw_uuid)}: {raw_uuid}")

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

        # Core arm joints and hand joints must always be loaded to extract manipulation states/actions
        for key in ["core", "left_hand", "right_hand"]:
            cols_to_load.append(f"obs/positions/{key}")
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

            ep_uuid_val = self._coerce_uuid_to_bytes(ep_uuid)
            dp = None

            scanner = self.main_ds.scanner(
                columns=cols_to_load,
                filter=f"chunk_in_episode = {chunk}"
            )
            table = scanner.to_table()

            if table.num_rows > 0:
                ep_uuids_col = table.column("episode_uuid").to_pylist()
                matched_idx = -1
                for idx_t, val in enumerate(ep_uuids_col):
                    if self._coerce_uuid_to_bytes(val) == ep_uuid_val:
                        matched_idx = idx_t
                        break

                if matched_idx >= 0:
                    main_row = table.to_pandas().iloc[matched_idx]
                    dp = self.get_datapoint(main_row, instr)

            if dp is not None:
                datapoints.append(dp)

        return datapoints


    def get_datapoint(self, main_row, instruction) -> dict | None:
        if self.processor is None:
            raise ValueError("Processor must be set before getting datapoints")

        video_data = {}
        for key in self.modality_configs.get("video", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys:
            if key == "image" or key == "primary_image_key": img_col = "obs/camera/left_image_256"
            elif key == "wrist_image": img_col = "obs/camera/wrist_left_image_256"
            else: img_col = f"obs/camera/{key}_image_256"

            if img_col in main_row:
                img_bytes = main_row[img_col]
                if pd.notna(img_bytes) and img_bytes is not None:
                    img = Image.open(io.BytesIO(img_bytes))
                    video_data[key] = [np.array(img)]

        chunk_len = len(self.action_delta_indices) if len(self.action_delta_indices) > 0 else 50

        states = {}
        actions = {}


        # 1. Gather all individual components
        s_core_col, a_core_col = "obs/positions/core", "action/q_target/core"
        s_l_col, a_l_col = "obs/positions/left_hand", "action/q_target/left_hand"
        s_r_col, a_r_col = "obs/positions/right_hand", "action/q_target/right_hand"

        left_arm_state, right_arm_state = None, None
        left_arm_action, right_arm_action = None, None

        s_arms, a_arms_delta = None, None
        s_l_hand, a_l_hand = None, None
        s_r_hand, a_r_hand = None, None

        # Parse Core Arms
        if s_core_col in main_row:
            arr = np.array(main_row[s_core_col], dtype=np.float32)
            if len(arr) % chunk_len == 0 and len(arr) > chunk_len: arr = arr.reshape(chunk_len, -1)
            elif len(arr) % 50 == 0 and len(arr) > 50: arr = arr.reshape(50, -1)

            if len(arr.shape) >= 2 and arr.shape[-1] >= 29:
                left_arm_state = arr[..., 15:22]
                right_arm_state = arr[..., 22:29]
                s_arms = arr[..., 15:29]
            elif len(arr.shape) == 1 and arr.shape[0] >= 29:
                left_arm_state = arr[..., 15:22]
                right_arm_state = arr[..., 22:29]
                s_arms = arr[..., 15:29]

        if a_core_col in main_row:
            arr = np.array(main_row[a_core_col], dtype=np.float32)
            if len(arr) % chunk_len == 0 and len(arr) > chunk_len: arr = arr.reshape(chunk_len, -1)
            elif len(arr) % 50 == 0 and len(arr) > 50: arr = arr.reshape(50, -1)

            if len(arr.shape) >= 2 and arr.shape[-1] >= 29:
                left_arm_action = arr[..., 15:22]
                right_arm_action = arr[..., 22:29]
                a_arms = arr[..., 15:29]
            elif len(arr.shape) == 1 and arr.shape[0] >= 29:
                left_arm_action = arr[..., 15:22]
                right_arm_action = arr[..., 22:29]
                a_arms = arr[..., 15:29]

            if s_arms is not None:
                a_arms_delta = a_arms - s_arms

        # Parse Left Hand (Gripper) - Binarized, No Deltas
        if s_l_col in main_row:
            arr = np.array(main_row[s_l_col], dtype=np.float32)
            if len(arr) % chunk_len == 0 and len(arr) > chunk_len: arr = arr.reshape(chunk_len, -1)
            elif len(arr) % 50 == 0 and len(arr) > 50: arr = arr.reshape(50, -1)
            s_l_hand = np.where(arr < 0.5, 0.0, 1.0).astype(np.float32)

        if a_l_col in main_row:
            arr = np.array(main_row[a_l_col], dtype=np.float32)
            if len(arr) % chunk_len == 0 and len(arr) > chunk_len: arr = arr.reshape(chunk_len, -1)
            elif len(arr) % 50 == 0 and len(arr) > 50: arr = arr.reshape(50, -1)
            a_l_hand = np.where(arr < 0.5, 0.0, 1.0).astype(np.float32)

        # Parse Right Hand (Gripper) - Binarized, No Deltas
        if s_r_col in main_row:
            arr = np.array(main_row[s_r_col], dtype=np.float32)
            if len(arr) % chunk_len == 0 and len(arr) > chunk_len: arr = arr.reshape(chunk_len, -1)
            elif len(arr) % 50 == 0 and len(arr) > 50: arr = arr.reshape(50, -1)
            s_r_hand = np.where(arr < 0.5, 0.0, 1.0).astype(np.float32)

        if a_r_col in main_row:
            arr = np.array(main_row[a_r_col], dtype=np.float32)
            if len(arr) % chunk_len == 0 and len(arr) > chunk_len: arr = arr.reshape(chunk_len, -1)
            elif len(arr) % 50 == 0 and len(arr) > 50: arr = arr.reshape(50, -1)
            a_r_hand = np.where(arr < 0.5, 0.0, 1.0).astype(np.float32)

        # 2. Concatenate into Unified Arrays
        s_unified, a_unified = None, None
        if s_arms is not None and s_l_hand is not None and s_r_hand is not None:
            if s_arms.ndim == s_l_hand.ndim == s_r_hand.ndim:
                s_unified = np.concatenate([s_arms, s_l_hand, s_r_hand], axis=-1)
            elif s_arms.ndim == 1 and s_l_hand.ndim == 1 and s_r_hand.ndim == 1:
                s_unified = np.concatenate([s_arms, s_l_hand, s_r_hand], axis=-1)
            else:
                s_unified = s_arms
        elif s_arms is not None:
            s_unified = s_arms

        if a_arms_delta is not None and a_l_hand is not None and a_r_hand is not None:
            if a_arms_delta.ndim == a_l_hand.ndim == a_r_hand.ndim:
                a_unified = np.concatenate([a_arms_delta, a_l_hand, a_r_hand], axis=-1)
            elif a_arms_delta.ndim == 1 and a_l_hand.ndim == 1 and a_r_hand.ndim == 1:
                a_unified = np.concatenate([a_arms_delta, a_l_hand, a_r_hand], axis=-1)
            else:
                a_unified = a_arms_delta
        elif a_arms_delta is not None:
            a_unified = a_arms_delta

        state_keys = self.modality_configs.get("state", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys
        action_keys = self.modality_configs.get("action", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys

        if s_unified is not None:
            for key in state_keys:
                if key not in ["left_eef_9d", "right_eef_9d"]:
                    states[key] = s_unified

        if a_unified is not None:
            for key in action_keys:
                if key not in ["left_eef_9d", "right_eef_9d"]:
                    actions[key] = a_unified

        # 3. EEF (from arms)
        if self.g1_fk is not None and left_arm_state is not None and right_arm_state is not None:
            if left_arm_state.ndim == 2:
                s_l_eef, s_r_eef = self.g1_fk.compute_eef_9d_batch(left_arm_state, right_arm_state)
            else:
                sl, sr = self.g1_fk.compute_eef_9d(left_arm_state, right_arm_state)
                s_l_eef, s_r_eef = np.expand_dims(sl, 0), np.expand_dims(sr, 0)
            states["left_eef_9d"] = s_l_eef
            states["right_eef_9d"] = s_r_eef

        if self.g1_fk is not None and left_arm_action is not None and right_arm_action is not None:
            if left_arm_action.ndim == 2:
                a_l_eef, a_r_eef = self.g1_fk.compute_eef_9d_batch(left_arm_action, right_arm_action)
            else:
                al, ar = self.g1_fk.compute_eef_9d(left_arm_action, right_arm_action)
                a_l_eef, a_r_eef = np.expand_dims(al, 0), np.expand_dims(ar, 0)
            actions["left_eef_9d"] = a_l_eef
            actions["right_eef_9d"] = a_r_eef


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
        if hasattr(self, "_cached_stats"): return self._cached_stats

        import pyarrow.compute as pc
        from collections import defaultdict
        import random

        state_keys = self.modality_configs.get("state", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys
        action_keys = self.modality_configs.get("action", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys

        cols_to_load = []
        for key in ["core", "left_hand", "right_hand"]:
            cols_to_load.append(f"obs/positions/{key}")
            cols_to_load.append(f"action/q_target/{key}")

        all_sampled_rows = []
        for rows in self.sharded_rows: all_sampled_rows.extend(rows)

        max_stat_samples = 5000
        if len(all_sampled_rows) > max_stat_samples: sampled_for_stats = random.sample(all_sampled_rows, max_stat_samples)
        else: sampled_for_stats = all_sampled_rows

        if len(sampled_for_stats) == 0: return {}

        all_state_data = defaultdict(list)
        all_action_data = defaultdict(list)

        for row in sampled_for_stats:
            ep_uuid = row["episode_uuid"]
            chunk = row["chunk_in_episode"]
            ep_uuid_val = self._coerce_uuid_to_bytes(ep_uuid)

            scanner = self.main_ds.scanner(columns=cols_to_load, filter=f"chunk_in_episode = {chunk}")
            table = scanner.to_table()

            if table.num_rows > 0:
                ep_uuids_col = table.column("episode_uuid").to_pylist()
                matched_idx = -1
                for idx_t, val in enumerate(ep_uuids_col):
                    if self._coerce_uuid_to_bytes(val) == ep_uuid_val:
                        matched_idx = idx_t
                        break

                if matched_idx >= 0:
                    df = table.to_pandas().iloc[matched_idx]

                    s_core_col, a_core_col = "obs/positions/core", "action/q_target/core"
                    if s_core_col in df.index and a_core_col in df.index:
                        s_core, a_core = df[s_core_col], df[a_core_col]
                        if s_core is not None and a_core is not None and len(s_core) > 0 and len(a_core) > 0:
                            s_arr = np.array(s_core, dtype=np.float32)
                            a_arr = np.array(a_core, dtype=np.float32)

                            chunk_len = len(self.action_delta_indices) if len(self.action_delta_indices) > 0 else 50
                            if len(s_arr) % chunk_len == 0 and len(s_arr) > chunk_len: s_arr = s_arr.reshape(chunk_len, -1)
                            elif len(s_arr) % 50 == 0 and len(s_arr) > 50: s_arr = s_arr.reshape(50, -1)
                            if len(a_arr) % chunk_len == 0 and len(a_arr) > chunk_len: a_arr = a_arr.reshape(chunk_len, -1)
                            elif len(a_arr) % 50 == 0 and len(a_arr) > 50: a_arr = a_arr.reshape(50, -1)

                            s_la, s_ra, a_la, a_ra = None, None, None, None

                            if len(s_arr.shape) >= 2 and s_arr.shape[-1] >= 29:
                                s_la, s_ra = s_arr[..., 15:22], s_arr[..., 22:29]
                                s_arr = s_arr[..., 15:29]
                            elif len(s_arr.shape) == 1 and s_arr.shape[0] >= 29:
                                s_la, s_ra = s_arr[..., 15:22], s_arr[..., 22:29]
                                s_arr = s_arr[..., 15:29]

                            if len(a_arr.shape) >= 2 and a_arr.shape[-1] >= 29:
                                a_la, a_ra = a_arr[..., 15:22], a_arr[..., 22:29]
                                a_arr = a_arr[..., 15:29]
                            elif len(a_arr.shape) == 1 and a_arr.shape[0] >= 29:
                                a_la, a_ra = a_arr[..., 15:22], a_arr[..., 22:29]
                                a_arr = a_arr[..., 15:29]

                            if self.g1_fk is not None and s_la is not None and s_ra is not None and a_la is not None and a_ra is not None:
                                if s_la.ndim == 2:
                                    s_le, s_re = self.g1_fk.compute_eef_9d_batch(s_la, s_ra)
                                    a_le, a_re = self.g1_fk.compute_eef_9d_batch(a_la, a_ra)
                                else:
                                    s_le, s_re = self.g1_fk.compute_eef_9d(s_la, s_ra)
                                    a_le, a_re = self.g1_fk.compute_eef_9d(a_la, a_ra)
                                    s_le, s_re = np.expand_dims(s_le, 0), np.expand_dims(s_re, 0)
                                    a_le, a_re = np.expand_dims(a_le, 0), np.expand_dims(a_re, 0)

                                all_state_data["left_eef_9d"].append(s_le)
                                all_state_data["right_eef_9d"].append(s_re)
                                all_action_data["left_eef_9d"].append(a_le)
                                all_action_data["right_eef_9d"].append(a_re)

                            all_state_data["core"].append(s_arr)
                            all_action_data["core"].append(a_arr - s_arr)

                    for hand in ["left_hand", "right_hand"]:
                        s_col, a_col = f"obs/positions/{hand}", f"action/q_target/{hand}"
                        g_key = "left_gripper" if hand == "left_hand" else "right_gripper"

                        if s_col in df.index and a_col in df.index:
                            s_arr, a_arr = df[s_col], df[a_col]
                            if s_arr is not None and a_arr is not None and len(s_arr) > 0 and len(a_arr) > 0:
                                s_h = np.array(s_arr, dtype=np.float32)
                                a_h = np.array(a_arr, dtype=np.float32)

                                chunk_len = len(self.action_delta_indices) if len(self.action_delta_indices) > 0 else 50
                                if len(s_h) % chunk_len == 0 and len(s_h) > chunk_len: s_h = s_h.reshape(chunk_len, -1)
                                elif len(s_h) % 50 == 0 and len(s_h) > 50: s_h = s_h.reshape(50, -1)
                                if len(a_h) % chunk_len == 0 and len(a_h) > chunk_len: a_h = a_h.reshape(chunk_len, -1)
                                elif len(a_h) % 50 == 0 and len(a_h) > 50: a_h = a_h.reshape(50, -1)

                                # Binarize gripper states: < 0.5 -> 0.0, else 1.0
                                s_h = np.where(s_h < 0.5, 0.0, 1.0).astype(np.float32)
                                # Grippers remain absolute, binarized actions
                                a_h = np.where(a_h < 0.5, 0.0, 1.0).astype(np.float32)

                                all_state_data[g_key].append(s_h)
                                all_action_data[g_key].append(a_h)

        stats = {"state": defaultdict(dict), "action": defaultdict(dict)}

        # We need to construct the unified "state" arrays (core + left_gripper + right_gripper) because the model expects them concatenated
        # To avoid misaligned concatenation, we iterate over zip lengths if all exist.
        if "core" in all_state_data and "left_gripper" in all_state_data and "right_gripper" in all_state_data:
            # Reconstruct unified arrays correctly for each chunk
            s_c_list, a_c_list = [], []
            for i in range(len(all_state_data["core"])):
                s_c = all_state_data["core"][i]
                s_l = all_state_data["left_gripper"][i] if i < len(all_state_data["left_gripper"]) else None
                s_r = all_state_data["right_gripper"][i] if i < len(all_state_data["right_gripper"]) else None

                a_c = all_action_data["core"][i]
                a_l = all_action_data["left_gripper"][i] if i < len(all_action_data["left_gripper"]) else None
                a_r = all_action_data["right_gripper"][i] if i < len(all_action_data["right_gripper"]) else None

                if s_l is not None and s_r is not None and a_l is not None and a_r is not None:
                    # They all exist for this row, concatenate them.
                    # Core arms + Left Gripper + Right Gripper
                    if s_c.ndim == s_l.ndim == s_r.ndim:
                        s_c_list.append(np.concatenate([s_c, s_l, s_r], axis=-1))
                        a_c_list.append(np.concatenate([a_c, a_l, a_r], axis=-1))
                    elif s_c.ndim == 1 and s_l.ndim == 1 and s_r.ndim == 1:
                        s_c_list.append(np.concatenate([s_c, s_l, s_r], axis=-1))
                        a_c_list.append(np.concatenate([a_c, a_l, a_r], axis=-1))
                    else:
                        s_c_list.append(s_c)
                        a_c_list.append(a_c)
                else:
                    # Fallback to core only if grippers are missing in some extremely rare chunk
                    s_c_list.append(s_c)
                    a_c_list.append(a_c)

            all_state_data["unified"] = s_c_list
            all_action_data["unified"] = a_c_list
        else:
            if "core" in all_state_data:
                all_state_data["unified"] = all_state_data["core"]
                all_action_data["unified"] = all_action_data["core"]


        for k in ["unified", "left_eef_9d", "right_eef_9d"]:
            if k in all_state_data and len(all_state_data[k]) > 0:
                s_c = np.concatenate(all_state_data[k], axis=0)
                a_c = np.concatenate(all_action_data[k], axis=0)
                if s_c.ndim == 1:
                    s_c = s_c.reshape(-1, s_c.shape[-1] if len(s_c.shape) > 1 else s_c.size)
                    a_c = a_c.reshape(-1, a_c.shape[-1] if len(a_c.shape) > 1 else a_c.size)

                target_k_s = [sk for sk in state_keys if sk not in ["left_eef_9d", "right_eef_9d"]] if k == "unified" else [k]
                target_k_a = [ak for ak in action_keys if ak not in ["left_eef_9d", "right_eef_9d"]] if k == "unified" else [k]

                for tk in target_k_s:
                    stats["state"][tk]["mean"] = np.mean(s_c, axis=0).tolist()
                    stats["state"][tk]["std"] = np.std(s_c, axis=0).tolist()
                    stats["state"][tk]["min"] = np.min(s_c, axis=0).tolist()
                    stats["state"][tk]["max"] = np.max(s_c, axis=0).tolist()
                    stats["state"][tk]["q01"] = np.quantile(s_c, 0.01, axis=0).tolist()
                    stats["state"][tk]["q99"] = np.quantile(s_c, 0.99, axis=0).tolist()

                for tk in target_k_a:
                    stats["action"][tk]["mean"] = np.mean(a_c, axis=0).tolist()
                    stats["action"][tk]["std"] = np.std(a_c, axis=0).tolist()
                    stats["action"][tk]["min"] = np.min(a_c, axis=0).tolist()
                    stats["action"][tk]["max"] = np.max(a_c, axis=0).tolist()
                    stats["action"][tk]["q01"] = np.quantile(a_c, 0.01, axis=0).tolist()
                    stats["action"][tk]["q99"] = np.quantile(a_c, 0.99, axis=0).tolist()

        stats["state"] = dict(stats["state"])
        stats["action"] = dict(stats["action"])
        self._cached_stats = dict(stats)
        return self._cached_stats
