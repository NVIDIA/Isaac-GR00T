import re

with open("gr00t/data/dataset/lance_dataset.py", "r") as f:
    content = f.read()

# Instead of scanning the entire dataset in `get_datapoint`, we can use PyArrow's take/filter over chunk arrays if possible.
# However, the Lance `Scanner` with filter is the standard way to do pushdown.
# The issue mentioned `table = scanner.to_table()` dynamically finding UUIDs per row is "prohibitively slow and bottleneck the GPU".
# To fix this, we can scan the chunk of rows once in `get_shard` and then iterate over them, rather than doing it row-by-row in `get_datapoint`.

old_get_shard = """
    def get_shard(self, idx: int) -> list:
        rows = self.sharded_rows[idx]
        datapoints = []
        for row in rows:
            dp = self.get_datapoint(row)
            if dp is not None:
                datapoints.append(dp)
        return datapoints

    def get_datapoint(self, row) -> dict | None:
        if self.processor is None:
            raise ValueError("Processor must be set before getting datapoints")

        episode_uuid = row["episode_uuid"]
        chunk_in_episode = row["chunk_in_episode"]
        instruction = row.get("instruction", "")

        # To avoid scanning the entire dataset dynamically, we use Lance's pushdown filtering
        import pyarrow.compute as pc

        if isinstance(episode_uuid, bytes):
            episode_uuid_val = episode_uuid
        else:
            episode_uuid_val = episode_uuid

        # Optimization: the main dataset should ideally be queried in batches, or indexed.
        # For a single row, the scanner with filter is the Lance way,
        # but to make it fast we need to make sure we project only needed columns.

        cols_to_load = []
        for key in self.modality_configs.get("video", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys:
            if key == "image" or key == "primary_image_key": cols_to_load.append("obs/camera/left_image_256")
            elif key == "wrist_image": cols_to_load.append("obs/camera/wrist_left_image_256")
            else: cols_to_load.append(f"obs/camera/{key}_image_256")

        for key in self.modality_configs.get("state", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys:
            cols_to_load.append(f"obs/positions/{key}")

        for key in self.modality_configs.get("action", ModalityConfig(delta_indices=[], modality_keys=[])).modality_keys:
            cols_to_load.append(f"action/q_target/{key}")

        scanner = self.main_ds.scanner(
            columns=cols_to_load,
            filter=(pc.field("episode_uuid") == episode_uuid_val) & (pc.field("chunk_in_episode") == chunk_in_episode),
        )
        table = scanner.to_table()
        if table.num_rows == 0:
            return None

        main_row = table.to_pandas().iloc[0]
"""

new_get_shard = """
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
"""
content = content.replace(old_get_shard, new_get_shard)

with open("gr00t/data/dataset/lance_dataset.py", "w") as f:
    f.write(content)
