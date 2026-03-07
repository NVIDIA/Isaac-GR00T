#!/usr/bin/env python
"""
LeRobot Dataset Loader

A simplified, clean implementation for loading LeRobot datasets with video support.
This module provides the core functionality for loading episodes from LeRobot format datasets,
handling metadata parsing, video decoding, and data preprocessing for VLA training.

The LeRobotEpisodeLoader serves as the foundation for higher-level dataset classes,
providing episode-level data access with support for multi-modal data including:
- Video frames from multiple camera views
- Proprioceptive state information
- Action sequences
- Language instructions/annotations

Returns messages with VLAStepData as defined in types.py.
"""

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd

from gr00t.data.types import ModalityConfig
from gr00t.utils.initial_actions import INITIAL_ACTIONS_FILENAME, load_initial_actions
from gr00t.utils.video_utils import get_frames_by_indices


# Import NVIDIA on-demand video decoder with graceful fallback
try:
    import accvlab.on_demand_video_decoder as nvc

    NVC_AVAILABLE = True
except ImportError:
    NVC_AVAILABLE = False

import torch


# LeRobot standard metadata filenames
LEROBOT_META_DIR_NAME = "meta"
LEROBOT_INFO_FILENAME = "info.json"
LEROBOT_EPISODES_FILENAME = "episodes.jsonl"
LEROBOT_TASKS_FILENAME = "tasks.jsonl"
LEROBOT_MODALITY_FILENAME = "modality.json"
LEROBOT_STATS_FILE_NAME = "stats.json"
LEROBOT_RELATIVE_STATS_FILE_NAME = "relative_stats.json"

ALLOWED_MODALITIES = ["video", "state", "action", "language"]
DEFAULT_COLUMN_NAMES = {
    "state": "observation.state",
    "action": "action",
}

LANG_KEYS = ["task", "sub_task"]


def _rec_defaultdict() -> defaultdict:
    """Factory that creates an infinitely nestable defaultdict."""
    return defaultdict(_rec_defaultdict)


def _to_plain_dict(tree):
    """Recursively turn a (nested) defaultdict into a regular dict."""
    if isinstance(tree, defaultdict):
        return {k: _to_plain_dict(v) for k, v in tree.items()}
    return tree


class LeRobotEpisodeLoader:
    """
    Episode-level data loader for LeRobot format datasets.

    This class handles the loading and preprocessing of individual episodes from LeRobot datasets.
    It manages metadata parsing, video decoding, and data extraction across multiple modalities
    (video, state, action, language) while maintaining compatibility with the VLA training pipeline.

    Key responsibilities:
    - Parse LeRobot metadata files (info.json, episodes.jsonl, etc.)
    - Load and decode video data using configurable backends
    - Extract and process multi-modal data according to modality configurations
    - Provide dataset statistics for normalization
    - Handle initial action loading for policy initialization

    Args:
        dataset_path: Path to dataset root directory containing meta/ and data files
        modality_configs: Dictionary mapping modality names to ModalityConfig objects
                         that specify temporal sampling and data keys to load
        video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
        video_backend_kwargs: Additional arguments for the video backend

    Example:
        >>> loader = LeRobotEpisodeLoader(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(
        ...             delta_indices=list(range(16)), modality_keys=["joint_velocities"]
        ...         ),
        ...     },
        ... )
        >>> episode_data = loader[0]  # Load first episode as DataFrame
    """

    def __init__(
        self,
        dataset_path: str | Path,
        modality_configs: dict[str, ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize LeRobot episode loader with dataset path and modality configurations.

        The initialization process involves:
        1. Loading all metadata files from the dataset
        2. Parsing and validating modality configurations
        3. Computing effective episode lengths based on action horizon
        """
        self.dataset_path = Path(dataset_path)
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs

        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        # Load metadata files and parse dataset structure
        self._load_metadata()

        # Set up modality configs after metadata is loaded
        self.modality_configs = self._parse_and_validate_modality_configs(modality_configs)

        # Compute effective episode lengths accounting for action horizon
        self.episode_lengths = self.get_episode_lengths()

        # Initialize NVC decoder if using nvc backend (lazy initialization)
        self._nvc_decoder = None
        if self.video_backend == "nvc":
            if not NVC_AVAILABLE:
                raise ImportError(
                    "accvlab.on_demand_video_decoder is not available. "
                    "Please install it or use a different video_backend."
                )

    def _load_metadata(self) -> None:
        """
        Load all metadata files including dataset statistics.

        Parses the standard LeRobot metadata structure:
        - info.json: Dataset configuration and file patterns
        - episodes.jsonl: Per-episode metadata (length, timestamps, etc.)
        - tasks.jsonl: Task descriptions and mappings
        - modality.json: Modality structure and data layout
        - stats.json: Dataset statistics for normalization
        """
        meta_dir = self.dataset_path / LEROBOT_META_DIR_NAME

        # Load dataset configuration
        info_path = meta_dir / LEROBOT_INFO_FILENAME
        with open(info_path, "r") as f:
            self.info_meta = json.load(f)

        # Load episode metadata (one episode per line)
        episodes_path = meta_dir / LEROBOT_EPISODES_FILENAME
        with open(episodes_path, "r") as f:
            self.episodes_metadata = [json.loads(line) for line in f]

        # Load task descriptions and create mapping
        tasks_path = meta_dir / LEROBOT_TASKS_FILENAME
        with open(tasks_path, "r") as f:
            tasks_data = [json.loads(line) for line in f]
            self.tasks_map = {task["task_index"]: task["task"] for task in tasks_data}

        # Load modality structure information
        modality_path = meta_dir / LEROBOT_MODALITY_FILENAME
        with open(modality_path, "r") as f:
            self.modality_meta = json.load(f)

        # Load dataset statistics for normalization
        stats_path = meta_dir / LEROBOT_STATS_FILE_NAME
        assert stats_path.exists(), (
            f"{stats_path} does not exist for {self.dataset_path}, please use gr00t/data/stats.py to generate it"
        )
        with open(stats_path, "r") as f:
            self.stats = json.load(f)

        relative_stats_path = meta_dir / LEROBOT_RELATIVE_STATS_FILE_NAME
        if relative_stats_path.exists():
            with open(relative_stats_path, "r") as f:
                self.stats["relative_action"] = json.load(f)

        # Extract key configuration parameters
        self.feature_config = self.info_meta.get("features", {})
        self.data_path_pattern = self.info_meta["data_path"]
        self.video_path_pattern = self.info_meta.get("video_path")
        self.chunk_size = self.info_meta["chunks_size"]
        self.fps = self.info_meta.get("fps", 30)

    def get_episode_lengths(self):
        """
        Compute original episode lengths.

        Returns:
            List of original episode lengths
        """
        episode_lengths = []
        for ep_meta in self.episodes_metadata:
            episode_lengths.append(int(ep_meta["length"]))
        return episode_lengths

    def get_episode_length(self, idx: int) -> int:
        """Get the length of a specific episode."""
        return self.episode_lengths[idx]

    def _parse_and_validate_modality_configs(
        self,
        modality_configs: dict[str, ModalityConfig],
    ) -> dict[str, ModalityConfig]:
        """
        Parse and validate modality configurations, filling in defaults where needed.

        For missing modality configs, creates default configurations:
        - video: All available camera views with single timestep
        - state: All available state keys with single timestep
        - action: All available action keys with 16-step horizon
        - language: Must be explicitly configured if needed

        Args:
            modality_configs: User-provided modality configurations

        Returns:
            Complete and validated modality configurations

        Raises:
            ValueError: If invalid modalities are specified
            AssertionError: If language modality configuration is invalid
        """
        # Validate all modality configurations
        for modality in modality_configs:
            if modality not in ALLOWED_MODALITIES:
                raise ValueError(f"Invalid modality: {modality}")
            if modality == "language":
                # Language modality has special constraints
                assert len(modality_configs[modality].modality_keys) == 1, (
                    "Language modality must have exactly one key"
                )
                assert modality_configs[modality].delta_indices == [0], (
                    "Only single timestep is supported for language modality"
                )

        # Validate video modality_keys against modality.json.
        # Each key in modality_configs["video"].modality_keys must exist in
        # modality.json["video"], otherwise _load_video_data will fail with
        # a confusing KeyError when trying to resolve the original video key.
        if "video" in modality_configs and "video" in self.modality_meta:
            config_keys = set(modality_configs["video"].modality_keys)
            meta_keys = set(self.modality_meta["video"].keys())
            missing_keys = config_keys - meta_keys
            if missing_keys:
                raise ValueError(
                    f"Video modality_keys {sorted(missing_keys)} in modality_config "
                    f"not found in modality.json. "
                    f"modality_config expects: {sorted(config_keys)}, "
                    f"modality.json defines: {sorted(meta_keys)}. "
                    f"Please ensure modality.json and your modality_config use the "
                    f"same video key names."
                )

        return modality_configs

    def __len__(self) -> int:
        """Return number of episodes in dataset."""
        return len(self.episodes_metadata)

    def _extract_joint_groups(
        self,
        df: pd.DataFrame,
        joint_groups: list[str],
        modality_type: str = "state",
    ) -> pd.DataFrame:
        """
        Extract specific joint groups from data arrays based on modality metadata.

        Uses the modality metadata to slice the appropriate indices from the raw data arrays,
        allowing for flexible joint group extraction (e.g., arm joints, gripper state).

        Args:
            df: DataFrame containing the raw episode data
            joint_groups: List of joint group names to extract (e.g., ["arm", "gripper"])
            modality_type: Type of modality ("state" or "action")

        Returns:
            DataFrame with columns for each requested joint group containing sliced arrays
        """
        modality_info = self.modality_meta.get(modality_type, {})
        joint_data = pd.DataFrame()

        for group_name in joint_groups:
            if group_name in modality_info:
                group_info = modality_info[group_name]
                start_idx = group_info["start"]
                end_idx = group_info["end"]
                original_key = group_info.get("original_key", DEFAULT_COLUMN_NAMES[modality_type])
                # Slice the array data for this joint group
                if isinstance(df[original_key].iloc[0], np.ndarray):
                    joint_data[group_name] = df[original_key].map(lambda x: x[start_idx:end_idx])
                else:
                    joint_data[group_name] = df[original_key]  # for strings and scalars
            else:
                print(
                    f"Warning: Joint group '{group_name}' not found in {modality_type} modality. Available groups: {list(modality_info.keys())}"
                )

        return joint_data

    def _load_parquet_data(self, episode_index: int) -> pd.DataFrame:
        """
        Load and process parquet data for a specific episode.

        Handles the complete data loading pipeline:
        1. Load raw parquet file based on chunking structure
        2. Process language annotations (convert task indices to strings)
        3. Extract state and action joint groups

        Args:
            episode_index: Index of the episode to load

        Returns:
            Processed DataFrame with all modality data
        """
        # Load raw parquet data using chunking pattern
        chunk_idx = episode_index // self.chunk_size
        parquet_filename = self.data_path_pattern.format(
            episode_chunk=chunk_idx, episode_index=episode_index
        )
        parquet_path = self.dataset_path / parquet_filename
        original_df = pd.read_parquet(parquet_path)
        loaded_df = pd.DataFrame()

        # Process language annotations (convert task indices to task strings)
        if "language" in self.modality_configs:
            for key in self.modality_configs["language"].modality_keys:
                # these keys will be loaded separately from episodes.jsonl
                if key in LANG_KEYS:
                    continue
                assert key.startswith("annotation.")
                subkey = key.replace("annotation.", "")
                assert subkey in self.modality_meta["annotation"], (
                    f"Key {subkey} not found in language modality"
                )
                original_key = self.modality_meta["annotation"][subkey].get("original_key", key)
                loaded_df[f"language.{key}"] = original_df[original_key].apply(
                    lambda x: self.tasks_map[x]
                )

        # Extract joint groups for state and action modalities
        for modality_type in ["state", "action"]:
            if modality_type not in self.modality_configs:
                continue
            joint_groups_df = self._extract_joint_groups(
                original_df,
                self.modality_configs[modality_type].modality_keys,
                modality_type,
            )
            for joint_group in joint_groups_df.columns:
                loaded_df[f"{modality_type}.{joint_group}"] = joint_groups_df[joint_group]

        return loaded_df

    def _load_video_data(self, episode_index: int, indices: np.ndarray) -> dict[str, np.ndarray]:
        """
        Load video data for all configured camera views at specified indices.

        Uses the configured video backend to decode video frames at the exact indices
        needed for the episode, supporting multiple camera views simultaneously.

        Args:
            episode_index: Index of the episode to load videos for
            indices: Array of indices to extract frames at

        Returns:
            Dictionary mapping camera view names to arrays of decoded frames
        """
        video_data = {}

        if not self.video_path_pattern or "video" not in self.modality_configs:
            return video_data

        chunk_idx = episode_index // self.chunk_size
        image_keys = self.modality_configs["video"].modality_keys

        # Use NVC backend for GPU-accelerated multi-camera decoding
        if self.video_backend == "nvc":
            return self._load_video_data_nvc(episode_index, indices, chunk_idx, image_keys)

        # Default path: decode each camera separately using standard backends
        for image_key in image_keys:
            # Resolve the original key used in video file naming
            original_key = self.modality_meta["video"][image_key].get(
                "original_key", f"observation.images.{image_key}"
            )
            assert original_key in self.feature_config, (
                f"Original key {original_key} not found in feature config"
            )

            # Construct video file path using pattern
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_idx,
                video_key=original_key,
                episode_index=episode_index,
            )
            video_path = self.dataset_path / video_filename

            # Decode video frames at specified timestamps
            video_data[image_key] = get_frames_by_indices(
                str(video_path),
                indices,
                video_backend=self.video_backend,
                video_backend_kwargs=self.video_backend_kwargs or {},
            )

        return video_data

    def _load_video_data_nvc(
        self,
        episode_index: int,
        indices: np.ndarray,
        chunk_idx: int,
        image_keys: list[str],
    ) -> dict[str, np.ndarray]:
        """
        Load video data using NVIDIA GPU-accelerated decoder (nvc backend).

        This method decodes frames from multiple camera videos simultaneously,
        leveraging GPU hardware decoding for optimal performance with sequential
        frame access patterns.

        Args:
            episode_index: Index of the episode to load videos for
            indices: Array of frame indices to extract
            chunk_idx: Chunk index for video file path pattern
            image_keys: List of camera view keys to load

        Returns:
            Dictionary mapping camera view names to arrays of decoded frames
        """
        torch.cuda.nvtx.range_push(f"nvc_load_video_ep{episode_index}")

        num_cameras = len(image_keys)
        num_frames = len(indices)

        # Build video paths for all cameras
        torch.cuda.nvtx.range_push("nvc_build_video_paths")
        video_paths = []
        for image_key in image_keys:
            original_key = self.modality_meta["video"][image_key].get(
                "original_key", f"observation.images.{image_key}"
            )
            assert original_key in self.feature_config, (
                f"Original key {original_key} not found in feature config"
            )

            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_idx, video_key=original_key, episode_index=episode_index
            )
            video_paths.append(str(self.dataset_path / video_filename))
        torch.cuda.nvtx.range_pop()  # nvc_build_video_paths

        # Lazy initialize NVC decoder
        if self._nvc_decoder is None:
            torch.cuda.nvtx.range_push("nvc_init_decoder")
            gpu_id = (self.video_backend_kwargs or {}).get("gpu_id", 0)
            self._nvc_decoder = nvc.CreateSampleReader(
                num_of_set=1,  # Sequential access to same video set
                num_of_file=num_cameras,  # Number of cameras
                iGpu=gpu_id,
            )
            torch.cuda.nvtx.range_pop()  # nvc_init_decoder

        # Decode all frames: iterate over frame indices, decode all cameras per frame
        # Temporary storage: list of frames for each camera
        frame_lists = {key: [] for key in image_keys}
        as_bgr = (self.video_backend_kwargs or {}).get("as_bgr", False)

        torch.cuda.nvtx.range_push(f"nvc_decode_frames_n{num_frames}")
        for frame_idx in indices:
            # Decode one frame from each camera simultaneously
            torch.cuda.nvtx.range_push(f"nvc_decode_frame_{frame_idx}")
            frame_indices = [int(frame_idx)] * num_cameras
            decoded_frames = self._nvc_decoder.DecodeN12ToRGB(video_paths, frame_indices, as_bgr)

            # Convert CUDA frames to numpy and append to each camera's list
            for i, image_key in enumerate(image_keys):
                frame_np = torch.as_tensor(decoded_frames[i]).cpu().numpy()
                frame_np = np.expand_dims(frame_np, axis=0)  # (H, W, C) -> (1, H, W, C)
                frame_lists[image_key].append(frame_np)

            torch.cuda.nvtx.range_pop()  # nvc_decode_frame_{frame_idx}
        torch.cuda.nvtx.range_pop()  # nvc_decode_frames_n{num_frames}

        # Concatenate frames into final video_data: dict[str, ndarray] with shape (N, H, W, C)
        torch.cuda.nvtx.range_push("nvc_concat_frames")
        video_data = {key: np.concatenate(frames, axis=0) for key, frames in frame_lists.items()}
        torch.cuda.nvtx.range_pop()  # nvc_concat_frames

        torch.cuda.nvtx.range_pop()  # nvc_load_video_ep{episode_index}

        return video_data

    def get_dataset_statistics(self) -> dict[str, Any]:
        """
        Extract dataset statistics for normalization from loaded metadata.

        Constructs a nested dictionary containing statistics (mean, std, min, max, q01, q99)
        for each joint group in state and action modalities. These statistics are used
        by processors for data normalization during training.

        Returns:
            Nested dictionary: {modality: {joint_group: {stat_type: values}}}
        """
        mapping = {"state": "observation.state", "action": "action"}
        dataset_statistics = _rec_defaultdict()

        for modality in mapping.keys():  # state, action
            for joint_key in self.modality_configs[modality].modality_keys:
                # Determine which statistics key to use
                if self.modality_meta[modality][joint_key].get("original_key", None) is not None:
                    stats_key = self.modality_meta[modality][joint_key]["original_key"]
                else:
                    stats_key = mapping[modality]

                # Extract the relevant slice of statistics
                start_idx, end_idx = (
                    self.modality_meta[modality][joint_key]["start"],
                    self.modality_meta[modality][joint_key]["end"],
                )
                for stat_type in self.stats[stats_key].keys():  # mean, std, min, max, q01, q99
                    dataset_statistics[modality][joint_key][stat_type] = self.stats[stats_key][
                        stat_type
                    ][start_idx:end_idx]
        stats = _to_plain_dict(dataset_statistics)
        # Directly add relative action stats
        if "relative_action" in self.stats:
            stats["relative_action"] = self.stats["relative_action"]
        return stats

    def create_language_from_meta(
        self, episode_meta: dict, nframes: int, lang_key: str
    ) -> list[str]:
        if lang_key == "task":
            meta_language = random.choice(episode_meta["tasks"])
            new_languages = [meta_language] * nframes
        elif lang_key == "sub_task":
            action_delta_indices = self.modality_configs["action"].delta_indices
            action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1
            new_languages = [[] for _ in range(nframes)]
            sub_tasks = episode_meta["sub_tasks"]
            for sub_task in sub_tasks:
                start_idx, end_idx, sub_text = (
                    sub_task["start"],
                    sub_task["end"],
                    sub_task["text"],
                )
                horizon = action_horizon // 2
                for i in range(start_idx - horizon, end_idx):
                    if i < 0:
                        continue
                    new_languages[i].append(sub_text)
            new_languages = [i if len(i) > 0 else [""] for i in new_languages]
            new_languages = [random.choice(i) for i in new_languages]
        else:
            raise ValueError(f"Language key {lang_key} not supported")
        return new_languages

    def __getitem__(self, idx: int) -> pd.DataFrame:
        """
        Load complete episode data as a processed DataFrame.

        Combines parquet data loading and video decoding to create a unified DataFrame
        containing all modality data for the episode. Video frames are converted to
        PIL Images and stored in the DataFrame.

        Args:
            idx: Episode index to load

        Returns:
            DataFrame with columns for all modalities and timestamps, with video frames
            as PIL Images ready for further processing

        Raises:
            IndexError: If episode index is out of bounds
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of bounds")

        episode_meta = self.episodes_metadata[idx]
        episode_id = episode_meta["episode_index"]
        nominal_length = episode_meta["length"]

        # Load and parse the parquet data
        df = self._load_parquet_data(episode_id)

        if "language" in self.modality_configs:
            lang_key = self.modality_configs["language"].modality_keys[0]
            if lang_key in LANG_KEYS:
                new_languages = self.create_language_from_meta(episode_meta, len(df), lang_key)
                df["language." + lang_key] = new_languages

        # Use actual dataframe length (might be less than nominal)
        actual_length = min(len(df), nominal_length)
        df = df.iloc[:actual_length]

        # Load synchronized video data
        video_data = self._load_video_data(episode_id, np.arange(actual_length))

        # Add video frames to dataframe as PIL Images
        for key in video_data.keys():
            assert len(video_data[key]) == len(df), (
                f"Video data for {key} has length {len(video_data[key])} but dataframe has length {len(df)}"
            )
            df[f"video.{key}"] = [frame for frame in video_data[key]]

        return df

    def _compute_required_frame_indices(
        self,
        step_indices: np.ndarray,
        max_length: int,
    ) -> np.ndarray:
        """
        Compute video frame indices required for the given step indices.

        This method ONLY considers delta_indices from the 'video' modality to determine
        which frames need to be decoded. Other modalities (state, action, language) get
        their data from parquet files, not from video decoding.

        Args:
            step_indices: Array of step indices that will be used for training
            max_length: Maximum valid frame index (episode length)

        Returns:
            Sorted array of unique frame indices that need to be decoded

        Example:
            If step_indices = [5, 10, 50] and video.delta_indices = [-1, 0]:
            Then required indices = {4, 5, 9, 10, 49, 50}
        """
        required_indices = set()

        # Only use video modality's delta_indices for computing required frames
        # Other modalities (state, action, language) get data from parquet, not video
        if "video" in self.modality_configs:
            video_delta_indices = self.modality_configs["video"].delta_indices
            for step_idx in step_indices:
                for delta in video_delta_indices:
                    frame_idx = int(step_idx) + delta
                    # Clamp to valid range [0, max_length - 1]
                    if 0 <= frame_idx < max_length:
                        required_indices.add(frame_idx)

        return np.array(sorted(required_indices), dtype=np.int64)

    def load_episode_sampled(
        self,
        episode_index: int,
        step_indices: np.ndarray,
    ) -> pd.DataFrame:
        """
        Load episode data with on-demand video decoding (nvc backend optimization).

        This method only decodes the video frames that are actually needed based on
        step_indices and modality delta_indices, saving ~90% decoding overhead when
        episode_sampling_rate is low (e.g., 0.1).

        The method supports delta_indices with length > 1 for all modalities:
        - video: e.g., delta_indices=[-1, 0] for previous + current frame
        - action: e.g., delta_indices=[0, 1, ..., 15] for 16-step action horizon
        - state: e.g., delta_indices=[0] for current state only

        Args:
            episode_index: Index of the episode to load
            step_indices: Array of step indices that will be used for training

        Returns:
            DataFrame with complete parquet data but sparse video data.
            Video columns only contain decoded frames at required indices,
            other positions are None.

        Note:
            This method is specifically designed for nvc backend. Other backends
            should continue using __getitem__ which decodes all frames.
        """
        if episode_index < 0 or episode_index >= len(self):
            raise IndexError(f"Episode index {episode_index} out of bounds")

        episode_meta = self.episodes_metadata[episode_index]
        episode_id = episode_meta["episode_index"]
        nominal_length = episode_meta["length"]

        # Load complete parquet data (state, action, language) - this is fast
        df = self._load_parquet_data(episode_id)

        # Process language annotations from metadata
        if "language" in self.modality_configs:
            lang_key = self.modality_configs["language"].modality_keys[0]
            if lang_key in LANG_KEYS:
                new_languages = self.create_language_from_meta(episode_meta, len(df), lang_key)
                df["language." + lang_key] = new_languages

        # Use actual dataframe length (might be less than nominal)
        actual_length = min(len(df), nominal_length)
        df = df.iloc[:actual_length].copy()

        # Set DataFrame index to frame numbers for .loc access compatibility
        df.index = pd.RangeIndex(actual_length)

        # Compute required frame indices based on step_indices and all modality delta_indices
        required_frame_indices = self._compute_required_frame_indices(step_indices, actual_length)

        # Decode only the required video frames (this is where we save ~90% overhead)
        video_data = self._load_video_data(episode_id, required_frame_indices)

        # Build sparse video columns (only required frames have values, others are None)
        for key in video_data.keys():
            # Initialize column with None for all positions
            video_column = [None] * actual_length
            # Fill only the decoded frames at their original positions
            for i, frame_idx in enumerate(required_frame_indices):
                video_column[frame_idx] = video_data[key][i]
            df[f"video.{key}"] = video_column

        return df

    def get_initial_actions(self):
        """
        Load initial actions for policy initialization if available.

        Returns:
            List containing initial action dictionaries, or empty list if not available
        """
        meta_dirpath = self.dataset_path / LEROBOT_META_DIR_NAME
        initial_actions_path = meta_dirpath / INITIAL_ACTIONS_FILENAME
        if initial_actions_path.exists():
            initial_actions = load_initial_actions(initial_actions_path)
            return initial_actions  # a single-element list of dict[str, dict[str, np.ndarray]]
        else:
            return []
