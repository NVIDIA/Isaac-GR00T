from dataclasses import dataclass, field
from typing import Any, List, Optional

from gr00t.data.types import ModalityConfig

from .embodiment_configs import MODALITY_CONFIGS


@dataclass
class SingleDatasetConfig:
    """Configuration for a single dataset in a mixed-training setup.

    A list of these objects can be supplied in ``DataConfig.datasets`` to mix
    multiple datasets at arbitrary ratios.  For convenience the *legacy*
    single-dataset fields still exist; if ``datasets`` is non-empty they take
    precedence.
    """

    # Path to the dataset root directory (can be strings or dicts for complex configs)
    dataset_paths: List[Any]

    # Robot embodiment identifier (e.g. "gr1", "franka")
    embodiment_tag: Optional[str] = None

    # Relative sampling probability (will be normalised across the list)
    mix_ratio: float = 1.0

    dataset_type: str = "physical_embodiment"

    # Optional validation dataset path for open-loop evaluation
    # If not provided, falls back to dataset_paths for evaluation
    val_dataset_path: Optional[str] = None


@dataclass
class DataConfig:
    """Dataset configuration (supports single or multiple datasets)."""

    # Leave empty by default for backwards-compatibility with the original
    # single-dataset workflow.  Users can supply one or more configs via CLI or
    # YAML when they need mixing.
    datasets: List[SingleDatasetConfig] = field(default_factory=list)

    # Modality configs
    # There are three sources of modality configs:
    # 1. Default modality configs in code: gr00t/configs/data/embodiment_configs.py
    # 2. Modality configs supplied through command line: --data.modality_configs (although rare and inconvenient)
    # 1 and 2 are unified through `config.data.modality_configs`.
    # 3. modality configs saved in the pretrained checkpoint.
    modality_configs: dict[str, dict[str, ModalityConfig]] = field(
        default_factory=lambda: MODALITY_CONFIGS
    )

    # Sharded dataset configuration
    download_cache: bool = False
    shard_size: int = 2**10
    episode_sampling_rate: float = 0.1
    num_shards_per_epoch: int = int(1e5)

    # Override statistics from the pretrained checkpoint
    override_pretraining_statistics: bool = False

    # General task / mode config (shared across datasets)
    mode: str = "single_turn"
    random_chop: float = 0.0
    mock_dataset_mode: bool = False  # if True, cache the first datapoint of each dataset and always return one of them to simulate best-case dataloading

    # Data loading
    shuffle: bool = True
    seed: int = 42
    multiprocessing_context: str = "fork"  # Options: "fork", "spawn", and "forkserver"
    allow_padding: bool = False

    # Subsample ratio for the dataset
    subsample_ratio: float = 1.0

    # DP Image Config
    image_crop_size: List[int] = field(default_factory=lambda: [244, 244])
    image_target_size: List[int] = field(default_factory=lambda: [224, 224])
    video_backend: str = "torchcodec"
