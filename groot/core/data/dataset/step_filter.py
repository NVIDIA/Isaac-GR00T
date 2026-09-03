"""Load the LeRobot exclusion-list step-filter format used by dataset overlays."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)
STEP_FILTER_FILENAME = "meta/step_filter.jsonl"


class EpisodeLengthSource(Protocol):
    """Minimal episode-loader contract needed to resolve step filters."""

    def get_episode_length(self, episode_index: int) -> int: ...


def load_step_indices_filter(
    dataset_path: str | Path, episode_loader: EpisodeLengthSource
) -> dict[int, np.ndarray] | None:
    """Convert excluded indices in ``step_filter.jsonl`` to allowed indices."""
    step_filter_path = Path(dataset_path) / STEP_FILTER_FILENAME
    if not step_filter_path.is_file():
        return None

    allowed_indices: dict[int, np.ndarray] = {}
    for line_number, line in enumerate(step_filter_path.read_text().splitlines(), start=1):
        record = json.loads(line)
        episode_index = int(record["episode_index"])
        if episode_index in allowed_indices:
            raise ValueError(
                f"Duplicate episode_index {episode_index} in {step_filter_path}:{line_number}"
            )
        episode_length = int(episode_loader.get_episode_length(episode_index))
        excluded = np.asarray(record["step_indices"], dtype=np.int64)
        if excluded.ndim != 1 or np.any(excluded < 0) or np.any(excluded >= episode_length):
            raise ValueError(
                f"Invalid excluded step indices for episode {episode_index} in {step_filter_path}"
            )
        allowed_indices[episode_index] = np.setdiff1d(
            np.arange(episode_length, dtype=np.int64), excluded
        )

    logger.info(
        "Loaded step filter %s: retained %s/%s episodes and %s steps",
        step_filter_path,
        sum(bool(len(indices)) for indices in allowed_indices.values()),
        len(allowed_indices),
        sum(len(indices) for indices in allowed_indices.values()),
    )
    return allowed_indices
