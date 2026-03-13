"""Validate and optionally repair GR00T-flavored LeRobot v2 datasets.

This script is intended as a lightweight preflight check for datasets used with
Isaac-GR00T. It verifies required metadata and can repair common gaps seen in
community datasets, including missing ``meta/episodes.jsonl`` and missing
``chunk_index`` fields in episode metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Missing dependency 'pyarrow'. Install project dependencies first (for example: `uv sync`)."
    ) from exc


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _scan_data_episodes(dataset_root: Path) -> list[dict[str, Any]]:
    """Build episode metadata directly from per-episode parquet files."""
    data_root = dataset_root / "data"
    parquet_files = sorted(data_root.glob("chunk-*/episode_*.parquet"))
    episodes = []
    for pq_path in parquet_files:
        episode_str = pq_path.stem.split("_")[-1]
        episode_index = int(episode_str)
        chunk_name = pq_path.parent.name
        chunk_index = int(chunk_name.split("-")[-1])
        row_count = pq.read_metadata(pq_path).num_rows
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [],
                "length": int(row_count),
                "chunk_index": chunk_index,
            }
        )
    return sorted(episodes, key=lambda ep: ep["episode_index"])


def _load_episodes(episodes_path: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    with episodes_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))
    return episodes


def _write_episodes(episodes_path: Path, episodes: list[dict[str, Any]]) -> None:
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    with episodes_path.open("w") as writer:
        for ep in episodes:
            writer.write(json.dumps(ep))
            writer.write("\n")


def _repair_episodes(
    episodes: list[dict[str, Any]],
    chunks_size: int,
) -> tuple[list[dict[str, Any]], int]:
    repaired = []
    n_fixes = 0
    for ep in episodes:
        out = dict(ep)
        episode_index = int(out["episode_index"])
        if "tasks" not in out:
            out["tasks"] = []
            n_fixes += 1
        if "chunk_index" not in out:
            out["chunk_index"] = episode_index // chunks_size
            n_fixes += 1
        if "length" in out:
            out["length"] = int(out["length"])
        repaired.append(out)
    return repaired, n_fixes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the local LeRobot v2 dataset root",
    )
    parser.add_argument(
        "--write-fixes",
        action="store_true",
        help="Write repaired metadata files in-place",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    meta_root = dataset_root / "meta"
    info_path = meta_root / "info.json"
    tasks_path = meta_root / "tasks.jsonl"
    modality_path = meta_root / "modality.json"
    episodes_path = meta_root / "episodes.jsonl"

    print(f"[check] dataset root: {dataset_root}")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not info_path.exists():
        raise FileNotFoundError(f"Missing required metadata file: {info_path}")
    if not tasks_path.exists():
        raise FileNotFoundError(f"Missing required metadata file: {tasks_path}")
    if not modality_path.exists():
        raise FileNotFoundError(f"Missing required metadata file: {modality_path}")

    info = _read_json(info_path)
    chunks_size = int(info.get("chunks_size", 1000))
    print(f"[ok] info.json found, chunks_size={chunks_size}")
    print("[ok] tasks.jsonl found")
    print("[ok] modality.json found")

    fixes = 0
    if episodes_path.exists():
        episodes = _load_episodes(episodes_path)
        print(f"[ok] episodes.jsonl found ({len(episodes)} entries)")
    else:
        episodes = _scan_data_episodes(dataset_root)
        print(
            "[warn] episodes.jsonl missing; reconstructed episode metadata from parquet files "
            f"({len(episodes)} entries)"
        )
        fixes += 1

    episodes, n_repairs = _repair_episodes(episodes, chunks_size=chunks_size)
    fixes += n_repairs
    if n_repairs > 0:
        print(f"[warn] repaired {n_repairs} fields in episode metadata")

    if args.write_fixes and fixes > 0:
        _write_episodes(episodes_path, episodes)
        print(f"[fix] wrote repaired episodes metadata to {episodes_path}")
    elif fixes > 0:
        print("[hint] rerun with --write-fixes to persist repairs")

    print("[done] dataset metadata validation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
