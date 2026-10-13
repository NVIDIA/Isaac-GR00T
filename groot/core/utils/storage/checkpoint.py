# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Node-local filesystem policy and opt-in async ZeRO-2 recovery checks."""

import hashlib
import json
import os
from pathlib import Path

CHECKPOINT_VIEW_METADATA_FILE = ".checkpoint-source.json"
CHECKPOINT_VIEW_CONFIG_FILES = (
    "config.json",
    "processor_config.json",
    "processor/processor_config.json",
    "experiment_cfg/final_processor_config.json",
)


def checkpoint_view_identity(metadata: dict, configs: dict[str, str], links: dict[str, str]) -> str:
    """Bind a localized view to its source, dependency cache, configs and link topology."""
    return hashlib.sha256(
        json.dumps([metadata, configs, links], sort_keys=True).encode()
    ).hexdigest()


def checkpoint_view_links(source: Path, configs: dict[str, str]) -> dict[str, str]:
    """Inventory links without reading weights or walking linked asset directories."""
    excluded = {
        "msc_resume_manifest.json",
        "SUCCESS",
        ".resume_download_complete",
        ".download_complete",
        CHECKPOINT_VIEW_METADATA_FILE,
    }
    links = {}
    for path in source.iterdir():
        if path.name in excluded or path.name in configs:
            continue
        if nested := {
            name.removeprefix(path.name + "/"): content
            for name, content in configs.items()
            if name.startswith(path.name + "/")
        }:
            links.update(
                {
                    f"{path.name}/{name}": link
                    for name, link in checkpoint_view_links(path, nested).items()
                }
            )
        else:
            links[path.name] = str(path.resolve())
    return links


def _is_checkpoint_view(path: Path) -> bool:
    return (
        path.parent.name == ".checkpoint-configs"
        and len(path.name) == 64
        and all(character in "0123456789abcdef" for character in path.name)
    )


def checkpoint_view_metadata(checkpoint: str | Path) -> dict | None:
    """Read a verified loader-view origin; ordinary checkpoints never redirect.

    The marker is only recognized in the resolver's content-addressed namespace,
    and its bytes must agree with the view's localized configs and symlink inventory.
    Older views have no reliable origin: callers must prepare their original checkpoint
    again, which creates a fresh view under the versioned identity scheme.
    """
    view = Path(checkpoint).resolve()
    if not _is_checkpoint_view(view):
        return None
    marker = view / CHECKPOINT_VIEW_METADATA_FILE
    if not marker.is_file():
        raise ValueError(
            "Checkpoint runtime view has no source metadata; resolve its original checkpoint "
            "again with ensure_cosmos_model_deps"
        )
    if marker.is_symlink() or marker.stat().st_size > 16 * 1024:
        raise ValueError("Invalid checkpoint runtime-view source metadata")
    metadata = json.loads(marker.read_text())
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"version", "source", "dependency_cache"}
        or type(metadata["version"]) is not int
        or metadata["version"] != 1
        or any(
            not isinstance(metadata[key], str) or not Path(metadata[key]).is_absolute()
            for key in ("source", "dependency_cache")
        )
    ):
        raise ValueError("Invalid checkpoint runtime-view source metadata")
    configs = {}
    for name in CHECKPOINT_VIEW_CONFIG_FILES:
        relative = Path(name)
        # A config under an unexpanded symlinked directory is source-owned too.
        if any(
            (view / Path(*relative.parts[:end])).is_symlink()
            for end in range(1, len(relative.parts) + 1)
        ):
            continue
        config = view / relative
        if config.is_file():
            configs[name] = config.read_text()
    if (
        checkpoint_view_identity(metadata, configs, checkpoint_view_links(view, configs))
        != view.name
    ):
        raise ValueError(
            "Checkpoint runtime-view source metadata or inventory does not match its identity"
        )
    source = Path(metadata["source"])
    if not source.is_dir() or source.resolve() != source or _is_checkpoint_view(source):
        raise ValueError(
            "Checkpoint runtime-view source is missing, redirected, or another runtime view"
        )
    return metadata


def checkpoint_source_path(checkpoint: str | Path) -> Path:
    """Return the source whose exact metadata and weights define checkpoint identity."""
    metadata = checkpoint_view_metadata(checkpoint)
    return Path(metadata["source"]) if metadata is not None else Path(checkpoint)


def node_local_checkpoint_output(output_dir: str | Path) -> bool:
    """Classify this path, not other paths used by a node-local training run."""
    node_local_root = Path("/tmp").resolve()  # noqa: S108 - classify, not create
    return Path(output_dir).resolve().is_relative_to(node_local_root)


def optimizer_reset_resume_enabled() -> bool:
    return os.environ.get("GROOT_ALLOW_OPTIMIZER_RESET_RESUME", "").lower() in {"1", "true", "yes"}


def checkpoint_has_optimizer_state(checkpoint: str | Path) -> bool:
    """Detect even incomplete optimizer artifacts; they must never imply a reset."""
    root = Path(checkpoint)
    return (
        any((root / name).exists() for name in ("optimizer.pt", "optimizer.bin", "dcp_state"))
        or any(root.glob("optimizer_rank*"))
        or any(root.glob("global_step*"))
        or any(root.glob("pytorch_optimizer*"))
    )


def _require_nonempty_files(paths: list[Path], description: str) -> None:
    if not paths or any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise ValueError(f"Incomplete {description}: missing or empty files")


def _validate_dcp_optimizer(root: Path) -> None:
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import MetadataIndex, TensorStorageMetadata

    _require_nonempty_files([root / ".metadata"], f"DCP optimizer metadata in {root}")
    try:
        metadata = FileSystemReader(str(root)).read_metadata()
    except Exception as exc:
        raise ValueError(f"Invalid DCP optimizer metadata in {root}") from exc
    storage = metadata.storage_data
    if not isinstance(storage, dict) or not storage:
        raise ValueError(f"DCP metadata has no optimizer data: {root}")
    if not any(
        key == "optimizer" or key.startswith("optimizer.") for key in metadata.state_dict_metadata
    ):
        raise ValueError(f"DCP metadata has no optimizer state: {root}")
    if set(metadata.state_dict_metadata) != {index.fqn for index in storage}:
        raise ValueError(f"DCP metadata has incomplete storage entries: {root}")
    for name, entry in metadata.state_dict_metadata.items():
        indices = (
            [MetadataIndex(name, chunk.offsets) for chunk in entry.chunks]
            if isinstance(entry, TensorStorageMetadata)
            else [MetadataIndex(name)]
        )
        if any(index not in storage for index in indices):
            raise ValueError(f"DCP metadata has incomplete storage entries: {root}")
    # DCP records byte ranges, so checking only that one .distcp file exists can
    # accept a truncated download or a checkpoint missing other hosts' shards.
    for entry in storage.values():
        name = entry.relative_path
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"Invalid DCP optimizer shard path in {root}")
        path = root / name
        if (
            type(entry.offset) is not int
            or type(entry.length) is not int
            or entry.offset < 0
            or entry.length <= 0
            or not path.is_file()
            or path.stat().st_size < entry.offset + entry.length
        ):
            raise ValueError(f"Incomplete DCP optimizer data: missing or truncated {path}")


def checkpoint_optimizer_mode(checkpoint: str | Path, *, world_size: int = 1) -> str:
    """Select the preferred recovery format and validate only its optimizer state."""
    root = Path(checkpoint)
    # Match loader precedence: incomplete selected state is an error, but an
    # unused async compatibility copy must not invalidate complete primary state.
    per_rank = list(root.glob("optimizer_rank*"))
    if per_rank:
        expected = {f"optimizer_rank{rank}.pt" for rank in range(world_size)}
        if {path.name for path in per_rank} != expected:
            raise ValueError(
                f"Incomplete per-rank optimizer state for world_size={world_size}: {root}"
            )
        _require_nonempty_files(per_rank, f"per-rank optimizer state in {root}")
        return "per_rank"
    dcp = root / "dcp_state"
    if dcp.exists():
        _validate_dcp_optimizer(dcp)
        return "dcp"
    if checkpoint_has_optimizer_state(root):
        raise ValueError("Explicit recovery supports only async ZeRO-2 per-rank or DCP state")
    return "reset"


def validate_explicit_resume_checkpoint(
    checkpoint: str | Path, *, require_success: bool = False, world_size: int = 1
) -> tuple[int, str]:
    """Check the selected async ZeRO-2 checkpoint, not ordinary legacy resumes."""
    root = Path(checkpoint)
    if require_success and not (root / "SUCCESS").is_file():
        raise ValueError(f"Resume checkpoint has no durable SUCCESS marker: {root}")
    state = json.loads((root / "trainer_state.json").read_text())
    if (
        not isinstance(state, dict)
        or type(state.get("global_step")) is not int
        or state["global_step"] < 0
    ):
        raise ValueError(f"Invalid trainer_state.json global_step: {root}")
    # Match the existing loader's preference for indexed safetensors.
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid model weight index: {index}")
        if any(
            not isinstance(name, str) or Path(name).name != name for name in weight_map.values()
        ):
            raise ValueError(f"Invalid model shard path: {index}")
        files = [root / name for name in set(weight_map.values())]
    else:
        files = [root / "model.safetensors"]
    _require_nonempty_files(files, f"safetensors model weights in {root}")
    mode = checkpoint_optimizer_mode(root, world_size=world_size)
    if mode == "reset":
        if not optimizer_reset_resume_enabled():
            raise ValueError(
                "Optimizer-free recovery requires GROOT_ALLOW_OPTIMIZER_RESET_RESUME=1"
            )
        _require_nonempty_files([root / "scheduler.pt"], f"reset scheduler state in {root}")
    return state["global_step"], mode


def coordinated_explicit_resume_checkpoint(checkpoint: str | Path) -> tuple[int, str]:
    """Agree before entering the existing optimizer loader's collectives."""
    import torch.distributed as dist

    distributed = dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    result, failure = None, None
    try:
        result = validate_explicit_resume_checkpoint(checkpoint, world_size=world_size)
    except Exception as exc:  # noqa: BLE001 - every rank must take the same failure path
        failure = f"{type(exc).__name__}: {exc}"
    values = [(result, failure)]
    if distributed:
        values = [None] * world_size
        dist.all_gather_object(values, (result, failure))
    failures = [f"Rank {rank}: {item[1]}" for rank, item in enumerate(values) if item[1]]
    if failures:
        raise ValueError("Invalid explicit checkpoint recovery: " + "; ".join(failures))
    if any(item[0] != values[0][0] for item in values):
        raise ValueError("Checkpoint step or optimizer recovery mode differs between ranks")
    return values[0][0]
