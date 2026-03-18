#!/usr/bin/env python3
"""
Patch an internal gr00t checkpoint for use with Isaac-GR00T (OSS).

Changes applied to config.json:
  - Rename class name "GrootN1d7" -> "Gr00tN1d7" (model_type, architectures)
  - Rename "vlm_model_path" -> "model_name"
  - Remove gr00t-only fields: vlm_backend, dit_latent_dim

Changes applied to processor_config.json:
  - Rename class name "GrootN1d7Processor" -> "Gr00tN1d7Processor"
  - Remove gr00t-only processor kwargs: vlm_backend, vlm_model_path,
    random_history_crop, do_human_interpolation, interpolation_steps,
    human_embodiment_tags, eagle_path, state_gaussian_noise_std
  - Strip unknown fields from each ModalityConfig in modality_configs:
    normalization_mode, exclude_state, normalize_rotation
  - Trim language modality_keys to a single entry (OSS supports one key)
  - Set use_relative_action=True when None (internal code hardcodes True)

Changes applied to statistics.json (if present):
  - Keep only the embodiment tags listed in processor_config modality_configs,
    dropping unrelated pretrain statistics that would cause a KeyError

Usage:
    python scripts/patch_checkpoint.py <checkpoint_dir>
    python scripts/patch_checkpoint.py <checkpoint_dir> --output <patched_dir>
"""

import argparse
import json
from pathlib import Path
import shutil


# Class name renames (applied as string replacements across all JSON values).
CLASS_RENAMES = {
    "GrootN1d7Processor": "Gr00tN1d7Processor",
    "GrootN1d7": "Gr00tN1d7",
}

# gr00t-only fields in config.json to remove entirely.
CONFIG_FIELDS_TO_REMOVE = {"vlm_backend", "dit_latent_dim"}

# gr00t-only keys in processor_config.json -> processor_kwargs to remove.
PROCESSOR_KWARGS_TO_REMOVE = {
    "vlm_backend",
    "vlm_model_path",
    "random_history_crop",
    "do_human_interpolation",
    "interpolation_steps",
    "human_embodiment_tags",
    "eagle_path",
    "state_gaussian_noise_std",
}

# Extra fields stored in each ModalityConfig by gr00t that are not in the OSS dataclass.
MODALITY_CONFIG_FIELDS_TO_REMOVE = {"normalization_mode", "exclude_state", "normalize_rotation"}


def _apply_class_renames(obj):
    """Recursively replace class-name strings in a JSON-decoded object."""
    if isinstance(obj, str):
        for old, new in CLASS_RENAMES.items():
            obj = obj.replace(old, new)
        return obj
    if isinstance(obj, dict):
        return {k: _apply_class_renames(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_apply_class_renames(v) for v in obj]
    return obj


def _patch_config(data: dict) -> dict:
    data = _apply_class_renames(data)
    # Rename vlm_model_path -> model_name
    if "vlm_model_path" in data and "model_name" not in data:
        data["model_name"] = data.pop("vlm_model_path")
    elif "vlm_model_path" in data:
        data.pop("vlm_model_path")
    # Remove gr00t-only fields
    for key in CONFIG_FIELDS_TO_REMOVE:
        data.pop(key, None)
    # Remove cross_attention_dim from diffusion_model_cfg: it is passed
    # explicitly by the model (as backbone_embedding_dim) and having it
    # here too causes "keyword argument specified twice".
    if "diffusion_model_cfg" in data:
        data["diffusion_model_cfg"].pop("cross_attention_dim", None)
    return data


def _strip_modality_configs(modality_configs: dict) -> dict:
    """Remove unknown fields from each ModalityConfig entry.

    Also trims language modality_keys to a single entry — the OSS code
    supports only one language key during inference.
    """
    result = {}
    for embodiment_tag, mods in modality_configs.items():
        result[embodiment_tag] = {}
        for modality, cfg in mods.items():
            if isinstance(cfg, dict):
                cfg = {k: v for k, v in cfg.items() if k not in MODALITY_CONFIG_FIELDS_TO_REMOVE}
                if modality == "language" and isinstance(cfg.get("modality_keys"), list):
                    cfg["modality_keys"] = cfg["modality_keys"][:1]
            result[embodiment_tag][modality] = cfg
    return result


def _patch_processor_config(data: dict) -> dict:
    data = _apply_class_renames(data)
    pk = data.get("processor_kwargs", {})
    # Remove gr00t-only top-level processor kwargs
    for key in PROCESSOR_KWARGS_TO_REMOVE:
        pk.pop(key, None)
    # Strip unknown fields from modality configs
    if "modality_configs" in pk:
        pk["modality_configs"] = _strip_modality_configs(pk["modality_configs"])
    # The internal groot processor hardcodes use_relative_action=True; ensure
    # the OSS processor does the same (None in the config is treated as False).
    if pk.get("use_relative_action") is None:
        pk["use_relative_action"] = True
    data["processor_kwargs"] = pk
    return data


def _write_json(path: Path, data: dict) -> bool:
    """Write JSON and return True if the file changed."""
    with open(path) as f:
        original = f.read()
    result = json.dumps(data, indent=2) + "\n"
    if result == original:
        return False
    with open(path, "w") as f:
        f.write(result)
    return True


def patch_checkpoint(checkpoint_dir: Path, output_dir: Path | None = None) -> None:
    checkpoint_dir = checkpoint_dir.resolve()
    if not checkpoint_dir.is_dir():
        raise ValueError(f"Not a directory: {checkpoint_dir}")

    if output_dir is not None:
        output_dir = output_dir.resolve()
        if output_dir.exists():
            raise ValueError(f"Output directory already exists: {output_dir}")
        print(f"Copying {checkpoint_dir} -> {output_dir}")
        shutil.copytree(checkpoint_dir, output_dir)
        target = output_dir
    else:
        target = checkpoint_dir

    patchers = {
        "config.json": _patch_config,
        "processor_config.json": _patch_processor_config,
    }
    changed_any = False
    for name, patcher in patchers.items():
        p = target / name
        if not p.exists():
            print(f"  [skip] {name} not found")
            continue
        with open(p) as f:
            data = json.load(f)
        patched = patcher(data)
        changed = _write_json(p, patched)
        print(f"  [{'patched' if changed else 'unchanged'}] {name}")
        changed_any = changed_any or changed

    # Patch statistics.json: drop entries for embodiment tags that are not in
    # the processor's modality_configs.  The internal training job saves stats
    # for every embodiment seen across the full pretrain/posttrain run, but the
    # OSS StateActionProcessor iterates over ALL statistics keys and tries to
    # look them up in modality_configs, raising KeyError for unknown tags.
    stats_path = target / "statistics.json"
    proc_path = target / "processor_config.json"
    if stats_path.exists() and proc_path.exists():
        with open(proc_path) as f:
            proc_data = json.load(f)
        known_tags = set(proc_data.get("processor_kwargs", {}).get("modality_configs", {}).keys())
        with open(stats_path) as f:
            stats = json.load(f)
        filtered = {k: v for k, v in stats.items() if k in known_tags}
        changed = _write_json(stats_path, filtered)
        print(f"  [{'patched' if changed else 'unchanged'}] statistics.json")
        changed_any = changed_any or changed

    if not changed_any:
        print("Nothing to patch — checkpoint may already be compatible.")
    else:
        print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "checkpoint_dir", type=Path, help="Path to the checkpoint directory to patch"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help="Write patched files to a new directory instead of modifying in-place",
    )
    args = parser.parse_args()
    patch_checkpoint(args.checkpoint_dir, args.output)


if __name__ == "__main__":
    main()
