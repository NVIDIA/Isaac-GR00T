# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Checkpoint-carried N2 numerics: ``action_clip_range`` and ``rope_fp32``.

Neither setting changes a weight shape, so a checkpoint loads silently under the wrong one:

- ``action_cfg.action_clip_range``: bound of the normalized-action label clip and of the decode
  clamp. Default 1.0.
- ``GrootN2Config.rope_fp32``: the action/state RoPE tables are fp32 constants that checkpoints
  do not store, and every bf16 cast is parameters-only (``groot/core/utils/precision.py``).
  Default False: stored tables and whole-model casts, as before.

``resolve_numerics`` gives training (warm start, resume) and serving the same answer. The
checkpoint's value comes from:

1. config.json: ``action_cfg.action_clip_range`` (processor_config.json's must agree) and
   ``rope_fp32`` (under ``base_model_cfg`` for GrootN2RL). Checkpoints trained with this module
   record both; older ones can have them added to config.json.
2. Otherwise, what the checkpoint itself proves. No ``action_clip_range`` means 1.0: every code
   path that trained another bound recorded it. RoPE tables stored as bf16 mean
   ``rope_fp32=False``. An action-head checkpoint that stores no RoPE tables cannot be told
   apart, because code that does not store the tables stores none under either cast: resolving
   it raises until config.json or an explicit value says which.

Sources that disagree raise. An explicit value that disagrees with the checkpoint's raises
unless the override is allowed explicitly. Standard library only, so it runs without torch.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path
import re
import struct
from typing import Any

DEFAULT_ACTION_CLIP_RANGE = 1.0
# The legacy persistent ActionRoPE1D buffers, also under GrootN2RL's ``wrapped_model.``.
_ROPE_TABLE_KEY = re.compile(r"(^|\.)(action|state)_rope\.(freqs|seq)$")
# Weights every GrootN2 action head stores, whatever its RoPE mode.
_ACTION_HEAD_KEY = re.compile(r"(^|\.)action_decoder\.")

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Setting:
    """One setting as a checkpoint records or proves it."""

    value: Any  # None: the checkpoint says nothing
    source: str
    # An explicit value that disagrees needs the override flag.
    binding: bool = False
    # The checkpoint cannot tell, and the default could be wrong: an explicit value is required.
    required: bool = False


@dataclasses.dataclass(frozen=True)
class N2Numerics:
    action_clip_range: float | None  # None: no action head
    rope_fp32: bool
    action_clip_range_source: str
    rope_fp32_source: str

    def describe(self) -> str:
        return (
            f"action_clip_range={self.action_clip_range} ({self.action_clip_range_source}), "
            f"rope_fp32={self.rope_fp32} ({self.rope_fp32_source})"
        )


def validate_action_clip_range(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{where}: action_clip_range must be a number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{where}: action_clip_range must be finite and > 0, got {value!r}")
    return float(value)


def validate_rope_fp32(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{where}: rope_fp32 must be true or false, got {value!r}")
    return value


def actor_config(config: dict) -> dict:
    """The GrootN2 part of a config.json: GrootN2RL nests it under ``base_model_cfg``."""
    base = config.get("base_model_cfg")
    return base if isinstance(base, dict) else config


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(size))


def stored_weight_evidence(checkpoint_dir: str | Path) -> tuple[dict[str, str], bool]:
    """``({RoPE table key: safetensors dtype}, stores action-head weights)``.

    Reads safetensors headers only, and with an index only the shards that hold RoPE tables.
    A checkpoint without safetensors weights gives no evidence.
    """
    checkpoint_dir = Path(checkpoint_dir)
    index = checkpoint_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        has_action_head = any(_ACTION_HEAD_KEY.search(key) for key in weight_map)
        shards = sorted({shard for key, shard in weight_map.items() if _ROPE_TABLE_KEY.search(key)})
    else:
        has_action_head = False
        shards = sorted(path.name for path in checkpoint_dir.glob("model*.safetensors"))
    dtypes = {}
    for shard in shards:
        for key, meta in _safetensors_header(checkpoint_dir / shard).items():
            has_action_head = has_action_head or bool(_ACTION_HEAD_KEY.search(key))
            if key != "__metadata__" and _ROPE_TABLE_KEY.search(key):
                dtypes[key] = meta["dtype"]
    return dtypes, has_action_head


def checkpoint_numerics(
    checkpoint_dir: str | Path, config: dict | None = None
) -> tuple[Setting, Setting]:
    """``(action_clip_range, rope_fp32)`` as the checkpoint records or proves them.

    ``config`` stands in for the checkpoint's config.json (to check an edit before writing it);
    by default that file is read. A weights-only checkpoint (no config.json) records
    nothing, so only its stored weights can prove anything.
    """
    checkpoint_dir = Path(checkpoint_dir)
    where = str(checkpoint_dir)
    if config is None:
        config = _read_json(checkpoint_dir / "config.json") or {}
    base = actor_config(config)
    action_cfg = base.get("action_cfg")

    recorded_mode = base.get("rope_fp32")
    if recorded_mode is not None:
        recorded_mode = validate_rope_fp32(recorded_mode, f"{where}/config.json")
    if config and action_cfg is None:
        if recorded_mode is None:
            rope = Setting(False, "default: video-only checkpoint records no rope_fp32")
        else:
            rope = Setting(recorded_mode, "checkpoint config.json", binding=True)
        return Setting(None, "checkpoint has no action head"), rope

    clips = {}
    processor = _read_json(checkpoint_dir / "processor_config.json") or {}
    processor_action_cfg = processor.get("processor_kwargs", {}).get("action_cfg") or {}
    if processor_action_cfg.get("action_clip_range") is not None:
        clips["processor_config.json"] = validate_action_clip_range(
            processor_action_cfg["action_clip_range"], f"{where}/processor_config.json"
        )
    if action_cfg and action_cfg.get("action_clip_range") is not None:
        clips["config.json"] = validate_action_clip_range(
            action_cfg["action_clip_range"], f"{where}/config.json"
        )
    if len(set(clips.values())) > 1:
        raise ValueError(f"{where}: action_clip_range disagrees across {clips}")
    if clips:
        source = "config.json" if "config.json" in clips else "processor_config.json"
        clip = Setting(clips[source], f"checkpoint {source}", binding=True)
    elif config:
        clip = Setting(DEFAULT_ACTION_CLIP_RANGE, "checkpoint records none, so 1.0", binding=True)
    else:
        clip = Setting(None, "checkpoint has no config.json")

    stored, has_action_head = stored_weight_evidence(checkpoint_dir)
    if recorded_mode is not None:
        if recorded_mode and "BF16" in stored.values():
            raise ValueError(
                f"{where}: config.json says rope_fp32=true, but the checkpoint stores bf16 RoPE "
                f"tables {stored}, which only the legacy whole-model cast produces"
            )
        rope = Setting(recorded_mode, "checkpoint config.json", binding=True)
    elif not stored and not has_action_head:
        # Nothing to go by (e.g. a partial or synthetic checkpoint): the legacy whole-model cast.
        rope = Setting(False, "default: no RoPE mode recorded and no action-head weights stored")
    elif not stored:
        rope = Setting(
            None,
            "the checkpoint stores no action/state RoPE tables and config.json records no "
            "rope_fp32. Code that does not store the tables trained it, with fp32 tables only if "
            "its bf16 cast was parameters-only (DeepSpeed >= 0.19.7 or the params-only patch)",
            required=True,
        )
    elif set(stored.values()) == {"BF16"}:
        rope = Setting(False, "checkpoint stores bf16 RoPE tables", binding=True)
    else:
        logger.warning(
            "[N2 numerics] %s stores RoPE tables as %s and records no rope_fp32: training "
            "likely kept them fp32 (no DeepSpeed bf16 module cast). Loading keeps the legacy "
            "whole-model cast; record rope_fp32 in config.json if that is wrong.",
            where,
            sorted(set(stored.values())),
        )
        rope = Setting(False, f"default: RoPE tables stored as {sorted(set(stored.values()))}")
    return clip, rope


def _choose(
    name: str,
    checkpoint: Setting,
    explicit: Any,
    default: Any,
    *,
    checkpoint_dir: str | Path | None,
    allow_override: bool,
    hint: str,
) -> tuple[Any, str]:
    if explicit is not None:
        if checkpoint.binding and checkpoint.value != explicit:
            message = (
                f"{name}={explicit!r} disagrees with {checkpoint.value!r} from "
                f"{checkpoint.source} ({checkpoint_dir})"
            )
            if not allow_override:
                raise ValueError(f"{message}. Drop the explicit value, or force it: {hint}")
            logger.warning("[N2 numerics] %s; overriding, as explicitly allowed", message)
            return explicit, f"explicit, overriding {checkpoint.value!r} from {checkpoint.source}"
        return explicit, "explicit" + (", = checkpoint" if checkpoint.value == explicit else "")
    if checkpoint.required:
        raise ValueError(
            f"{checkpoint_dir}: cannot tell {name}: {checkpoint.source}. Record {name} in the "
            f"checkpoint's config.json, or set it explicitly: {hint}"
        )
    if checkpoint.value is not None:
        return checkpoint.value, checkpoint.source
    return default, "default"


def resolve_numerics(
    checkpoint_dir: str | Path | None,
    *,
    has_action_head: bool,
    action_clip_range: float | None = None,
    rope_fp32: bool | None = None,
    allow_override: bool = False,
    hint: str = "",
) -> N2Numerics:
    """Effective numerics for a model loaded from ``checkpoint_dir`` (None: fresh model).

    ``action_clip_range`` / ``rope_fp32`` are explicit values (None: not set). ``hint`` says how
    the caller sets and forces them, for error messages.
    """
    if checkpoint_dir is None:
        clip_checkpoint = rope_checkpoint = Setting(None, "no checkpoint")
    else:
        clip_checkpoint, rope_checkpoint = checkpoint_numerics(checkpoint_dir)
    kwargs = {"checkpoint_dir": checkpoint_dir, "allow_override": allow_override, "hint": hint}
    if has_action_head:
        if action_clip_range is not None:
            action_clip_range = validate_action_clip_range(action_clip_range, "explicit")
        clip, clip_source = _choose(
            "action_clip_range",
            clip_checkpoint,
            action_clip_range,
            DEFAULT_ACTION_CLIP_RANGE,
            **kwargs,
        )
    elif action_clip_range is not None:
        raise ValueError("action_clip_range is set for a model without an action head")
    else:
        clip, clip_source = None, "no action head"
    if rope_fp32 is not None:
        rope_fp32 = validate_rope_fp32(rope_fp32, "explicit")
    rope, rope_source = _choose("rope_fp32", rope_checkpoint, rope_fp32, False, **kwargs)
    return N2Numerics(clip, rope, clip_source, rope_source)
