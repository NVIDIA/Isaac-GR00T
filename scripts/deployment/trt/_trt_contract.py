# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single source of truth for the TRT engine's ``action_horizon`` and
``batch_size``, read back from ``export_metadata.json``.

When a model is exported to ONNX/TensorRT, two numbers are baked into the
engine and recorded in ``export_metadata.json`` next to it:

* ``action_horizon`` — the predicted action-chunk length. It also determines
  the engine's static ``sa_seq_len`` (``1 + action_horizon``).
* ``batch_size`` — baked as a *static* shape (the exporter registers only the
  sequence dim in ``dynamic_axes``), so the engine only accepts that exact
  batch at runtime.

The same two numbers are then re-stated independently elsewhere: in the
loaded model's config, in the ``--batch-size`` flag of the verify / benchmark
scripts, and in the ``--action-horizon`` open-loop stride of the standalone
inference script. When any copy drifts from the engine, the failure is silent
or cryptic — a foreign or stale ``.engine`` dropped into the bundle, or a
typo'd ``--batch-size``, surfaces only as a generic ``Invalid input shape``
raised deep inside the engine's ``forward()``, naming neither the engine's
baked value nor the user's flag.

The helpers here read the baked values back from ``export_metadata.json`` and
validate each re-stated copy against them up-front, with error messages that
name both sides. If the metadata file is missing (older bundles) the checks
degrade to a warning rather than failing.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any


logger = logging.getLogger(__name__)

_METADATA_FILENAME = "export_metadata.json"

# Bumped when export_metadata.json changes incompatibly (a key the build/runtime
# readers depend on is renamed, removed, or repurposed). The exporter stamps
# this; the build reader rejects a bundle whose version it does not recognize.
EXPORT_METADATA_SCHEMA_VERSION = 1

# Keys the build reader (build_tensorrt_engine.build_full_pipeline) needs to size
# the TRT shape profiles, plus the values the runtime contract re-checks. A bundle
# missing any of these cannot be built without guessing a sequence/patch shape.
# ``schema_version`` is intentionally not here: the version gate in
# validate_export_metadata handles its absence before the missing-keys check runs.
REQUIRED_EXPORT_METADATA_KEYS = (
    "sa_seq_len",
    "vl_seq_len",
    "llm_seq_len",
    "num_patches",
    "num_merged_patches",
    "num_vis_tokens",
    "action_horizon",
    "batch_size",
    "precision",
)


def validate_export_metadata(
    metadata: dict[str, Any],
    *,
    source: str = "export metadata",
    engine_path: str = "",
) -> None:
    """Raise unless ``metadata`` is a current-schema, build-ready bundle.

    Checks ``schema_version`` equals :data:`EXPORT_METADATA_SCHEMA_VERSION` and
    every :data:`REQUIRED_EXPORT_METADATA_KEYS` entry is present, so a stale
    bundle or a renamed/dropped field fails here — naming the cause — instead of
    silently defaulting to a wrong sequence/patch hint deep in the TRT build. The
    message states the problem; the caller decides the remedy.
    """
    where = f" at {engine_path}" if engine_path else ""
    version = metadata.get("schema_version")
    if version != EXPORT_METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"{source}: {_METADATA_FILENAME}{where} has schema_version={version!r}, but "
            f"this build expects {EXPORT_METADATA_SCHEMA_VERSION}"
        )
    missing = [k for k in REQUIRED_EXPORT_METADATA_KEYS if k not in metadata]
    if missing:
        raise ValueError(
            f"{source}: {_METADATA_FILENAME}{where} is missing required key(s) {missing}"
        )


def _candidate_metadata_paths(engine_path: str) -> list[str]:
    """Locations to look for ``export_metadata.json`` given an engine path.

    ``engine_path`` may be an engine directory or a single ``.engine`` file
    (dit_only mode). The metadata is written by ``export_onnx_n1d7`` into the
    ONNX output dir and copied next to the engines by ``build_trt_pipeline``,
    so we also check a sibling ``onnx/`` dir for un-copied legacy layouts.
    """
    base = engine_path
    if os.path.isfile(engine_path) or engine_path.endswith(".engine"):
        base = os.path.dirname(engine_path)
    candidates = [
        os.path.join(base, _METADATA_FILENAME),
        os.path.join(os.path.dirname(base.rstrip("/")), "onnx", _METADATA_FILENAME),
    ]
    return candidates


def load_export_metadata(engine_path: str) -> dict[str, Any] | None:
    """Return the export metadata for an engine bundle, or ``None`` if absent.

    A missing *or unreadable* (corrupt JSON / IO error) metadata file returns
    ``None`` so callers can degrade to a warning uniformly rather than crashing
    on a malformed file.
    """
    for path in _candidate_metadata_paths(engine_path):
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "Failed to read export metadata %s (%s); treating as absent.",
                    path,
                    e,
                )
                return None
    return None


def _policy_action_horizon(policy: Any) -> int | None:
    """Best-effort read of the loaded policy's action horizon."""
    action_head = getattr(getattr(policy, "model", None), "action_head", None)
    if action_head is None:
        return None
    cfg = getattr(action_head, "config", None)
    if cfg is not None and getattr(cfg, "action_horizon", None) is not None:
        return int(cfg.action_horizon)
    if getattr(action_head, "action_horizon", None) is not None:
        return int(action_head.action_horizon)
    return None


def assert_engine_matches_policy(
    policy: Any,
    engine_path: str,
    *,
    source: str = "setup_tensorrt_engines",
) -> dict[str, Any] | None:
    """Validate that an engine bundle was built for the loaded policy.

    Compares the engine's recorded ``action_horizon`` (and the derived
    ``sa_seq_len == 1 + action_horizon``) against the loaded policy's action
    head. A mismatch — e.g. a foreign or stale ``.engine`` dropped into the
    bundle — raises here, naming both values, instead of surfacing as a
    generic ``Invalid input shape`` deep inside ``Engine.forward()``.

    When ``export_metadata.json`` is absent the contract cannot be checked; we
    log a warning and return ``None`` rather than failing (older bundles).
    """
    metadata = load_export_metadata(engine_path)
    if metadata is None:
        logger.warning(
            "%s: no %s found next to %s; cannot validate that the engine's "
            "action_horizon / batch_size match the loaded policy. A "
            "mismatched engine will fail later as a cryptic 'Invalid input "
            "shape' inside Engine.forward().",
            source,
            _METADATA_FILENAME,
            engine_path,
        )
        return None

    engine_ah = metadata.get("action_horizon")
    engine_sa = metadata.get("sa_seq_len")
    if engine_ah is not None and engine_sa is not None and engine_sa != engine_ah + 1:
        raise ValueError(
            f"{source}: corrupt {_METADATA_FILENAME} for {engine_path}: "
            f"sa_seq_len={engine_sa} but action_horizon={engine_ah} "
            f"(expected sa_seq_len == 1 + action_horizon == {engine_ah + 1})."
        )

    policy_ah = _policy_action_horizon(policy)
    if engine_ah is not None and policy_ah is not None and engine_ah != policy_ah:
        sa_note = f" (baked into sa_seq_len={engine_sa})" if engine_sa is not None else ""
        raise ValueError(
            f"{source}: TRT engine bundle at {engine_path} was built for "
            f"action_horizon={engine_ah}{sa_note}, but the loaded policy has "
            f"action_horizon={policy_ah}. The engine and policy disagree on "
            "chunk size; re-export/rebuild the engines for this model, or load "
            "the model the engines were built from."
        )
    return metadata


def assert_engine_bundle_present(
    engine_path: str,
    required_files,
    *,
    mode: str = "n17_full_pipeline",
    source: str = "setup_tensorrt_engines",
) -> None:
    """Fail fast, with build instructions, when a TRT engine bundle is missing.

    ``setup_tensorrt_engines`` swaps in several ``.engine`` files that have no
    PyTorch fallback (the action head's state/action encoders, the DiT, and the
    action decoder). If the ``--trt-engine-path`` directory does not exist, or
    exists but is missing one of those files, the loader would otherwise die with
    a bare ``FileNotFoundError`` deep inside ``Engine.load`` — giving the user no
    hint that they simply have not built the engines yet. Raise an actionable
    error here instead, naming the missing directory / files and the build step.
    """
    build_hint = (
        "Build the engines first, e.g.:\n"
        "  python scripts/deployment/build_trt_pipeline.py \\\n"
        "      --model-path <model> --dataset-path <dataset> \\\n"
        "      --embodiment-tag <TAG> --output-dir ./gr00t_trt_deployment\n"
        "then pass --trt-engine-path ./gr00t_trt_deployment/engines "
        "(see scripts/deployment/ for the full deployment guide)."
    )
    if not os.path.isdir(engine_path):
        raise FileNotFoundError(
            f"{source}: inference-mode '{mode}' needs a TensorRT engine "
            f"directory, but none exists at {engine_path!r}.\n{build_hint}"
        )
    missing = [f for f in required_files if not os.path.exists(os.path.join(engine_path, f))]
    if missing:
        raise FileNotFoundError(
            f"{source}: inference-mode '{mode}' requires these TensorRT engine "
            f"file(s) in {engine_path!r}, which are missing: "
            f"{', '.join(sorted(missing))}.\n{build_hint}"
        )


def resolve_batch_size(
    engine_path: str,
    requested: int | None = None,
    *,
    source: str = "TRT runtime",
) -> int:
    """Resolve the runtime batch size against the engine's build-time batch.

    ``export_onnx_n1d7`` bakes the batch dim as a static shape (only
    ``seq_len`` is in ``dynamic_axes``), so the engine only accepts the exact
    batch it was built at. This reads that value from ``export_metadata.json``
    and validates the requested batch against it:

    - ``requested is None`` -> return the engine's build batch.
    - ``requested != build batch`` -> raise, naming both (a typo'd
      ``--batch-size`` otherwise fails as a cryptic ``Invalid input shape``).
    """
    metadata = load_export_metadata(engine_path)
    built = metadata.get("batch_size") if metadata else None

    if requested is None:
        if built is None:
            return 1
        return int(built)

    if built is not None and int(requested) != int(built):
        raise ValueError(
            f"{source}: requested batch_size={requested} but the TRT engine at "
            f"{engine_path} was built (statically) for batch_size={built}. "
            "The export pipeline does not register the batch dim in "
            "dynamic_axes, so the engine only accepts its build batch. Pass "
            f"--batch-size {built}, or rebuild the engines at batch_size="
            f"{requested}."
        )
    return int(requested)


def assert_grid_thw_matches(
    baked_grid: Any,
    runtime_grid_thw: Any,
    *,
    source: str = "ViT TRT forward",
) -> None:
    """Validate a runtime ``image_grid_thw`` against the ViT engine's baked grid.

    The ViT export pre-computes position/rotary embeddings for the captured
    ``grid_thw`` and freezes them as buffers; the engine's only input is
    ``pixel_values``. Those buffers depend on each view's ``[t, h, w]``
    *layout*, not on how many views are present: batching tiles the same
    per-view grid, so the runtime view count scales with batch size while the
    layouts stay fixed (and a real total-shape mismatch is already rejected by
    the static ``pixel_values`` shape). So we require every runtime view's
    layout to be one the engine baked; a view with an unbaked layout (e.g. H/W
    swapped, different resolution or temporal span) would get the wrong
    embeddings and is rejected, while a different view *count* (batch) is fine.

    ``baked_grid`` is ``None`` for engine bundles built before ``vit_grid_thw``
    was recorded; skip the check then (degrade like a missing metadata file).
    """
    if baked_grid is None:
        return
    rt = runtime_grid_thw
    if hasattr(rt, "detach"):
        rt = rt.detach().cpu()
    if hasattr(rt, "tolist"):
        rt = rt.tolist()
    rt_rows = [tuple(int(x) for x in row) for row in rt]
    baked_layouts = {tuple(int(x) for x in row) for row in baked_grid}
    unbaked = sorted({row for row in rt_rows if row not in baked_layouts})
    if unbaked:
        raise ValueError(
            f"{source}: ViT TRT engine baked position/rotary buffers for "
            f"image_grid_thw layouts {sorted(baked_layouts)}, but this observation "
            f"has view layout(s) {[list(r) for r in unbaked]} that were never baked. "
            f"The engine ignores runtime grid_thw (pixel_values is its only input) "
            f"and would silently produce wrong vision features. Re-export/rebuild "
            f"the ViT engine for this image configuration, or run with a baked layout."
        )


def assert_exec_horizon_within_model(
    *,
    exec_horizon: int,
    model_action_horizon: int,
    source: str = "inference",
) -> None:
    """Validate an open-loop execution stride against the model's chunk size.

    ``standalone_inference_script --execution-horizon`` is the number of actions
    consumed per predicted chunk; it must not exceed the model's
    ``action_horizon`` (the predicted chunk length), otherwise indexing the
    chunk by ``range(exec_horizon)`` runs past the end.
    """
    if not (1 <= exec_horizon <= model_action_horizon):
        raise ValueError(
            f"{source}: --execution-horizon={exec_horizon} must satisfy "
            f"1 <= execution_horizon <= model action_horizon={model_action_horizon} "
            "(= the predicted chunk length). A larger stride indexes past the "
            "predicted action chunk."
        )


__all__ = [
    "EXPORT_METADATA_SCHEMA_VERSION",
    "REQUIRED_EXPORT_METADATA_KEYS",
    "validate_export_metadata",
    "load_export_metadata",
    "assert_engine_matches_policy",
    "assert_engine_bundle_present",
    "resolve_batch_size",
    "assert_grid_thw_matches",
    "assert_exec_horizon_within_model",
]
