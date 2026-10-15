# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Per-sample tactile F6 and deform-map inputs for GrootN2 (pure numpy).

``GrootN2Processor`` calls ``process_tactile_modalities`` once per sample. The outputs
are RAW encoder inputs that ride the collator's generic numpy stack; the frozen
Tactile-VAE and HB1 encoders run inside the model.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from groot.core.training.models.n2.tactile.guards import required_input_keys

# Every key this module can emit. The collator raises when a key in
# required_batch_keys(model_cfg) is missing from part of a batch and drops the others,
# so it never stacks a short batch.
# A tuple, not a set, so every rank and dataloader worker iterates it in the same order.
TACTILE_BATCH_KEYS = ("tactile_raw", "tactile_raw_future", "tactile_future_n_padded", "deform_raw")


def required_batch_keys(model_cfg: Any) -> frozenset[str]:
    """Training batch keys the model built from ``model_cfg`` consumes (empty if non-tactile)."""
    return frozenset(required_input_keys(model_cfg, training=True))


def process_tactile(
    modality: Any,
    tactile: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    training: bool = True,
) -> dict[str, np.ndarray]:
    """Stack per-finger F6 windows into RAW Tactile-VAE inputs.

    Args:
        modality: The tag's "tactile" ModalityConfig. ``modality_keys`` are the 10
            fingers, hand-major (left_* then right_*). Deltas must be 20 history
            frames (<= 0), optionally followed by 20 future frames (> 0).
        tactile: finger key -> [T, 6] window in delta_indices order.
        metadata: Step metadata; only ``_padded_mask_tactile`` is read.
        training: Require the full configured window. Evaluation also accepts its
            history-only projection, as requested by the policy at serving time.

    Returns:
        ``tactile_raw`` float32 [2, 20, 5, 6]. A 40-frame window (present + future) also
        returns ``tactile_raw_future`` float32 [2, 20, 5, 6] and
        ``tactile_future_n_padded`` (np.int64), the number of future frames the loader
        clamped past the episode end.
    """
    # Lazy: configs.model imports every model config.
    from groot.core.training.configs.model import n2_tactile as pinned

    n = pinned.TACTILE_INPUT_WINDOW
    deltas = np.asarray(modality.delta_indices)
    if len(deltas) not in (n, 2 * n):
        raise ValueError(
            f"Unexpected tactile window length T={len(deltas)}; expected {n} (present) or "
            f"{2 * n} (present + future)"
        )
    if not np.all(deltas[:n] <= 0) or not np.all(deltas[n:] > 0):
        raise ValueError(
            f"tactile delta_indices must contain {n} history frames (<= 0), optionally "
            f"followed by {n} future frames (> 0); got {list(modality.delta_indices)}"
        )
    win = np.stack(
        [np.asarray(tactile[k], dtype=np.float32) for k in modality.modality_keys], axis=1
    )
    n_frames = win.shape[0]
    if n_frames != len(deltas) and not (not training and n_frames == n):
        raise ValueError(
            f"tactile input window length T={n_frames} does not match "
            f"configured delta_indices length {len(deltas)}"
        )
    # [T, 10, 6] -> [n_hands, T, n_fingers, 6]; the present/future split is on axis 1.
    win = win.reshape(
        n_frames, pinned.TACTILE_N_HANDS, pinned.TACTILE_N_FINGERS, pinned.TACTILE_F6_DIM
    )
    win = np.transpose(win, (1, 0, 2, 3))

    if n_frames == n:
        return {"tactile_raw": np.ascontiguousarray(win, dtype=np.float32)}
    # With allow_padding the loader clamps past-end indices to the last frame and flags
    # them per delta frame; present frames are never flagged. No mask: nothing clamped.
    pad_mask = metadata.get("_padded_mask_tactile", None)
    n_padded = 0 if pad_mask is None else int(np.asarray(pad_mask)[n:].sum())
    return {
        "tactile_raw": np.ascontiguousarray(win[:, :n], dtype=np.float32),
        "tactile_raw_future": np.ascontiguousarray(win[:, n:], dtype=np.float32),
        # The model turns this into per-sample loss validity via future_pad_tolerance.
        "tactile_future_n_padded": np.int64(n_padded),
    }


def process_deform(modality: Any, deform: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Take the current frame of each per-finger deform video as RAW HB1 input.

    Args:
        modality: The tag's "deform" ModalityConfig (10 keys, hand-major, delta [0]).
        deform: key -> decoded frames, a list or a [T, H, W(, C)] array. Frames are
            grayscale-replicated HWC uint8, or already 2-D.

    Returns:
        ``deform_raw`` float32 [n_keys, 240, 240] in [0, 255]; HB1 consumes raw values,
        so there is no /255.
    """
    from groot.core.training.configs.model import n2_tactile as pinned  # lazy, as above

    if list(modality.delta_indices) != [0]:
        # Deformation conditions on the current frame, never a past/future frame.
        raise ValueError(
            f"deform delta_indices must be exactly [0], got {list(modality.delta_indices)}"
        )
    hw = pinned.DEFORM_INPUT_HW
    maps = []
    for key in modality.modality_keys:
        frame = np.asarray(deform[key][0], dtype=np.float32)
        if frame.ndim == 3:
            frame = frame[..., 0]  # R=G=B, keep one channel
        if frame.shape != (hw, hw):
            raise ValueError(f"deform key {key!r}: frame shape {frame.shape} != ({hw}, {hw})")
        maps.append(frame)
    return {"deform_raw": np.ascontiguousarray(np.stack(maps, axis=0), dtype=np.float32)}


def process_tactile_modalities(
    modality_configs_for_tag: Mapping[str, Any], content: Any, *, training: bool = True
) -> dict[str, np.ndarray]:
    """Tactile and deform inputs for one sample, in that key order.

    A group is skipped when the tag does not declare it or the step carries no data for
    it (including objects without ``tactile``/``deform`` attributes). Whether a missing
    key is an error is decided downstream, by the collator (required_batch_keys) and the
    model's input checks.
    """
    result = {}
    modality = modality_configs_for_tag.get("tactile", None)
    tactile = getattr(content, "tactile", None)
    if modality is not None and tactile:
        result.update(
            process_tactile(modality, tactile, getattr(content, "metadata", {}), training=training)
        )
    modality = modality_configs_for_tag.get("deform", None)
    deform = getattr(content, "deform", None)
    if modality is not None and deform:
        result.update(process_deform(modality, deform))
    return result
