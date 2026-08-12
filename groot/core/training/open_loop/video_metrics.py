# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Open-loop *video* (world-model) reconstruction metrics: PSNR / SSIM / LPIPS.

The single subtlety these helpers exist to handle is the **crop**. A GrootN2
world model only ever sees and predicts a ``crop_fraction`` (typically 0.95)
*center-crop* of the frame (set by ``image_crop_fraction`` in the processor; the
model input is crop→resize to ``image_target_size``). So comparing the model's
output against the *full-frame* GT penalizes a field-of-view the model was never
asked to reproduce — historically this has deflated ego PSNR by ~10 dB and
saturated the ego comparison (see the ego-wam wiki EVAL_GUIDE.md
§"PSNR/SSIM/LPIPS scoring convention").

:func:`video_frame_metrics` therefore reports a **single** config-appropriate
``psnr`` / ``ssim`` / ``lpips`` triple:

* when the model trained on a crop (``crop_fraction < 1``, e.g. 0.95) it scores
  vs the GT center-cropped the same way the model input is — the model's own
  frame, true-reconstruction scale;
* when there is no crop (``crop_fraction >= 1``) it scores vs the GT naively
  resized to the predicted size.

The opposite (full-FOV) variant is intentionally *not* emitted: comparing a
crop prediction against the full-frame GT penalizes a field of view the model
was never asked to reproduce (historically ~10 dB deflated, ego-saturated) and
is meaningless for ranking.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "center_crop_resize",
    "decode_video_latents",
    "naive_resize",
    "video_frame_metrics",
    "whole_path_video_chunks",
]


def decode_video_latents(policy) -> list:
    """Decode a policy's buffered video latents into per-view ``(T, H, W, C)`` uint8 frame
    arrays, using the policy's EXISTING ``model.decode_latents`` — so the open-loop video
    metrics need no extra method on ``Gr00tN2Policy``.

    Duck-typed on ``policy.video_latents_buffer`` (list of per-chunk latents) and
    ``policy.model`` (with ``max_n_cameras`` + ``decode_latents``); both a raw
    ``Gr00tN2Policy`` and the :class:`~groot.core.training.open_loop.live_policy.LivePolicyAdapter`
    (which exposes them via properties) satisfy it. Returns ``[]`` when nothing was buffered
    (video prediction off)."""
    buf = getattr(policy, "video_latents_buffer", None)
    model = getattr(policy, "model", None)
    if not buf or model is None:
        return []
    per_view: list[list] = [[] for _ in range(model.max_n_cameras)]
    for step_latents in buf:
        decoded_views = model.decode_latents(step_latents)
        if not decoded_views:
            continue
        for v_idx, view_tensor in enumerate(decoded_views):
            per_view[v_idx].append(view_tensor[0].cpu().numpy())
    if not per_view[0]:
        return []
    return [np.concatenate(frames, axis=0) for frames in per_view]


def whole_path_video_chunks(
    n_frames_total: int,
    *,
    n_cond: int,
    n_total: int,
    pred_deltas: list[int],
    start_step: int,
    action_horizon: int,
    traj_length: int,
) -> tuple[list[tuple[int, int, list[int]]], int]:
    """Index math for stitching *every* rollout chunk into a whole-path video score.

    A world-model policy buffers one decoded video chunk per ``get_action`` hop
    (stride ``action_horizon``), so :func:`decode_video_latents` returns
    ``ego_full`` of shape ``(n_total * n_chunks, H, W, C)`` — every chunk
    concatenated. The legacy metric used only chunk 0 (``ego_full[n_cond:n_total]``);
    this returns the slice for *all* chunks so they can be scored and aggregated,
    mirroring how the action metric re-queries the policy at each hop.

    For each chunk ``k`` it gives the predicted-frame slice ``(pred_lo, pred_hi)``
    into ``ego_full`` and the matching ground-truth frame positions, anchored — like
    the action loop — at ``chunk_start = start_step + k * action_horizon`` and
    clamped to ``[0, traj_length - 1]``.

    Args:
        n_frames_total: ``ego_full.shape[0]`` — total decoded frames across chunks.
        n_cond: conditioning (``<= 0``-delta) frames per chunk.
        n_total: total frames per chunk (cond + predicted) = ``len(deltas)``.
        pred_deltas: sorted strictly-positive video deltas (the predicted frames).
        start_step: rollout start step of the trajectory window.
        action_horizon: rollout stride (steps between policy queries).
        traj_length: trajectory length, for clamping GT positions.

    Returns:
        ``(chunks, n_clamped)`` where ``chunks[k] = (pred_lo, pred_hi, gt_positions)``
        and ``n_clamped`` counts GT positions clamped past the trajectory end (the
        last-chunk edge effect, worth logging).
    """
    if n_total <= 0:
        raise ValueError(f"whole_path_video_chunks: n_total must be > 0, got {n_total}")
    if n_frames_total % n_total != 0:
        raise ValueError(
            f"whole_path_video_chunks: n_frames_total={n_frames_total} not divisible by "
            f"n_total={n_total} (decoded frames don't tile into whole chunks)"
        )
    n_chunks = n_frames_total // n_total
    chunks: list[tuple[int, int, list[int]]] = []
    n_clamped = 0
    for k in range(n_chunks):
        pred_lo = k * n_total + n_cond
        pred_hi = (k + 1) * n_total
        chunk_start = start_step + k * action_horizon
        gt_positions: list[int] = []
        for d in pred_deltas:
            raw = chunk_start + d
            clamped = min(max(raw, 0), traj_length - 1)
            if clamped != raw:
                n_clamped += 1
            gt_positions.append(clamped)
        chunks.append((pred_lo, pred_hi, gt_positions))
    return chunks, n_clamped


def naive_resize(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a (H, W, 3) uint8 frame to (target_h, target_w) — full FOV, squished."""
    import cv2

    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def center_crop_resize(
    frame: np.ndarray, crop_fraction: float, target_h: int, target_w: int
) -> np.ndarray:
    """Replicate the processor's eval transform: center-crop ``crop_fraction`` then resize.

    Puts the GT into the model's own (cropped) frame so the metric is apples-to-apples.
    frame: (H, W, 3) uint8.
    """
    import cv2

    h, w = frame.shape[:2]
    ch, cw = max(1, int(h * crop_fraction)), max(1, int(w * crop_fraction))
    y, x = (h - ch) // 2, (w - cw) // 2
    return cv2.resize(frame[y : y + ch, x : x + cw], (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def video_frame_metrics(
    pred: np.ndarray,
    gt_raw: np.ndarray,
    crop_fraction: float = 0.95,
    lpips_net=None,
    crop_transform=None,
) -> dict[str, float]:
    """Per-frame PSNR/SSIM(/LPIPS) of predicted vs GT video, scored in the model's frame.

    Only the **single config-appropriate variant** is returned (not a full-vs-crop pair).
    The GT is brought into the model's own frame via the processor's ``crop_transform``
    whenever it is supplied — the **exact** input pipeline (crop+resize for a crop ckpt,
    plain resize for a no-crop ckpt), so the comparison matches training/inference in
    BOTH cases. Only when no transform is available does it fall back to a generic
    ``crop_fraction`` center-crop (``crop_fraction < 1``) or a plain resize (``>= 1``).

    Scoring a 0.95-crop prediction against the full-FOV GT penalizes a field of view the
    model was never asked to reproduce (historically ~10 dB deflation) and is meaningless,
    so that variant is no longer emitted — see the module docstring.

    Args:
        pred:   (T, Hp, Wp, 3) uint8 — predicted frames at the model output resolution.
        gt_raw: (T, Hg, Wg, 3) uint8 — GT frames at the stored (raw) resolution.
        crop_fraction: the model's ``image_crop_fraction`` (e.g. 0.95). ``>= 1`` → no crop.
        lpips_net: optional LPIPS module (e.g. ``lpips.LPIPS(net="alex")`` on a device).
                   If None, the ``lpips`` key is omitted.
        crop_transform: optional callable replicating the processor's eval crop+resize.

    Returns a dict of means over T: ``psnr, ssim[, lpips]``.
    Higher PSNR/SSIM and lower LPIPS = better.
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    n = min(len(pred), len(gt_raw))
    if n == 0:
        return {}
    hp, wp = pred.shape[1:3]
    use_crop = crop_fraction is not None and float(crop_fraction) < 1.0

    psnrs: list[float] = []
    ssims: list[float] = []
    lpipss: list[float] = []
    torch = None
    if lpips_net is not None:
        import torch  # noqa: PLC0415

        lp_device = next(lpips_net.parameters()).device

    for t in range(n):
        p = pred[t]
        # Bring the GT into the model's frame. ALWAYS prefer the processor's OWN eval
        # transform (``crop_transform``) when available — it is exactly what the model
        # saw for its input (crop+resize for a crop ckpt, plain resize for a no-crop
        # ckpt), so the comparison matches the training/inference pipeline in BOTH cases.
        # Only fall back to a generic crop/resize when the processor exposes no transform.
        if crop_transform is not None:
            g = np.asarray(crop_transform(gt_raw[t]), dtype=np.uint8)
            if g.ndim == 3 and g.shape[0] == 3 and g.shape[2] != 3:  # CHW -> HWC
                g = np.transpose(g, (1, 2, 0))
            if g.shape[:2] != (hp, wp):
                g = naive_resize(g, hp, wp)
        elif use_crop:
            g = center_crop_resize(gt_raw[t], float(crop_fraction), hp, wp)
        else:
            g = naive_resize(gt_raw[t], hp, wp)
        psnrs.append(float(peak_signal_noise_ratio(g, p, data_range=255)))
        ssims.append(float(structural_similarity(g, p, channel_axis=2, data_range=255)))
        if lpips_net is not None:
            with torch.no_grad():
                pt = torch.from_numpy(p).permute(2, 0, 1).unsqueeze(0).float().to(lp_device) / 127.5 - 1
                gt_t = torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).float().to(lp_device) / 127.5 - 1
                lpipss.append(float(lpips_net(pt, gt_t).item()))

    out: dict[str, float] = {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}
    if lpipss:
        out["lpips"] = float(np.mean(lpipss))
    return out
