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
"""Gradio UI for steerability inspection.

Workflow:
1. Enter a checkpoint path (any local path with config.json + processor_config.json)
2. Pick a dataset + traj_id; full episode loads automatically
3. Pick a frame, write a prompt, click Generate
4. See denoising progress bar live
5. Compare GT future video vs generated video side-by-side (matched length)

Run on the GB200 SSH node:
    cd /mnt/aws-lfs-02/home/ruijiez/gr00t
    pip install --break-system-packages gradio
    PYTHONPATH=src python3 -m groot.core.training.open_loop.steerability_ui

Then on your Mac, port-forward:
    ssh -p 2118 -L 7860\\:127.0.0.1\\:7860 -N root@localhost
"""
from copy import deepcopy
import json
import logging
import os
import subprocess
import time

import gradio as gr
import numpy as np
import torch
from torchvision.io import write_video

from groot.core.data.schema import EmbodimentTag
from groot.core.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from groot.core.data.dataset.sharded_single_step_dataset import extract_step_data
from groot.core.data.types import ModalityConfig
from groot.core.policy.n2_policy import Gr00tN2Policy
from groot.core.training.open_loop.open_loop_eval import parse_observation_gr00t


logging.basicConfig(level=logging.INFO)


# Known checkpoint examples (user can paste any path though)
EXAMPLE_CHECKPOINTS = {
    "YAM 86k action-loss": "/mnt/aws-lfs-02/shared/checkpoints/ruijiez/n2_yam_pretrain-subtask-video-action-bidrectional-full-beta-batch2048-200k_lr3e-5_256gpu_bs8_sd1.0/checkpoint-86000",
    "Long-context 44k video-only": "/mnt/aws-lfs-02/shared/checkpoints/ruijiez/n2_egoscale_soup-bidrectional-batch2048-100k-video-only-long-context/checkpoint-44000",
}

DEFAULT_DATASET = "/mnt/aws-lfs-02/shared/datasets/yam_v7/xdof.yam_v7_subtask_only_merged_global_task"
SAVE_DIR = "/tmp/steerability_ui"
os.makedirs(SAVE_DIR, exist_ok=True)

# Caches
_policy_cache: dict[str, Gr00tN2Policy] = {}
_full_dataset_cache: dict[tuple, LeRobotEpisodeLoader] = {}
_episode_preview_cache: dict[tuple, str] = {}


def _detect_embodiment_tag(model_path: str) -> EmbodimentTag:
    """Read processor_config.json from a checkpoint dir and return the (single) embodiment tag."""
    with open(os.path.join(model_path, "processor_config.json")) as f:
        cfg = json.load(f)
    embs = list(cfg["processor_kwargs"]["modality_configs"].keys())
    if len(embs) != 1:
        # Multi-embodiment ckpt: pick the first XDOF one
        xdof = [e for e in embs if "xdof" in e]
        emb = xdof[0] if xdof else embs[0]
    else:
        emb = embs[0]
    return EmbodimentTag(emb)


def _full_modality_configs(
    model_path: str,
    embodiment_tag: EmbodimentTag,
    dataset_path: str | None = None,
) -> dict:
    """Read the FULL training-time modality config (including future delta_indices)
    from the checkpoint's processor_config.json. This is the un-truncated config that
    we use for ground-truth video extraction so GT and generated mp4s have the same length.

    If `dataset_path` is provided, language modality_keys are intersected with the
    keys actually present in the dataset's `meta/modality.json` annotation section.
    Some checkpoints register `annotation.task_recaption_{1,2,3}` for cross-dataset
    soup training; loading those against a YAM dataset (which only has `sub_task`
    and `task`) would otherwise crash the LeRobot loader.
    """
    with open(os.path.join(model_path, "processor_config.json")) as f:
        cfg = json.load(f)
    emb_cfg = cfg["processor_kwargs"]["modality_configs"][embodiment_tag.value]

    available_lang_keys: set[str] | None = None
    if dataset_path is not None:
        modality_path = os.path.join(dataset_path, "meta", "modality.json")
        if os.path.exists(modality_path):
            with open(modality_path) as f:
                ds_modality = json.load(f)
            ann = ds_modality.get("annotation", {}) or {}
            available_lang_keys = {f"annotation.{k}" for k in ann.keys()}

    out = {}
    for m in ["video", "state", "language", "action"]:
        if m not in emb_cfg:
            continue
        sub = emb_cfg[m]
        keys = sub.get("modality_keys") or []
        if m == "language" and available_lang_keys is not None:
            filtered = [k for k in keys if k in available_lang_keys]
            if not filtered:
                raise gr.Error(
                    f"None of the ckpt's language keys {keys} exist in dataset "
                    f"{dataset_path}'s meta/modality.json (available: {sorted(available_lang_keys)}). "
                    f"Did you pick the right embodiment tag?"
                )
            keys = filtered
        out[m] = ModalityConfig(
            delta_indices=sub["delta_indices"],
            modality_keys=keys,
        )
    return out


def get_policy(model_path: str, embodiment_tag: EmbodimentTag) -> Gr00tN2Policy:
    key = f"{model_path}::{embodiment_tag.value}"
    if key in _policy_cache:
        return _policy_cache[key]
    os.environ["GROOT_N2_NUM_VIDEO_STEPS"] = "32"
    logging.info(f"Loading policy from {model_path}")
    policy = Gr00tN2Policy(
        embodiment_tag=embodiment_tag,
        model_path=model_path,
        device="cuda",
        num_inference_timesteps=32,
        save_video_dir=SAVE_DIR,
    )
    _policy_cache[key] = policy
    return policy


def get_full_dataset(dataset_path: str, modality_configs: dict) -> LeRobotEpisodeLoader:
    """Loader keyed on (dataset_path, video_delta_indices) so we keep one loader
    per modality-config variant.
    """
    di = tuple(modality_configs["video"].delta_indices)
    key = (dataset_path, di)
    if key in _full_dataset_cache:
        return _full_dataset_cache[key]
    logging.info(f"Loading dataset {dataset_path} with delta_indices={di}")
    loader = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )
    _full_dataset_cache[key] = loader
    return loader


def build_full_episode_preview(dataset_path: str, traj_id: int) -> str:
    """Tile per-camera videos for an entire episode horizontally.

    LeRobot stores one mp4 per camera per episode under
    `videos/chunk-NNN/observation.images.<view>/episode_NNNNNN.mp4`. We decode
    each via torchvision.io (which bundles libx264 via PyAV — the system
    ffmpeg here lacks libx264 encoder), hstack as numpy, and write_video.
    """
    from torchvision.io import read_video

    key = (dataset_path, traj_id)
    if key in _episode_preview_cache and os.path.exists(_episode_preview_cache[key]):
        return _episode_preview_cache[key]

    chunk_dir = os.path.join(dataset_path, "videos")
    chunks = sorted(os.listdir(chunk_dir))
    if not chunks:
        raise gr.Error(f"No videos chunk dir under {chunk_dir}")
    # Read chunks_size from meta/info.json so we know which chunk an episode lives in.
    info_path = os.path.join(dataset_path, "meta", "info.json")
    chunks_size = 1000
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        chunks_size = int(info.get("chunks_size", 1000))
    chunk_idx = int(traj_id) // chunks_size
    chosen_chunk = f"chunk-{chunk_idx:03d}"
    if chosen_chunk not in chunks:
        # Compute total episodes from the meta file for a clearer error
        total_eps = "?"
        ep_path = os.path.join(dataset_path, "meta", "episodes.jsonl")
        if os.path.exists(ep_path):
            with open(ep_path) as f:
                total_eps = sum(1 for _ in f)
        raise gr.Error(
            f"traj_id {traj_id} maps to {chosen_chunk}, but dataset only has chunks "
            f"{chunks[:3]}{'...' if len(chunks)>3 else ''} (total episodes: {total_eps}). "
            f"Did you mean to switch dataset?"
        )
    view_root = os.path.join(chunk_dir, chosen_chunk)
    all_views = sorted(d for d in os.listdir(view_root) if d.startswith("observation.images."))
    if not all_views:
        raise gr.Error(f"No observation.images.* dirs under {view_root}")
    # If multiple resolutions exist (e.g. YAM ships both raw + _320_240),
    # prefer the inference-resolution variant. Datasets like Mecka have only
    # one resolution per view; the filter falls through.
    preferred = [d for d in all_views if "320" in d or "_320_240" in d]
    chosen_views = preferred or all_views

    # Heuristic ordering: top, left, right (YAM); otherwise alphabetical.
    def _view_order(name: str):
        if "top" in name:
            return (0, name)
        if "left" in name:
            return (1, name)
        if "right" in name:
            return (2, name)
        return (3, name)
    chosen_views = sorted(chosen_views, key=_view_order)

    ep_str = f"episode_{int(traj_id):06d}.mp4"
    frames_per_view: list[np.ndarray] = []
    fps_seen: list[float] = []
    for vd in chosen_views:
        p = os.path.join(view_root, vd, ep_str)
        if not os.path.exists(p):
            raise gr.Error(f"Missing per-view video: {p}")
        # read_video returns (T, H, W, C) uint8 by default (pts_unit='pts'); we
        # convert directly to numpy. info has 'video_fps'.
        video, _, info = read_video(p, pts_unit="sec")
        frames_per_view.append(video.numpy())
        fps_seen.append(float(info.get("video_fps", 30.0)))

    # Match lengths if uneven (rare, but defensive).
    min_len = min(f.shape[0] for f in frames_per_view)
    frames_per_view = [f[:min_len] for f in frames_per_view]
    # Heights should already match; concat along width.
    tiled = np.concatenate(frames_per_view, axis=2)  # (T, H, V*W, C)
    fps = int(round(fps_seen[0]))

    out_path = os.path.join(SAVE_DIR, f"preview_traj{traj_id}.mp4")
    write_video(out_path, torch.from_numpy(tiled), fps=fps)
    _episode_preview_cache[key] = out_path
    return out_path


def build_gt_mp4(data_point, save_path: str, fps: int = 8) -> str:
    """Tile per-camera images horizontally and save mp4 — matches the
    Gr00tN2Policy.save_video tiling so GT and generated are comparable.
    """
    views = list(data_point.images.values())
    if not views:
        raise gr.Error("No images in data_point.")
    arr = np.concatenate([np.asarray(v) for v in views], axis=2)  # (T, H, V*W, C)
    write_video(save_path, torch.from_numpy(arr), fps=fps)
    return save_path


def load_episode_preview(dataset_path: str, traj_id):
    """Triggered when user changes dataset_path or traj_id."""
    if not dataset_path or traj_id is None:
        return None, "(enter a dataset path)"
    try:
        out = build_full_episode_preview(dataset_path, int(traj_id))
        # Read traj length from meta/episodes.jsonl for the slider max
        ep_path = os.path.join(dataset_path, "meta", "episodes.jsonl")
        info = "loaded"
        if os.path.exists(ep_path):
            with open(ep_path) as f:
                for line in f:
                    d = json.loads(line)
                    if d.get("episode_index") == int(traj_id):
                        info = f"length={d['length']} | task={d['tasks'][0] if d.get('tasks') else '?'}"
                        break
        return out, info
    except Exception as e:
        return None, f"error: {e}"


def autodetect_embodiment(model_path: str):
    if not model_path or not os.path.isdir(model_path):
        return None
    try:
        return _detect_embodiment_tag(model_path).value
    except Exception as e:
        logging.warning(f"autodetect failed: {e}")
        return None


def run(
    model_path: str,
    embodiment_tag_str: str,
    dataset_path: str,
    traj_id,
    frame_idx,
    prompt: str,
    video_guidance_scale: float,
    denoising_steps,
    progress=gr.Progress(),
):
    if not model_path:
        raise gr.Error("Please enter a checkpoint path")
    if not embodiment_tag_str:
        raise gr.Error("Embodiment tag is empty — try Autodetect")

    progress(0.0, desc="Loading policy")
    embodiment_tag = EmbodimentTag(embodiment_tag_str)
    policy = get_policy(model_path, embodiment_tag)
    policy.video_guidance_scale = float(video_guidance_scale)
    # Per-call override of denoising steps. Both schedulers use the same value
    # (k=1 in joint-denoise + kv-cache mode requires num_action % num_video == 0).
    n_steps = int(denoising_steps)
    policy.num_video_steps = n_steps
    policy.num_inference_steps = n_steps

    # FULL modality config from training (un-truncated) for GT extraction.
    progress(0.05, desc="Loading dataset (with future indices)")
    full_modality = _full_modality_configs(model_path, embodiment_tag, dataset_path=dataset_path)
    obs_modality_configs = {k: v for k, v in full_modality.items() if k != "action"}
    full_dataset = get_full_dataset(dataset_path, obs_modality_configs)
    if int(traj_id) >= len(full_dataset):
        raise gr.Error(f"traj_id {traj_id} >= dataset length {len(full_dataset)}")
    full_traj = full_dataset[int(traj_id)]
    if int(frame_idx) >= len(full_traj):
        raise gr.Error(f"frame_idx {frame_idx} >= traj length {len(full_traj)}")

    progress(0.10, desc=f"Extracting frame {frame_idx}")
    full_data_point = extract_step_data(
        full_traj, int(frame_idx), obs_modality_configs, embodiment_tag, allow_padding=True
    )

    progress(0.15, desc="Building GT video")
    ts = int(time.time() * 1000)
    gt_path = os.path.join(SAVE_DIR, f"gt_{ts}.mp4")
    build_gt_mp4(full_data_point, gt_path)

    progress(0.20, desc="Preparing observation")
    # The policy's modality_config truncates video to past-only indices (e.g. 5 frames).
    # Slice the FULL data_point's video arrays down to the first len(policy past indices)
    # frames so the policy's expected horizon matches.
    policy_modality = policy.get_modality_config()
    n_cond = len(policy_modality["video"].delta_indices)
    base_obs: dict = {}
    for k, v in full_data_point.states.items():
        # State delta_indices are usually [0]; align state similarly to past-only.
        s = np.asarray(v)
        base_obs[f"state.{k}"] = s[: len(policy_modality["state"].delta_indices)]
    for k, v in full_data_point.images.items():
        arr = np.asarray(v)
        base_obs[f"video.{k}"] = arr[:n_cond]
    for lk in policy_modality["language"].modality_keys:
        base_obs[lk] = prompt
    parsed_obs = parse_observation_gr00t(base_obs, policy_modality)

    policy.video_latents_buffer.clear()
    policy._cached_prompt_embs = None
    policy._cached_language_dict = None

    total_steps = policy.num_video_steps

    def step_cb(step: int, total: int):
        frac = 0.25 + 0.65 * (step / max(total, 1))
        progress(frac, desc=f"Denoising {step}/{total}")

    policy.step_callback = step_cb
    progress(0.25, desc="Running model")
    try:
        policy.get_action(parsed_obs)
    finally:
        policy.step_callback = None

    progress(0.92, desc="Decoding video latents")
    gen_path = os.path.join(SAVE_DIR, f"gen_{ts}.mp4")
    policy.save_video(gen_path, traj_id=0)

    # The model tiles `max_n_cameras` views horizontally in its saved mp4. For
    # embodiments with fewer real views (e.g. Mecka has 1 ego view; YAM has 3),
    # the remaining slots are garbage. Crop to the actual view count using the
    # model's max_n_cameras as the per-view tile width (NOT a hardcoded 3 —
    # max_n_cameras=4 ckpts produced wrong crops with the old logic).
    n_actual_views = len(policy_modality["video"].modality_keys)
    max_cams = int(getattr(policy.model, "max_n_cameras", n_actual_views))
    if n_actual_views < max_cams:
        from torchvision.io import read_video
        video, _, info = read_video(gen_path, pts_unit="sec")
        per_view_w = video.shape[2] // max_cams
        cropped = video[:, :, : n_actual_views * per_view_w, :]
        fps_in = int(round(float(info.get("video_fps", 8.0))))
        write_video(gen_path, cropped, fps=fps_in)

    progress(1.0, desc="Done")
    return gt_path, gen_path


def fill_example(label: str):
    return EXAMPLE_CHECKPOINTS[label]


def build_ui():
    with gr.Blocks(title="N2 Steerability") as demo:
        gr.Markdown("# N2 Steerability Inspector")
        gr.Markdown(
            "Enter a checkpoint path, pick a dataset + traj, then a frame + prompt. "
            "GT future and generated video are matched in length for fair side-by-side comparison."
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Checkpoint**")
                model_path = gr.Textbox(
                    label="Checkpoint path (local)",
                    value=EXAMPLE_CHECKPOINTS["Long-context 44k video-only"],
                    placeholder="/mnt/aws-lfs-02/shared/checkpoints/.../checkpoint-NNNNN",
                )
                with gr.Row():
                    for label in EXAMPLE_CHECKPOINTS:
                        btn = gr.Button(label, size="sm")
                        btn.click(lambda l=label: EXAMPLE_CHECKPOINTS[l], outputs=model_path)
                with gr.Row():
                    embodiment_tag_str = gr.Textbox(
                        label="Embodiment tag", value="xdof_relative_eef_relative_joint_subtask"
                    )
                    detect_btn = gr.Button("Autodetect from ckpt", size="sm")
                detect_btn.click(autodetect_embodiment, inputs=model_path, outputs=embodiment_tag_str)

                gr.Markdown("**Dataset & trajectory**")
                dataset_path = gr.Textbox(label="Dataset path", value=DEFAULT_DATASET)
                traj_id = gr.Number(label="Trajectory ID", value=22, precision=0, minimum=0)
                load_btn = gr.Button("Load full episode preview", size="sm")
                ep_info = gr.Textbox(label="Episode info", interactive=False, value="(click load)")

                gr.Markdown("**Inference**")
                frame_idx = gr.Number(label="Frame index (conditioning)", value=60, precision=0, minimum=0)
                prompt = gr.Textbox(
                    label="Language prompt",
                    value="The left arm moves from its starting position to hover over the mango.",
                    lines=2,
                )
                video_cfg = gr.Slider(
                    label="Video CFG scale", minimum=1.0, maximum=10.0, value=1.0, step=0.5
                )
                denoising_steps = gr.Radio(
                    label="Denoising steps (video & action)",
                    choices=[1, 2, 4, 8, 16, 32],
                    value=32,
                    info="Higher = better quality, slower. Both schedulers use this value.",
                )
                run_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("**Full episode (preview)**")
                episode_video = gr.Video(label="Full episode (3 camera tile)", autoplay=False)
                gr.Markdown("**Side-by-side: GT vs generated (matched length)**")
                with gr.Row():
                    gt_video = gr.Video(label="Ground truth (dataset)", autoplay=True, loop=True)
                    gen_video = gr.Video(label="Generated", autoplay=True, loop=True)

        load_btn.click(
            fn=load_episode_preview,
            inputs=[dataset_path, traj_id],
            outputs=[episode_video, ep_info],
        )

        run_btn.click(
            fn=run,
            inputs=[
                model_path, embodiment_tag_str, dataset_path,
                traj_id, frame_idx, prompt, video_cfg, denoising_steps,
            ],
            outputs=[gt_video, gen_video],
        )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
