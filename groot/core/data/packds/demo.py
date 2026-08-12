# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PackDS by Example: Why a Step-Table Format Works Well
#
# Robotics datasets are often split across multiple storage layers: videos in one place,
# observations and actions in another, and metadata in sidecar files that have to stay in
# sync. PackDS takes a different approach:
#
# - **one timestep = one row**
# - **one dataset = one table**
# - **video, state, action, and annotations stay aligned by construction**
#
# This notebook focuses on the parts that make PackDS distinctive:
#
# 1. A single synchronized row model for multimodal robot data
# 2. Random access and selective columnar reads from the same dataset
# 3. Transparent video storage choices: JPEG vs. video blobs
# 4. Analytics and metadata access without sidecar bookkeeping
#
# We use a tiny YamSim dataset so the examples stay fast and concrete.
#
# **Prerequisites:**
# ```bash
# uv sync --extra notebook
# ```

# %% [markdown]
# ## 1. Setup

# %%
from __future__ import annotations

import os
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SVT_LOG", "0")
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from gymnasium.envs import registration
from tqdm.auto import tqdm

from groot.control.envs.yam.yam_sim_env import SimDummyPolicy
from groot.core.data import packds

try:
    registration.register(
        id="YamSim-v0",
        entry_point="groot.control.envs.yam.yam_sim_env:YamSimEnv",
    )
except Exception:
    pass

WORK_DIR = Path(tempfile.mkdtemp(prefix="packds_demo_"))
print(f"Working directory: {WORK_DIR}")

# %% [markdown]
# ## 2. Build a tiny PackDS dataset
#
# We collect a small amount of simulated data. The simulator is just a convenient source of
# realistic multimodal robot steps; the main point is the resulting PackDS layout.

# %%
CONTROL_FREQ = 30.0
NUM_EPISODES = 2
STEPS_PER_EPISODE = 48
TASKS = ["pick up cup", "place on shelf"]

env = gym.make("YamSim-v0")
policy = SimDummyPolicy(env.action_space)

CAM_KEYS = sorted(k for k in env.observation_space.spaces if k.endswith("_camera_image"))
STATE_KEYS = sorted(k for k in env.observation_space.spaces if not k.endswith("_camera_image"))
ACTION_KEYS = sorted(env.action_space.spaces.keys())

print(f"Cameras: {CAM_KEYS}")
print(f"State keys: {STATE_KEYS}")
print(f"Action keys: {ACTION_KEYS}")


# %%
def collect_episode(
    env: gym.Env,
    policy: SimDummyPolicy,
    episode_index: int,
    task_name: str,
    num_steps: int = STEPS_PER_EPISODE,
) -> list[packds.PackDSStep]:
    """Roll out one episode and return PackDSStep rows."""
    episode_id = str(uuid.uuid4())
    obs, _ = env.reset(seed=episode_index)
    steps: list[packds.PackDSStep] = []

    for step_index in range(num_steps):
        action, _ = policy.get_action(obs)
        next_obs, _, _, _, _ = env.step(action)
        t_ns = round(step_index * 1e9 / CONTROL_FREQ)

        steps.append(
            packds.PackDSStep(
                episode_id=episode_id,
                episode_index=episode_index,
                step_index=step_index,
                step_time_ns=t_ns,
                original_step_time_ns=t_ns,
                has_valid_action=step_index < num_steps - 1,
                annotations={"task": task_name},
                decoded_video={k: obs[k] for k in CAM_KEYS},
                state={k: obs[k].astype(np.float32) for k in STATE_KEYS},
                action={k: action[k].astype(np.float32) for k in ACTION_KEYS},
            )
        )
        obs = next_obs

    return steps


all_episodes = [
    collect_episode(env, policy, episode_index=i, task_name=TASKS[i])
    for i in tqdm(range(NUM_EPISODES), desc="Collecting episodes")
]
raw_episodes = deepcopy(all_episodes)
close_env = getattr(env, "close", None)
if callable(close_env):
    close_env()
renderer = getattr(getattr(env, "unwrapped", None), "mujoco_renderer", None)
close_renderer = getattr(renderer, "close", None)
if callable(close_renderer):
    close_renderer()

dataset_path = WORK_DIR / "demo_blob_h264.packds"
with packds.PackDS(
    dataset_path,
    mode="w",
    steps_per_second=CONTROL_FREQ,
    overwrite=True,
    encode_jpeg=False,
    encode_video_blob=True,
    video_blob_codec="h264",
) as ds:
    for episode_steps in tqdm(all_episodes, desc="Writing PackDS"):
        ds.write_steps(episode_steps)

size_mb = sum(f.stat().st_size for f in dataset_path.rglob("*") if f.is_file()) / 1e6
print(f"Dataset written to: {dataset_path}")
print(f"Approx. size on disk: {size_mb:.2f} MB")

# %% [markdown]
# ## 3. One row = one synchronized robot timestep
#
# The key PackDS idea is that a row is not just a frame or just a state vector. It is one
# synchronized robot timestep containing:
#
# - episode identity and timing
# - all camera views for that moment
# - state
# - action
# - annotations such as the task name
#
# That means consumers do not have to join across separate files to recover alignment.

# %%
ds = packds.PackDS(dataset_path)
print(f"Episodes: {ds.num_episodes}")
print(f"Steps: {ds.num_steps}")
print(f"Video keys: {ds.video_keys}")
print(f"State keys: {ds.state_keys}")
print(f"Action keys: {ds.action_keys}")

ep_id = ds.get_episode_ids()[0]
batch = packds.from_table(ds.read_episodes([ep_id])[ep_id]).decode_video()
step = batch[10]

state_shapes = {k: v.shape for k, v in step.state.items()}
action_shapes = {k: v.shape for k, v in step.action.items()}

print("One PackDS row contains:")
print(f"  episode_id: {step.episode_id}")
print(f"  episode_index: {step.episode_index}")
print(f"  step_index: {step.step_index}")
print(f"  step_time_ns: {step.step_time_ns}")
print(f"  annotations: {step.annotations}")
print(f"  first camera frame shape: {step.decoded_video[ds.video_keys[0]].shape}")
print(f"  state shapes: {state_shapes}")
print(f"  action shapes: {action_shapes}")

# %% [markdown]
# A quick visual check: one step, all cameras, same moment in time.

# %%
fig, axes = plt.subplots(1, len(ds.video_keys), figsize=(4 * len(ds.video_keys), 4))
if len(ds.video_keys) == 1:
    axes = [axes]
for ax, cam in zip(axes, ds.video_keys):
    ax.imshow(step.decoded_video[cam])
    ax.set_title(cam)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Random access and selective reads
#
# PackDS is useful not only because the data is aligned, but because it remains easy to
# access in multiple ways:
#
# - whole-episode scans for sequence models
# - arbitrary row reads for random sampling
# - selective column reads when you do not need video bytes

# %%
# Episode-oriented read
ep_id_1 = ds.get_episode_ids()[1]
ep_table = ds.read_episodes([ep_id_1])[ep_id_1]
ep_batch = packds.from_table(ep_table)
print(f"Episode 1 length: {len(ep_batch)} steps")
print(f"First task label: {ep_batch[0].annotations['task']}")

# Arbitrary global row access
indices = [0, 3, 11, 25, 60, 75]
raw_table = ds.read_raw_indices(indices)
batch = packds.from_table(raw_table)
print(f"Random batch size: {len(batch)}")
print(f"Episode IDs in batch: {batch.episode_id}")
print(f"Step indices in batch: {batch.step_index.tolist()}")

# %% [markdown]
# PackDS also makes columnar access explicit through the underlying Lance table. Here we
# read only scalar metadata plus a single state column, skipping video columns entirely.

# %%
selected_columns = [
    "episode_index",
    "step_index",
    "annotation:task",
    packds.utils.key_to_column(packds.constants.STATE_PREFIX, ds.state_keys[0]),
]
small_table = ds.steps_table.to_lance().to_table(columns=selected_columns)
print(f"Selected columns: {small_table.column_names}")
print(f"Rows read: {small_table.num_rows}")
small_table.slice(0, 5).to_pandas()

# %% [markdown]
# Predicate pushdown works on the same table, so filtering stays cheap and direct.

# %%
filtered = ds.steps_table.to_lance().to_table(
    columns=["episode_index", "step_index", "annotation:task"],
    filter="episode_index = 0 AND step_index < 5",
)
filtered.to_pandas()

# %% [markdown]
# ## 5. Same logical API, different video encodings
#
# One of PackDS's nicest properties is that user code does not have to care much about how
# video is stored. We can write the same steps in different encodings and still decode them
# the same way.
#
# Here we compare:
# - JPEG per frame
# - H.264 video blobs
# - AV1 video blobs

# %%
def reset_video_storage(episodes: list[list[packds.PackDSStep]]) -> None:
    for episode in episodes:
        for s in episode:
            s.encoded_video = None
            s.encoded_video_blob = None
            s.video_blob_frame_index = None


# %%
def write_dataset(path: Path, episodes: list[list[packds.PackDSStep]], **kwargs) -> float:
    reset_video_storage(episodes)
    with packds.PackDS(path, mode="w", steps_per_second=CONTROL_FREQ, overwrite=True, **kwargs) as ds_out:
        for episode in episodes:
            ds_out.write_steps(episode)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


jpeg_path = WORK_DIR / "demo_jpeg.packds"
blob_h264_path = WORK_DIR / "demo_blob_h264_compare.packds"
blob_av1_path = WORK_DIR / "demo_blob_av1.packds"

size_jpeg = write_dataset(
    jpeg_path,
    deepcopy(raw_episodes),
    encode_jpeg=True,
    encode_video_blob=False,
)
size_h264 = write_dataset(
    blob_h264_path,
    deepcopy(raw_episodes),
    encode_jpeg=False,
    encode_video_blob=True,
    video_blob_codec="h264",
)
size_av1 = write_dataset(
    blob_av1_path,
    deepcopy(raw_episodes),
    encode_jpeg=False,
    encode_video_blob=True,
    video_blob_codec="av1",
)

print(f"{'Mode':<18} {'Size (MB)':>10}")
print("-" * 30)
print(f"{'JPEG':<18} {size_jpeg:>10.2f}")
print(f"{'Video blob H.264':<18} {size_h264:>10.2f}")
print(f"{'Video blob AV1':<18} {size_av1:>10.2f}")

# %% [markdown]
# Regardless of encoding, decoding uses the same high-level API.

# %%
for path in [jpeg_path, blob_h264_path, blob_av1_path]:
    ds_mode = packds.PackDS(path)
    mode_ep_id = ds_mode.get_episode_ids()[0]
    mode_batch = packds.from_table(ds_mode.read_episodes([mode_ep_id])[mode_ep_id]).decode_video()
    mode_steps = list(mode_batch)
    cam = ds_mode.video_keys[0]
    print(f"{path.name}: decoded {cam} frame shape = {mode_steps[0].decoded_video[cam].shape}")
    ds_mode.close()

# %% [markdown]
# This is a core PackDS advantage: storage layout can change, while read-side user code can
# stay essentially the same.

# %% [markdown]
# ## 6. Columnar analytics without format conversion pain
#
# Because PackDS is fundamentally tabular, the same dataset is easy to inspect with Polars.
# We can do lightweight analysis without inventing a separate export pipeline.

# %%
analytics_table = ds.steps_table.to_lance().to_table(
    columns=["episode_index", "step_index", "annotation:task", "has_valid_action"]
)
analytics_df = pl.from_arrow(analytics_table)
analytics_df

# %%
analytics_df.group_by("episode_index").agg(
    pl.len().alias("num_steps"),
    pl.col("annotation:task").first().alias("task"),
    pl.col("has_valid_action").sum().alias("valid_action_steps"),
).sort("episode_index")

# %% [markdown]
# ## 7. Self-describing datasets: schema metadata and stats
#
# PackDS keeps important dataset-level information in schema metadata instead of relying on
# separate bookkeeping files.

# %%
metadata_keys = [
    "packds:version",
    "packds:steps_per_second",
    "packds:jpeg_quality",
    "packds:video_blob_codec",
    "packds:video_blob_crf",
    "packds:video_blob_group_size",
]

print("Selected schema metadata:")
for key in metadata_keys:
    value = ds.schema.metadata.get(key.encode()) if ds.schema and ds.schema.metadata else None
    if value is not None:
        print(f"  {key}: {value.decode()}")

# %% [markdown]
# PackDS also computes normalization statistics when the dataset is finalized.

# %%
print("A few stored stats:")
for key in sorted(ds.stats)[: min(4, len(ds.stats))]:
    stat = ds.stats[key]
    print(
        f"  {key}: "
        f"mean≈{np.mean(stat['mean']):+.3f}, "
        f"std≈{np.mean(stat['std']):.3f}, "
        f"range=[{np.min(stat['min']):+.3f}, {np.max(stat['max']):+.3f}]"
    )

# %% [markdown]
# ## 8. Takeaways
#
# The main reason to use PackDS is not just that it stores robot data — it stores it in a
# way that keeps multimodal timesteps aligned while still behaving like a columnar table.
#
# **What stood out in this demo:**
#
# - **Unified multimodal rows**: video, state, action, timing, and annotations live in one row
# - **Flexible access patterns**: episode scans, random batches, and selected-column reads all work naturally
# - **Encoding flexibility**: JPEG and video blobs share the same high-level decode path
# - **Self-describing datasets**: schema metadata and normalization stats travel with the dataset
# - **Analytics-friendly layout**: Lance/PyArrow/Polars interop works directly on the stored data
#
# For the full API and on-disk spec, see:
# - `groot/vla/data/packds/README.md`
# - `groot/vla/data/packds/SPEC.md`

# %%
ds.close()
