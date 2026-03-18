<div align="center">

  <img src="media/header_compress.png" width="800" alt="NVIDIA Isaac GR00T N1.7 Header">

  <!-- --- -->
  
  <p style="font-size: 1.2em;">
    <a href="https://developer.nvidia.com/isaac/gr00t"><strong>Website</strong></a> | 
    <a href="https://huggingface.co/nvidia/GR00T-N1.7-3B"><strong>Model</strong></a> |
    <a href="https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"><strong>Dataset</strong></a> |
    <a href="https://arxiv.org/abs/2503.14734"><strong>Paper</strong></a>
  </p>
</div>

## NVIDIA Isaac GR00T

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td style="width:33.33%; text-align:center;">
      <img src="media/unitree_g1.gif" style="max-width:100%; height:auto;">
    </td>
    <td style="width:33.33%; text-align:center;">
      <img src="media/agibot_g1.gif" style="max-width:100%; height:auto;">
    </td>
    <td style="width:33.33%; text-align:center;">
      <img src="media/yam.gif" style="max-width:100%; height:auto;">
    </td>
  </tr>
</table>

> We just released GR00T N1.7, the latest version of GR00T N1 with a new VLM backbone (Cosmos-Reason2-2B / Qwen3-VL) and improved performance.

> To use older versions: [N1.6](https://github.com/NVIDIA/Isaac-GR00T/) | [N1.5](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release)

NVIDIA Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot skills. This cross-embodiment model takes multimodal input, including language and images, to perform manipulation tasks in diverse environments.

GR00T N1.7 is trained on a diverse mixture of robot data including bimanual, semi-humanoid and an expansive humanoid dataset. It is adaptable through post-training for specific embodiments, tasks and environments.

The neural network architecture of GR00T N1.7 is a combination of vision-language foundation model and diffusion transformer head that denoises continuous actions. Here is a schematic diagram of the architecture:

<div align="center">
<img src="media/model-architecture.png" width="800" alt="model-architecture">
</div>

Here is the general procedure to use GR00T N1.7:

1. We assume the user has already collected a dataset of robot demonstrations in the form of (video, state, action) triplets for a specific task.
2. The user will first convert the demonstration data into the LeRobot compatible data schema (more info in [`getting_started/data_preparation.md`](getting_started/data_preparation.md)), which is compatible with the upstream [Huggingface LeRobot Dataset V2](https://github.com/huggingface/lerobot).
3. Our repo provides convenient scripts to validate zero-shot performance of the pretrained model (see [Policy API Guide](getting_started/policy.md) and [DROID Inference](examples/DROID/README.md)).
4. Our repo provides examples of different configurations for training with different robot embodiments (see [`examples/`](examples/) and [Fine-tuning Guide](getting_started/finetune_new_embodiment.md)).
5. Our repo provides convenient scripts for finetuning the pre-trained GR00T N1.7 model on user's data, and running inference, see [`examples`](examples).
6. Our repo provides convenient scripts to run academic simulation benchmarks with finetuned checkpoints (see [Evaluation](#4-evaluation)).
7. The user will need to connect the `Gr00tPolicy` to the robot controller to execute actions on their target hardware.

## What's New in GR00T N1.7

GR00T N1.7 builds on N1.6 with a new VLM backbone and code-level improvements.

### Key Changes from N1.6

- **New VLM backbone:** Cosmos-Reason2-2B (Qwen3-VL architecture), replacing the Eagle backbone used in N1.6. Supports flexible resolution and encodes images in their native aspect ratio without padding.
- **State dropout regularization:** `state_dropout_prob` defaults to 0.8 (was 0.0 in N1.6) for improved generalization.
- **Percentile normalization support:** Added `use_percentiles` option for action/state normalization.
- Simplified data processing pipeline (`processing_gr00t_n1d7.py`).
- Added full pipeline export to ONNX and TensorRT with improved frequency.
- Added more supported robot embodiments.

### Inherited from N1.6 (vs N1.5)

The following improvements were introduced in N1.6 and carry forward in N1.7:
- 2x larger DiT action head (32 layers vs 16 in N1.5).
- Removed N1.5's post-VLM 4-layer transformer adapter; instead unfreezes top 4 VLM layers during pretraining.
- State-relative action chunks for most embodiments (vs absolute joint angles / EEF positions in N1.5).
- Expanded pretraining data: bimanual YAM arms, AGIBot Genie1, simulated Galaxea R1 Pro (BEHAVIOR), Unitree G1 whole-body locomanipulation.
- Faster dataloader with sharded dataloader support.
- Flexible training configuration.

## Target Audience

GR00T N1.7 is intended for researchers and professionals in robotics. This repository provides tools to:

- Leverage a pre-trained foundation model for robot control
- Fine-tune on small, custom datasets
- Adapt the model to specific robotics tasks with minimal data
- Deploy the model for inference

The focus is on enabling customization of robot behaviors through finetuning.

## Installation Guide

### Clone the Repository

GR00T relies on submodules for certain dependencies. Include them when cloning:

Note: `git-lfs` may be required to download parquet data files in `/demo_data`. To install it, `sudo apt install git-lfs`.
```sh
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```

If you've already cloned without submodules, initialize them separately:

```sh
git submodule update --init --recursive
```

### Set Up the Environment

GR00T uses [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management. Each supported platform has its own dependency configuration under `scripts/deployment/`.

#### dGPU (Non-Jetson) — Default

#### System Dependencies

Install FFmpeg (required by `torchcodec`, the default video backend):

```sh
sudo apt-get update && sudo apt-get install -y ffmpeg
```

#### Python Environment

After installing uv, create the environment and install GR00T:

```sh
uv sync --python 3.10
```
GPU dependencies (flash-attn, TensorRT, etc.) are included in the default install.

#### Jetson AGX Thor

> **flash-attn on older systems (e.g., Ubuntu 20.04 with glibc < 2.35):** The pre-built `flash-attn` wheel may fail with `ImportError: glibc_compat.so: cannot open shared object file`. To fix this, build from source:
> ```sh
> uv pip install flash-attn==2.7.4.post1 --no-binary flash-attn --no-cache
> ```
> This compiles locally (~10-30 minutes) and avoids the glibc compatibility issue. Verify with:
> ```sh
> python -c "import flash_attn; print(flash_attn.__version__)"
> ```

> **CUDA 13.x Users:** PyTorch 2.7 pins Triton to 3.3.1, which does not recognize CUDA major version 13+. This causes a `RuntimeError` in Triton's `ptx_get_version()`. To fix this, locate Triton's compiler file (typically at `<your-venv>/lib/python3.10/site-packages/triton/backends/nvidia/compiler.py`) and add the following branch **before** the existing `if major == 12:` line:
> ```python
> if major == 13:
>     return 90 + minor
> ```

Tested with JetPack 7.1.

```bash
bash scripts/deployment/thor/install_deps.sh
source .venv/bin/activate
source scripts/activate_thor.sh
```

See the [Thor setup guide](scripts/deployment/README.md#jetson-thor-setup) for Docker and bare metal details.

#### DGX Spark

Tested with DGX Spark GB10.

```bash
bash scripts/deployment/spark/install_deps.sh
source .venv/bin/activate
source scripts/activate_spark.sh
```

See the [Spark setup guide](scripts/deployment/README.md#dgx-spark-setup) for Docker and bare metal details.

#### Jetson Orin

Tested with JetPack 6.2.

```bash
bash scripts/deployment/orin/install_deps.sh
source .venv/bin/activate
source scripts/activate_orin.sh
```

See the [Orin setup guide](scripts/deployment/README.md#jetson-orin-setup) for Docker and bare metal details.

> **CUDA 13.x Users:** PyTorch 2.7 pins Triton to 3.3.1, which does not recognize CUDA major version 13+. This causes a `RuntimeError` in Triton's `ptx_get_version()`. To fix this, locate Triton's compiler file (typically at `<your-venv>/lib/python3.10/site-packages/triton/backends/nvidia/compiler.py`) and add the following branch **before** the existing `if major == 12:` line:
> ```python
> if major == 13:
>     return 90 + minor
> ```

For a containerized setup that avoids system-level dependency conflicts, see our [Docker Setup Guide](docker/README.md).

For training and inference hardware recommendations, see the [Hardware Recommendation Guide](getting_started/hardware_recommendation.md).

## Model Checkpoints

### Base Models
We provide pre-trained base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data and can be used for finetuning on downstream tasks.

| Model | Use Case | Description | Checkpoint Path | Branch |
| ----- | -------- | ----------- | --------------- | ------ |
| GR00T N1.7 | Finetuning | Base GR00T N1.7 model (3B parameters) | [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) | [main](https://github.com/NVIDIA/Isaac-GR00T) |
| GR00T N1.6 | Finetuning | Base [GR00T N1.6 model](https://research.nvidia.com/labs/gear/gr00t-n1_6/) (3B parameters) | [nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) | [n1.6-release](https://github.com/NVIDIA/Isaac-GR00T) |
| GR00T N1.5 | Finetuning | Base [GR00T N1.5 model](https://research.nvidia.com/labs/gear/gr00t-n1_5/) (3B parameters) | [nvidia/GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B) | [n1.5-release](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release) |

### Finetuned Models
We also provide finetuned checkpoints for various robot platforms and benchmarks. These models are finetuned from the base models above and can be used directly for evaluation or as starting points for further finetuning.

| Model | Base Model | Description | Checkpoint Path | Example |
| ----- | ---------- | ----------- | --------------- | ------- |
| GR00T-N1.7-LIBERO | [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) | Fine-tuned on [LIBERO](https://libero-project.github.io/) benchmark for Franka Panda robot on manipulation tasks | [nvidia/GR00T-N1.7-LIBERO](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) | [LIBERO](examples/LIBERO/README.md) |

> **Note:** Additional N1.7 finetuned checkpoints are coming soon. N1.6 finetuned checkpoints are available on the [n1.6-release](https://github.com/NVIDIA/Isaac-GR00T) branch.

## Quick Start

### DROID Inference

Start the policy server with the pre-trained N1.7 checkpoint:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
```

See [DROID](examples/DROID/README.md) for the full inference and control setup.

### LIBERO Inference (Server-Client)

Run inference with the LIBERO finetuned checkpoint using the server-client architecture:

First, download the finetuned model to a local directory (HuggingFace does not support nested repo paths directly):
```bash
uv run hf download nvidia/GR00T-N1.7-LIBERO --include "libero_10/config.json" "libero_10/embodiment_id.json" "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" "libero_10/processor_config.json" "libero_10/statistics.json" --local-dir checkpoints/GR00T-N1.7-LIBERO
```

**Terminal 1 — Start the policy server:**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --embodiment-tag LIBERO_PANDA \
    --device cuda:0
```

> **Tip:** If you get `ZMQError: Address already in use`, the default port 5555 is occupied. Use `--port <other_port>` (e.g., `--port 5556`).

**Terminal 2 — Run open-loop evaluation against the server:**
```bash
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path demo_data/libero_demo \
    --embodiment-tag LIBERO_PANDA \
    --host 127.0.0.1 \
    --port 5555 \
    --traj-ids 0 \
    --action-horizon 16 \
    --steps 5
```

See [LIBERO](examples/LIBERO/README.md) for finetuning and evaluation details.

## Getting started with this repo

We provide accessible Jupyter notebooks and detailed documentation in the [`./getting_started`](getting_started) folder.

## 1. Data Preparation

Please refer to the [data preparation guide](getting_started/data_preparation.md) for more details.

## 2. Inference

After data is prepared, the GR00T model can be used to generate output actions with the below simple inference script:

```bash
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag LIBERO_PANDA \
  --traj-ids 0 1 2 \
  --inference-mode pytorch \
  --action-horizon 8
```

GR00T-N1.7-3B inference timing on H100 (4 denoising steps, single view):

| Mode | Data Processing | Backbone | Action Head | E2E | Frequency |
|------|-----------------|----------|-------------|-----|-----------|
| PyTorch Eager | 4 ms | 49 ms | 95 ms | 148 ms | 6.8 Hz |
| torch.compile | 4 ms | 48 ms | 11 ms | 63 ms | 15.8 Hz |
| **TRT Full Pipeline** | **4 ms** | **7 ms** | **13 ms** | **23 ms** | **42.6 Hz** |

For more details including faster inference with TensorRT, see the [Deployment & Inference Guide](scripts/deployment/README.md).

## 3. Finetuning

### Supported Embodiment Tags

GR00T N1.7 supports the following embodiment tags:

**Pretrain tags** (zero-shot inference, or as base for finetuning):

| Tag | Robot | Value |
|-----|-------|-------|
| `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` | DROID | `oxe_droid_relative_eef_relative_joint` |
| `ROBOCASA_PANDA_OMRON` | RoboCasa Panda + Omron base | `robocasa_panda_omron` |
| `AGIBOT` | AgiBot | `agibot` |
| `XDOF` | Generic X-DOF | `xdof` |

**Posttrain tags with built-in modality configs** (can finetune directly):

| Tag | Robot | Value |
|-----|-------|-------|
| `UNITREE_G1` | Unitree G1 | `unitree_g1_full_body_with_waist_height_nav_cmd` |
| `BEHAVIOR_R1_PRO` | Galaxea R1 Pro (BEHAVIOR) | `sim_behavior_r1_pro` |
| `LIBERO_PANDA` | LIBERO Panda | `libero_sim` |

**Posttrain tags for evaluation only** (require `--modality-config-path` for finetuning):

| Tag | Robot | Value |
|-----|-------|-------|
| `SIMPLER_ENV_GOOGLE` | SimplerEnv Google Robot | `simpler_env_google` |
| `SIMPLER_ENV_WIDOWX` | SimplerEnv WidowX | `simpler_env_widowx` |

**Generic tag** for any new robot: `NEW_EMBODIMENT` (requires `--modality-config-path`)

> **Note:** Pretrain tags (e.g., `ROBOCASA_PANDA_OMRON`) are embedded in the model checkpoint and used for inference. They do not have entries in `MODALITY_CONFIGS`, so finetuning with a pretrain tag requires using `NEW_EMBODIMENT` with a `--modality-config-path` instead.

### Fine-tuning

> **Reproducing LIBERO benchmark results?** See [examples/LIBERO/README.md](examples/LIBERO/README.md) for the exact dataset download and finetune commands using the built-in `LIBERO_PANDA` embodiment.
>
> **Fine-tuning on your own robot?** Continue below.

### Fine-tune on Custom Embodiments ("NEW_EMBODIMENT")

To finetune GR00T on your own robot data and configuration, follow the detailed tutorial available at [`getting_started/finetune_new_embodiment.md`](getting_started/finetune_new_embodiment.md).

#### Prerequisites

Ensure your input data follows the **GR00T-flavored LeRobot v2 format**, and specify your modality configuration at `modality_config_path`.

#### Run Fine-tuning Script

Here is an example using the included SO100 demo data (`demo_data/cube_to_bowl_5`):

**Single GPU:**
```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/cube_to_bowl_5 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so100_config.py \
    --num-gpus 1 \
    --output-dir /tmp/test_finetune \
    --max-steps 2000 \
    --global-batch-size 32 \
    --dataloader-num-workers 4
```

**Multi-GPU (e.g., 8xH100):**
```bash
uv run torchrun --nproc_per_node=8 --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/cube_to_bowl_5 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so100_config.py \
    --num-gpus 8 \
    --output-dir /tmp/test_finetune_8gpu \
    --max-steps 2000 \
    --global-batch-size 32 \
    --dataloader-num-workers 4
```

Replace `demo_data/cube_to_bowl_5` and `examples/SO100/so100_config.py` with your own dataset and modality config. See [`examples/SO100`](examples/SO100/README.md) for a complete walkthrough.

> **Note:** Use `uv run torchrun` (not bare `torchrun`) to ensure the correct virtual environment is used. Add `--use-wandb` to enable Weights & Biases logging. For more extensive configuration, use `gr00t/experiment/launch_train.py`.

### Recommended Fine-tuning Configuration

For optimal results, maximize your batch size based on available hardware and train for a few thousand steps.

#### Hardware Performance Considerations

**Fine-tuning Performance**
- We recommend using 1 H100 node or L40 node for optimal finetuning performance
- Other hardware configurations (e.g., A6000) will also work but may require longer training time
- Optimal batch size depends on your hardware and which model components are being tuned

#### Training Variance

Users may observe some variance in post-training results across runs, even when using the same configuration, seed, and dropout settings. In our experiments, we have observed performance differences as large as 5-6% between runs. This variance may be attributed to non-deterministic operations in image augmentations or other stochastic components. When comparing results to reported benchmarks, please keep this inherent variance in mind.

## 4. Evaluation

We recommend a two-stage evaluation approach: open-loop evaluation followed by simulation evaluation to comprehensively assess model quality.

### 4.1 Open-Loop Evaluation

Open-loop evaluation provides an offline assessment by comparing the model's predicted actions against ground truth data from your dataset.

#### Running the Evaluation

Execute the evaluation script with your newly trained model:
```bash
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path <DATASET_PATH> \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path <CHECKPOINT_PATH> \
    --traj-ids 0 \
    --action-horizon 16  # ensure this is within the delta_indices of action's modality config.
```

#### Interpreting Results

The evaluation generates a visualization saved at `/tmp/open_loop_eval/traj_{traj_id}.jpeg`, which includes:
- Ground truth actions vs. predicted actions
- Unnormalized mean squared error (MSE) metrics

These plots provide a quick indicator of the policy's accuracy on the training dataset distribution.

### 4.2 Closed-Loop Evaluation

After validating performance through open-loop evaluation, test your model in closed-loop environments.

#### Understanding the Policy API

After training your model, you'll use the `Gr00tPolicy` class to load and run inference. The policy expects observations in a specific format (nested dictionaries with video, state, and language modalities) and returns actions ready for execution.

**Quick Start with Server-Client Architecture:**

```bash
# On GPU server: Start the policy server
uv run python gr00t/eval/run_gr00t_server.py \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path <CHECKPOINT_PATH> \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port 5555
```

```python
from gr00t.policy.server_client import PolicyClient

policy = PolicyClient(host="localhost", port=5555) # Connect to the policy server
env = YourEnvironment() # Create an environment
obs, info = env.reset() # Reset the environment
if not policy.ping(): # Verify connection
    raise RuntimeError("Cannot connect to policy server!")
action, info = policy.get_action(obs) # Run inference
obs, reward, done, truncated, info = env.step(action) # Execute the action
```

**Debugging with ReplayPolicy:**

When developing a new environment integration or debugging your inference loop, you can use `ReplayPolicy` to replay recorded actions from an existing dataset. This helps verify that your environment setup, observation formatting, and action execution work correctly—without needing a trained model.

```bash
# Start server with ReplayPolicy (replays actions from dataset)
uv run python gr00t/eval/run_gr00t_server.py \
    --dataset-path <DATASET_PATH> \
    --embodiment-tag NEW_EMBODIMENT \
    --execution-horizon 8  # should match the executed action horizon in the environment
```

The server will replay actions from the first episode of the dataset. Use `policy.reset(options={"episode_index": N})` on the client to switch to a different episode.

**For detailed documentation on:**
- How to adapt the policy to your own environment
- Server-client architecture for remote inference
- Observation and action formats
- Querying modality configurations
- Batched inference
- Troubleshooting common errors

See the complete [Policy API Guide](getting_started/policy.md).

#### Evaluation Examples

We support evaluation on available public benchmarks and our internal benchmarks. Our evaluation framework uses a server-client architecture that communicates via RESTful API. Both the policy server and simulation environment client use the same IP (usually localhost) and port to run simulation evaluation.

For the policy server, we reuse the project root's uv environment (same as finetuning) to run `run_gr00t_server`. For simulation environment clients, we provide individual setup scripts to configure uv environments, as they typically conflict with each other when using a single shared environment.

You can use [the verification script](scripts/eval/check_sim_eval_ready.py) to verify that all dependencies and environments for simulation evaluation are properly configured.

Please refer to each benchmark link below for more details.

#### Adding a New Sim Benchmark

Each sim benchmark registers its environments under a gym env_name with the format `{prefix}/{task_name}` (e.g., `libero_sim/LIVING_ROOM_SCENE2_put_soup_in_basket`). The evaluation framework uses the prefix to look up the corresponding `EmbodimentTag` via a mapping in [`gr00t/eval/sim/env_utils.py`](gr00t/eval/sim/env_utils.py).

> **Important:** The env_name prefix and the `EmbodimentTag` value are often different. For example, `sim_behavior_r1_pro` maps to `EmbodimentTag.BEHAVIOR_R1_PRO` (`"sim_behavior_r1_pro"`). Do not assume they match.

To add a new benchmark:

1. Add an entry to `ENV_PREFIX_TO_EMBODIMENT_TAG` in `gr00t/eval/sim/env_utils.py`:
   ```python
   ENV_PREFIX_TO_EMBODIMENT_TAG = {
       ...
       "my_new_benchmark": EmbodimentTag.MY_ROBOT,
   }
   ```
2. If the benchmark has multiple env_name prefixes (e.g., `my_benchmark_v1`, `my_benchmark_v2`), all related prefixes **must** map to the same `EmbodimentTag`.
3. Add corresponding test cases in `tests/gr00t/eval/sim/test_env_utils.py` and update the `test_all_known_prefixes_present` test.

**Zero-shot Evaluation** (evaluate without finetuning):
- **DROID**: [Instructions](examples/DROID/README.md)

**Finetuned Evaluation** (test after task-specific finetuning):
- **LIBERO**: [Instructions](examples/LIBERO/README.md)
- **SO-100**: [Instructions](examples/SO100/README.md)


# Contributions

During Early Access we are not accepting pull requests while the codebase stabilizes. If you encounter issues or have suggestions, please open an [Issue](https://github.com/NVIDIA/Isaac-GR00T/issues) in this repository.

# Support

Support during Early Access is best-effort. We will continue iterating toward a more stable General Availability (GA) release.


## License

- **Code:** Apache 2.0 — see [LICENSE](LICENSE)
- **Model weights:** [NVIDIA Software and Model Evaluation License](https://developer.nvidia.com/downloads/license/nvidia-software-model-evaluation-license)

```
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
```


## Citation

[Paper Site](https://research.nvidia.com/labs/lpr/publication/gr00tn1_2025/)
```bibtex
@inproceedings{gr00tn1_2025,
  archivePrefix = {arxiv},
  eprint     = {2503.14734},
  title      = {{GR00T} {N1}: An Open Foundation Model for Generalist Humanoid Robots},
  author     = {NVIDIA and Johan Bjorck and Fernando Castañeda, Nikita Cherniadev and Xingye Da and Runyu Ding and Linxi "Jim" Fan and Yu Fang and Dieter Fox and Fengyuan Hu and Spencer Huang and Joel Jang and Zhenyu Jiang and Jan Kautz and Kaushil Kundalia and Lawrence Lao and Zhiqi Li and Zongyu Lin and Kevin Lin and Guilin Liu and Edith Llontop and Loic Magne and Ajay Mandlekar and Avnish Narayan and Soroush Nasiriany and Scott Reed and You Liang Tan and Guanzhi Wang and Zu Wang and Jing Wang and Qi Wang and Jiannan Xiang and Yuqi Xie and Yinzhen Xu and Zhenjia Xu and Seonghyeon Ye and Zhiding Yu and Ao Zhang and Hao Zhang and Yizhou Zhao and Ruijie Zheng and Yuke Zhu},
  month      = {March},
  year       = {2025},
  booktitle  = {ArXiv Preprint},
}
```
