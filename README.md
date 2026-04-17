<div align="center">

  <img src="media/header_compress.png" width="800" alt="NVIDIA Isaac GR00T N1.7 Header">

  <!-- --- -->

  <p style="font-size: 1.2em;">
    <a href="https://developer.nvidia.com/isaac/gr00t"><strong>Website</strong></a> |
    <a href="https://huggingface.co/collections/nvidia/gr00t-n17"><strong>Model</strong></a> |
    <a href="https://huggingface.co/collections/nvidia/physical-ai"><strong>Dataset</strong></a> |
    <a href="https://arxiv.org/abs/2503.14734"><strong>Paper</strong></a> |
    <a href="https://developer.nvidia.com/isaac"><strong>NVIDIA Isaac</strong></a> |
    <a href="#gr00t-n17-faq"><strong>FAQ</strong></a>
  </p>
</div>

## Table of Contents

- [NVIDIA Isaac GR00T](#nvidia-isaac-gr00t)
- [What's New in GR00T N1.7](#whats-new-in-gr00t-n17)
- [Installation](#installation)
- [Model Checkpoints & Embodiment Tags](#model-checkpoints--embodiment-tags)
- [Data Format](#data-format)
- [Inference](#inference)
- [Fine-tuning](#fine-tuning)
- [Evaluation](#evaluation)
- [FAQ](#gr00t-n17-faq)
- [Contributions](#contributions)
- [License](#license)
- [Citation](#citation)

---

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

> We just released GR00T N1.7 Early Access, the latest version of GR00T N1 with a new VLM backbone (Cosmos-Reason2-2B / Qwen3-VL) and improved performance.

> **This is an Early Access (EA) release.** You are welcome to download the model, explore the codebase, and begin building on the stack, with the understanding that support and stability guarantees are limited until the GA release.
>
> **What's available:**
> - Pre-trained GR00T N1.7 model weights and reference code
> - Fine-tuning and inference with custom robot data or demonstrations
> - Experimentation, prototyping, and research use cases
>
> **Available at GA:**
> - Production deployment with commercial support
> - Complete benchmarks and a fully validated, stable feature set
> - Pull request contributions
>
> We welcome feedback - please feel free to raise issues in this repository.

> To use older versions: [N1.6](https://github.com/NVIDIA/Isaac-GR00T/releases/tag/n1.6-release) | [N1.5](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release)

NVIDIA Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot skills. This cross-embodiment model takes multimodal input, including language and images, to perform manipulation tasks in diverse environments.

GR00T N1.7 is trained on a diverse mixture of robot data including bimanual, semi-humanoid and an expansive humanoid dataset. It is adaptable through post-training for specific embodiments, tasks and environments.

GR00T N1.7 is fully commercially licensable under Apache 2.0. It delivers comparable performance to N1.6, with improved generalization and language-following capabilities driven by the inclusion of 20K hours of EgoScale human video data in pretraining.

The neural network architecture of GR00T N1.7 is a combination of vision-language foundation model and diffusion transformer head that denoises continuous actions. Here is a schematic diagram of the architecture:

<div align="center">
<img src="media/model-architecture.png" width="800" alt="model-architecture">
</div>

### Workflow Overview

1. **Prepare data** — Collect robot demonstrations (video, state, action) and convert them to the [GR00T LeRobot format](#data-format). Demo datasets are included for quick testing.
2. **Run inference** — Try zero-shot inference with the base model on [pretrain embodiments](#embodiment-tags), or use a [finetuned checkpoint](#checkpoints) for benchmark tasks.
3. **Fine-tune** — Adapt the model to your robot using [`launch_finetune.py`](#fine-tuning) with your own data and modality config.
4. **Evaluate** — Validate with [open-loop evaluation](#open-loop-evaluation), then test in [simulation benchmarks](#benchmark-examples) or on real hardware via the [Policy API](getting_started/policy.md).
5. **Deploy** — Connect `Gr00tPolicy` to your robot controller, optionally accelerated with [TensorRT](scripts/deployment/README.md).

## What's New in GR00T N1.7

GR00T N1.7 builds on N1.6 with a new VLM backbone and code-level improvements.

### Key Changes from N1.6

- **New VLM backbone:** Cosmos-Reason2-2B (Qwen3-VL architecture), replacing the Eagle backbone used in N1.6. Supports flexible resolution and encodes images in their native aspect ratio without padding.
- Simplified data processing pipeline (`processing_gr00t_n1d7.py`).
- Added full pipeline export to ONNX and TensorRT with improved frequency.

---

## Installation

### Hardware Requirements

**Inference:** 1 GPU with 16 GB+ VRAM (e.g., RTX 4090, L40, H100, Jetson AGX Thor/Orin, DGX Spark).

**Fine-tuning:** 1 or more GPUs with 40 GB+ VRAM recommended. We recommend H100 or L40 nodes for optimal performance. Other hardware (e.g., A6000) works but may require longer training time. See the [Hardware Recommendation Guide](getting_started/hardware_recommendation.md) for detailed specs.

### Clone the Repository

GR00T relies on submodules for certain dependencies. Include them when cloning:

**Note:** `git-lfs` is **required** to download parquet data files in `/demo_data`. Install it before cloning: `sudo apt install git-lfs && git lfs install`.
```sh
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```

If you've already cloned without submodules, initialize them separately:

```sh
git submodule update --init --recursive
```

### Set Up the Environment

GR00T uses [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management. Install uv first:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### dGPU (x86_64) — Default

Install FFmpeg (required by `torchcodec`, the default video backend):
```sh
sudo apt-get update && sudo apt-get install -y ffmpeg
```

Create the environment and install GR00T:
```sh
uv sync --python 3.10
```
GPU dependencies (flash-attn, TensorRT, etc.) are included in the default install.

Verify the installation:
```sh
uv run python -c "import gr00t; print('GR00T installed successfully')"
```

> **`flash-attn` message on every `uv run`:** You may see `Installing flash-attn...` each time you run `uv run`. This is a known `uv` behavior with URL-pinned wheel sources — `uv` re-validates the cached wheel against the source URL on each invocation. It is **not** rebuilding from source; the wheel is already cached locally and the operation takes 2-3 seconds. This only affects x86_64 platforms. 
> To suppress it, remove the `flash-attn` entries under `[tool.uv.sources]` in your local `pyproject.toml` after the initial install. But that will break `uv lock` and cause flash-attn to build from source on next lock regeneration.

<details>
<summary><strong>Alternative: pip install (without uv)</strong></summary>

If you prefer pip/conda over uv, create a Python 3.10 virtualenv and install:
```sh
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e .
```
Note: GPU dependencies (flash-attn, TensorRT) may require manual installation with pip. The `uv` workflow handles these automatically.
</details>

> **If fine-tuning fails with `CUDA_HOME is unset`:** Run `bash scripts/deployment/dgpu/install_deps.sh` once to configure CUDA paths, or manually `export CUDA_HOME=/usr/local/cuda`.

> **CUDA 13.x Users (Thor, Spark, and other CUDA 13+ platforms):** PyTorch 2.7 pins Triton to 3.3.1, which does not recognize CUDA major version 13+. This causes a `RuntimeError` in Triton's `ptx_get_version()`. Run the patch script to fix:
> ```sh
> uv run bash scripts/patch_triton_cuda13.sh
> ```

> **GB300 (sm_103) Users:** Triton 3.3.1 (pinned by PyTorch 2.7) does not support the GB300 GPU architecture (sm_103). `torch.compile` will fail on GB300. Use PyTorch eager mode or TensorRT inference instead. Triton 3.5.1+ adds sm_103 support but is not yet compatible with the pinned PyTorch version.

> **aarch64 Video Backend:** On aarch64 platforms (Thor, Orin), `torchcodec` is the required video backend. Pre-built wheels are not available for aarch64, so it is built from source during `install_deps.sh`. If you encounter `NotImplementedError` from the video backend, ensure `torchcodec` was built successfully during setup. Other backends (decord, pyav) are not supported on aarch64.

<details>
<summary><strong>DGX Spark</strong> (tested with DGX Spark GB10)</summary>

```bash
bash scripts/deployment/spark/install_deps.sh
source .venv/bin/activate
source scripts/activate_spark.sh
```

See the [Spark setup guide](scripts/deployment/README.md#dgx-spark-setup) for Docker and bare metal details.
</details>

<details>
<summary><strong>Jetson AGX Thor</strong> (tested with JetPack 7.1)</summary>

> **flash-attn on older systems (e.g., Ubuntu 20.04 with glibc < 2.35):** The pre-built `flash-attn` wheel may fail with `ImportError: glibc_compat.so: cannot open shared object file`. To fix this, build from source:
> ```sh
> uv pip install flash-attn==2.7.4.post1 --no-binary flash-attn --no-cache
> ```
> This compiles locally (~10-30 minutes) and avoids the glibc compatibility issue.

```bash
bash scripts/deployment/thor/install_deps.sh
source .venv/bin/activate
source scripts/activate_thor.sh
```

See the [Thor setup guide](scripts/deployment/README.md#jetson-thor-setup) for Docker and bare metal details.
</details>

<details>
<summary><strong>Jetson Orin</strong> (tested with JetPack 6.2)</summary>

```bash
bash scripts/deployment/orin/install_deps.sh
source .venv/bin/activate
source scripts/activate_orin.sh
```

See the [Orin setup guide](scripts/deployment/README.md#jetson-orin-setup) for Docker and bare metal details.
</details>

For a containerized setup that avoids system-level dependency conflicts, see our [Docker Setup Guide](docker/README.md).

---

## Model Checkpoints & Embodiment Tags

### Checkpoints

| Checkpoint | Type | Embodiment Tag | Description |
|------------|------|---------------|-------------|
| [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) | Base | See [pretrain tags](getting_started/policy.md#--embodiment-tag) | Base model (3B params) — zero-shot inference on pretrain embodiments, or finetune for new tasks |
| [`nvidia/GR00T-N1.7-LIBERO`](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) | Finetuned | `LIBERO_PANDA` | Finetuned on [LIBERO](https://libero-project.github.io/) benchmark (Franka Panda) |
| [`nvidia/GR00T-N1.7-DROID`](https://huggingface.co/nvidia/GR00T-N1.7-DROID) | Finetuned | `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` | Finetuned on [DROID](https://droid-dataset.github.io/) dataset |
| [`nvidia/GR00T-N1.7-SimplerEnv-Bridge`](https://huggingface.co/nvidia/GR00T-N1.7-SimplerEnv-Bridge) | Finetuned | `SIMPLER_ENV_WIDOWX` | Finetuned on SimplerEnv Bridge (WidowX) |
| [`nvidia/GR00T-N1.7-SimplerEnv-Fractal`](https://huggingface.co/nvidia/GR00T-N1.7-SimplerEnv-Fractal) | Finetuned | `SIMPLER_ENV_GOOGLE` | Finetuned on SimplerEnv Fractal (Google Robot) |

> Older versions: [N1.6 checkpoints](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.6-release) | [N1.5 checkpoints](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release)

### Embodiment Tags

Every inference or finetuning command requires an `--embodiment-tag`. The tag determines which modality config (state/action keys, normalization) the model uses. Tags are **case-insensitive**.

For the full list of pretrain and posttrain tags, see the [Policy API Guide — Embodiment Tags](getting_started/policy.md#--embodiment-tag).

---

## Data Format

GR00T uses a flavor of the [LeRobot v2 dataset format](https://github.com/huggingface/lerobot) with an additional `meta/modality.json` file that describes state/action/video structure. A dataset looks like:

```
my_dataset/
  meta/
    info.json            # dataset metadata
    episodes.jsonl       # episode index and lengths
    tasks.jsonl          # language task descriptions
    modality.json        # state/action/video key mapping (GR00T-specific)
  data/chunk-000/        # parquet files (state, action per timestep)
  videos/chunk-000/      # mp4 video files per episode
```

The `modality.json` maps how the concatenated state/action arrays split into named fields (e.g., `x`, `y`, `z`, `gripper`) and which video keys are available. This is what the embodiment tag uses to interpret the data.

**Included demo datasets** (ready to use, no download needed):

| Dataset | Robot | Embodiment Tag | Use Case |
|---------|-------|---------------|----------|
| `demo_data/droid_sample` | DROID (3 episodes) | `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` | Zero-shot inference with base model |
| `demo_data/libero_demo` | LIBERO Panda (5 episodes) | `LIBERO_PANDA` | Inference with finetuned checkpoint |
| `demo_data/cube_to_bowl_5` | SO100 arm (5 episodes) | `NEW_EMBODIMENT` | Fine-tuning custom embodiment example |

> To generate more DROID episodes: `python scripts/download_droid_sample.py --num-episodes 10`

**Using your own data:** Convert your demonstrations to the format above. If coming from LeRobot v3, use the conversion script: `python scripts/lerobot_conversion/convert_v3_to_v2.py`. See the full [Data Preparation Guide](getting_started/data_preparation.md) for schema details and examples.

---

## Inference

### Zero-Shot Inference (Base Model)

The included `demo_data/droid_sample` dataset works with the base model out of the box — no finetuning or checkpoint download needed:

```bash
uv run python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 1 2 \
    --inference-mode pytorch \
    --action-horizon 8
```

This runs open-loop inference on 2 DROID episodes, comparing predicted actions against ground truth. The base model downloads automatically from HuggingFace on first run (~6 GB).

### Finetuned Inference

For posttrain embodiments, use a finetuned checkpoint. Most finetuned checkpoints (e.g., DROID, SimplerEnv) have a flat file structure and can be passed directly as a HuggingFace model ID — no manual download needed:

```bash
uv run python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.7-DROID \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 1 2 \
    --inference-mode pytorch \
    --action-horizon 8
```

Some checkpoints (e.g., LIBERO) use a nested folder structure with model files under a subfolder. HuggingFace does not support nested repo paths in `--model-path`, so you must download first:

```bash
uv run hf download nvidia/GR00T-N1.7-LIBERO \
    --include "libero_10/config.json" "libero_10/embodiment_id.json" \
    "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" \
    "libero_10/processor_config.json" "libero_10/statistics.json" \
    --local-dir checkpoints/GR00T-N1.7-LIBERO
```

```bash
uv run python scripts/deployment/standalone_inference_script.py \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --dataset-path demo_data/libero_demo \
    --embodiment-tag LIBERO_PANDA \
    --traj-ids 0 1 2 \
    --inference-mode pytorch \
    --action-horizon 8
```

### Server-Client Inference (for Deployment)

For real-world deployment or simulation evaluation, use the server-client architecture. The policy runs on a GPU server; a lightweight client sends observations and receives actions over ZMQ.

**Terminal 1 — Start the policy server:**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --device cuda:0
```

**Terminal 2 — Run open-loop evaluation as a client:**
```bash
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --host 127.0.0.1 \
    --port 5555 \
    --traj-ids 1 2 \
    --action-horizon 8
```

> **Tip:** If you get `ZMQError: Address already in use`, the default port 5555 is occupied. Use `--port <other_port>`.

For connecting to a real robot (e.g., DROID hardware), see [examples/DROID/README.md](examples/DROID/README.md). For faster inference with TensorRT, see the [Deployment & Inference Guide](scripts/deployment/README.md).

See the complete [Policy API Guide](getting_started/policy.md) for documentation on observation/action formats, batched inference, and troubleshooting.

---

## Fine-tuning

### Reproducing Benchmark Results

Each benchmark has a self-contained README with dataset download, finetune, and evaluation commands:

| Benchmark | Embodiment | Guide |
|-----------|-----------|-------|
| LIBERO | `LIBERO_PANDA` | [examples/LIBERO/README.md](examples/LIBERO/README.md) |
| SimplerEnv (Fractal) | `SIMPLER_ENV_GOOGLE` | [examples/SimplerEnv/README.md](examples/SimplerEnv/README.md) |
| SimplerEnv (Bridge) | `SIMPLER_ENV_WIDOWX` | [examples/SimplerEnv/README.md](examples/SimplerEnv/README.md) |
| SO100 | `NEW_EMBODIMENT` | [examples/SO100/README.md](examples/SO100/README.md) |

### Fine-tune on Your Own Robot ("NEW_EMBODIMENT")

To finetune GR00T on your own robot data and configuration, follow the detailed tutorial at [`getting_started/finetune_new_embodiment.md`](getting_started/finetune_new_embodiment.md).

Ensure your input data follows the [GR00T LeRobot format](#data-format), and specify your modality configuration via `--modality-config-path`.

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

### Training Tips

- Maximize batch size for your hardware and train for a few thousand steps.
- Users may observe 5-6% variance between runs due to non-deterministic image augmentations. Keep this in mind when comparing to reported benchmarks.
- **`--state_dropout_prob`** (default: 0.8 in model config, 0.0 in finetune CLI): Randomly drops state inputs during training to improve generalization and reduce state-dependency. The LIBERO and SimplerEnv finetune scripts set this to 0.8. If your task relies heavily on proprioceptive state, consider lowering this value.

---

## Evaluation

### Open-Loop Evaluation

Compare predicted actions against ground truth from your dataset:

```bash
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path <DATASET_PATH> \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path <CHECKPOINT_PATH> \
    --traj-ids 0 \
    --action-horizon 16
```

This generates a visualization at `/tmp/open_loop_eval/traj_{traj_id}.jpeg` with ground truth vs. predicted actions and MSE metrics. Use `--save-plot-path <dir>` to save plots to a custom location.

### Closed-Loop Evaluation

Test your model in simulation or on real hardware using the server-client architecture:

```bash
# Start the policy server
uv run python gr00t/eval/run_gr00t_server.py \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path <CHECKPOINT_PATH> \
    --device cuda:0 \
    --host 0.0.0.0 --port 5555
```

```python
from gr00t.policy.server_client import PolicyClient

policy = PolicyClient(host="localhost", port=5555)
env = YourEnvironment()
obs, info = env.reset()
action, info = policy.get_action(obs)
obs, reward, done, truncated, info = env.step(action)
```

**Debugging with ReplayPolicy:** To verify your environment setup without a trained model, start the server with `--dataset-path <DATASET_PATH>` (omit `--model-path`) to replay recorded actions from the dataset.

See the complete [Policy API Guide](getting_started/policy.md) for observation/action formats, batched inference, and troubleshooting.

### Benchmark Examples

We support evaluation on public benchmarks using a server-client architecture. The policy server reuses the project root's uv environment; simulation clients have individual setup scripts.

You can use [the verification script](scripts/eval/check_sim_eval_ready.py) to verify that all dependencies are properly configured.

**Zero-shot** (evaluate with the base model, no finetuning):
- [DROID](examples/DROID/README.md) — real-world DROID robot

**Finetuned** (evaluate with finetuned checkpoints):
- [LIBERO](examples/LIBERO/README.md) — LIBERO benchmark (Franka Panda)
- [SimplerEnv](examples/SimplerEnv/README.md) — Google Robot (Fractal) and WidowX (Bridge)
- [SO100](examples/SO100/README.md) — SO100 custom embodiment workflow

<details>
<summary><strong>Adding a New Sim Benchmark</strong></summary>

Each sim benchmark registers its environments under a gym env_name with the format `{prefix}/{task_name}` (e.g., `libero_sim/LIVING_ROOM_SCENE2_put_soup_in_basket`). The evaluation framework uses the prefix to look up the corresponding `EmbodimentTag` via a mapping in [`gr00t/eval/sim/env_utils.py`](gr00t/eval/sim/env_utils.py).

> **Important:** The env_name prefix and the `EmbodimentTag` value are often different. For example, `libero_sim` maps to `EmbodimentTag.LIBERO_PANDA` (`"libero_sim"`). Do not assume they match.

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
</details>



# GR00T N1.7 FAQ

## Infrastructure & Hardware

### Is the data loader GPU-accelerated?

No, the current data loader is CPU-based. However, it has been heavily optimized for multimodal data to ensure it does not become a training bottleneck. We validated this on various configurations, including GB200, H100, and local desktops with RTX 4090 GPUs. We are actively exploring GPU-accelerated approaches for future releases.

### Is the same data loader used for both pre-training and post-training?

Yes, the data loading pipeline is unified across both training stages.

### What is the role of the Policy Remote Server in the deployment diagram?

The Policy Remote Server decouples inference from the physical robot. This allows users to run the policy on a high-compute cluster (e.g., H100s) for faster inference while the robot operates in a separate environment. It separates dependencies and enables scaling beyond the robot's onboard compute.

## Workflow & Architecture

### How is the Whole-Body Control (WBC) stack used with GR00T N models?

The WBC and the GR00T VLA policy are trained independently and coupled at deployment. The typical workflow for integrating them involves four key steps:

1. **Train WBC**: Train a low-level controller for the specific robot hardware.
2. **Collect Data**: Use the trained WBC to collect "loco-manipulation" data.
3. **Fine-tune**: Train the GR00T model on this data.
4. **Deploy**: Run the fine-tuned GR00T policy, which outputs commands to the WBC via an adapter interface.

See the [Loco-Manipulation workflow example](https://github.com/NVlabs/GR00T-WholeBodyControl) for details.

### Why retain only specific LLM layers (e.g., 16 layers) during fine-tuning?

This configuration was empirically tuned for the backbone (e.g., Eagle/Cosmos-Reason). Research suggests early layers capture grammatical structure, while middle-to-late layers are highly expressive. However, the very last layers are often over-optimized for next-token prediction; pruning or freezing them can sometimes yield better representations for vision-language-action alignment.

### How do you verify if the language model is successfully aligned with the action space?

We evaluate this end-to-end via downstream task success. We design evaluation tasks that are ambiguous without language instructions (e.g., "pick the pear" from a bowl of mixed fruit). If the robot succeeds, it confirms the model is correctly grounding language commands into physical actions.

## Data Strategy & Volume

### How much data is required for post-training on a new embodiment or task?

Data requirements depend heavily on task complexity and scene variation. Typical guidelines include:

- **Simple, fixed-location tasks (Pick & Place):** ~100 trajectories.
- **Complex scenes or multi-step tasks:** ~500+ trajectories.
- **Whole-Body Control (High DoF):** ~2,000+ trajectories (e.g., shelf-picking with G1).
- **Fine manipulation:** ~100–500 episodes, ideally with human motion pre-training.

### What is the recommended strategy for improving success rates on hard tasks?

We recommend an iterative approach: start with ~100 teleoperated demonstrations, train a policy, and then use HG-DAgger (Human Gated Dataset Aggregation). Run the policy, intervene when it fails, and add the corrections from those trajectories to the dataset. This helps the model cover out-of-distribution states that pure behavior cloning (BC) might miss, and recover from partial failure states (e.g., a grip slipping or imprecise item placement).

### Does including real-robot data from other embodiments help if I only care about one robot?

Yes. Even if cross-embodiment generalization is not your goal, including diverse real-robot data adds visual diversity and robustness to the VLA's backbone, improving performance on your specific target robot.

### Does GR00T N1.7 support synthetic data generation via Cosmos?

While research models (like DreamGen) show promise, a robust, product-ready pipeline for generating synthetic training data via Cosmos is currently in development and not yet part of the standard release.

## Model Capabilities

### Can the model handle lighting changes or different object colors?

VLMs can struggle with drastic appearance changes (e.g., hard shadows or significant hue shifts). While we haven't released specific lighting ablations, we strongly recommend using color jitter augmentation during training and collecting diverse data (20–50 episodes) under different lighting conditions to prevent overfitting.

### Can GR00T models perform reasoning or Visual Question Answering (VQA)?

The GR00T N1.x series is optimized specifically for action generation, not open-ended reasoning or VQA. Capabilities requiring complex semantic reasoning are targeted for future N2 releases.

### Can the model learn "retry" behaviors?

The current architecture is stateless and does not inherently "know" if a previous attempt failed. While some retry behavior may emerge from high-quality data, explicit recovery strategies are best achieved through DAgger (collecting data on recovery from failure) or Reinforcement Learning (RL), rather than pure Imitation Learning.

### Does the model distinguish between left and right arms in bimanual tasks?

Yes, provided the training data is distinct or annotated (e.g., instructions specifying "left arm" vs. "right arm"). If the dataset contains mixed, unannotated data where both arms perform identical tasks indiscriminately, the model may struggle to distinguish them.

### Is there a zero-shot cross-embodiment VLA model?

No. While cross-embodiment data improves generalization, a true "zero-shot" model (one that works perfectly on a new robot without *any* fine-tuning) does not currently exist in the open VLA landscape.

### Will differences in object shape between training and deployment cause the success rate to drop?

It depends on the degree of deviation. If the target object's shape differs drastically from the training data, performance will likely drop significantly. However, if the shape variation is minor and shares a similar grasping affordance (e.g., a slightly different bottle shape that is still grasped from the side), the model may still succeed, though with potentially lower reliability than on the original objects.

### Has the impact of large viewpoint changes (e.g., head movement) on task difficulty been studied?

Yes. Large viewpoint changes effectively change the observation distribution, which can complicate simple tasks. For example, a "simple" handover becomes complex if the robot's head moves significantly, altering the camera's perspective of its own hands.

- **Current Status:** Most public GR00T demos feature a relatively fixed head position to stabilize observations.
- **Mitigation:** To handle natural head movement, we recommend training with aggressive camera pose augmentation or collecting data that explicitly includes head motion to ensure the policy becomes robust to viewpoint shifts.



---

# Contributions

During Early Access we are not accepting pull requests while the codebase stabilizes. If you encounter issues or have suggestions, please open an [Issue](https://github.com/NVIDIA/Isaac-GR00T/issues) in this repository.

# Support

Support during Early Access is best-effort. We will continue iterating toward a more stable General Availability (GA) release.


## License

- **Code:** Apache 2.0 — see [LICENSE](LICENSE)
- **Model weights:** [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)

```
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

