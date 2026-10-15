# Installation

Run the commands in this guide from the package root, the directory containing
`README.md` and `pyproject.toml`. Choose the environment matching your platform
and workflow:

| Platform and workflow | Setup | Continue with |
| --- | --- | --- |
| x86_64 Linux: post-training or the full smoke test | [Training](#x86-training) | [Minimal end-to-end example](minimal_example.md) or the README training commands |
| x86_64 Linux: PyTorch evaluation or policy serving | [Inference](#x86-inference) | The README evaluation and serving commands |
| x86_64 Linux: TensorRT evaluation or policy serving | [Optional TensorRT](#optional-tensorrt-on-x86) | [TensorRT deployment guide](../scripts/deployment/README.md) |
| Jetson AGX Thor: TensorRT export, build, validation, and serving | [Thor](#jetson-thor) | [TensorRT deployment guide](../scripts/deployment/README.md) |

The training environment also supports PyTorch evaluation and policy serving.
Install one environment for your workflow; a training user does not need to
install the separate inference extra.

## x86 prerequisites

- x86_64 Linux with an NVIDIA GPU and a current NVIDIA driver
- CUDA 12.8 development tools
- FFmpeg 4, 5, 6, or 7 with shared libraries
- Python 3.10, 3.11, or 3.12
- Access to the model repositories referenced by your checkpoint or experiment;
  set `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or `HUGGINGFACE_TOKEN` when
  authentication is required
- Writable local storage for model caches and outputs

The full N2 training smoke test requires an H100-class GPU with at least 80 GB
of memory and at least 200 GB of free local disk. Inference memory and storage
requirements depend on the checkpoint and workload.

For a new container, a known-good public starting point is
`nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`. An equivalent x86_64 Linux
environment with CUDA 12.8 development tools also works. To build the training
environment below as an image from that base, see
[docker/README.md](../docker/README.md); the frozen uv environment remains the
software contract.

Apple Silicon Docker containers run as ARM64 Linux by default. They cannot run
this x86 CUDA workflow; use an x86_64 NVIDIA Linux host. Jetson Thor has a
separate AArch64/CUDA 13 environment described below.

On Ubuntu or Debian, install the build tools and shared libraries needed by the
locked packages. FlashAttention and, for training, DeepSpeed are built from
source during installation.

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates curl git python3-dev python3-venv \
  ffmpeg yasm cmake libtool pkg-config libass-dev libfreetype6-dev \
  libvorbis-dev autoconf automake texinfo
ffmpeg -version
```

When running as `root`, omit `sudo`. Install uv if it is not already available,
then make it available in the current shell:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Use the locked packages rather than a `--system-site-packages` environment or
manual package replacements. The project configures uv to build FlashAttention
against the matching PyTorch version and install CUDA-enabled TorchCodec.
Sync the selected extras before running workflow commands. `uv run` performs
inexact syncing by default and retains installed extra packages.

## x86 training

Complete the [x86 prerequisites](#x86-prerequisites), then install the locked
training environment:

```bash
uv sync --frozen --extra training
```

Verify GPU access and the compiled and data-loading dependencies:

```bash
uv run python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
uv run python -c 'import torchcodec; import torchcodec.decoders; print(torchcodec.__version__)'
uv run python -c 'import deepspeed, lance; print(deepspeed.__version__, lance.__version__)'
```

Continue with the [minimal end-to-end example](minimal_example.md) to prepare
the sample dataset and verify training, evaluation, and serving, or use the
[README](../README.md#agent-or-cli-quickstart) to post-train on your own dataset.

## x86 inference

Complete the [x86 prerequisites](#x86-prerequisites), then install the locked
inference environment for evaluation or policy serving:

```bash
uv sync --frozen --extra deployment
```

Verify GPU access and video decoding:

```bash
uv run python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
uv run python -c 'import torchcodec; import torchcodec.decoders; print(torchcodec.__version__)'
```

Use the [README](../README.md#agent-or-cli-quickstart) evaluation or policy-server
commands with an existing checkpoint. Open-loop evaluation also requires a
local dataset; serving does not.

## Optional TensorRT on x86

Choose the command matching your existing environment. Include its original
extra so syncing retains the training or inference dependencies:

```bash
# Inference only
uv sync --frozen --extra deployment --extra tensorrt

# Training followed by accelerated evaluation or serving
uv sync --frozen --extra training --extra tensorrt
```

Verify TensorRT:

```bash
uv run python -c 'import torch, tensorrt; print(torch.__version__, tensorrt.__version__, torch.cuda.get_device_name())'
```

Add `--enable-trt` to evaluation, train-eval, or policy-server commands to build
and validate a checkpoint-specific bundle on the target GPU, then reuse it.
Automatic bundles require four denoising steps. The minimal smoke test uses
two denoising steps and PyTorch inference; follow its commands as written.

TensorRT also requires a compatible checkpoint contract. Unsupported contracts
or failed validation stop the command without a PyTorch fallback. See the
[deployment guide](../scripts/deployment/README.md) for qualified configurations,
checkpoint constraints, bundle caching, and explicit engine builds. TensorRT
accelerates inference; training continues to use PyTorch.

## Jetson Thor

The supported Thor workflow is TensorRT export, build, validation, and serving
on bare metal. It requires Ubuntu 24.04 on AArch64, Python 3.12, and the CUDA 13
toolkit and driver. Run `sudo -v` first if sudo requires a password.

```bash
# One-time install: validates AArch64/Python 3.12 and creates the locked CUDA 13 environment
bash scripts/deployment/thor/install_deps.sh

# In each new shell
source scripts/deployment/thor/activate_thor.sh
```

Activation selects the Thor project with `UV_PROJECT` and its project-local
`.venv`. Both plain `python` and `uv run` use the same environment. Verify it
before continuing with the [deployment guide](../scripts/deployment/README.md):

```bash
uv run python -c 'import torch, tensorrt; print(torch.__version__, tensorrt.__version__, torch.cuda.get_device_name())'
```
