# Installation

Run the commands in this guide from the package root, the directory containing
`README.md` and `pyproject.toml`.

On x86_64 Linux and on Linux aarch64 GB300 / DGX Station, the supported
environment is the [GR00T container](../docker/README.md). It runs the NGC
PyTorch stack that GR00T's internal training uses, and it is the environment CI
tests. Jetson AGX Thor has its own supported [host setup](#jetson-thor).

| Platform and workflow | Setup | Continue with |
| --- | --- | --- |
| x86_64 Linux or GB300 / DGX Station: post-training, PyTorch evaluation, or policy serving | [Container](../docker/README.md) | [Minimal end-to-end example](minimal_example.md) or the README commands |
| x86_64 Linux or GB300 / DGX Station: TensorRT evaluation or policy serving | [Container](../docker/README.md) with the `tensorrt` extra | [TensorRT deployment guide](../scripts/deployment/README.md) |
| Jetson AGX Thor: TensorRT export, build, validation, and serving | [Thor](#jetson-thor) | [TensorRT deployment guide](../scripts/deployment/README.md) |

The training environment also supports PyTorch evaluation and policy serving.

## Host installation (best effort)

The x86 and GB300 sections below install the whole lock directly on a host,
without the container. This path is best effort: it installs the lock's own
PyTorch and CUDA builds, which differ from the container's, and CI does not test
it. Use it when you cannot run the container, for example with a driver older
than CUDA 13 (R580) requires.

## x86 prerequisites

- x86_64 Linux with an NVIDIA GPU and a current NVIDIA driver
- CUDA 12.8 development tools
- FFmpeg 4, 5, 6, or 7 with shared libraries
- Python 3.11 or 3.12
- Access to the model repositories referenced by your checkpoint or experiment;
  set `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or `HUGGINGFACE_TOKEN` when
  authentication is required
- Writable local storage for model caches and outputs

The full N2 training smoke test requires an H100-class GPU with at least 80 GB
of memory and at least 200 GB of free local disk. On 48 GB GPUs such as L40S,
training the full N2 model with the default AdamW optimizer runs out of memory.
Inference memory and storage requirements depend on the checkpoint and
workload.

For a new container, a known-good public starting point is
`nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`. An equivalent x86_64 Linux
environment with CUDA 12.8 development tools also works.

Apple Silicon Docker containers run as ARM64 Linux by default. They cannot run
this x86 CUDA workflow; use an x86_64 NVIDIA Linux host. GB300 and Jetson Thor
have separate AArch64/CUDA 13 setups described below.

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

## Straggler detection

Multi-GPU and multi-node training reports ranks that run slower than the others
through NVIDIA Resiliency Extension (NVRx), which the `training` extra installs.
The lock installs NVRx 0.4.1 on x86_64, built for CUDA 12, and NVRx 0.6.0 on
aarch64, built for CUDA 13 and requiring glibc 2.39 (see the
[GB300 prerequisites](#gb300-prerequisites)). Training logs a report every 60 seconds; disable it
with `--set training.straggler.enabled=false`. If a report's collective fails,
training stops with `Straggler report failed in a collective` instead of hanging;
rerun with detection disabled.

## GB300 prerequisites

Linux aarch64 GB300 and DGX Station use the same lock and the same commands as
x86_64. The lock selects the platform's stack:

| Platform | Locked training stack |
| --- | --- |
| x86_64 Linux | PyTorch 2.7.1+cu128, torchvision 0.22.1, torchcodec 0.5, FlashAttention 2.8.0.post2, TensorRT cu12 10.16.1.11 |
| Linux aarch64 (GB300, DGX Station) | PyTorch 2.11.0 CUDA 13, torchvision 0.26.0, torchcodec 0.15.0, Triton 3.6.0, FlashAttention 2.8.0.post2 (built from source), TensorRT cu13 10.16.1.11 |

Before installing:

- Use Ubuntu 24.04 or another distribution with glibc 2.39 or newer. The training
  extra fails to install on any older Linux aarch64 host, such as Ubuntu 22.04,
  because the only aarch64 NVRx wheel requires glibc 2.39.
- Install a current NVIDIA driver. Driver 595 open was verified on GB300
  (compute capability 10.3). Grace GPU memory must be onlined as movable
  before that driver loads, or `nvidia-smi` reports no device. Persist
  `memhp_default_state=online_movable` on the kernel command line.
- Install CUDA 13 `nvcc` so that `/usr/local/cuda` exists. uv compiles
  FlashAttention with it, since no aarch64 wheel is published, and DeepSpeed
  checks it on import. On Ubuntu 24.04 sbsa, `cuda-nvcc-13-0` from the NVIDIA
  CUDA apt repository provides that path. For a new container, a known-good
  public starting point is `nvidia/cuda:13.0.2-devel-ubuntu24.04`.
- Install the apt packages and uv from the [x86 prerequisites](#x86-prerequisites),
  including `python3-dev`. Triton JIT-compiles attention kernels and needs
  `Python.h`.

Not supported on Linux aarch64:

- Closed-loop RoboDojo. Its supported Isaac Sim version, **5.1.0**, runs on
  x86_64 Linux only; `isaacsim-kernel==5.1.0.0` publishes no aarch64 artifact.
- A validated TensorRT path for a real N2 checkpoint. GB300 CI builds small
  synthetic bf16 ONNX models into engines with
  `groot.deployment.n2_trt.tools.build_engines` and executes them through
  open-loop eval, CUDA graphs, the policy server, and the `closed-loop-eval`
  entrypoint with a fixture backend (no simulator). Exporting a real N2
  checkpoint to ONNX, building full-size engines, the numeric validation step,
  and the end-to-end `build-trt` / `--enable-trt` path are not yet validated.

## GB300 training

Complete the [GB300 prerequisites](#gb300-prerequisites), then install and
verify the locked training environment with the same commands as
[x86 training](#x86-training). `FLASH_ATTN_CUDA_ARCHS=100` builds only the
FlashAttention kernels GB300 runs; the build takes about six minutes on 72
Grace cores and is cached for later syncs. uv does not rebuild the cached
FlashAttention when `FLASH_ATTN_CUDA_ARCHS` changes, so after changing it run
`uv cache clean flash-attn` and sync with `--reinstall-package flash-attn`. If
the compiler runs out of memory, limit the parallel jobs with `MAX_JOBS`, for
example `MAX_JOBS=8`.

```bash
export FLASH_ATTN_CUDA_ARCHS=100
uv sync --frozen --extra training
uv run python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
uv run python -c 'import torchcodec; import torchcodec.decoders; print(torchcodec.__version__)'
uv run python -c 'import deepspeed, lance; print(deepspeed.__version__, lance.__version__)'
```

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
