# Docker Setup for Isaac GR00T 2

`docker/Dockerfile` builds the [x86 training environment](../docs/installation.md#x86-training)
or the [GB300 training environment](../docs/installation.md#gb300-training) as
an image. It starts from the documented public base for the target
architecture, installs the documented system packages, and runs
`uv sync --frozen --extra training` into `/opt/gr00t-venv`. The frozen uv
environment remains the software contract; the image only records the
installation steps.

| Architecture | Base image |
| --- | --- |
| x86_64 (`linux/amd64`) | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` |
| Linux aarch64 GB300 / DGX Station (`linux/arm64`) | `nvidia/cuda:13.0.2-devel-ubuntu24.04` |

Jetson Thor uses the bare-metal [Thor setup](../docs/installation.md#jetson-thor).

## Prerequisites

- x86_64 Linux, or Linux aarch64 on GB300 / DGX Station, with an NVIDIA GPU and
  a current NVIDIA driver. GB300 hosts also need the driver and memory settings
  in the [GB300 prerequisites](../docs/installation.md#gb300-prerequisites).
- Docker 23 or newer, with BuildKit
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- About 30 GB of free disk for the image, plus the model cache and outputs
  described in the [installation guide](../docs/installation.md#x86-prerequisites)

## Build

From the package root:

```bash
bash docker/build.sh
```

This builds for the host architecture and tags the image `gr00t-n2`; set
`GR00T_IMAGE` to choose another tag. To build for the other architecture, set
`GR00T_PLATFORM` to `linux/amd64` or `linux/arm64`; Docker then builds under
QEMU emulation, which is very slow. Extra arguments are passed to
`docker build`. To include other extras, override `UV_EXTRAS`:

```bash
bash docker/build.sh --build-arg UV_EXTRAS="--extra training --extra tensorrt"
```

The image contains no source checkout. Its build context is limited to
`pyproject.toml` and `uv.lock`.

The build needs internet access: it pulls the BuildKit Dockerfile frontend, the
CUDA base image, and the uv image, and downloads Ubuntu packages and the locked
Python packages, including the prebuilt FlashAttention wheel from GitHub on
x86_64. On arm64 the build compiles FlashAttention for GB300 instead; for other
arm64 GPUs, pass their architecture, for example
`--build-arg FLASH_ATTN_CUDA_ARCHS=90` for GH200.

## Run

From the package root, mount the checkout and a directory for datasets,
caches, and outputs:

```bash
export GROOT2_HOME="${GROOT2_HOME:-$HOME/gr00t2}"
mkdir -p "$GROOT2_HOME"

docker run -it --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN \
  -e GROOT2_HOME=/workspace/gr00t2 \
  -v "$GROOT2_HOME:/workspace/gr00t2" \
  -v "$PWD:/workspace/gr00t" \
  -w /workspace/gr00t \
  gr00t-n2
```

The container runs as `root`, so files it writes to the mounted directories are
owned by `root`.

To reach a policy server in the container from a client outside it, publish
the server port. The policy server in the minimal example listens on
`0.0.0.0:5555`, so start the container with `-p 5555:5555`:

```bash
docker run -it --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 5555:5555 \
  -e HF_TOKEN \
  -e GROOT2_HOME=/workspace/gr00t2 \
  -v "$GROOT2_HOME:/workspace/gr00t2" \
  -v "$PWD:/workspace/gr00t" \
  -w /workspace/gr00t \
  gr00t-n2
```

Match the published port to the server's `--port`. Training, evaluation, and a
client running in the same container need no published port.

Inside the container, the image environment is active, and `uv run` uses it
because `UV_PROJECT_ENVIRONMENT=/opt/gr00t-venv`. Verify it with the same
checks as the installation guide:

```bash
uv run python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
uv run python -c 'import torchcodec; import torchcodec.decoders; print(torchcodec.__version__)'
uv run python -c 'import deepspeed, lance; print(deepspeed.__version__, lance.__version__)'
```

Then continue with the [minimal end-to-end example](../docs/minimal_example.md)
from step 2, or the [README](../README.md#agent-or-cli-quickstart) commands.

## Matching the checkout

The image records the hashes of the `pyproject.toml` and `uv.lock` it was
built from. Check that the mounted checkout uses the same files:

```bash
(cd /workspace/gr00t && sha256sum -c /opt/gr00t-venv/.gr00t-project.sha256)
```

**If the check fails, rebuild the image from the checkout before running
`uv run`.** Otherwise `uv run` silently syncs the image environment to the
checkout's lock.

## Troubleshooting

- GPU not visible: check the NVIDIA Container Toolkit with
  `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`
  on x86_64, or `nvidia/cuda:13.0.2-base-ubuntu24.04` on GB300.
- Apple Silicon and other ARM64 hosts without an NVIDIA GPU cannot use this
  image. It builds there, but the container has no NVIDIA GPU.
- Rebuild from scratch: `bash docker/build.sh --no-cache`.
