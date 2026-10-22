# Docker Setup for Isaac GR00T 2

`docker/Dockerfile` builds the supported GR00T environment for x86_64 and for
Linux aarch64 GB300 / DGX Station, on the NGC PyTorch container
`nvcr.io/nvidia/pytorch:25.11-py3`, the same base as GR00T's internal training
images. Both architectures then run the same stack:

| Component | Source |
| --- | --- |
| PyTorch 2.10 (NGC build), torchvision, Triton, CUDA 13.0, cuDNN, NCCL | NGC container |
| FlashAttention 2.7.4 | NGC container |
| TorchCodec 0.9.1 | Built against NGC PyTorch |
| FFmpeg | Ubuntu on x86_64; built from source with NVDEC on arm64 |
| All other Python packages | `uv.lock`, synced into `/opt/gr00t-venv` |

The venv uses `--system-site-packages`, and `gr00t-sync` installs the lock
without its PyTorch stack and CUDA libraries, so NGC's builds stay in use. On
x86_64 the lock pins CUDA 12 builds of NVRx and TensorRT for host installs on
CUDA 12.8; `gr00t-sync` installs the CUDA 13 builds the lock records for aarch64
instead. The best-effort
[host installation](../docs/installation.md#host-installation-best-effort)
installs the whole lock, including its own PyTorch, and CI does not test it.

Jetson Thor uses its own host [Thor setup](../docs/installation.md#jetson-thor).

The image is built from the NGC container, so its use is subject to the
[NVIDIA Deep Learning Container License](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf).

## Prerequisites

- x86_64 Linux, or Linux aarch64 on GB300 / DGX Station, with an NVIDIA GPU and
  a driver that supports CUDA 13.0 (R580 or newer). GB300 hosts also need the
  driver and memory settings in the
  [GB300 prerequisites](../docs/installation.md#gb300-prerequisites).
- Docker 23 or newer, with BuildKit
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- About 50 GB of free disk for the image (the NGC base alone is about 30 GB),
  plus the model cache and outputs described in the
  [installation guide](../docs/installation.md#x86-prerequisites)

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
`pyproject.toml`, `uv.lock`, and `docker/gr00t-sync`.

The build needs internet access: it pulls the BuildKit Dockerfile frontend, the
NGC container, and the uv image, clones TorchCodec (and on arm64 FFmpeg) from
GitHub, and downloads Ubuntu packages and the locked Python packages. On arm64,
FFmpeg's CUDA filters are compiled for GB300 (`sm_100`).

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
without syncing because `UV_PROJECT_ENVIRONMENT=/opt/gr00t-venv` and
`UV_NO_SYNC=1`. To add extras, run `gr00t-sync` from the package root instead of
`uv sync`, which would replace NGC's PyTorch with the lock's:

```bash
gr00t-sync --extra training --extra tensorrt
```

Verify the environment with the same checks as the installation guide:

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

**If the check fails, rebuild the image from the checkout, or run
`gr00t-sync --extra training` to sync the image environment to it.** `uv run`
does not sync, so it otherwise runs the image's packages.

## Troubleshooting

- GPU not visible: check the NVIDIA Container Toolkit with
  `docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi`.
- Apple Silicon and other ARM64 hosts without an NVIDIA GPU cannot use this
  image. It builds there, but the container has no NVIDIA GPU.
- Rebuild from scratch: `bash docker/build.sh --no-cache`.
