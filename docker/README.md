# Docker Setup for NVIDIA Isaac GR00T

Docker configuration for building and running a containerized GR00T environment with all dependencies pre-installed. A single `Dockerfile` supports both x86_64 and aarch64 (GB200, Grace Hopper) architectures. On aarch64, `torchcodec` is installed from the prebuilt wheel shipped under `scripts/deployment/dgpu/wheels/`; the build falls back to a source compile only if the wheel is missing.

## Prerequisites

- Docker (version 20.10+) and [perform post-installation setup](https://docs.docker.com/engine/install/linux-postinstall/) so you can run Docker commands without sudo. If you skip this setup, prefix the Docker commands below with `sudo`.
- NVIDIA Container Toolkit ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- NVIDIA GPU with compatible drivers
- Bash shell
- Sufficient disk space (several GB)

## Building the Docker Image

From the repository root:

```bash
bash docker/build.sh
```

This builds from `nvidia/cuda:12.8.0-devel-ubuntu22.04` and installs all dependencies into `/opt/gr00t-venv`. The image does not include a working source checkout; for normal use, start the image and then clone or pull the repo you want to run inside the container.

## Running the Container

**Recommended workflow: run the image, then clone or update the repo inside it.**

Start an interactive shell:

```bash
docker run -it --rm --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    gr00t
```

Then, inside the container:

```bash
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T /workspace/Isaac-GR00T
cd /workspace/Isaac-GR00T
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c "import gr00t; print('GR00T ready')"
```

The image venv is active by default (`/opt/gr00t-venv`; `/workspace/.venv` is a compatibility symlink), and uv is configured with `UV_PROJECT_ENVIRONMENT=/opt/gr00t-venv`. After setting `PYTHONPATH` to the checked-out repo, both `python ...` and `uv run ...` use the global image venv instead of creating a checkout-local `.venv`. If you are working on an existing checkout in the container, run `git pull --ff-only` from that checkout instead of cloning again.

The global venv records the `uv.lock` hash it was built from. If your checked-out repo uses a different lockfile, create a checkout-local venv before running commands. Reusing a uv cache keeps this path from starting cold:

```bash
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/uv-cache}"
export UV_LINK_MODE=copy
UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync
source .venv/bin/activate
```

Do not run a bare `uv sync` unless you intend to update the global image venv. Use `UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync` when you want an isolated per-checkout environment.

Avoid bind-mounting over `/workspace`, because that can hide the image's `/workspace/.venv` compatibility symlink. If you need to mount local source for live editing, mount it under a subdirectory:

```bash
docker run -it --rm --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd):/workspace/Isaac-GR00T" \
    gr00t bash -c 'cd /workspace/Isaac-GR00T && export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" && bash'
```

## Edge Device Containers

### Thor Container (Jetson Thor / CUDA 13)

The `gr00t-thor` image is built from `scripts/deployment/thor/Dockerfile` for Jetson Thor with CUDA 13 support:

```bash
bash docker/build.sh --profile=thor
```

For full Thor usage instructions (inference, benchmarks, bare metal setup), see the [Deployment & Inference Guide](../scripts/deployment/README.md#jetson-thor-setup).

### Spark Container (DGX Spark / CUDA 13)

The `gr00t-spark` image is built from `scripts/deployment/spark/Dockerfile` for DGX Spark with CUDA 13 support:

```bash
bash docker/build.sh --profile=spark
```

For full Spark usage instructions (inference, benchmarks, bare metal setup), see the [Deployment & Inference Guide](../scripts/deployment/README.md#dgx-spark-setup).

### Orin Container (Jetson Orin / CUDA 12.6)

The `gr00t-orin` image is built from `scripts/deployment/orin/Dockerfile` for Jetson Orin (JetPack 6.2, CUDA 12.6, Python 3.10):

```bash
bash docker/build.sh --profile=orin
```

For full Orin usage instructions (inference, benchmarks, bare metal setup), see the [Deployment & Inference Guide](../scripts/deployment/README.md#jetson-orin-setup).

## Troubleshooting

**GPU not detected:**
- Verify NVIDIA Container Toolkit: `nvidia-container-toolkit --version`
- Restart Docker: `sudo systemctl restart docker`
- Test GPU access: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`

**Permission errors:**
- Use `sudo` with Docker commands, or add your user to the `docker` group: `sudo usermod -aG docker $USER`

**Build failures:**
- Check disk space: `df -h`
- Clean Docker: `docker system prune -a`
- Rebuild: `bash docker/build.sh --no-cache`
