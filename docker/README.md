# Docker Setup for NVIDIA Isaac GR00T

Docker configuration for building and running a containerized GR00T environment with all dependencies pre-installed. A single `Dockerfile` supports both x86_64 and aarch64 (GB200, Grace Hopper) architectures. On aarch64, torchcodec is built from source since no prebuilt wheel is available.

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

This builds from `nvidia/cuda:12.8.0-devel-ubuntu22.04`, installs all dependencies via `uv sync`, and sets up the GR00T codebase at `/workspace/`.

## Running the Container

**Interactive shell (uses code baked into image):**
```bash
docker run -it --rm --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    gr00t
```

**Development mode (mounts local codebase for live editing):**
```bash
docker run -it --rm --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/workspace \
    gr00t bash -c "uv pip install -e . && bash"
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
