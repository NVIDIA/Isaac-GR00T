# Docker Setup for NVIDIA Isaac GR00T

Docker configuration for building and running a containerized GR00T environment with all dependencies pre-installed. The image (`gr00t-dev`) is based on NVIDIA's PyTorch container and includes CUDA support, Python dependencies, PyTorch3D, and the GR00T codebase.

## Prerequisites

- Docker (version 20.10+) and [perform post-installation setup](https://docs.docker.com/engine/install/linux-postinstall/) so you can run Docker commands without sudo. If you skip this setup, prefix the Docker commands below with `sudo`.
- NVIDIA Container Toolkit ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- NVIDIA GPU with compatible drivers
- Bash shell
- Sufficient disk space (several GB)

## Building the Docker Image

Make sure you are using a bash environment:

```bash
bash build.sh
```

The build process uses `nvcr.io/nvidia/pytorch:25.04-py3` as the base image, installs all dependencies, and sets up the GR00T codebase at `/workspace/gr00t/`.

## Running the Container

**Interactive shell (uses code baked into image):**
```bash
docker run -it --rm --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    gr00t-dev
```

**Development mode (mounts local codebase for live editing):**
```bash
docker run -it --rm --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd)/..:/workspace/gr00t \
    gr00t-dev bash -c "uv pip install -e . && bash"
```

## Thor Container (Jetson Thor / CUDA 13)

The `gr00t-thor` image is built from `scripts/deployment/thor/Dockerfile` for Jetson Thor with CUDA 13 support:

```bash
bash build.sh --profile=thor
```

For full Thor usage instructions (inference, benchmarks, bare metal setup), see the [Deployment & Inference Guide](../scripts/deployment/README.md#jetson-thor-setup).

## Spark Container (DGX Spark / CUDA 13)

The `gr00t-spark` image is built from `scripts/deployment/spark/Dockerfile` for DGX Spark with CUDA 13 support:

```bash
bash build.sh --profile=spark
```

For full Spark usage instructions (inference, benchmarks, bare metal setup), see the [Deployment & Inference Guide](../scripts/deployment/README.md#dgx-spark-setup).

## Orin Container (Jetson Orin / CUDA 12.6)

The `gr00t-orin` image is built from `scripts/deployment/orin/Dockerfile` for Jetson Orin (JetPack 6.2, CUDA 12.6, Python 3.10):

```bash
bash build.sh --profile=orin
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
- Rebuild: `sudo bash build.sh --no-cache`
