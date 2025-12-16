# Docker Setup for NVIDIA Isaac GR00T

Docker configuration for building and running a containerized GR00T environment with all dependencies pre-installed. The image (`gr00t-dev`) is based on NVIDIA's PyTorch container and includes CUDA support, Python dependencies, PyTorch3D, and the GR00T codebase.

## Prerequisites

- Docker (version 20.10+)
- NVIDIA Container Toolkit ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- NVIDIA GPU with compatible drivers
- Bash shell
- Sufficient disk space (several GB)

## Building the Docker Image

Make sure you are using a bash environment:

```bash
sudo bash build.sh
```

The build process uses `nvcr.io/nvidia/pytorch:25.04-py3` as the base image, installs all dependencies, and sets up the GR00T codebase at `/workspace/gr00t/`.

## Running the Container

**Interactive shell (uses code baked into image):**
```bash
sudo docker run -it --rm --gpus all gr00t-dev /bin/bash
```

**Development mode (mounts local codebase for live editing):**
```bash
sudo docker run -it --rm --gpus all \
    -v $(pwd)/..:/workspace/gr00t \
    gr00t-dev /bin/bash
```
**Run this from the `docker/` directory. Changes to your local GR00T code will be immediately reflected inside the container.**


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
