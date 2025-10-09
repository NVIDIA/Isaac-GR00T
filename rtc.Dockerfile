# This Dockerfile is based on https://github.com/NVIDIA/Isaac-GR00T/blob/main/Dockerfile
# The original base image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel was replaced by nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 because the pytorch image includes conda.
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Configure environment variables
ARG USERNAME=ubuntu
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt update && \
    apt install -y tzdata && \
    ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && \
    apt install -y netcat dnsutils && \
    apt-get update && \
    apt-get install -y libgl1-mesa-glx git libvulkan-dev \
    zip unzip wget curl git git-lfs build-essential cmake python3-dev \
    vim less sudo htop ca-certificates man tmux ffmpeg tensorrt \
    # Add OpenCV system dependencies
    libglib2.0-0 libsm6 libxext6 libxrender-dev

WORKDIR /workspace/Isaac-GR00T

# Create the user if it does not exist and allow it to execute sudo commands without a password
RUN useradd -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R $USERNAME:$USERNAME /workspace
USER $USERNAME

# skips downloading large files during uv sync
ENV GIT_LFS_SKIP_SMUDGE=1

# Install uv 
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/$USERNAME/.local/bin:$PATH"

# Copy only pyproject.toml for dependency installation
COPY pyproject.toml .
RUN UV_PROJECT_ENVIRONMENT=/workspace/venv uv sync --extra base

# The actual code will be mounted as a volume at runtime

ENTRYPOINT [ "./rtc_entrypoint.sh" ]
