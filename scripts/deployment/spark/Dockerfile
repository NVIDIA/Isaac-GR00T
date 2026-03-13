FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential yasm cmake libtool git pkg-config curl \
    libass-dev libfreetype6-dev libvorbis-dev \
    autoconf automake texinfo tmux ffmpeg

COPY src/gr00t/ /workspace/gr00t/

# install_deps.sh copies spark/pyproject.toml to repo root before uv sync
ENV DOCKER_CONTAINER=1
ENV UV_PROJECT_ENVIRONMENT=/opt/gr00t-venv
RUN cd /workspace/gr00t && bash scripts/deployment/spark/install_deps.sh

# Activate the venv by default and expose nvidia pip package libs (cudnn, cudss, cublas)
ENV VIRTUAL_ENV=/opt/gr00t-venv
ENV PATH="$VIRTUAL_ENV/bin:/usr/local/cuda/bin:$PATH"
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV CUDA_PATH=/usr/local/cuda-13.0
ENV CPATH="/usr/local/cuda-13.0/include:${CPATH}"
ENV C_INCLUDE_PATH="/usr/local/cuda-13.0/include:${C_INCLUDE_PATH}"
ENV CPLUS_INCLUDE_PATH="/usr/local/cuda-13.0/include:${CPLUS_INCLUDE_PATH}"
ENV LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudss/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

WORKDIR /workspace/gr00t
