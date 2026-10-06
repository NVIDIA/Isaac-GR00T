FROM nvidia/cuda:13.2.1-devel-ubuntu24.04

ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential yasm cmake libtool git pkg-config curl \
    libass-dev libfreetype6-dev libvorbis-dev \
    autoconf automake texinfo tmux ffmpeg \
    libopenblas-dev

COPY . /workspace/

# install_deps.sh consumes the shared JetPack 7.2 pyproject in place.
ENV DOCKER_CONTAINER=1
ENV UV_PROJECT_ENVIRONMENT=/opt/gr00t-venv
RUN cd /workspace && bash scripts/deployment/jetson/install_deps.sh

# Activate the venv by default and expose the CUDA, PyTorch, and packaged TensorRT runtimes.
ENV VIRTUAL_ENV=/opt/gr00t-venv
ENV PATH="$VIRTUAL_ENV/bin:/usr/local/cuda-13.2/bin:/usr/local/cuda/bin:$PATH"
ENV TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas
ENV CUDA_HOME=/usr/local/cuda-13.2
ENV CUDA_PATH=/usr/local/cuda-13.2
ENV CPATH="/usr/local/cuda-13.2/include:${CPATH:-}"
ENV C_INCLUDE_PATH="/usr/local/cuda-13.2/include:${C_INCLUDE_PATH:-}"
ENV CPLUS_INCLUDE_PATH="/usr/local/cuda-13.2/include:${CPLUS_INCLUDE_PATH:-}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64:$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/tensorrt_libs:${LD_LIBRARY_PATH:-}"

WORKDIR /workspace/gr00t
