ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
        curl \
        pkg-config \
        cmake \
        ninja-build \
        yasm \
        nasm \
        python3-pip \
        python3-dev \
        python3-setuptools \
        python3-wheel \
        pybind11-dev \
        zlib1g-dev \
        libssl-dev \
        libbz2-dev \
        liblzma-dev \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace/gr00t
COPY src/gr00t/ .

# Set to get precompiled jetson wheels
RUN export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126 && \
    export PIP_TRUSTED_HOST=pypi.jetson-ai-lab.io && \
    pip3 install --upgrade pip setuptools wheel && \
    pip3 install -e .[orin]

ARG FFMPEG_VERSION=n4.4.2
ARG TORCHCODEC_VERSION=v0.4.0

# Build and install ffmpeg
RUN git clone --branch ${FFMPEG_VERSION} --depth 1 https://git.ffmpeg.org/ffmpeg.git /tmp/ffmpeg && \
    cd /tmp/ffmpeg && \
    ./configure --enable-shared --enable-pic --prefix=/usr && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/ffmpeg

# Build and install decord
RUN git clone --recursive https://github.com/dmlc/decord /tmp/decord && \
    cd /tmp/decord && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    cd ../python && \
    python3 setup.py install --user && \
    rm -rf /tmp/decord

# Set decord library path environment variable
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.local/decord/

# Build and install torchcodec
RUN pip3 install --upgrade pybind11 && \
    git clone --branch ${TORCHCODEC_VERSION} --depth 1 https://github.com/pytorch/torchcodec.git /tmp/torchcodec && \
    cd /tmp/torchcodec && \
    export MAX_JOBS=$(nproc) && \
    export CUDA_HOME=/usr/local/cuda && \
    export TORCH_CUDA_ARCH_LIST="8.7" && \
    export I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 && \
    pip3 install --no-build-isolation . && \
    rm -rf /tmp/torchcodec
