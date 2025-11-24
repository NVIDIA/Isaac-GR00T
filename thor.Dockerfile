ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      libsm6 \
      libxext6 \
      ffmpeg \
      libhdf5-serial-dev \
      libtesseract-dev \
      libgtk-3-0 \
      libtbb12 \
      libgl1 \
      libatlas-base-dev \
      libopenblas-dev \
      build-essential \
      python3-setuptools \
      make \
      cmake \
      nasm \
      yasm \
      pkg-config \
      git \
      libgnutls28-dev \
      libvpx-dev \
      libopus-dev \
      libvorbis-dev \
      libmp3lame-dev \
      libfreetype-dev \
      libass-dev \
      libaom-dev \
      libdav1d-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

# Copy the entire project (especially the gr00t/ source directory)
COPY . .

# Set to get precompiled jetson wheels
RUN export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/sbsa/cu130 && \
    export PIP_TRUSTED_HOST=pypi.jetson-ai-lab.io && \
    pip3 install .[thor]

RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Build FFmpeg 4.4.2 for decord compatibility
RUN cd /tmp && \
    git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg-4.4.2 && \
    cd ffmpeg-4.4.2 && \
    git checkout n4.4.2 -b n4.4.2 && \
    ./configure --enable-shared --enable-pic --prefix=/usr && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && \
    rm -rf /tmp/ffmpeg-4.4.2

# Build and install decord
RUN cd /tmp && \
    git clone --recursive https://github.com/dmlc/decord && \
    cd decord && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make && \
    cd ../python && \
    python3 setup.py install --user && \
    cd /tmp && \
    rm -rf /tmp/decord

# Set decord library path environment variable
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.local/decord/
