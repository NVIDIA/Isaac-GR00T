ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
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
      libtbb2 \
      libatlas-base-dev \
      libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

COPY pyproject.toml .

#Set to get precompiled jetson wheels
RUN export PIP_INDEX_URL=https://pypi.jetson-ai-lab.dev/jp6/cu126 && \
    export PIP_TRUSTED_HOST=pypi.jetson-ai-lab.dev && \
    pip3 install --upgrade pip setuptools && \
    pip3 install -e .[orin]
