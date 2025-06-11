# Isaac-GROOT Docker Setup

## Overview
This repository contains a Dockerfile to set up and run the NVIDIA Isaac-GROOT framework in a containerized environment. The setup includes CUDA 12.4, Miniconda, and all required dependencies for running Isaac-GROOT.

## Prerequisites
Before building and running the Docker container, ensure you have the following installed on your system:

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

## Build Instructions
Follow these steps to build the Docker image:

1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. Build the Docker image:
   ```sh
   docker build -t isaac-groot .
   ```

## Running the Container
To run the container with GPU support, use:
```sh
docker run --gpus all -it --rm isaac-groot
```
This will start a bash session inside the container.

## Features Included
- **CUDA 12.4**: The image is based on `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`.
- **Miniconda**: Installed to manage Python dependencies.
- **Isaac-GROOT Clone**: The repository is cloned directly inside the container.
- **Python 3.10 Environment**: A Conda virtual environment `gr00t` is created.
- **Jupyter Support**: Installed within the Conda environment.
- **FFmpeg and OpenCV**: Installed to handle media processing requirements.

## Accessing Jupyter Notebook (Optional)
If you want to run Jupyter notebooks inside the container, use:
```sh
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
Then, access it from your browser at `http://localhost:8888`.

## Notes
- The `ENTRYPOINT` and `CMD` are set to start a bash session.
- Make sure your system has the necessary NVIDIA drivers for GPU acceleration.

## License
This repository follows the [MIT License](LICENSE) unless stated otherwise.

## Support
For any issues, feel free to open an issue on GitHub or refer to the [Isaac-GROOT documentation](https://github.com/nvidia/Isaac-GR00T).


