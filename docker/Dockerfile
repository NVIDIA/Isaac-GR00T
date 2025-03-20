# Creating a Dockerfile to run the Isaac-GROOT by NVIDIA
# Importing nvidia/cuda image with CUDA12.4 support 
# Making sure we have Ubuntu image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Setting the frontend to be non-interactive
# This is to avoid any user input required during the installation of packages
ENV DEBIAN_FRONTEND=noninteractive

# Updating repositories and installing necessary packages for installing Conda and Python Libraries
RUN apt-get update && apt-get install wget -y\
        ffmpeg \
        libopencv-dev \
        git

# Create a miniconda directory to store the miniconda files
# Download and install miniconda3
RUN mkdir -p miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh
RUN bash /miniconda3/miniconda.sh -b -u -p miniconda3
RUN rm miniconda3/miniconda.sh

# Clone the GR00T repo from the NVIDIA GitHub Page
RUN git clone https://github.com/nvidia/Isaac-GR00T.git /GR00T

# Entering the GR00T directory
WORKDIR /GR00T

# Setting the environment paths to use the conda binary
ENV PATH="/miniconda3/bin:$PATH"
ARG PATH="/miniconda3/bin:$PATH"

# Create a virtual conda environment on miniconda with the required prerequisites
RUN conda create -n gr00t python=3.10

# Source the env path now to install the pip packages in the environment
ENV PATH="/miniconda3/envs/gr00t/bin:$PATH"
ARG PATH="/miniconda3/envs/gr00t/bin:$PATH"

# Now run all the conda related commands to make sure conda gets initialized in the bash script
# Install all necessary python packages for gr00t
RUN <<EOT bash
conda init
source /root/.bashrc
conda activate gr00t
pip install --upgrade setuptools
pip install -e .
pip install --no-build-isolation flash-attn==2.7.1.post4
EOT

# Set the default conda environment while using the /bin/bash to gr00t
RUN echo "conda activate gr00t" >> /root/.bashrc

# Install the jupyter extension for conda so that you can use jupyter when required to run the code
RUN conda install anaconda::jupyter

# Setting the Entrypoint and Command to /bin/bash whenever executed
ENTRYPOINT ["/bin/bash"]
CMD ["/bin/bash"]


