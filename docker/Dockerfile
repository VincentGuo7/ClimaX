# tags: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags?quick-deploy=false

FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-7 \
    git \
    gpg \
    curl \
    vim \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    openssh-client \
    openssh-server \
    libsm6 \
    libxext6 \
    ffmpeg \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/environment.yml

# Create environment
ENV PATH=/opt/conda/bin:${PATH}

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

RUN conda update -n base -y conda && \
    conda env create -f /tmp/environment.yml

SHELL ["conda", "run", "-n", "climaX", "/bin/bash", "-c"]


ENV PATH /opt/conda/envs/climaX/bin:$PATH
ENV CONDA_DEFAULT_ENV climaX
