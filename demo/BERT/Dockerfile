FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update && \
    apt-get install -y curl git wget bzip2 zlib1g-dev pkg-config build-essential ninja-build g++ \
    libnvinfer7=7.0.0-1+cuda10.2 libnvinfer-dev=7.0.0-1+cuda10.2 \
    libnvinfer-plugin7=7.0.0-1+cuda10.2 libnvinfer-plugin-dev=7.0.0-1+cuda10.2 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/miniconda3/bin:${PATH} CONDA_PREFIX=/opt/miniconda3
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -p /opt/miniconda3 -b && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    conda update -y conda && \
    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch && \
    conda install curl conda-verify conda-build mkl-include cmake -c anaconda && \
    conda install git git-lfs docopt -c conda-forge  && \
    conda clean -afy

RUN pip --no-cache-dir install contexttimer future transformers docopt