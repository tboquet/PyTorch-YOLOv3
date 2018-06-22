FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         wget \
         unzip \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include && \
     /opt/conda/bin/conda install -c pytorch magma-cuda90 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
# This must be done before pip so that requirements.txt is available
# CUDA 9.0-specific steps
RUN conda install pytorch torchvision cuda90 -c pytorch

# Install HDF5 Python bindings
RUN conda install -y \
    h5py \
    && conda clean -ya

RUN pip install h5py-cache

# YOLO specific parts

COPY . /srv/app
WORKDIR /srv/app

RUN cd /srv/app && \
    pip install -r requirements.txt
