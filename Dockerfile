FROM nvidia/cuda:9.1-base-ubuntu16.04

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
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
# This must be done before pip so that requirements.txt is available
# CUDA 9.0-specific steps
RUN conda install -y -c pytorch \
    cuda91=1.0 \
    magma-cuda91=2.3.0 \
    pytorch=0.4.0 \
    torchvision=0.2.1 \
    && conda clean -ya
# Install HDF5 Python bindings
RUN conda install -y \
    h5py \
    && conda clean -ya

RUN pip install h5py-cache

# YOLO specific parts
COPY . /srv/app
WORKDIR /srv/app

RUN mkdir /data && mkdir /config && mkdir /config/weights && \
    mkdir /output

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN pip install \
    scikit-image \
    numpy \
    torch>=0.4.0 \
    torchvision \
    pillow \
    matplotlib \
    click

RUN cd /srv/app && \
    # pip install -r requirements.txt && \
    python setup.py install

RUN yolov3 --help
