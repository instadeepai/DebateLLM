FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Ensure no installs try launch interactive screen
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python3.9 && \
    apt install -y python3.9-dev && \
    apt-get install -y python3-pip && \
    apt-get install -y python3.9-venv && \
    apt-get install -y git && \
    apt-get install -y wget
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 10

# Check python v
RUN python -V

# Setup virtual env
RUN python -m venv dyn-net
ENV VIRTUAL_ENV /dyn-net
ENV PATH /dyn-net/bin:$PATH
RUN pip install --upgrade pip setuptools wheel

# Neptune key - not needed on cluster, already an env variable.
ARG NEPTUNE_API_TOKEN
ENV NEPTUNE_API_TOKEN=${NEPTUNE_API_TOKEN}

ARG GOOGLE_APPLICATION_CREDENTIALS
ENV GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}

# Tensorflow
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV TF_CPP_MIN_LOG_LEVEL=3

# JAX
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# working directory
WORKDIR /home/app/debatellm
ENV PYTHONPATH=$PWD:$PYTHONPATH

COPY . /home/app/debatellm

# ENV PYTHONPATH=$PYTHONPATH:./medAlpaca/:./

ADD requirements.txt .
RUN pip install -r requirements.txt

# Install medAlpaca
# RUN git clone https://github.com/kbressem/medAlpaca.git

# Fix package versions
RUN pip install peft==0.2.0 transformers==4.28.0


EXPOSE 6006
