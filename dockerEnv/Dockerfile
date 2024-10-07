ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=12.2.2

#
# Docker builder stage.
#
#FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 as builder

#ARG COLMAP_GIT_COMMIT=main
ARG CUDA_ARCHITECTURES=89
## Extract CUDA_ARCHITECTURES from nvidia-smi
ENV QT_XCB_GL_INTEGRATION=xcb_egl

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libboost-filesystem-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev \
        wget

## Build and install GLOMAP.
#RUN git clone https://github.com/colmap/glomap.git
#RUN cd glomap && \
#    mkdir build && \
#    cd build && \
#    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
#        -DCMAKE_INSTALL_PREFIX=/colmap_installed && \
#    ninja install

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    /opt/miniconda/bin/conda init && \
    rm Miniconda3-latest-Linux-x86_64.sh