FROM ubuntu:22.04

ARG REPO_BRANCH="main"
ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system

# Install pip and git with apt
RUN apt-get update && \
    apt-get install -y python3-pip git zip unzip curl ffmpeg libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# We upgrade pip and setuptools
RUN python3 -m pip install --no-cache-dir pip setuptools --upgrade

WORKDIR /tmp/

# # Copy pyproject.toml first so that we done need to reinstall in case another file
# # is changing after rebuiding docker image
RUN git clone --branch ${REPO_BRANCH} --single-branch https://github.com/ComputationalPhysiology/mps_motion.git
RUN python3 -m pip install --no-cache-dir ./mps_motion[docker]

# We remove the contents of the temporary directory to minimize the size of the image
RUN rm -rf /tmp/*
