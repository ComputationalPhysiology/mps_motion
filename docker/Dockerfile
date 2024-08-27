FROM ubuntu:22.04

ARG REPO_BRANCH="main"
ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system

# Install pip and git with apt
RUN apt-get update && \
    apt-get install -y gcc python3-dev pkg-config libhdf5-dev python3-pip git zip unzip curl ffmpeg libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# We upgrade pip and setuptools
RUN python3 -m pip install --no-cache-dir pip setuptools --upgrade
RUN python3 -m pip install --no-cache-dir git+https://github.com/ComputationalPhysiology/mps_motion.git
# Install the headless version of opencv in docker
RUN python3 -m pip uninstall -y opencv-python opencv-contrib-python
RUN python3 -m pip install --no-cache-dir opencv-python-headless opencv-contrib-python-headless
