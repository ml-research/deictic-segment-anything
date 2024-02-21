# Select the base image

### To run with GPUs, use the following:
FROM nvcr.io/nvidia/pytorch:21.10-py3
###


### To run without GPUs, use the following
# FROM ubuntu:18.04
# RUN apt-get update
# RUN apt-get install -y software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get install -y python3.8-dev python3-pip
# RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3
# RUN python3 --version
# RUN pip3 --version
# RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
###


### other setup
ENV TZ=Europe/Berlin
ARG DEBIAN_FRONTEND=noninteractive

# Select the working directory
WORKDIR  /Workspace

# Install system libraries required by OpenCV.
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
# Install Python requirements
# RUN pip install opencv-python==4.5.5.64
RUN pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install networkx==3.0 lark-parser joblib scikit-learn torchsummary setuptools tensorboard numpy>=1.18.5 tqdm>=4.41.0 matplotlib>=3.2.2 opencv-python>=4.1.2 Pillow PyYAML>=5.3.1 scipy>=1.4.1 seaborn pandas rtpt

RUN pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
RUN pip install torch-geometric
RUN pip install wandb anytree

# RUN python -m pip install -e Grounded-Segment-Anything/GroundingDINO
#RUN cd GroundingDINO
#RUN pip install -e .
#RUN cd ..
RUN pip install "opencv-python-headless<4.3"
RUN pip install --upgrade diffusers[torch]
RUN pip install openai==0.28.1
RUN pip install pydantic==1.9.0
RUN pip install visual_genome

# fix opencv
#RUN pip uninstall opencv-python
#RUN pip uninstall opencv-contrib-python
#RUN pip uninstall opencv-contrib-python-headless
#RUN pip3 install opencv-contrib-python==4.5.5.62
