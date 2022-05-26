ARG CUDA="11.1"
ARG CUDNN="8"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

# try use updated key
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
    && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
    && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libyaml-dev vim zsh wget htop tmux \
    && apt-get install -y ffmpeg libgl-dev unzip zip

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x /miniconda.sh \
    && /miniconda.sh -b -p /miniconda \
    && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.7 environment
RUN /miniconda/bin/conda install -y conda-build \
    && /miniconda/bin/conda create -y --name py37 python=3.7 \
    && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython h5py nltk joblib jupyter pandas scipy
RUN pip install requests ninja cython yacs>=0.1.8 numpy>=1.19.5 cython matplotlib opencv-python tqdm \
    protobuf tensorboardx pymongo sklearn boto3 scikit-image cityscapesscripts
RUN pip install azureml-defaults>=1.0.45 azureml.core inference-schema
RUN pip --no-cache-dir install --force-reinstall -I pyyaml

RUN python -m nltk.downloader punkt

# Install latest PyTorch 1.7.1
ARG CUDA
# RUN conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch \
#     && conda clean -ya
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia \
    && conda clean -ya
RUN conda install -y -c conda-forge timm einops

# install pycocotools
# RUN git clone https://github.com/cocodataset/cocoapi.git \
#  && cd cocoapi/PythonAPI \
#  && python setup.py build_ext install
RUN conda install -y -c conda-forge pycocotools

# install cityscapesScripts
RUN python -m pip install cityscapesscripts

# RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

# install gdown for download dataset from google drive
RUN pip install gdown

# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}

RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
RUN echo """export ZSH="/root/.oh-my-zsh"\nZSH_THEME="gentoo"\nplugins=(git)\nsource \$ZSH/oh-my-zsh.sh""" > /root/.zshrc
RUN echo """syntax on\nfiletype indent on\nset autoindent\nset number\ncolorscheme desert""" > /root/.vimrc

CMD [ "zsh" ]

RUN git clone https://github.com/iamzifei/scene_graph_benchmark.git

WORKDIR /scene_graph_benchmark

# download the waterbird dataset and unzip it
RUN gdown 1TKiSXtY-x4Q51KxCyLL2QMoDKSU_1ku9
RUN unzip waterbirds.zip

RUN rm waterbirds.zip

RUN mkdir pretrained_model
RUN wget -P ./pretrained_model https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth
RUN mkdir visualgenome
RUN wget -P ./visualgenome https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/VG-SGG-dicts-vgoi6-clipped.json

# finish build the image, the following code is for running inside the container

# sudo docker run --rm --gpus all -it --entrypoint bash scene_graph_benchmark

# python setup.py build develop

# docker cp 5889ed174720:/scene_graph_benchmark/output/woman_fish_x152c4.obj.jpg Downloads