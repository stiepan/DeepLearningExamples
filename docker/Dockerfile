#base on NVIDIA DL for Pytorch
FROM nvcr.io/nvidia/pytorch:22.12-py3

WORKDIR /
RUN git clone https://github.com/klecki/DeepLearningExamples.git joc
WORKDIR /joc
RUN git checkout dali-aa

RUN pip list | grep dali
# The container always lacks this
RUN pip install nvidia-dllogger

# currently container has cuda 11
RUN pip uninstall -y nvidia-dali-cuda110
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda110

WORKDIR /joc/PyTorch/Classification/ConvNets/