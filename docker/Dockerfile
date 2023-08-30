FROM supervisely/base-py-sdk:6.69.66

RUN rm -rf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y git
RUN apt-get install -y ffmpeg libsm6 libxext6  
RUN apt-get install ninja-build

RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install -U openmim
RUN mim install mmcv-full==1.7.1

RUN git clone https://github.com/ViTAE-Transformer/ViTPose.git /ViTPose
WORKDIR /ViTPose
RUN pip3 install -v -e .
RUN pip3 install timm==0.4.9 einops
WORKDIR /
COPY __init__.py /ViTPose/mmpose/
RUN pip3 install pyyaml==6.0
RUN pip3 install ffmpeg-python==0.2.0

RUN python3 -m pip install supervisely==6.69.66
