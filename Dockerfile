FROM python:3.10.8

RUN apt-get -y update && apt-get -y install ffmpeg
# RUN apt-get -y update && apt-get -y install git wget python-dev python3-dev libopenmpi-dev python-pip zlib1g-dev cmake python-opencv

ENV CODE_DIR /root/code

COPY . $CODE_DIR/PPO
WORKDIR $CODE_DIR/PPO

# Clean up pycache and pyc files
RUN rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install torch

CMD /bin/bash