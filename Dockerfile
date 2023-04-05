FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS stage1
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt install -y curl wget git
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
USER user
ENV HOME=/home/user
WORKDIR $HOME
RUN mkdir $HOME/.cache $HOME/.config && chmod -R 777 $HOME
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh
RUN chmod +x Miniconda3-py39_23.1.0-1-Linux-x86_64.sh
RUN ./Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -b -p /home/user/miniconda
ENV PATH="$HOME/miniconda/bin:$PATH"
RUN conda init
RUN conda install python=3.9.13
RUN python3 -m pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

FROM stage1 AS stage2
RUN mkdir /home/user/ai-voice-cloning
WORKDIR /home/user/ai-voice-cloning
COPY --chown=user:user modules modules

FROM stage2 AS stage3
RUN python3 -m pip install -r ./modules/tortoise-tts/requirements.txt
RUN python3 -m pip install -e ./modules/tortoise-tts/
RUN python3 -m pip install -r ./modules/dlas/requirements.txt
RUN python3 -m pip install -e ./modules/dlas/
ADD requirements.txt requirements.txt
RUN python3 -m pip install -r ./requirements.txt
ADD --chown=user:user . /home/user/ai-voice-cloning

CMD ["python", "./src/main.py", "--listen", "0.0.0.0:7680"]
