#!/bin/bash
git submodule init
git submodule update --remote

python3 -m venv venv
source ./venv/bin/activate
python3 -m pip install --upgrade pip
# ROCM
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.1.1 # 5.2 does not work for me desu
python3 -m pip install -r ./dlas/requirements.txt
python3 -m pip install -r ./tortoise-tts/requirements.txt
python3 -m pip install -r ./requirements.txt
python3 -m pip install -e ./tortoise-tts/
deactivate
