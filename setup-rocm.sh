#!/bin/bash
# get local dependencies
git submodule init
git submodule update --remote
# setup venv
python3 -m venv venv
source ./venv/bin/activate
python3 -m pip install --upgrade pip # just to be safe
# ROCM
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.1.1 # 5.2 does not work for me desu
# install requirements
python3 -m pip install -r ./requirements.txt # install local requirements
python3 -m pip install -r ./modules/tortoise-tts/requirements.txt # install TorToiSe requirements
python3 -m pip install -e ./modules/tortoise-tts/ # install TorToiSe
python3 -m pip install -r ./modules/dlas/requirements.txt # instal DLAS requirements

python -m pip install -U einops==0.6.0
python -m pip install -U librosa==0.8.1

rm *.bat

./setup-rocm-bnb.sh

deactivate