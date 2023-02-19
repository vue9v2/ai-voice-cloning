#!/bin/bash
git submodule init
git submodule update

python -m venv venv
source ./venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r ./dlas/requirements.txt
python -m pip install -r ./tortoise-tts/requirements.txt
python -m pip install -r ./requirements.txt
python -m pip install -e ./tortoise-tts/
deactivate
