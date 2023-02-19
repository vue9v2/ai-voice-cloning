#!/bin/bash
python -m venv venv
source ./venv/bin/activate
python -m pip install --upgrade pip
# ROCM
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.1.1 # 5.2 does not work for me desu
./setup-training.sh
./setup-tortoise.sh
python -m pip install -r ./requirements.txt
deactivate
