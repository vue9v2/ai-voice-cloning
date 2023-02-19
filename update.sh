#!/bin/bash
git pull
git submodule update

python -m venv venv
source ./venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r ./dlas/requirements.txt
python -m pip install -r ./tortoise-tts/requirements.txt
python -m pip install -e ./tortoise-tts 
python -m pip install -r ./requirements.txt


deactivate