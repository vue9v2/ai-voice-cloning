#!/bin/bash
git pull
python -m venv venv
source ./venv/bin/activate

git clone https://git.ecker.tech/mrq/DL-Art-School dlas
cd dlas
git pull
cd ..

git clone https://git.ecker.tech/mrq/tortoise-tts
cd tortoise-tts
git pull
cd ..

python -m pip install --upgrade pip
python -m pip install -r ./dlas/requirements.txt
python -m pip install -r ./tortoise-tts/requirements.txt
python -m pip install -e ./tortoise-tts 
python -m pip install -r ./requirements.txt


deactivate