#!/bin/bash
git fetch --all
git reset --hard origin/master

./update.sh

# force install requirements
python3 -m venv venv
source ./venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r ./requirements.txt
python3 -m pip install -r ./tortoise-tts/requirements.txt
python3 -m pip install -e ./tortoise-tts 
python3 -m pip install -r ./dlas/requirements.txt

deactivate