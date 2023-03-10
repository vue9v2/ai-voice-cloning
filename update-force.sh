#!/bin/bash
git fetch --all
git reset --hard origin/master

./update.sh

# force install requirements
python3 -m venv venv
source ./venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -U -r ./requirements.txt
python3 -m pip install --force-reinstall -U -r ./modules/tortoise-tts/requirements.txt
python3 -m pip install --force-reinstall -U -e ./modules/tortoise-tts 
python3 -m pip install --force-reinstall -U -r ./modules/dlas/requirements.txt

deactivate