#!/bin/bash
source ./venv/bin/activate
python3 ./src/train.py -opt "$1"
deactivate
