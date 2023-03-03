#!/bin/bash
source ./venv/bin/activate

GPUS=$1
CONFIG=$2
PORT=1234

if (( $GPUS > 1 )); then
	python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT ./src/train.py -opt "$CONFIG" --launcher=pytorch
else
	python3 ./src/train.py -opt "$CONFIG"
fi
deactivate
