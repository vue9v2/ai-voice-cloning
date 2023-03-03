#!/bin/bash
source ./venv/bin/activate
git clone https://git.ecker.tech/mrq/bitsandbytes-rocm
cd bitsandbytes-rocm
make hip
CUDA_VERSION=gfx1030 python setup.py install # assumes you're using a 6XXX series card
python3 -m bitsandbytes # to validate it works
cd ..