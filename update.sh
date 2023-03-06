#!/bin/bash
git pull
git submodule update --remote

if python -m pip show whispercpp &>/dev/null; then python -m pip install -U git+https://git.ecker.tech/lightmare/whispercpp.py; fi
if python -m pip show whisperx &>/dev/null; then python -m pip install -U git+https://github.com/m-bain/whisperx.git; fi

deactivate