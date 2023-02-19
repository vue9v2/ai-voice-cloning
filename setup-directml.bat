git submodule init
git submodule update

python -m venv venv
call .\venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio torch-directml==0.1.13.1.dev230119
python -m pip install -r .\dlas\requirements.txt
python -m pip install -r .\tortoise-tts\requirements.txt
python -m pip install -r .\requirements.txt
python -m pip install -e .\tortoise-tts\
deactivate
pause