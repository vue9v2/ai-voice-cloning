git submodule init
git submodule update --remote

python -m venv venv
call .\venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio torch-directml
python -m pip install -r .\requirements.txt
python -m pip install -r .\tortoise-tts\requirements.txt
python -m pip install -e .\tortoise-tts\
python -m pip install -r .\dlas\requirements.txt

pause
deactivate