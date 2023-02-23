git submodule init
git submodule update --remote

python -m venv venv
call .\venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r .\dlas\requirements.txt
python -m pip install -r .\tortoise-tts\requirements.txt
python -m pip install -r .\requirements.txt
python -m pip install -e .\tortoise-tts\

copy .\dlas\bitsandbytes_windows\* .\venv\Lib\site-packages\bitsandbytes\. /Y

deactivate
pause