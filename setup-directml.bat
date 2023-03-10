git submodule init
git submodule update --remote

python -m venv venv
call .\venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio torch-directml
python -m pip install -r .\requirements.txt
python -m pip install -r .\modules\tortoise-tts\requirements.txt
python -m pip install -e .\modules\tortoise-tts\
python -m pip install -r .\modules\dlas\requirements.txt

python -m pip install -U -r einops==0.6.0
python -m pip install -U -r librosa==0.8.1

del *.sh

pause
deactivate