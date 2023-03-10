git fetch --all
git reset --hard origin/master
call .\update.bat

python -m venv venv
call .\venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install -U -r .\requirements.txt
python -m pip install -U -r .\modules\tortoise-tts\requirements.txt
python -m pip install -U -e .\modules\tortoise-tts 
python -m pip install -U -r .\modules\dlas\requirements.txt
python -m pip install -U einops==0.6.0
python -m pip install -U librosa==0.8.1

pause
deactivate