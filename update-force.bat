git fetch --all
git reset --hard origin/master
call .\update.bat

python -m venv venv
call .\venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install -U -r .\dlas\requirements.txt
python -m pip install -U -r .\tortoise-tts\requirements.txt
python -m pip install -U -e .\tortoise-tts 
python -m pip install -U -r .\requirements.txt

pause
deactivate