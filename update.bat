git pull
python -m venv venv
call .\venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r ./requirements.txt
python -m pip install -r ./dlas/requirements.txt
deactivate

cd dlas
git pull
cd ..
pause