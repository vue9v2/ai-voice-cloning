call .\venv\Scripts\activate.bat
set PYTHONUTF8=1
python ./src/train.py -opt "%1"
pause
deactivate