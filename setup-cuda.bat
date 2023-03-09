git submodule init
git submodule update --remote

python -m venv venv
call .\venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r .\requirements.txt
python -m pip install -r .\modules\tortoise-tts\requirements.txt
python -m pip install -e .\modules\tortoise-tts\
python -m pip install -r .\modules\dlas\requirements.txt

xcopy .\modules\dlas\bitsandbytes_windows\* .\venv\Lib\site-packages\bitsandbytes\. /Y
xcopy .\modules\dlas\bitsandbytes_windows\cuda_setup\* .\venv\Lib\site-packages\bitsandbytes\cuda_setup\. /Y
xcopy .\modules\dlas\bitsandbytes_windows\nn\* .\venv\Lib\site-packages\bitsandbytes\nn\. /Y

del *.sh

pause
deactivate