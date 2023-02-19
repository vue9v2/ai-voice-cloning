python -m venv venv
call .\venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio torch-directml==0.1.13.1.dev230119
.\setup-tortoise.bat
.\setup-training.bat
python -m pip install -r ./requirements.txt
deactivate
pause