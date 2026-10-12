@echo off
setlocal
title TrainerBoi
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo TrainerBoi is not installed yet. Running setup first...
    call setup.bat
    if errorlevel 1 exit /b 1
)

set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PYTHONUNBUFFERED=1"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "CUDA_MODULE_LOADING=LAZY"

echo Starting TrainerBoi at http://127.0.0.1:7860  (close this window to stop it)
".venv\Scripts\python.exe" studio_server.py --port 7860 --open
pause
