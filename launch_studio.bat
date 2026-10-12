@echo off
cd /d "%~dp0"
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
echo Starting TrainerBoi at http://127.0.0.1:7860 ...
".venv\Scripts\python.exe" studio_server.py --port 7860 --open
pause
