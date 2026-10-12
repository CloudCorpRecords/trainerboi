@echo off
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set CUDA_MODULE_LOADING=LAZY
call "%~dp0.venv\Scripts\activate.bat"
echo TrainerBoi environment activated.
python --version
