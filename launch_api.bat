@echo off
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "CUDA_MODULE_LOADING=LAZY"

echo ========================================================================
echo        TRAINERBOI - LOCAL OPENAI-COMPATIBLE API SERVER
echo ========================================================================
echo   Endpoint:  http://127.0.0.1:8000/v1
echo   Swagger:   http://127.0.0.1:8000/docs
echo   Compatible with Cursor, Continue.dev, LM Studio, LibreChat
echo ========================================================================
echo.

"%~dp0.venv\Scripts\python.exe" "%~dp0api_server.py"
pause

