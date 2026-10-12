@echo off
:: One-time setup from source: Python 3.11 environment + the right PyTorch for your GPU + all libraries + the UI.
:: (The desktop installer does all of this for you automatically - this is for running from a git checkout.)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1"
exit /b %errorlevel%
