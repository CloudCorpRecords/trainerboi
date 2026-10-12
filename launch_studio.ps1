# Start TrainerBoi in your browser
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
Write-Host "Starting TrainerBoi on http://127.0.0.1:7860 ..." -ForegroundColor Cyan
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\studio_server.py" --port 7860 --open
