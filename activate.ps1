# Activate the TrainerBoi virtual environment
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:CUDA_MODULE_LOADING = "LAZY"
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"
Write-Host "TrainerBoi environment activated." -ForegroundColor Green
Write-Host "Python: $(python --version)"
