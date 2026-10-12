# TrainerBoi setup from source (Windows). Needs `uv` (https://docs.astral.sh/uv/) - everything else is downloaded for you.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is required. Install it with:  winget install --id astral-sh.uv   (or see https://docs.astral.sh/uv/)" -ForegroundColor Yellow
    exit 1
}

# Pick the PyTorch build that matches the NVIDIA driver (CPU build if there is no usable NVIDIA GPU).
$index = "https://download.pytorch.org/whl/cpu"
$gpu = $false
try {
    $driver = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
    if ($driver) {
        $major = [int]($driver.Split(".")[0])
        if ($major -ge 580) { $index = "https://download.pytorch.org/whl/cu130"; $gpu = $true }
        elseif ($major -ge 570) { $index = "https://download.pytorch.org/whl/cu128"; $gpu = $true }
        elseif ($major -ge 560) { $index = "https://download.pytorch.org/whl/cu126"; $gpu = $true }
        Write-Host "NVIDIA driver $driver -> $index"
    }
} catch { }
if (-not $gpu) { Write-Host "No usable NVIDIA GPU found: installing the CPU build (training will be very slow)." -ForegroundColor Yellow }

Write-Host "`n[1/4] Python 3.11 environment"
uv venv .venv --python 3.11
$py = ".venv\Scripts\python.exe"

Write-Host "`n[2/4] PyTorch"
uv pip install --python $py torch torchvision torchaudio --index-url $index

Write-Host "`n[3/4] TrainerBoi libraries"
uv pip install --python $py -r requirements-app.txt
if ($gpu) {
    try {
        uv pip install --python $py -r requirements-nvidia.txt
        uv pip install --python $py --no-deps liger-kernel==0.8.4
    } catch { Write-Host "Optional GPU speed-ups failed to install - TrainerBoi still works without them." -ForegroundColor Yellow }
}

Write-Host "`n[4/4] User interface"
if (Test-Path ui\dist\index.html) { Write-Host "already built" }
elseif (Get-Command npm -ErrorAction SilentlyContinue) { npm --prefix ui install; npm --prefix ui run build }
else { Write-Host "Node.js/npm not found - install Node 20+ from https://nodejs.org, then run:  cd ui && npm install && npm run build" -ForegroundColor Yellow }

Write-Host "`nDone. Start TrainerBoi with START.bat" -ForegroundColor Green
