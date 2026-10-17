#!/usr/bin/env bash
# Start TrainerBoi in your browser (macOS / Linux).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "TrainerBoi is not installed yet. Running setup first..."
    ./setup.sh
fi

export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_ENABLE_MPS_FALLBACK=1
if [ "$(uname -s)" = "Darwin" ] && [ -f ".venv/lib/libomp.dylib" ]; then
    export DYLD_LIBRARY_PATH="$PWD/.venv/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

echo "Starting TrainerBoi at http://127.0.0.1:7860  (Ctrl+C to stop)"
exec ".venv/bin/python" studio_server.py --port 7860 --open
