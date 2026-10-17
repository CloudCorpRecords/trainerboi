#!/usr/bin/env bash
# Start TrainerBoi in your browser (assumes setup.sh already ran).
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
if [ "$(uname -s)" = "Darwin" ] && [ -f ".venv/lib/libomp.dylib" ]; then
    export DYLD_LIBRARY_PATH="$PWD/.venv/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
echo "Starting TrainerBoi at http://127.0.0.1:7860 ..."
exec ".venv/bin/python" studio_server.py --port 7860 --open
