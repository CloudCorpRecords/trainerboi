#!/usr/bin/env bash
# Start the TrainerBoi OpenAI-compatible API server (macOS / Linux).
set -euo pipefail
cd "$(dirname "$0")"

export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_ENABLE_MPS_FALLBACK=1
if [ "$(uname -s)" = "Darwin" ] && [ -f ".venv/lib/libomp.dylib" ]; then
    export DYLD_LIBRARY_PATH="$PWD/.venv/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

echo "========================================================================"
echo "       TRAINERBOI - LOCAL OPENAI-COMPATIBLE API SERVER"
echo "========================================================================"
echo "  Endpoint:  http://127.0.0.1:8000/v1"
echo "  Swagger:   http://127.0.0.1:8000/docs"
echo "  Compatible with Cursor, Continue.dev, LM Studio, LibreChat"
echo "========================================================================"
echo ""

exec ".venv/bin/python" api_server.py
