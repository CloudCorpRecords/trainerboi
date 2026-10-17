#!/usr/bin/env bash
# TrainerBoi setup from source (macOS / Linux). Needs `uv` (https://docs.astral.sh/uv/) - everything else is downloaded for you.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it with one of:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh   (then restart your terminal)"
    echo "  brew install uv"
    echo "See https://docs.astral.sh/uv/ for details."
    exit 1
fi

OS="$(uname -s)"
TORCH_ARGS=(torch torchvision torchaudio)
GPU_KIND="cpu"

if [ "$OS" = "Darwin" ]; then
    # Default PyPI wheels include Apple Metal (MPS) support - no special index needed.
    if [ "$(uname -m)" = "arm64" ]; then
        GPU_KIND="mps"
        echo "Apple Silicon detected: installing PyTorch with Metal (MPS) acceleration."
    else
        echo "Intel Mac detected: installing the CPU build (training will be slow, tiny models only)."
    fi
else
    # Linux: pick the PyTorch build that matches the NVIDIA driver, if any.
    DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    if [ -n "$DRIVER" ]; then
        MAJOR="${DRIVER%%.*}"
        if [ "$MAJOR" -ge 580 ] 2>/dev/null; then INDEX="https://download.pytorch.org/whl/cu130"; GPU_KIND="cuda";
        elif [ "$MAJOR" -ge 570 ] 2>/dev/null; then INDEX="https://download.pytorch.org/whl/cu128"; GPU_KIND="cuda";
        elif [ "$MAJOR" -ge 560 ] 2>/dev/null; then INDEX="https://download.pytorch.org/whl/cu126"; GPU_KIND="cuda";
        fi
        if [ "$GPU_KIND" = "cuda" ]; then
            echo "NVIDIA driver $DRIVER -> $INDEX"
            TORCH_ARGS+=(--index-url "$INDEX")
        fi
    fi
    if [ "$GPU_KIND" = "cpu" ]; then
        echo "No usable NVIDIA GPU found: installing the CPU build (training will be very slow)."
        TORCH_ARGS+=(--index-url "https://download.pytorch.org/whl/cpu")
    fi
fi

echo ""
echo "[1/4] Python 3.11 environment"
uv venv .venv --python 3.11
PY=".venv/bin/python"

echo ""
echo "[2/4] PyTorch"
uv pip install --python "$PY" "${TORCH_ARGS[@]}"

echo ""
echo "[3/4] TrainerBoi libraries"
uv pip install --python "$PY" -r requirements-app.txt
if [ "$OS" = "Darwin" ]; then
    # XGBoost needs the OpenMP runtime (libomp.dylib). Homebrew users already have it;
    # everyone else reuses the copy bundled with PyTorch/scikit-learn (no Homebrew needed).
    if ! "$PY" -c "import xgboost" >/dev/null 2>&1; then
        OMP_SRC="$("$PY" -c "import torch, os; p = os.path.join(os.path.dirname(torch.__file__), 'lib', 'libomp.dylib'); print(p if os.path.exists(p) else '')" 2>/dev/null || true)"
        if [ -z "$OMP_SRC" ]; then
            OMP_SRC="$("$PY" -c "import sklearn, os; p = os.path.join(os.path.dirname(sklearn.__file__), '.dylibs', 'libomp.dylib'); print(p if os.path.exists(p) else '')" 2>/dev/null || true)"
        fi
        if [ -n "$OMP_SRC" ]; then
            mkdir -p .venv/lib
            cp "$OMP_SRC" .venv/lib/libomp.dylib
            echo "Installed libomp.dylib into .venv/lib (the launch scripts put it on the library path)."
        else
            echo "WARNING: no bundled libomp.dylib found - tabular training needs: brew install libomp"
        fi
    fi
fi
if [ "$GPU_KIND" = "cuda" ]; then
    # Optional speed-ups (Linux here; Windows uses requirements-nvidia.txt via setup.bat).
    # triton-windows has no Linux build, so install the Linux equivalents individually.
    uv pip install --python "$PY" bitsandbytes torchao unsloth unsloth-zoo triton || \
        echo "Optional GPU speed-ups failed to install - TrainerBoi still works without them."
    uv pip install --python "$PY" --no-deps liger-kernel || true
fi

echo ""
echo "[4/4] User interface"
if [ -f ui/dist/index.html ]; then
    echo "already built"
elif command -v npm >/dev/null 2>&1; then
    npm --prefix ui install && npm --prefix ui run build
else
    echo "Node.js/npm not found - install Node 20+ from https://nodejs.org, then run:  cd ui && npm install && npm run build"
fi

echo ""
echo "Done. Start TrainerBoi with ./START.sh"
