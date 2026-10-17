# Activate the TrainerBoi virtual environment (macOS / Linux).
# Usage:  source ./activate.sh
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_ENABLE_MPS_FALLBACK=1
_TB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "$_TB_DIR/.venv/bin/activate"
if [ -f "$_TB_DIR/.venv/lib/libomp.dylib" ]; then
    export DYLD_LIBRARY_PATH="$_TB_DIR/.venv/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
unset _TB_DIR
echo "TrainerBoi environment activated."
python --version
