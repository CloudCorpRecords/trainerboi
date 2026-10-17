#!/usr/bin/env bash
# One-time setup from source: Python 3.11 environment + PyTorch (Metal/MPS on Apple Silicon) + libraries + UI.
# (The desktop installer does all of this for you automatically - this is for running from a git checkout.)
set -euo pipefail
cd "$(dirname "$0")"
bash scripts/setup.sh
