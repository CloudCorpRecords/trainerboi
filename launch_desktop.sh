#!/usr/bin/env bash
# Start the TrainerBoi Electron desktop app in dev mode (macOS / Linux).
set -euo pipefail
cd "$(dirname "$0")/desktop"
if [ ! -d node_modules ]; then
    npm install
fi
npm start
