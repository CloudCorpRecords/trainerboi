#!/usr/bin/env bash
# Create a double-clickable TrainerBoi launcher on the macOS Desktop.
set -euo pipefail
cd "$(dirname "$0")"
TARGET="$(pwd)/START.sh"
SHORTCUT="$HOME/Desktop/TrainerBoi.command"

cat > "$SHORTCUT" <<EOF
#!/usr/bin/env bash
cd "$(dirname "$TARGET")"
exec ./START.sh
EOF
chmod +x "$SHORTCUT"

if [ -f "$SHORTCUT" ]; then
    echo ""
    echo "[SUCCESS] Launcher created on your Desktop:"
    echo "$SHORTCUT"
    echo "(First launch: right-click it and choose Open to approve it.)"
    echo ""
else
    echo "[NOTICE] Could not create the Desktop launcher. Run ./START.sh in this folder instead."
fi
