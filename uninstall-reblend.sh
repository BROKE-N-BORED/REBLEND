#!/usr/bin/env bash
# Reblend uninstaller (double-click to run, no terminal needed).
# Removes the program, its app-menu entry and its launcher. Your
# scenes (saved wherever you saved them) and any install backups
# are KEPT.
set -u
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The installer always installs to this one location, wherever this
# script itself was run from.
TARGET="${HOME:-}/.local/share/reblend"

BACKEND=""
if command -v zenity >/dev/null 2>&1; then
    BACKEND="zenity"
elif python3 -c "import tkinter" 2>/dev/null; then
    BACKEND="tkinter"
fi
if [ -z "$BACKEND" ]; then
    exit 1
fi
confirm() {
    if [ "$BACKEND" = "zenity" ]; then
        zenity --question --title="$1" --text="$2" --width=420 2>/dev/null
    else
        python3 "$HERE/installer-dialogs.py" confirm "$1" "$2"
    fi
}
info() {
    if [ "$BACKEND" = "zenity" ]; then
        zenity --info --title="$1" --text="$2" --width=420 2>/dev/null
    else
        python3 "$HERE/installer-dialogs.py" info "$1" "$2"
    fi
}

if ! confirm "Uninstall Reblend" \
    "This removes the Reblend program, its app-menu entry and its launcher.

Your scenes are kept (they live where you saved them), and so are any install backups."; then
    exit 1
fi
rm -rf "$TARGET"
rm -f "$HOME/.local/share/applications/reblend.desktop"
rm -f "$HOME/.local/bin/reblend"
command -v update-desktop-database >/dev/null 2>&1 && \
    update-desktop-database "$HOME/.local/share/applications" 2>/dev/null
info "Reblend removed" \
    "Reblend is removed. Your scenes were kept."
exit 0
