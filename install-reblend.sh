#!/usr/bin/env bash
# Reblend v0.2 — T11 (mesh import/export) — double-click GUI installer
# for Linux Mint. No terminal needed: double-click "Install Reblend"
# and answer the dialogs. Installs Reblend as a real program:
#   program files -> ~/.local/share/reblend
#   app menu      -> "Reblend" (Graphics), with its icon
#   launcher      -> ~/.local/bin/reblend
# The install itself is Reblend's own install.sh (virtualenv +
# PySide6_Essentials); this script only wraps it in dialogs, copies
# the files, and creates the app-menu entry Reblend's install.sh
# does not make. A previous install is backed up first (code only —
# Reblend keeps no data inside the program folder; scenes live
# wherever you save them and are never touched).
# Full log: /tmp/reblend-install.log and, after the copy step,
# ~/.local/share/reblend/install.log
#
# NOTE: the icon (assets/reblend.png) is PROVISIONAL — a plain cube
# drawn in code. Joshua has not picked the Reblend logo yet; when he
# does, this icon gets replaced with the real mark.
set -u
SRC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${HOME:-}/.local/share/reblend"
LOG="/tmp/reblend-install.log"
DESKTOP_FILE="${HOME:-}/.local/share/applications/reblend.desktop"
VERSION="Reblend v0.2 — T11 (mesh import/export)"
: > "$LOG" 2>/dev/null || LOG="$SRC/install.log"
echo "Reblend installer started: $(date)" >> "$LOG"
echo "Source: $SRC" >> "$LOG"

# ------------------------------------------------------------------
# Dialogs: zenity first (ships with Mint), tkinter fallback.
# ------------------------------------------------------------------
DIALOGS="$SRC/installer-dialogs.py"
BACKEND=""
if command -v zenity >/dev/null 2>&1; then
    BACKEND="zenity"
elif python3 -c "import tkinter" 2>/dev/null; then
    BACKEND="tkinter"
fi
if [ -z "$BACKEND" ]; then
    echo "No dialog tool found (need zenity or python3-tk)." >> "$LOG"
    exit 1
fi
echo "Dialog backend: $BACKEND" >> "$LOG"

dlg_confirm() { # title, text -> 0 Yes, 1 No
    if [ "$BACKEND" = "zenity" ]; then
        zenity --question --title="$1" --text="$2" --width=420 2>/dev/null
    else
        python3 "$DIALOGS" confirm "$1" "$2"
    fi
}
dlg_info() { # title, text
    if [ "$BACKEND" = "zenity" ]; then
        zenity --info --title="$1" --text="$2" --width=420 2>/dev/null
    else
        python3 "$DIALOGS" info "$1" "$2"
    fi
}
dlg_error() { # title, text
    if [ "$BACKEND" = "zenity" ]; then
        zenity --error --title="$1" --text="$2" --width=420 2>/dev/null
    else
        python3 "$DIALOGS" error "$1" "$2"
    fi
}
dlg_progress() { # pid; reads status lines ("# text") on stdin
    if [ "$BACKEND" = "zenity" ]; then
        zenity --progress --pulsate --title="Installing Reblend" \
            --text="Installing Reblend..." --auto-close --width=420 2>/dev/null
    else
        python3 "$DIALOGS" progress "$1"
    fi
}
fail() { # title, text — log it, show it, copy the log into the target
    echo "FAILED: $1 — $2" >> "$LOG"
    [ -d "$TARGET" ] && cp "$LOG" "$TARGET/install.log" 2>/dev/null
    dlg_error "$1" "$2"
    exit 1
}

# A child that already exited stays a zombie until waited for, and
# `kill -0` still succeeds on a zombie — so liveness is read from
# /proc state (Z = finished) instead.
child_alive() {
    [ -d "/proc/$1" ] || return 1
    [ "$(cut -d' ' -f3 "/proc/$1/stat" 2>/dev/null)" != "Z" ]
}

# ------------------------------------------------------------------
# Confirm
# ------------------------------------------------------------------
if ! dlg_confirm "Install Reblend" \
    "This installs $VERSION on this computer.

It copies the program to $TARGET, downloads its interface library (about 100 MB, a few minutes on first install), and adds Reblend to the app menu with its icon.

Your scenes are kept — they live where you saved them, not in the program folder."; then
    echo "Cancelled at the confirm step." >> "$LOG"
    exit 1
fi

# ------------------------------------------------------------------
# Back up a previous install (code only — the venv is regenerable
# and scenes live outside the program folder), keep the last 2.
# ------------------------------------------------------------------
if [ -d "$TARGET" ] && [ -n "$(ls -A "$TARGET" 2>/dev/null)" ]; then
    BACKUP="${TARGET}-backup-$(date +%Y%m%d-%H%M%S)"
    echo "Backing up previous install to $BACKUP" >> "$LOG"
    # --preserve=mode,timestamps, not -a: -a also preserves ownership,
    # which fails (and fails the whole copy) wherever chown is not
    # allowed, even though every file copied fine. Modes are what a
    # backup actually needs — the scripts must stay executable.
    BACKUP_OK=0
    if cp -r --preserve=mode,timestamps "$TARGET" "$BACKUP" >>"$LOG" 2>&1; then
        BACKUP_OK=1
    else
        rm -rf "$BACKUP"
        if cp -r "$TARGET" "$BACKUP" >>"$LOG" 2>&1; then
            BACKUP_OK=1
        fi
    fi
    if [ "$BACKUP_OK" -eq 1 ]; then
        rm -rf "$BACKUP/.venv"
        # Prune older backups, keep the two newest.
        ls -dt "${TARGET}"-backup-* 2>/dev/null | tail -n +3 | \
            while IFS= read -r old; do rm -rf "$old"; done
    else
        rm -rf "$BACKUP"
        echo "Backup failed (continuing anyway)." >> "$LOG"
    fi
fi

# ------------------------------------------------------------------
# Copy the program files in. Overlay only — nothing already in the
# target is deleted, so a re-install never wipes user files.
# ------------------------------------------------------------------
mkdir -p "$TARGET" || fail "Install failed" \
    "Could not create $TARGET. See $LOG for details."
echo "Copying program files to $TARGET" >> "$LOG"
# Same --preserve choice as the backup above: -a fails the whole copy
# wherever chown is not allowed, even when every file copied fine.
if ! cp -r --preserve=mode,timestamps "$SRC/." "$TARGET/" >>"$LOG" 2>&1; then
    fail "Install failed" \
        "Could not copy the program files. See $LOG for details."
fi
find "$TARGET" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
# The icon ships as a PNG in the Release zip; a plain repo checkout
# carries only the base64 text, so decode it when the PNG is absent.
if [ ! -f "$TARGET/assets/reblend.png" ] \
    && [ -f "$TARGET/assets/reblend.png.b64" ]; then
    echo "Decoding icon from base64." >> "$LOG"
    base64 -d "$TARGET/assets/reblend.png.b64" \
        > "$TARGET/assets/reblend.png" 2>>"$LOG" || \
        echo "Icon decode failed (continuing without an icon)." >> "$LOG"
fi
# Same for the sample .glb: the Release zip has the real file; a
# plain repo checkout carries only the base64 text.
if [ ! -f "$TARGET/examples/sample.glb" ] \
    && [ -f "$TARGET/examples/sample.glb.b64" ]; then
    echo "Decoding sample.glb from base64." >> "$LOG"
    base64 -d "$TARGET/examples/sample.glb.b64" \
        > "$TARGET/examples/sample.glb" 2>>"$LOG" || \
        echo "sample.glb decode failed (the import example will be missing)." >> "$LOG"
fi

# ------------------------------------------------------------------
# Run Reblend's own install.sh (venv + PySide6_Essentials) with a
# progress dialog fed by the log. It runs in its own process group
# so Cancel stops it and everything it started. Reblend's install
# needs no sudo — nothing is installed outside the program folder.
# ------------------------------------------------------------------
RUN=""
command -v setsid >/dev/null 2>&1 && RUN="setsid"
echo "Running install.sh (log: $LOG)" >> "$LOG"
$RUN bash "$TARGET/install.sh" >>"$LOG" 2>&1 &
INSTALL_PID=$!
(
    while child_alive "$INSTALL_PID"; do
        tail -n 1 "$LOG" 2>/dev/null | sed 's/^/# /'
        sleep 1
    done
    echo "# Finishing up..."
) | dlg_progress "$INSTALL_PID"
PROGRESS_STATUS=$?
if [ "$PROGRESS_STATUS" -ne 0 ]; then
    # Cancelled: stop the install and everything it started.
    kill -- "-$INSTALL_PID" 2>/dev/null || kill "$INSTALL_PID" 2>/dev/null
    wait "$INSTALL_PID" 2>/dev/null
    cp "$LOG" "$TARGET/install.log" 2>/dev/null
    dlg_info "Install cancelled" \
        "The install was stopped. Run Install Reblend again to finish it."
    exit 1
fi
wait "$INSTALL_PID"
INSTALL_STATUS=$?
echo "install.sh exit status: $INSTALL_STATUS" >> "$LOG"
if [ "$INSTALL_STATUS" -ne 0 ]; then
    TAIL="$(tail -n 5 "$LOG" 2>/dev/null)"
    fail "Install failed" \
        "The install step did not finish. The last lines of the log were:

$TAIL

Full log: $LOG"
fi

# ------------------------------------------------------------------
# Post-checks: the system Qt libraries (without them the window
# cannot open), the Python environment, then the menu entry.
# ------------------------------------------------------------------
probe_qt_libs() {
    QT_MISSING=()
    ldconfig -p 2>/dev/null | grep -q "libEGL.so.1" || QT_MISSING+=(libegl1)
    ldconfig -p 2>/dev/null | grep -q "libGL.so.1" || QT_MISSING+=(libgl1)
    ldconfig -p 2>/dev/null | grep -q "libxkbcommon.so.0" || QT_MISSING+=(libxkbcommon0)
    # Qt 6.5+ refuses to load the xcb platform plugin at all without
    # xcb-cursor — the window cannot open, and Qt's error names it.
    ldconfig -p 2>/dev/null | grep -q "libxcb-cursor.so.0" || QT_MISSING+=(libxcb-cursor0)
}
probe_qt_libs
if [ "${#QT_MISSING[@]}" -gt 0 ]; then
    echo "Qt libraries still missing: ${QT_MISSING[*]}" >> "$LOG"
    if dlg_confirm "System packages still missing" \
        "These system packages are still missing: ${QT_MISSING[*]}. Without them Reblend cannot open its window.

Try installing them now? (You will be asked for your password.)"; then
        if command -v pkexec >/dev/null 2>&1; then
            pkexec apt-get install -y "${QT_MISSING[@]}" >>"$LOG" 2>&1
        elif command -v sudo >/dev/null 2>&1; then
            export SUDO_ASKPASS="$TARGET/askpass-zenity.sh"
            sudo -A apt-get install -y "${QT_MISSING[@]}" >>"$LOG" 2>&1
        fi
        probe_qt_libs
    fi
    if [ "${#QT_MISSING[@]}" -gt 0 ]; then
        fail "Install incomplete" \
            "These system packages are still missing: ${QT_MISSING[*]}. Reblend cannot open its window without them.

Open Software Manager and install them, or run Install Reblend again. Full log: $LOG"
    fi
fi
if [ ! -x "$TARGET/.venv/bin/python" ]; then
    fail "Install incomplete" \
        "The Python environment was not created. Check your internet connection and run Install Reblend again. Full log: $LOG"
fi
if ! "$TARGET/.venv/bin/python" -c "import PySide6" >>"$LOG" 2>&1; then
    fail "Install incomplete" \
        "Reblend's interface library (PySide6) did not install. Check your internet connection and run Install Reblend again. Full log: $LOG"
fi

# ------------------------------------------------------------------
# App-menu entry. Reblend's own install.sh does not create one, so
# the GUI installer writes it here (absolute paths — the menu
# entry points at the installed copy, not at this folder).
# ------------------------------------------------------------------
mkdir -p "$(dirname "$DESKTOP_FILE")" || fail "Install failed" \
    "Could not create the app-menu folder. See $LOG for details."
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Reblend
Comment=Reblend v0.2 — lightweight 3D engine
Exec="$TARGET/run.sh"
Icon=$TARGET/assets/reblend.png
Terminal=false
Categories=Graphics;3DGraphics;
EOF
chmod +x "$DESKTOP_FILE"
if [ ! -f "$DESKTOP_FILE" ]; then
    fail "Install incomplete" \
        "The app-menu entry was not created. Full log: $LOG"
fi

# ------------------------------------------------------------------
# Finish: launcher symlink, menu refresh, log copy.
# ------------------------------------------------------------------
mkdir -p "$HOME/.local/bin" 2>/dev/null
ln -sf "$TARGET/run.sh" "$HOME/.local/bin/reblend" 2>/dev/null
command -v update-desktop-database >/dev/null 2>&1 && \
    update-desktop-database "$HOME/.local/share/applications" >>"$LOG" 2>&1
cp "$LOG" "$TARGET/install.log" 2>/dev/null
echo "Install complete: $(date)" >> "$LOG"

if dlg_confirm "Reblend is installed" \
    "Reblend is installed — find it in the app menu under Graphics, or search for Reblend.

Launch Reblend now?"; then
    nohup "$TARGET/run.sh" >/dev/null 2>&1 &
    disown
fi
exit 0
