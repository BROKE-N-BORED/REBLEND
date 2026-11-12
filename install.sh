#!/bin/sh
# Light Engine v0.1 installer. Run:  sh install.sh
# Checks Python >= 3.10, creates a virtualenv, installs PySide6_Essentials
# (Qt Widgets + OpenGL only — no WebEngine, no PDF; the full PySide6
# metapackage pulls a 116 MB web engine this app never uses).
# No sudo, nothing installed outside this folder. --check only reports.
set -e
cd "$(dirname "$0")"

check_only=0
if [ "$1" = "--check" ]; then check_only=1; fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed. Install it first, then run sh install.sh again."
  exit 1
fi

py_ok=$(python3 -c "import sys; print(1 if sys.version_info >= (3, 10) else 0)")
if [ "$py_ok" != "1" ]; then
  python3 --version
  echo "Light Engine needs Python 3.10 or newer."
  exit 1
fi

echo "Python: $(python3 --version)"

if [ -d ".venv" ] && [ -x ".venv/bin/python" ]; then
  echo "Virtualenv: present"
else
  echo "Virtualenv: missing"
  if [ "$check_only" = "1" ]; then exit 1; fi
fi

if [ -d ".venv" ] && [ -x ".venv/bin/python" ]; then
  if .venv/bin/python -c "import PySide6" 2>/dev/null; then
    echo "PySide6: installed ($(.venv/bin/python -c "import PySide6; print(PySide6.__version__)"))"
    if [ "$check_only" = "1" ]; then exit 0; fi
    echo "Already installed. Run it with:  sh run.sh"
    exit 0
  else
    echo "PySide6: missing"
  fi
fi

if [ "$check_only" = "1" ]; then exit 1; fi

if [ ! -d ".venv" ]; then
  echo "Creating virtualenv..."
  python3 -m venv .venv
fi

echo "Installing PySide6_Essentials (this downloads once)..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install PySide6_Essentials

echo "Checking the install..."
.venv/bin/python -m unittest discover -s tests -v 2>&1 | tail -3
echo "Done. Run it with:  sh run.sh"
