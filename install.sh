#!/bin/sh
# Reblend v0.2 installer. Run:  bash install.sh
# Checks Python >= 3.10, creates a virtualenv, installs PySide6_Essentials
# (Qt Widgets + OpenGL only — no WebEngine, no PDF; the full PySide6
# metapackage pulls a 116 MB web engine this app never uses).
# No sudo, nothing installed outside this folder. --check only reports.
set -e
cd "$(dirname "$0")"

check_only=0
if [ "$1" = "--check" ]; then check_only=1; fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed. Install it first, then run bash install.sh again."
  exit 1
fi

py_ok=$(python3 -c "import sys; print(1 if sys.version_info >= (3, 10) else 0)")
if [ "$py_ok" != "1" ]; then
  python3 --version
  echo "Reblend needs Python 3.10 or newer."
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
    echo "Already installed. Run it with:  bash run.sh"
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

echo "Installing PySide6_Essentials 6.12.0 (this downloads once)..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install PySide6_Essentials==6.12.0

echo "Checking the install..."
# The suite's exit code gates the install: it used to run through
# `tail -3`, whose exit status is always 0, so a red suite still
# printed Done. Output goes to a log (inside .venv, never committed);
# on failure the last lines print so the failure is visible.
if .venv/bin/python -m unittest discover -s tests -v > .venv/test-output.log 2>&1; then
  tail -3 .venv/test-output.log
  echo "Done. Run it with:  bash run.sh"
else
  echo "The self-check failed, so the install did not finish cleanly."
  echo "Reblend is installed, but its tests did not all pass:"
  tail -20 .venv/test-output.log
  echo "Full output is in .venv/test-output.log"
  exit 1
fi
