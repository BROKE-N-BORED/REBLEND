#!/bin/sh
# Light Engine v0.1 launcher. Run:  sh run.sh
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "Not installed yet. Run:  sh install.sh"
  exit 1
fi
exec .venv/bin/python main.py "$@"
