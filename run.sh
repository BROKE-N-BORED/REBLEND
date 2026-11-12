#!/bin/sh
# Reblend v0.2 launcher. Run:  bash run.sh
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "Not installed yet. Run:  bash install.sh"
  exit 1
fi
exec .venv/bin/python main.py "$@"
