#!/bin/sh
# Reblend v0.2 tests. Run:  bash test.sh
# Core tests run headless. Set LIGHT_ENGINE_QT_TESTS=1 to also run the
# viewport tests (needs a display: use xvfb-run -a bash test.sh).
cd "$(dirname "$0")"
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi
exec "$PY" -m unittest discover -s tests -v
