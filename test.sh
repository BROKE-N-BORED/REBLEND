#!/bin/sh
# Light Engine v0.1 tests. Run:  sh test.sh
# Core tests run headless. Set LIGHT_ENGINE_QT_TESTS=1 to also run the
# viewport tests (needs a display: use xvfb-run -a sh test.sh).
cd "$(dirname "$0")"
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi
exec "$PY" -m unittest discover -s tests -v
