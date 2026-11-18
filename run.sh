#!/bin/sh
# Reblend v0.2 launcher. Run:  bash run.sh
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "Not installed yet. Run:  bash install.sh"
  exit 1
fi
# UPD: a staged update (downloaded by the running app, verified
# against its checksum) applies here, before the app starts, so
# no file changes under a running copy. The marker test is one
# shell builtin, so an up-to-date launch costs nothing; the
# apply itself never blocks the launch — on any failure the
# old version starts as usual. After an apply, re-exec: run.sh
# itself may have been replaced by the update.
STAGING_DIR="$(dirname "$(pwd)")/$(basename "$(pwd)")-update"
if [ -f "$STAGING_DIR/pending.json" ]; then
  if .venv/bin/python apply_update.py; then
    # Absolute path: $0 may be a bare "run.sh" (bash run.sh),
    # and exec searches PATH for a bare name instead of the
    # current directory.
    exec "$(pwd)/run.sh" "$@"
  fi
fi
exec .venv/bin/python main.py "$@"
