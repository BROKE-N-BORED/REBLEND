#!/usr/bin/env python3
"""Apply a staged Reblend update (UPD). Called by run.sh before
the app launches: a staged update (downloaded by the running
app, verified against its checksum) is extracted over the app
folder here, so no file changes under a running copy.

Exit 0 only when an update was applied — run.sh then re-execs
itself, because run.sh may have been replaced. Exit 1 when
nothing is staged or the apply failed; the launch carries on
with the old version either way. A failure is printed, never
silent, and the failed staging is moved aside so the next
launch does not retry it forever.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from light_engine.updater import apply_pending_update  # noqa: E402


def main():
    result = apply_pending_update(HERE)
    if result["applied"]:
        print("Reblend updated to %s." % result["tag"])
        return 0
    if result["error"]:
        print("Reblend update not applied: %s" % result["error"])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
