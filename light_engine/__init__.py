"""Reblend v0.2: a lightweight native 3D app for Linux.

Standard library + PySide6 only. No web tech, no path tracing.
"""
# UPD: the one version constant. It always equals the tag of
# the release this tree ships as, without the leading "v"
# (this tree ships as v0.2-upd) — the updater compares the
# running tag against the latest release tag, so a stale
# version here would offer the app its own release forever.
# Each task bumps it to its own release tag; F3 keeps it
# single-source.
__version__ = "0.2-r2"
