#!/usr/bin/env python3
"""Reblend v0.2 entry point. Usage:
  python3 main.py [--scene path/to/scene.json]
With no scene given, the example tabletop scene opens.

One copy only (UI3, U27): a second launch raises the running
window instead of starting a second copy. The guard is a QLockFile
plus a QLocalServer that carries the raise message; a stale lock
from a crashed run is taken over, never a block.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from light_engine.scene import load_scene, new_scene  # noqa: E402

# UI3 (U5): where QSettings lands. The organization + application
# names are set on the QApplication in main().
SETTINGS_ORGANIZATION = "BrokeNBored"
SETTINGS_APPLICATION = "Reblend"

# UI3 (U27): the single-instance name. REBLEND_INSTANCE_NAME
# overrides it so tests and verify scripts can run their own
# instance group without touching a real Reblend.
INSTANCE_ENV_VAR = "REBLEND_INSTANCE_NAME"
DEFAULT_INSTANCE_NAME = "reblend"


def instance_name():
    """The single-instance name for this launch."""
    return os.environ.get(INSTANCE_ENV_VAR, DEFAULT_INSTANCE_NAME)


def instance_lock_path(name):
    """The lock file for one instance name, per user (the uid in
    the name keeps two users' Reblends from sharing a lock)."""
    from PySide6 import QtCore
    tmp = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.TempLocation)
    return os.path.join(tmp, "%s-%d.lock" % (name, os.getuid()))


def instance_server_name(name):
    """The local-server name for one instance name, per user."""
    return "%s-%d" % (name, os.getuid())


def acquire_instance_lock(name):
    """Take the single-instance lock. Returns the QLockFile on
    success, None when another copy holds it. A lock from a dead
    owner is taken over: the stale window is 0, so a crashed run
    never blocks the next launch (the pid check still protects a
    live owner). tryLock takes its timeout argument — bare
    tryLock() raises TypeError in this PySide6 build."""
    from PySide6 import QtCore
    lock = QtCore.QLockFile(instance_lock_path(name))
    lock.setStaleLockTime(0)
    if lock.tryLock(0):
        return lock
    return None


def remove_stale_instance_lock(name):
    """Drop a stale lock file (dead owner) so the next tryLock can
    succeed. Returns True when a stale lock was removed."""
    from PySide6 import QtCore
    lock = QtCore.QLockFile(instance_lock_path(name))
    lock.setStaleLockTime(0)
    return lock.removeStaleLockFile()


def notify_existing_instance(name):
    """Ask the running copy to raise its window. True when the
    message was delivered (a copy is running), False when nothing
    answered (no copy, or a stale lock from a crashed run)."""
    from PySide6 import QtNetwork
    socket = QtNetwork.QLocalSocket()
    socket.connectToServer(instance_server_name(name))
    if not socket.waitForConnected(1000):
        return False
    socket.write(b"raise")
    socket.waitForBytesWritten(1000)
    socket.disconnectFromServer()
    return True


def create_instance_server(name):
    """The local server a first copy listens on for raise requests.
    A crashed run leaves its socket behind, so the old socket is
    removed before listening or it fails with Address in use.
    Returns the server, or None when it could not listen."""
    from PySide6 import QtNetwork
    server_name = instance_server_name(name)
    QtNetwork.QLocalServer.removeServer(server_name)
    server = QtNetwork.QLocalServer()
    if server.listen(server_name):
        return server
    return None


def handle_raise_request(server, window):
    """A second launch asked us to come forward: read its message
    and raise the window."""
    while server.hasPendingConnections():
        socket = server.nextPendingConnection()
        socket.waitForReadyRead(1000)
        if socket.readAll().data() == b"raise":
            window.raise_()
            window.activateWindow()
        socket.disconnectFromServer()
        socket.deleteLater()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reblend v0.2")
    parser.add_argument("--scene", default=None,
                        help="Scene JSON file to open")
    args = parser.parse_args(argv)
    try:
        from PySide6 import QtCore, QtWidgets
        from light_engine import style
        from light_engine.window import MainWindow
    except ImportError as exc:
        print("Reblend needs PySide6, and it is not installed here.")
        print("Run:  bash install.sh")
        print(f"(import error was: {exc})")
        return 2
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName(SETTINGS_ORGANIZATION)
    app.setApplicationName(SETTINGS_APPLICATION)
    # One copy only (UI3, U27): a second launch raises the running
    # window and exits. A lock with no answering server is stale
    # (a crashed run left it), so take it over and carry on.
    name = instance_name()
    lock = acquire_instance_lock(name)
    if lock is None:
        if notify_existing_instance(name):
            return 0
        remove_stale_instance_lock(name)
        lock = acquire_instance_lock(name)
        if lock is None:
            print("Reblend is already running, and the other copy "
                  "did not answer.")
            return 1
    # `lock` is held for the whole run: dropping it would let a
    # second copy start.
    server = create_instance_server(name)
    scene = None
    scene_path = None
    if args.scene:
        try:
            scene = load_scene(args.scene)
            scene_path = args.scene
        except Exception as exc:
            print(f"Could not open scene {args.scene}: {exc}")
            return 2
    else:
        example = HERE / "examples" / "tavern-tabletop.json"
        if example.exists():
            try:
                scene = load_scene(str(example))
            except Exception:
                scene = None
    if scene is None:
        scene = new_scene()
    # One look for the whole app (UI8): the stylesheet plus the
    # press-dip filter, applied before the window is born so every
    # window and dialog starts styled.
    style.install(app)
    window = MainWindow(scene=scene, scene_path=scene_path,
                        settings=QtCore.QSettings())
    if server is not None:
        server.newConnection.connect(
            lambda: handle_raise_request(server, window))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
