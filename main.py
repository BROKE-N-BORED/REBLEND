#!/usr/bin/env python3
"""Reblend v0.2 entry point. Usage:
  python3 main.py [--scene path/to/scene.json]
With no scene given, the example tabletop scene opens.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from light_engine.scene import load_scene, new_scene  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reblend v0.2")
    parser.add_argument("--scene", default=None,
                        help="Scene JSON file to open")
    args = parser.parse_args(argv)
    try:
        from PySide6 import QtWidgets
        from light_engine.window import MainWindow
    except ImportError as exc:
        print("Reblend needs PySide6, and it is not installed here.")
        print("Run:  bash install.sh")
        print(f"(import error was: {exc})")
        return 2
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
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Reblend")
    window = MainWindow(scene=scene, scene_path=scene_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
