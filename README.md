# Reblend

A lightweight 3D app for Linux: still scenes, modeling, and animation in
one native desktop window. Built for low-spec machines: geometry is
tessellated once and kept in GPU vertex buffers, rendering is a
fixed-function rasterizer with PBR-lite materials, and there is no path
tracing.

## Requirements

- Linux, Python 3.10 or newer
- PySide6_Essentials (installed by `install.sh` into a local virtualenv)
- No other dependencies. No network use at runtime.
- Video export needs ffmpeg installed (`sudo apt install ffmpeg`);
  everything else works without it.

## Install and run

```
bash install.sh
bash run.sh
```

`bash run.sh --scene path/to/scene.json` opens a scene file. With no scene
given, the example tabletop scene opens.

## Controls

- Left-drag: orbit. Right-drag: pan. Scroll: zoom.
- Click an object: select it. Frame button: fit the view to the scene.
- Gizmo toolbar above the viewport: Move (W), Rotate (E), Scale (R)
  switch the gizmo drawn on the selected object. Drag a handle to
  move, rotate, or scale the selected object along that axis; the
  hovered handle brightens, Shift snaps (10 cm / 15 degrees / 0.1),
  and the inspector follows the drag live. Each drag is a single
  undo step, however far it moves. Exported frames never
  include the gizmo.
- Add panel: add a primitive. Inspector: move, rotate, scale, and edit
  the selected object's material. Duplicate / Delete act on the selection.
- Timeline: Play / Pause, scrub the slider, Add key / Delete key store
  the selected object's transform at the current frame.
- File menu: New, Open, Save, Import mesh (.obj / .glb), Export mesh,
  Export scene as .obj, Export mesh as .glb, Export scene as .glb,
  Render still to PNG, Export frames to a folder (one PNG per
  frame), Export video (MP4 / WebM via ffmpeg, with a progress
  bar and Cancel; a cancelled export leaves no file).
- Edit menu: Undo / Redo (up to 50 steps each).

## Scene format

Scenes are JSON files: objects (shape, size, position, rotation, scale,
material), cameras (name, position, look at), lights (key / fill / rim /
ambient), and animation (frames per second, end frame, keyframes).
Shapes: box, sphere, cylinder, cone, torus, plane. Materials carry a
base color, roughness, metalness, and one procedural texture
(none, wood_grain, brushed_metal, worn_leather). Procedural textures are
generated in code; no image files are used.

Emitters (T16) are objects of kind `emitter`: a particle source with
a shape (point / box / sphere / cone), rate, lifetime, speed, size,
colors, opacity fade, gravity factor, drag, wind response, seed, and
preset (fire / smoke / sparks / rain / snow / dust / water). The
particle motion is an analytic closed-form sim — scrubbing to any
frame gives identical particles — and the scene's top-level `world`
section holds the gravity and wind it runs under. Particle rendering
arrives in T17; until then an emitter shows as its small marker box.

## Budgets

The status line shows the object count and triangle count. Past a soft
100,000-triangle budget a warning appears. Nothing is ever blocked for
size: the warning is information only.

## Tests

```
bash test.sh
```

Core tests run headless. The viewport tests need a display and run when
`LIGHT_ENGINE_QT_TESTS=1` is set (under Xvfb: `xvfb-run -a bash test.sh`).

## What v0.1 does not do yet

- No sculpting, no physics, no particles.
- No video export (PNG frame sequences only).
- No gizmo dragging in the viewport (transforms are numeric).
- No shader-based rendering (fixed-function OpenGL 2.1).
