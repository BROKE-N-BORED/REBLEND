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
- Add panel: add a primitive or an emitter (fire preset by default).
  Inspector: collapsible sections (Object, Transform, Material) with
  a plain-words line under every field; the Transform rows carry X/Y/Z
  axis labels and the texture dropdown groups 20 procedural styles
  by family (wood, stone and marble, brick and concrete, metal,
  fabric and leather, ground), each named in plain words. Each
  style shows its own knobs under the dropdown — ring spacing and
  grain strength for wood, brick length and mortar width for
  brick, vein spacing for marble — and a knob change re-renders
  live (the texture re-bakes once, a short pause after the last
  change, so a drag stays responsive). By default only the knobs
  that change the look most show; "Show all settings" reveals
  every knob for the style, and switching back never loses a
  value. Emitter
  objects get an Emitter section
  with the preset and key facts, plus an Edit emitter button that
  opens the dedicated editor window (double-clicking an emitter in
  the object list opens it too): preset picker plus every emitter
  setting, grouped and explained — shape, emission, appearance over
  life, forces, seed. Picking a preset fills the settings; editing
  one by hand marks the picker custom. The object list keeps its
  scroll position and selection across edits. Duplicate / Delete act
  on the selection.
- Timeline: Play / Pause, scrub the slider, Add key / Delete key store
  the selected object's transform at the current frame.
- File menu: New, Open, Save, Import mesh (.obj / .glb), Export mesh,
  Export scene as .obj, Export mesh as .glb, Export scene as .glb,
  Render still to PNG, Export frames to a folder (one PNG per
  frame, with a progress bar and Cancel; cancelling keeps the
  frames already written), Export video (MP4 / WebM via ffmpeg,
  with a progress bar and Cancel; a cancelled export leaves no
  file).
- Everything in the interface is written in plain words: the Import,
  Export, and Video dialogs explain each field under it, and a file
  that cannot be read says so in plain words before the technical
  detail.
- Unsaved work is protected: the title shows a * while the scene
  has unsaved changes, and New, Open, and Quit ask Save / Discard /
  Cancel (Cancel is the default). Opening a scene keeps the old one
  on the undo stack, so an accidental Open is recoverable with undo.
- The status bar shows what the app is doing: a busy dot plus plain
  words while a mesh is being read or frames or a video are being
  exported, and a brief green done moment when a video finishes.
- One look across the whole app: dark surfaces with a warm cream
  accent, buttons that dip when pressed, and a visible focus ring
  on every control. The spin and combo arrows are cream chevrons
  (generated into the user's cache dir at startup, so no image
  files ship with the app). Reduced motion is respected: with the
  desktop's animation setting off, panels and windows jump to
  their final state instead of animating.
- The info row under the viewport shows the measured frame rate
  (frames per second and milliseconds per frame, a rolling average
  over the last 30 paints) next to the object and triangle counts,
  once enough paints have been measured. Behind it, the viewport
  tunes its own cost silently: if frames run slow it steps down
  (smaller shadow map, then shadows off, then a lower render
  scale) and steps back up when frames are fast again, so a heavy
  scene stays interactive. There is no setting for it — the
  viewport looks after itself. Exported frames and video always
  render at full quality, whatever level the viewport sits at.
- Edit menu: Undo / Redo (up to 50 steps each).

## Scene format

Scenes are JSON files: objects (shape, size, position, rotation, scale,
material), cameras (name, position, look at), lights (key / fill / rim /
ambient), and animation (frames per second, end frame, keyframes).
Shapes: box, sphere, cylinder, cone, torus, plane. Materials carry a
base color, roughness, metalness, and one procedural texture style
(none, or a style from the material library: 20 styles in 6 families —
wood, stone and marble, brick and concrete, metal, fabric and leather,
ground). Procedural textures are generated in code; no image files are
used. Scenes written before the library still load: the three old
texture names (wood_grain, brushed_metal, worn_leather) map to their
matching new styles.

Emitters (T16/T17/T18) are objects of kind `emitter`: a particle source
with a shape (point / box / sphere / cone), rate, lifetime, speed,
size, colors, opacity fade, gravity factor, drag, wind response,
seed, and preset (fire / smoke / sparks / rain / snow / dust /
water). The particle motion is an analytic closed-form sim —
scrubbing to any frame gives identical particles — and the scene's
top-level `world` section holds the gravity and wind it runs under.
An emitter that bounces off the floor or pulls toward another object
runs in stepped mode instead (T18): a fixed-timestep deterministic
re-simulation from frame 0, memoized per frame, so scrubbing stays
exact in both directions. Stepped mode adds true curl turbulence
(a divergence-free noise field, so smoke swirls instead of just
wobbling), floor bounce with restitution (a slow bounce settles
into a rest on the floor instead of micro-bouncing), and attraction
or repulsion toward a target object (a spring pull, so sparks can
orbit it). The mode is automatic from the parameters — nothing to
pick. Particles render in the viewport (T17): one merged draw per
frame, alpha blending for smoke / dust / snow / rain / water,
additive blending for fire / sparks, so fire glows and smoke fades
in and out over its life. An emitter still shows its small marker
box — that is the selection handle. Open
`examples/particles-demo.json` for a fire and a smoke emitter over
a plain ground.

Materials carry physics too (T19): every material has a density
(g/cm^3), friction, and restitution alongside its color, roughness,
metalness, and texture, so a steel ball and a wooden ball will fall
and bounce differently once the rigid-body sim lands (T20). The
Material section has a preset picker (balsa / pine / oak wood,
cork, steel, aluminium, copper, gold, stone, glass, rubber,
plastic, ice, cloth, water) that fills the three numbers in one
tap; editing any number by hand flips the picker to custom, and the
preset name is never stored — only the numbers are, so a scene file
stays self-describing. Ticking "also set look" before picking a
preset also sets the roughness and metalness from the preset's
visual hint; unticked, the look is never touched. Old scenes load
with the defaults (1.0 / 0.5 / 0.3).

Hovering a preset in that picker — or a texture in the texture
dropdown — previews it on the selected object in the viewport
(UI12): see the material before committing to it. Click applies;
moving away restores the previous material. A preview pushes no
undo step and never touches the scene — it lives only in the
viewport until the dropdown closes or a pick applies.

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
The headless suite needs no PySide6: on a bare system the Qt tests skip
and the rest run.

`test.sh` runs the suite in groups, not one process: a single run grows
too big under load (a Qt test process grows about 50 MB per test and
gets killed), so each group runs in its own process and the script
fails if any group fails. The groups are A2/B2 (headless) and A1/C1/C2/
C3/D1/D2/C6/C7/C8/C9/C10/C11/C12/C13/C14/C15/C16/C17/C18/C19/C20/C21/C22/C23/C24/C25/C26/C27/C28 (Qt, only with `LIGHT_ENGINE_QT_TESTS=1`); the drag
tests run as two method lists (D1/D2) for the same reason. CI runs the
headless suite on every push (`.github/workflows/ci.yml`).

## What v0.1 does not do yet

- No sculpting, no physics, no particles.
- No video export (PNG frame sequences only).
- No gizmo dragging in the viewport (transforms are numeric).
- No shader-based rendering (fixed-function OpenGL 2.1).
