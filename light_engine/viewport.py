"""The live 3D viewport: a real OpenGL view in a Qt window. No web tech.

OpenGL 3.3 Core profile, shader-based rendering (T1). The orbit camera
(drag orbit, right-drag pan, scroll zoom) and the grid + axes follow the
pattern in the Design Shell v1.0 Bench viewport (design-shell/
designshell/qtbench.py). This is new code written for this engine;
nothing is imported from Design Shell. What differs here: geometry lives
in GPU vertex buffers (VBOs) with one VAO each, rebuilt only when an
object changes, so paintGL never re-tessellates the scene. Objects
routed through the indexed path — smooth-shaded objects (T3) and
anything forced via set_indexed — also get an index buffer (IBO) and
draw with glDrawElements; everything else draws with glDrawArrays
from the triangle soup. Mesh objects (T7) always draw indexed, from
the .mesh.json asset file (see meshfile.py), cached once per scene
by resolved path; they cast and receive shadows through the same
VAOs as primitives. One shadow map (T5): the key light only, a
1024x1024 depth map rendered each paint from the light's viewpoint,
sampled in the main pass with a 3x3 PCF. Per-object cast/receive
flags decide who casts and who receives. Gizmo (T12): a Move/Rotate/
Scale gizmo on the selected object, drawn with the lines program only
(cones and cubes are wireframe outlines), constant screen size from
the camera distance, depth test off so it is never hidden inside the
object. Renders (render_frame_to) suppress it — it is
interactive-viewport only. Gizmo dragging (T13): the handles are
pickable (ray vs cylinder / ring / AABB, from the same constants
the draw uses), a press on a handle drags that axis — Move on a
camera-perpendicular plane, Rotate on the ring plane, Scale by
distance from the centre — Shift snaps (10 cm / 15 deg / 0.1), the
hovered handle brightens, and the inspector follows live. Drags
write the transform directly; one undo step per drag (T14), armed
on press and committed on release through the window's live-edit
snapshot. Particles (T17): every live particle of every emitter is
merged into one interleaved VBO per frame (position + color + size +
alpha) and drawn with the points program in two calls by blend mode
(alpha for smoke/dust/snow/rain/water, additive for fire/sparks),
depth test on, depth write off. A scene with no live particles skips
the draw entirely, so the tavern renders exactly as before.

Frame stats (UI10): every paint is timed (one perf_counter pair per
paint — nothing measurable) and the last 30 costs feed a rolling
average, shared through frame_stats() for the window's FPS readout
and the tuner below. Self-tuning quality (UI11): when paints arrive
steadily and the rolling average stays over the 24 fps budget, the
viewport steps down a quality ladder (smaller shadow map, shadows
off, lower render scale) and steps back up when frames are fast
again — hysteresis-guarded, no user-visible setting. Exports always
render at full quality (full_quality), whatever level the tuner
sits at.

Units are cm, z-up.
"""
from __future__ import annotations

import contextlib
import math
import os
import time
from array import array
from collections import deque

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject,
                              QOpenGLFunctions_3_3_Core, QOpenGLShader,
                              QOpenGLShaderProgram, QOpenGLTexture,
                              QOpenGLVertexArrayObject)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
import shiboken6

from . import animation as anim_mod
from . import meshfile as meshfile_mod
from . import particles as particles_mod
from . import shaders as shader_mod
from .geometry import (GIZMO_MIN_SCALE, GIZMO_MODES, GIZMO_MOVE_TIP,
                       GIZMO_PICK_TOLERANCE, GIZMO_RING_RADIUS,
                       GIZMO_SCALE_CUBE_CENTER, GIZMO_SCALE_CUBE_HALF,
                       GIZMO_SNAP_MOVE, GIZMO_SNAP_ROTATE, GIZMO_SNAP_SCALE,
                       _gizmo_axis_frame, apply_matrix, compose_matrix,
                       gizmo_lines, gizmo_size, local_bounds, mat_inverse,
                       ray_hits_aabb, ray_vs_cylinder_segment,
                       ray_vs_plane, ray_vs_ring, tessellate,
                       tessellate_indexed)
from .scene import default_smooth_shading, hex_to_rgb
from .textures import (build_qimage, layout_affects_bake,
                       layout_shader_mode, merged_layout_size,
                       merged_params, resolve_layout, resolve_style,
                       wear_uniforms)

# GL enum values (kept numeric, as in the Bench pattern, so the calls read
# the same as the OpenGL docs). Core profile only — no fixed-function
# lighting or material enums exist here by design.
GL_DEPTH_TEST = 0x0B71
GL_PROGRAM_POINT_SIZE = 0x8642
GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_TRIANGLES = 0x0004
GL_LINES = 0x0001
GL_FLOAT = 0x1406
GL_UNSIGNED_INT = 0x1405
# Particle rendering (T17) enums: points + blending.
GL_POINTS = 0x0000
GL_BLEND = 0x0BE2
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_ONE = 0x0001

# Interleaved particle VBO layout (T17): position (3) + color (3) +
# size (1) + alpha (1) = 8 floats, matching particles.PARTICLE_FLOATS
# and the points program's attribute locations 0-3.
PARTICLE_STRIDE = 32
# The points vertex shader attenuates size by distance with
# uPointScale = viewport_height / (2 * tan(fov / 2)); the viewport's
# field of view is 45 degrees (see _matrices).
POINTS_FOV_DEGREES = 45.0
# Shadow map (T5) enums.
GL_TEXTURE0 = 0x84C0
GL_TEXTURE1 = 0x84C1
GL_TEXTURE_2D = 0x0DE1
# Render-scale blit (UI11): linear filtering on the scaled FBO's
# texture, so the upscale is smooth, not blocky.
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_MAG_FILTER = 0x2800
GL_LINEAR = 0x2601
GL_FRAMEBUFFER = 0x8D40
GL_DEPTH_ATTACHMENT = 0x8D00
GL_FRAMEBUFFER_COMPLETE = 0x8CD5
GL_NONE = 0x0
GL_POLYGON_OFFSET_FILL = 0x8037

# Interleaved object VBO layout: position (3) + normal (3) + uv (2).
VERTEX_STRIDE = 32

# One shadow map, key light only, 1024x1024 (T5).
SHADOW_MAP_SIZE = 1024

# Frame stats (UI10) + self-tuning quality (UI11).
# The rolling average covers the last 30 paints; the readout appears
# once 10 paints are measured (a single-frame grab shows the
# placeholder, never a half-formed number).
FRAME_STATS_WINDOW = 30
FRAME_STATS_MIN_SAMPLES = 10
# The fps floor is 24 fps (T20's promise): a paint slower than this
# budget cannot hold it. The step-up threshold sits comfortably
# below the budget, so the tuner cannot flicker between levels.
FRAME_BUDGET_MS = 1000.0 / 24.0
FRAME_FAST_MS = 20.0
# Hysteresis: 30 consecutive slow samples to step down, 60
# consecutive fast samples to step back up (recovery is twice as
# slow as relief, so a scene hovering near the budget does not
# oscillate — and the dead band between the two thresholds does the
# rest). After a level change, 30 paints are ignored while the
# rolling window flushes the old level's costs.
TUNER_DOWN_FRAMES = 30
TUNER_UP_FRAMES = 60
TUNER_SETTLE_FRAMES = FRAME_STATS_WINDOW
# Only sustained painting counts toward the tuner: a paint is a
# load sample only when the app barely idled before it — the gap
# from the END of the previous paint to the START of this one
# must be under 500 ms. Playback at 24 fps idles ~0 ms between
# paints (the next frame is requested as soon as the last
# finishes); isolated demand paints — an orbit, a grab — idle
# for seconds and never engage the tuner. The gap is
# end-to-start, not start-to-start, so a slow paint does not
# count against itself: a heavy scene painting back-to-back at
# 150 ms a frame idles ~0 ms and is clearly sustained load,
# while a light scene painted once every 400 ms idles 400 ms
# and is not. (A start-to-start 200 ms window was tried first
# and failed on the real heavy scene: the drive loop idles
# ~195 ms mean / ~261 ms max between paints under sandbox
# load, so every paint read as isolated and the tuner never
# engaged. 500 ms has 2x margin over that max and is far below
# the seconds separating isolated paints.)
TUNER_CONTINUOUS_GAP_MS = 500.0
# Quality ladder (UI11): (shadow map size or None for off, render
# scale). Each level is strictly cheaper than the one before it:
# full quality, smaller shadow map, shadows off, then two steps of
# lower render scale.
QUALITY_LEVELS = ((1024, 1.0), (512, 1.0), (None, 1.0),
                  (None, 0.75), (None, 0.5))

# Render-scale blit (UI11): a fullscreen triangle that samples the
# scaled framebuffer's texture, upscaling it to the widget. No
# vertex attributes — positions come from gl_VertexID — so it draws
# with an empty VAO bound. Compiled inline here (viewport.py), not
# in shaders.py: the blit is a viewport mechanism, not a scene
# program.
_BLIT_VERT = """#version 330 core
out vec2 vUv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vUv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""
_BLIT_FRAG = """#version 330 core
in vec2 vUv;
uniform sampler2D uTex;
out vec4 fragColor;
void main() { fragColor = texture(uTex, vUv); }
"""

# The grid + axes sit 0.5 cm below the ground plane (z = 0). Drawn at
# exactly z = 0 they z-fight with any floor plane there and the grid
# lines bleed through the floor — visible in v0.1 too, once you look.
# 0.5 cm is far above depth-buffer precision at scene scale and far
# below anything a user can see.
GRID_Z = -0.5


def _interleaved(mesh):
    """Pack position/normal/uv soup into one float array (8 floats/vert)."""
    pos, nor, uv = mesh["positions"], mesh["normals"], mesh["uvs"]
    verts = len(pos) // 3
    flat = array("f", [0.0]) * (verts * 8)
    for i in range(verts):
        base = i * 8
        flat[base] = pos[i * 3]
        flat[base + 1] = pos[i * 3 + 1]
        flat[base + 2] = pos[i * 3 + 2]
        flat[base + 3] = nor[i * 3]
        flat[base + 4] = nor[i * 3 + 1]
        flat[base + 5] = nor[i * 3 + 2]
        flat[base + 6] = uv[i * 2]
        flat[base + 7] = uv[i * 2 + 1]
    return flat, verts


def _set_scalar(gl, prog, name, value):
    """Set a float/int uniform by name, with raw GL calls.

    PySide6's QOpenGLShaderProgram.setUniformValue silently drops
    float values set by location: the call resolves to the color
    overload (whose signature accepts typing.Any), which a float
    uniform ignores — roughness, metalness and exposure all read as
    0 that way. T4 caught it because an exposure of 0 renders the
    whole scene black; the same bug had left roughness and metalness
    stuck at 0 since T1. The raw glUniform1f/glUniform1i calls set
    the value for real. The program must be bound when this runs,
    as it is in _draw_objects."""
    loc = prog.uniformLocation(name)
    if loc < 0:
        return
    if isinstance(value, int):
        gl.glUniform1i(loc, value)
    else:
        gl.glUniform1f(loc, float(value))


def _line_vao(gl, vertices):
    """Build a VAO+VBO for lines from interleaved pos(3)+color(3)
    floats. Returns (vao, vbo, vertex count)."""
    flat = array("f", vertices)
    vao = QOpenGLVertexArrayObject()
    vao.create()
    vao.bind()
    buf = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
    buf.create()
    buf.bind()
    buf.allocate(flat.tobytes(), len(flat.tobytes()))
    # PySide6 quirk (verified in the sandbox): normalized must be an int
    # (0), not False, and the offset must be a shiboken VoidPtr — a plain
    # int offset is rejected even though the signature prints as int.
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, GL_FLOAT, 0, 24, shiboken6.VoidPtr(0))
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 3, GL_FLOAT, 0, 24, shiboken6.VoidPtr(12))
    buf.release()
    vao.release()
    return vao, buf, len(flat) // 6


class Viewport(QOpenGLWidget):
    object_selected = QtCore.Signal(str)  # object id, or "" for none
    # Fired on every gizmo-drag step (T13): the window refreshes the
    # inspector from the object, so the spin boxes follow the drag
    # live. No payload — the selected object is the one that moved.
    transform_changed = QtCore.Signal()
    # Gizmo-drag undo (T14): the window arms its live-edit snapshot
    # when a drag starts and commits it as one undo step when the
    # drag ends. No payload — the snapshot covers the whole scene.
    gizmo_drag_started = QtCore.Signal()
    gizmo_drag_finished = QtCore.Signal()
    # Frame stats (UI10): fired after every paint that was measured.
    # The window updates its FPS readout from frame_stats(); no
    # payload — the numbers live on the viewport.
    frame_stats_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = QtGui.QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        self.setFormat(fmt)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        # Mouse tracking (T13): hover highlighting needs move events
        # with no button held, which Qt only sends when tracking is
        # on. (QTest sends the events directly, so tests pass either
        # way — this is for real use.)
        self.setMouseTracking(True)
        self._scene = None
        self._base_path = None  # folder the scene was loaded from;
        # mesh_file paths resolve against it (T7). None for a fresh
        # unsaved scene, which has no mesh files to resolve.
        self._gl = None
        # Mesh assets (T7): _mesh_assets maps a resolved asset path to
        # its loaded mesh dict (loaded once per scene, shared by every
        # object that references the file); _mesh_paths maps an object
        # id to its resolved path (or None when the file cannot load —
        # a paint never crashes on a mesh, it just skips the object).
        self._mesh_assets = {}
        self._mesh_paths = {}
        self._programs = {}  # name -> QOpenGLShaderProgram
        self._shader_error = None  # plain message if init failed, else None
        self._shader_error_shown = False
        # object id -> (vbo, vao, ibo or None, draw count, indexed, key)
        # draw count is a vertex count for soup objects (glDrawArrays) or
        # an index count for indexed objects (glDrawElements).
        self._vbos = {}
        # Object ids routed through the indexed mesh path (T2 test hook).
        # Mesh objects are indexed by definition (T7, via _is_indexed)
        # and never need this hook.
        self._indexed = set()
        self._textures = {}  # (texture name, base color, params, bond) -> QOpenGLTexture
        # Hover-preview (UI12): (object id, material overrides) or
        # None. While set, that one object renders with the
        # overrides instead of its stored material; the scene is
        # untouched, so leaving the preview restores the object
        # exactly. The window sets this on combo highlight and
        # clears it when the popup closes or a pick applies.
        self._material_preview = None
        # Shadow map (T5): one 1024x1024 depth map from the key light.
        # _shadow_fbo is a QOpenGLFramebufferObject (NoAttachment) whose
        # handle carries the depth texture; _shadow_tex is the depth
        # texture itself. _shadow_error is set (plain words) when the
        # map could not be built — the scene still renders, shadowless.
        # _shadow_size is the size the map is built at now (UI11: the
        # tuner rebuilds it smaller, or drops it entirely).
        self._shadow_fbo = None
        self._shadow_tex = None
        self._shadow_error = None
        self._shadow_size = SHADOW_MAP_SIZE
        # Frame stats (UI10) + self-tuning quality (UI11).
        # _frame_times holds the last FRAME_STATS_WINDOW paint costs
        # in ms. _skip_next_frame_time drops the first paint after a
        # scene load: it builds every VBO at once (a one-time setup
        # cost, ~1 s on the tavern in the sandbox), which says
        # nothing about the scene's sustained frame rate.
        self._frame_times = deque(maxlen=FRAME_STATS_WINDOW)
        self._skip_next_frame_time = True
        self._last_paint_end = None
        # Test hook (UI11): extra seconds slept inside each paint,
        # so a test can drive the tuner with a slow scene without
        # building one. Always 0 in real use.
        self._paint_delay = 0.0
        # Tuner state (UI11): the current ladder level, the
        # consecutive slow/fast streaks, and the settle countdown
        # after a level change. _quality_override counts nested
        # full_quality() uses (exports); while it is nonzero the
        # effective level is 0 and the tuner does not count.
        self._quality_level = 0
        self._tuner_slow = 0
        self._tuner_fast = 0
        self._tuner_settle = 0
        self._quality_override = 0
        # Render scale (UI11): the scaled FBO + blit program, built
        # lazily on the first scaled paint and rebuilt when the
        # widget size or the scale changes. _target_fbo/_target_size
        # are the framebuffer + viewport the current paint draws
        # into (the widget's own FBO at full scale); the shadow pass
        # restores them, not the widget FBO directly.
        self._scale_fbo = None
        self._scale_fbo_size = None
        self._blit_program = None
        self._blit_vao = None
        self._target_fbo = None
        self._target_size = (0, 0)
        self._grid = None    # (vao, vbo, vertex count, span key)
        self._sel_vao = None  # unit-cube edge VAO for the selection box
        self._sel_vbo = None
        self._sel_count = 0
        # Particles (T17): one VAO + one VBO, created lazily on the
        # first paint that has live particles and reused after — the
        # VBO's storage is re-uploaded every frame, the VAO's
        # attribute bindings never change. _particle_count is the
        # live count from the last paint (0 for a scene with no
        # emitters, or before the first paint).
        self._particle_vao = None
        self._particle_vbo = None
        self._particle_count = 0
        self._frame = 0
        self.selected_id = None
        # Gizmo (T12): the transform mode, default Move. One VAO per
        # (mode, highlight) pair is built lazily and cached in
        # _gizmo_vaos (T13 adds the highlight dimension: the hovered
        # handle draws brighter, so each mode has one plain VAO plus
        # one per hovered axis).
        # _gizmo_suppressed hides the gizmo while a render is being
        # grabbed (render_frame_to sets it): the gizmo is
        # interactive-viewport only and never appears in an exported
        # frame. The selection box is NOT suppressed — it has always
        # been part of the grabbed viewport, and T12 does not change
        # that.
        # _gizmo_hover (T13) is the axis under the pointer ("x", "y",
        # "z") or None; _gizmo_drag is the drag state dict while a
        # handle is being dragged, else None.
        self.gizmo_mode = "move"
        self._gizmo_vaos = {}
        self._gizmo_suppressed = False
        self._gizmo_hover = None
        self._gizmo_drag = None
        # Exposure for the ACES tone mapping (T4). Fixed at 1.0 for
        # now — an exposure control is a later item, not T4.
        self._exposure = 1.0
        # Orbit camera: target + distance + azimuth/elevation (degrees).
        self.target = (0.0, 0.0, 0.0)
        self.distance = 400.0
        self.azimuth = -126.0
        self.elevation = 29.0
        self._drag_button = None
        self._drag_pos = None
        self._press_pos = None

    # -- scene ----------------------------------------------------------
    def set_scene(self, scene, base_path=None):
        self._scene = scene
        self._base_path = base_path
        self._frame = 0
        self.selected_id = None
        self._mesh_assets = {}
        self._mesh_paths = {}
        # A new scene invalidates a preview (UI12): the previewed
        # object belongs to the old scene.
        self._material_preview = None
        self._drop_vbos()
        # A new scene has its own load: the old scene's frame times
        # and tuner streaks describe nothing about it, so they go —
        # and the next paint (a full VBO rebuild) is skipped, the
        # same as the first paint after construction.
        self._frame_times.clear()
        self._tuner_slow = 0
        self._tuner_fast = 0
        self._tuner_settle = 0
        self._skip_next_frame_time = True
        self.frame()
        self.update()

    # -- mesh assets (T7) ----------------------------------------------
    def _mesh_for(self, obj):
        """The loaded mesh dict for a mesh object, or None when the
        asset cannot load (missing file, corrupt file, no base path).
        Each asset loads once per scene and is cached by resolved
        path; the per-object memo means a failed load is not retried
        on every paint."""
        if obj.get("shape") != "mesh":
            return None
        obj_id = obj["id"]
        if obj_id in self._mesh_paths:
            path = self._mesh_paths[obj_id]
            return self._mesh_assets.get(path) if path else None
        mesh_file = obj.get("mesh_file")
        path = None
        if mesh_file and self._base_path:
            path = mesh_file if os.path.isabs(mesh_file) \
                else os.path.join(self._base_path, mesh_file)
            try:
                if path not in self._mesh_assets:
                    self._mesh_assets[path] = meshfile_mod.load_mesh(path)
            except (OSError, ValueError):
                path = None
        self._mesh_paths[obj_id] = path
        return self._mesh_assets.get(path) if path else None

    def mesh_triangle_counts(self):
        """{object id: triangle_count} for the scene's mesh objects,
        from the assets' recorded counts. Loads any asset not yet
        cached — the window's triangle readout uses this."""
        counts = {}
        if self._scene:
            for obj in self._scene["objects"]:
                if obj.get("shape") == "mesh":
                    mesh = self._mesh_for(obj)
                    if mesh is not None:
                        counts[obj["id"]] = mesh["triangle_count"]
        return counts

    def _object_bounds(self, obj):
        """Local bounds for picking, framing, selection and shadows:
        primitives use their analytic bounds; mesh objects use the
        bounds of the asset's vertices (an unloadable mesh falls back
        to the same 100 cm box local_bounds uses for unknown
        shapes)."""
        if obj.get("shape") == "mesh":
            mesh = self._mesh_for(obj)
            if mesh is not None:
                return mesh["bounds"]
            return (-50.0, -50.0, -50.0), (50.0, 50.0, 50.0)
        return local_bounds(obj["shape"], obj.get("params", {}))

    def refresh(self):
        """Rebuild changed buffers on the next paint. Cheap: only objects
        whose shape/params changed get re-tessellated."""
        self.update()

    def set_frame(self, frame):
        self._frame = int(frame)
        self.update()

    def current_frame(self):
        return self._frame

    # -- gizmo (T12) -----------------------------------------------------
    def set_gizmo_mode(self, mode):
        """Switch the gizmo mode: "move", "rotate", or "scale".
        Raises ValueError on an unknown mode. No-op when the mode is
        already current — the toolbar buttons and shortcuts both call
        here, and a redundant call must not repaint."""
        if mode not in GIZMO_MODES:
            raise ValueError(
                f"unknown gizmo mode: {mode!r} "
                f"(expected one of {', '.join(GIZMO_MODES)})")
        if mode == self.gizmo_mode:
            return
        self.gizmo_mode = mode
        self.update()

    def set_gizmo_suppressed(self, suppressed):
        """Hide the gizmo while a render is being grabbed.
        render_frame_to sets this around its grab and clears it
        after; either change repaints, so the gizmo comes back in
        the live view as soon as the grab is done."""
        self._gizmo_suppressed = bool(suppressed)
        self.update()

    def set_material_preview(self, obj_id, overrides):
        """Hover-preview (UI12): render one object with these
        material values instead of its stored ones — a fragment
        of a material dict (roughness, metalness, texture,
        base_color). The scene is untouched: the preview lives
        only in the viewport until clear_material_preview. The
        overrides are copied, so the caller keeps its own dict."""
        self._material_preview = (obj_id, dict(overrides))
        self.update()

    def clear_material_preview(self):
        """End a hover-preview (UI12): the object renders from
        its stored material again. Idempotent — safe to call
        from every exit path, and a no-op repaint when nothing
        is previewed."""
        if self._material_preview is not None:
            self._material_preview = None
            self.update()

    def _gizmo_center(self):
        """Sampled world position of the selected object — the point
        the gizmo is drawn on (T12) and picked and dragged around
        (T13) — or None with no selection. Sampled, not the base
        transform, so the gizmo follows the animation."""
        if not self._scene or not self.selected_id:
            return None
        obj = next((o for o in self._scene["objects"]
                    if o["id"] == self.selected_id), None)
        if obj is None:
            return None
        model = self._model_matrix(obj)
        return (model[0][3], model[1][3], model[2][3])

    def _destroy_entry(self, entry):
        """Destroy one object's VAO + VBO + optional IBO. Caller must
        have the GL context current (paint time, or via the
        makeCurrent pattern in _drop_vbos / cleanup_gl)."""
        buf, vao, ibo = entry[0], entry[1], entry[2]
        for obj in (vao, buf, ibo):
            if obj is None:
                continue
            try:
                obj.destroy()
            except Exception:
                pass

    def _drop_vbos(self):
        # GL objects must be destroyed with the context current, or Qt
        # warns at shutdown ("destroyed without a current context").
        made_current = False
        try:
            if self.context() is not None:
                self.makeCurrent()
                made_current = True
        except Exception:
            pass
        for entry in self._vbos.values():
            self._destroy_entry(entry)
        self._vbos = {}
        if made_current:
            try:
                self.doneCurrent()
            except Exception:
                pass

    def _prune_vbos(self):
        """Destroy entries for objects no longer in the scene. Called
        from _draw_objects, where the context is current. Without this,
        deleting an object leaked its VAO/VBO until the next scene
        change — the one lifecycle gap T1 left behind."""
        if self._scene is None:
            return
        live = {o["id"] for o in self._scene["objects"]}
        stale = [oid for oid in self._vbos if oid not in live]
        for oid in stale:
            self._destroy_entry(self._vbos.pop(oid))
            self._indexed.discard(oid)
            self._mesh_paths.pop(oid, None)

    def _is_indexed(self, obj):
        """True when the object draws through the indexed path: mesh
        objects (T7) always do — their asset data is indexed by
        definition. Smooth shading (T3) routes through it, and the
        set_indexed hook can force any object onto it. A missing flag
        falls back to the shape default, so hand-built object dicts
        behave."""
        if obj.get("shape") == "mesh":
            return True
        if obj["id"] in self._indexed:
            return True
        return bool(obj.get("smooth_shading",
                            default_smooth_shading(obj.get("shape", "box"))))

    def set_indexed(self, object_id, indexed=True):
        """Force one object onto (or off) the indexed mesh path (T2
        test hook). Smooth-shaded objects and mesh objects are
        indexed regardless of this hook. The mesh key covers the
        indexed flag, so the next paint rebuilds the object's buffers
        through the other builder. No GL calls here — the rebuild
        happens at paint time."""
        if indexed:
            self._indexed.add(object_id)
        else:
            self._indexed.discard(object_id)
        self.update()

    def cleanup_gl(self):
        """Destroy all GL objects with the context current. Called on
        window close so shutdown is quiet."""
        made_current = False
        try:
            if self.context() is not None:
                self.makeCurrent()
                made_current = True
        except Exception:
            pass
        for entry in self._vbos.values():
            self._destroy_entry(entry)
        self._vbos = {}
        self._indexed = set()
        if self._grid is not None:
            try:
                self._grid[0].destroy()
            except Exception:
                pass
            try:
                self._grid[1].destroy()
            except Exception:
                pass
            self._grid = None
        if self._sel_vao is not None:
            try:
                self._sel_vao.destroy()
            except Exception:
                pass
            self._sel_vao = None
        if self._sel_vbo is not None:
            try:
                self._sel_vbo.destroy()
            except Exception:
                pass
            self._sel_vbo = None
        for vao, buf, _count in self._gizmo_vaos.values():
            try:
                vao.destroy()
            except Exception:
                pass
            try:
                buf.destroy()
            except Exception:
                pass
        self._gizmo_vaos = {}
        if self._particle_vao is not None:
            try:
                self._particle_vao.destroy()
            except Exception:
                pass
            self._particle_vao = None
        if self._particle_vbo is not None:
            try:
                self._particle_vbo.destroy()
            except Exception:
                pass
            self._particle_vbo = None
        self._particle_count = 0
        for tex in self._textures.values():
            try:
                tex.destroy()
            except Exception:
                pass
        self._textures = {}
        if self._shadow_fbo is not None:
            try:
                self._shadow_fbo.destroy()
            except Exception:
                pass
            self._shadow_fbo = None
        if self._shadow_tex is not None:
            try:
                self._shadow_tex.destroy()
            except Exception:
                pass
            self._shadow_tex = None
        # Render scale (UI11): the scaled FBO + blit program have no
        # destroy() in this build — dropping the references frees
        # them. The blit VAO does have one.
        self._scale_fbo = None
        self._scale_fbo_size = None
        self._blit_program = None
        if self._blit_vao is not None:
            try:
                self._blit_vao.destroy()
            except Exception:
                pass
            self._blit_vao = None
        self._programs = {}
        self._mesh_assets = {}
        self._mesh_paths = {}
        if made_current:
            try:
                self.doneCurrent()
            except Exception:
                pass

    def _mesh_key(self, obj):
        # The indexed flag is part of the key, so switching an object
        # between the soup and indexed paths — by toggling smooth
        # shading or via set_indexed — rebuilds its buffers. A mesh
        # object's key is its asset file: re-pointing it at another
        # file rebuilds the buffers.
        if obj.get("shape") == "mesh":
            return ("mesh", obj.get("mesh_file", ""), True)
        return (obj["shape"], repr(sorted(obj.get("params", {}).items())),
                self._is_indexed(obj))

    def _ensure_vbo(self, obj):
        key = self._mesh_key(obj)
        entry = self._vbos.get(obj["id"])
        if entry is not None and entry[5] == key:
            return entry
        if entry is not None:
            self._destroy_entry(entry)
        indexed = self._is_indexed(obj)
        if obj.get("shape") == "mesh":
            # Mesh geometry comes from the asset (T7), always indexed.
            # An unloadable mesh draws nothing: an empty mesh builds
            # an empty VAO/IBO and glDrawElements with count 0 is a
            # no-op, so a paint never crashes on a bad mesh file.
            mesh = self._mesh_for(obj) or {"positions": [], "normals": [],
                                           "uvs": [], "indices": []}
        elif indexed:
            mesh = tessellate_indexed(obj["shape"], obj.get("params", {}))
        else:
            mesh = tessellate(obj["shape"], obj.get("params", {}))
        flat, verts = _interleaved(mesh)
        buf = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        buf.create()
        vao = QOpenGLVertexArrayObject()
        vao.create()
        vao.bind()
        buf.bind()
        buf.allocate(flat.tobytes(), len(flat.tobytes()))
        gl = self._gl
        # Attribute locations match the pbr_lite shader: 0 position,
        # 1 normal, 2 uv. See _line_vao for the PySide6 arg quirks.
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, GL_FLOAT, 0, VERTEX_STRIDE,
                                 shiboken6.VoidPtr(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, GL_FLOAT, 0, VERTEX_STRIDE,
                                 shiboken6.VoidPtr(12))
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 2, GL_FLOAT, 0, VERTEX_STRIDE,
                                 shiboken6.VoidPtr(24))
        buf.release()
        ibo = None
        count = verts
        if indexed:
            # The element-buffer binding is part of VAO state, so the
            # IBO must be bound while the VAO is bound — and released
            # only AFTER the VAO is released, or the release would
            # unbind it from the VAO.
            ibo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
            ibo.create()
            ibo.bind()
            idx = array("I", mesh["indices"])
            ibo.allocate(idx.tobytes(), len(idx.tobytes()))
            count = len(mesh["indices"])
        vao.release()
        if ibo is not None:
            ibo.release()
        entry = (buf, vao, ibo, count, indexed, key)
        self._vbos[obj["id"]] = entry
        return entry

    def _ensure_texture(self, name, base_color, params=None, layout=None):
        # The cache key includes the merged params (PO2): a knob
        # change bakes a new texture instead of reusing the old
        # one. Merged (defaults + overrides), so "no params" and
        # "params equal to the defaults" share one entry — and
        # the key is built from the same merge the bake uses, so
        # the two can never disagree. Each distinct param set
        # costs one 128x128 texture (~64 KB); the cache is
        # cleared on set_scene, so a long session stays small.
        # PO4: the key also carries the brick bond — the layout
        # changes the baked tile for brick only (other styles lay
        # out in the shader), so the bond is None for every other
        # style and those keys are unchanged apart from the extra
        # element. A bond change bakes a new texture; a shader
        # layout change needs no bake at all.
        merged = merged_params(name, params)
        bond = layout if layout_affects_bake(name) else None
        key = (name, base_color, tuple(sorted(merged.items())), bond)
        tex = self._textures.get(key)
        if tex is not None:
            return tex
        img = build_qimage(name, hex_to_rgb(base_color), size=128,
                           params=merged, layout=bond)
        tex = QOpenGLTexture(img)
        tex.setMinificationFilter(QOpenGLTexture.LinearMipMapLinear)
        tex.setMagnificationFilter(QOpenGLTexture.Linear)
        tex.setWrapMode(QOpenGLTexture.Repeat)
        self._textures[key] = tex
        return tex

    def span(self):
        if not self._scene or not self._scene["objects"]:
            return 200.0
        span = 0.0
        for obj in self._scene["objects"]:
            mn, mx = self._object_bounds(obj)
            size = max(mx[i] - mn[i] for i in range(3))
            span = max(span, size)
            pos = obj.get("position", [0, 0, 0])
            span = max(span, math.sqrt(sum(v * v for v in pos)) + size)
        return max(span, 50.0)

    def frame(self):
        if self._scene and self._scene["objects"]:
            cx = sum(o.get("position", [0, 0, 0])[0]
                     for o in self._scene["objects"]) / len(self._scene["objects"])
            cy = sum(o.get("position", [0, 0, 0])[1]
                     for o in self._scene["objects"]) / len(self._scene["objects"])
            cz = sum(o.get("position", [0, 0, 0])[2]
                     for o in self._scene["objects"]) / len(self._scene["objects"])
            self.target = (cx, cy, cz)
        else:
            self.target = (0.0, 0.0, 0.0)
        self.distance = self.span() * 1.6
        self.azimuth = -126.0
        self.elevation = 29.0
        self.update()

    # -- camera math -----------------------------------------------------
    def camera_position(self):
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        d = self.distance
        return (self.target[0] + d * math.cos(el) * math.cos(az),
                self.target[1] + d * math.cos(el) * math.sin(az),
                self.target[2] + d * math.sin(el))

    def set_camera_position(self, position, look_at):
        self.target = (float(look_at[0]), float(look_at[1]), float(look_at[2]))
        dx = float(position[0]) - self.target[0]
        dy = float(position[1]) - self.target[1]
        dz = float(position[2]) - self.target[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-6:
            return
        self.distance = dist
        self.elevation = math.degrees(math.asin(max(-1.0, min(1.0, dz / dist))))
        self.azimuth = math.degrees(math.atan2(dy, dx))
        self.update()

    def current_camera(self, name):
        eye = self.camera_position()
        return {"name": name,
                "position": [round(v, 2) for v in eye],
                "look_at": [round(v, 2) for v in self.target]}

    def _view_dir(self):
        eye = self.camera_position()
        dx = self.target[0] - eye[0]
        dy = self.target[1] - eye[1]
        dz = self.target[2] - eye[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        return (dx / length, dy / length, dz / length)

    def _pan(self, dx_pixels, dy_pixels):
        vx, vy, vz = self._view_dir()
        rx, ry = vy, -vx
        rlen = math.sqrt(rx * rx + ry * ry) or 1.0
        rx, ry = rx / rlen, ry / rlen
        ux, uy, uz = ry * vz, -rx * vz, rx * vy - ry * vx
        ulen = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
        ux, uy, uz = ux / ulen, uy / ulen, uz / ulen
        scale = self.distance * 0.0012
        self.target = (self.target[0] - rx * dx_pixels * scale
                       + ux * dy_pixels * scale,
                       self.target[1] - ry * dx_pixels * scale
                       + uy * dy_pixels * scale,
                       self.target[2] + uz * dy_pixels * scale)
        self.update()

    # -- GL --------------------------------------------------------------
    def initializeGL(self):
        gl = QOpenGLFunctions_3_3_Core()
        if not gl.initializeOpenGLFunctions():
            self._shader_error = (
                "This computer's graphics driver did not provide "
                "OpenGL 3.3, which Reblend needs to draw. Check for a "
                "graphics driver update.")
            QtCore.QTimer.singleShot(0, self._show_shader_error)
            return
        self._gl = gl
        gl.glEnable(GL_DEPTH_TEST)
        gl.glEnable(GL_PROGRAM_POINT_SIZE)
        gl.glClearColor(0.09, 0.09, 0.10, 1.0)
        try:
            self._programs = shader_mod.compile_all()
        except shader_mod.ShaderError as exc:
            self._shader_error = str(exc)
            QtCore.QTimer.singleShot(0, self._show_shader_error)
            return
        self._build_selection_vao()
        self._build_shadow_map()

    def _show_shader_error(self):
        """Plain error dialog, once, when the renderer could not start.
        Deferred out of initializeGL so it never pops during a paint."""
        if self._shader_error is None or self._shader_error_shown:
            return
        self._shader_error_shown = True
        QtWidgets.QMessageBox.critical(
            self, "Graphics error",
            "Reblend could not start its 3D renderer:\n\n"
            + self._shader_error)

    def resizeGL(self, width, height):
        if self._gl is not None:
            self._gl.glViewport(0, 0, max(1, width), max(1, height))

    def _matrices(self):
        width, height = max(1, self.width()), max(1, self.height())
        eye = self.camera_position()
        proj = QtGui.QMatrix4x4()
        proj.perspective(45.0, width / height,
                         max(0.1, self.distance / 100.0),
                         self.distance * 20.0 + self.span() * 10.0)
        view = QtGui.QMatrix4x4()
        view.lookAt(QtGui.QVector3D(*eye),
                    QtGui.QVector3D(*self.target),
                    QtGui.QVector3D(0.0, 0.0, 1.0))
        return proj, view

    def paintGL(self):
        gl = self._gl
        if gl is None or self._shader_error is not None:
            return
        # Frame stats (UI10): one perf_counter pair around the paint
        # — the measurement itself costs nothing measurable.
        start = time.perf_counter()
        try:
            self._paint_scene(gl)
        finally:
            self._record_frame(start)

    def _paint_scene(self, gl):
        # Test hook (UI11): a slow scene, simulated.
        if self._paint_delay:
            time.sleep(self._paint_delay)
        # The effective quality may have changed since the last
        # paint (a tuner step, or an export override): the shadow
        # map is rebuilt lazily here, where the context is current.
        shadow_size, scale = self._effective_quality()
        if shadow_size != self._shadow_size:
            self._rebuild_shadow_map(shadow_size)
        # The target this paint draws into: the widget's own FBO at
        # full scale, the scaled FBO below it (only when the blit
        # program is available — without it there is no way to get
        # the scaled image onto the widget, so the paint stays full
        # size instead of showing a stale frame).
        fbo = None
        if scale < 1.0 and self._ensure_blit_program() is not None:
            fbo = self._ensure_scale_fbo(scale)
        if fbo is not None:
            self._target_fbo = fbo.handle()
            self._target_size = self._scale_fbo_size
            gl.glBindFramebuffer(GL_FRAMEBUFFER, fbo.handle())
            gl.glViewport(0, 0, *self._scale_fbo_size)
        else:
            self._target_fbo = self.defaultFramebufferObject()
            self._target_size = (max(1, self.width()),
                                 max(1, self.height()))
        # Shadow pass first (T5): the scene from the key light into
        # the depth map. It runs on every paint, which is also every
        # scene change — Qt only repaints when something asked for
        # one — and it costs one more draw of the scene's triangles.
        light_space = None
        if self._scene and self._shadow_fbo is not None:
            light_space = self._light_space_matrix()
            self._render_shadow_map(gl, light_space)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        proj, view = self._matrices()
        self._draw_grid(gl, proj, view)
        if self._scene:
            self._draw_objects(gl, proj, view, light_space)
            # Particles (T17) draw after the opaque objects and
            # before the selection box and gizmo: they are transparent
            # scene content, so they need the depth buffer filled and
            # must not write to it themselves.
            self._draw_particles(gl, proj, view)
            if self.selected_id:
                self._draw_selection(gl, proj, view)
                if not self._gizmo_suppressed:
                    self._draw_gizmo(gl, proj, view)
        if fbo is not None:
            self._blit_to_widget(gl, fbo)

    # -- frame stats (UI10) + self-tuning quality (UI11) -------------
    def frame_stats(self):
        """(fps, ms_per_frame, samples): the rolling average paint
        cost over the last FRAME_STATS_WINDOW paints, as the frame
        rate the viewport can sustain. (None, None, n) while fewer
        than FRAME_STATS_MIN_SAMPLES paints are measured — the
        readout shows nothing until the average means something."""
        n = len(self._frame_times)
        if n < FRAME_STATS_MIN_SAMPLES:
            return None, None, n
        avg_ms = sum(self._frame_times) / n
        if avg_ms <= 0:
            return None, avg_ms, n
        return 1000.0 / avg_ms, avg_ms, n

    def quality_info(self):
        """(level, shadow_size, render_scale) for the effective
        quality right now — level 0 while an export override is
        active. Tests and the verify script read this; the UI never
        shows it (UI11 has no user-visible setting)."""
        shadow_size, scale = self._effective_quality()
        level = 0 if self._quality_override else self._quality_level
        return level, shadow_size, scale

    def _effective_quality(self):
        if self._quality_override:
            return QUALITY_LEVELS[0]
        return QUALITY_LEVELS[self._quality_level]

    def _record_frame(self, start):
        now = time.perf_counter()
        if self._skip_next_frame_time:
            self._skip_next_frame_time = False
            self._last_paint_end = now
            return
        # Sustained load is measured by IDLE time: how long the
        # app sat between the previous paint finishing and this
        # one starting. A slow paint does not count against
        # itself (see TUNER_CONTINUOUS_GAP_MS).
        continuous = (self._last_paint_end is not None
                      and (start - self._last_paint_end) * 1000.0
                      <= TUNER_CONTINUOUS_GAP_MS)
        self._last_paint_end = now
        self._record_frame_time((now - start) * 1000.0, continuous)

    def _record_frame_time(self, cost_ms, continuous):
        """Feed one measured paint cost (ms) into the rolling
        average and the tuner. Split from _record_frame so tests
        can drive the tuner with synthetic costs instead of real
        paints; continuous says whether this paint arrived close
        after the previous one (sustained load) or on its own."""
        self._frame_times.append(cost_ms)
        if self._quality_override == 0:
            self._tuner_step(continuous)
        self.frame_stats_changed.emit()

    def _tuner_step(self, continuous):
        if not continuous:
            # An isolated paint is not sustained load: it breaks
            # both streaks, so a single slow paint (an orbit, a
            # grab) can never step the quality down.
            self._tuner_slow = 0
            self._tuner_fast = 0
            return
        if self._tuner_settle > 0:
            self._tuner_settle -= 1
            return
        _fps, avg_ms, _n = self.frame_stats()
        if avg_ms is None:
            return
        if avg_ms > FRAME_BUDGET_MS:
            self._tuner_slow += 1
            self._tuner_fast = 0
            if (self._tuner_slow >= TUNER_DOWN_FRAMES
                    and self._quality_level < len(QUALITY_LEVELS) - 1):
                self._set_quality_level(self._quality_level + 1)
        elif avg_ms < FRAME_FAST_MS:
            self._tuner_fast += 1
            self._tuner_slow = 0
            if (self._tuner_fast >= TUNER_UP_FRAMES
                    and self._quality_level > 0):
                self._set_quality_level(self._quality_level - 1)
        else:
            # Between the thresholds: hold. Neither streak grows,
            # so a scene hovering near the budget never flickers.
            self._tuner_slow = 0
            self._tuner_fast = 0

    def _set_quality_level(self, level):
        level = max(0, min(len(QUALITY_LEVELS) - 1, level))
        if level == self._quality_level:
            return
        self._quality_level = level
        self._tuner_slow = 0
        self._tuner_fast = 0
        self._tuner_settle = TUNER_SETTLE_FRAMES
        # The shadow map + scaled FBO change lazily on the next
        # paint, where the GL context is current; this just asks
        # for that paint.
        self.update()

    @contextlib.contextmanager
    def full_quality(self):
        """Force full quality (level 0) for the duration: exports
        render at full quality even when the tuner has stepped
        down. Re-entrant — nested uses share one override, so an
        export loop and its per-frame grab can both hold it. The
        tuner does not count paints while the override is active
        (an export is a batch render, not interactive load), and
        its streaks restart when the override lifts."""
        self._quality_override += 1
        if self._quality_override == 1:
            self._tuner_slow = 0
            self._tuner_fast = 0
            self.update()
        try:
            yield
        finally:
            self._quality_override -= 1
            if self._quality_override == 0:
                self._tuner_slow = 0
                self._tuner_fast = 0
                self._tuner_settle = TUNER_SETTLE_FRAMES
                self.update()

    def _light_uniforms(self, prog):
        """Set the pbr_lite light uniforms from the scene lights."""
        lights = (self._scene or {}).get("lights", []) if self._scene else []
        ambient = (0.35, 0.35, 0.35)
        slots = {}
        for light in lights:
            kind = light.get("kind")
            if kind == "ambient":
                r, g, b = hex_to_rgb(light.get("color", "#ffffff"))
                inten = float(light.get("intensity", 0.35))
                ambient = (r * inten, g * inten, b * inten)
            elif kind in ("key", "fill", "rim"):
                slots[kind] = light
        prog.setUniformValue("uAmbient", QtGui.QVector3D(*ambient))
        eye = self.camera_position()
        prog.setUniformValue("uViewPos", QtGui.QVector3D(*eye))
        for i, slot in enumerate(("key", "fill", "rim")):
            light = slots.get(slot)
            if light is None:
                pos, color = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
            else:
                r, g, b = hex_to_rgb(light.get("color", "#ffffff"))
                inten = float(light.get("intensity", 0.8))
                p = light.get("position", [0, 0, 0])
                pos = (float(p[0]), float(p[1]), float(p[2]))
                color = (r * inten, g * inten, b * inten)
            prog.setUniformValue(f"uLightPos[{i}]", QtGui.QVector3D(*pos))
            prog.setUniformValue(f"uLightColor[{i}]", QtGui.QVector3D(*color))

    # -- shadow map (T5) -------------------------------------------------
    def _build_shadow_map(self, size=SHADOW_MAP_SIZE):
        """Create the depth-only FBO + depth texture at (size, size).

        PySide6 gap, found while building this: this PySide6 build
        binds no glGenFramebuffers / glGenTextures on any
        QOpenGLFunctions class, so the FBO name comes from a
        QOpenGLFramebufferObject (NoAttachment) and the texture from
        a QOpenGLTexture in a depth format; the attach itself is the
        one raw call, glFramebufferTexture2D, which IS bound. If
        anything here fails, the scene still renders — shadowless —
        and _shadow_error says why in plain words.
        """
        gl = self._gl
        try:
            tex = QOpenGLTexture(QOpenGLTexture.Target2D)
            tex.setFormat(QOpenGLTexture.D32F)
            tex.setSize(size, size)
            tex.setMinificationFilter(QOpenGLTexture.Nearest)
            tex.setMagnificationFilter(QOpenGLTexture.Nearest)
            tex.setWrapMode(QOpenGLTexture.ClampToEdge)
            tex.allocateStorage()
            if not tex.isCreated():
                raise RuntimeError("the depth texture did not allocate")
            fbo = QOpenGLFramebufferObject(
                QtCore.QSize(size, size),
                QOpenGLFramebufferObject.NoAttachment)
            if not fbo.isValid():
                raise RuntimeError("the shadow framebuffer did not create")
            gl.glBindFramebuffer(GL_FRAMEBUFFER, fbo.handle())
            gl.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                      GL_TEXTURE_2D, tex.textureId(), 0)
            gl.glDrawBuffer(GL_NONE)
            gl.glReadBuffer(GL_NONE)
            status = gl.glCheckFramebufferStatus(GL_FRAMEBUFFER)
            gl.glBindFramebuffer(GL_FRAMEBUFFER,
                                 self.defaultFramebufferObject())
            if status != GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(
                    f"the shadow framebuffer is incomplete "
                    f"(status {status:#x})")
            if gl.glGetError() != 0:
                raise RuntimeError("a graphics error building the map")
        except Exception as exc:  # noqa: BLE001 - degraded, not fatal
            self._shadow_fbo = None
            self._shadow_tex = None
            self._shadow_error = (
                "Shadows are off: the shadow map could not be built "
                f"({exc}). The scene still renders without them.")
            return
        self._shadow_fbo = fbo
        self._shadow_tex = tex
        self._shadow_error = None

    def _rebuild_shadow_map(self, size):
        """Rebuild the shadow map at a new size (UI11), or drop it
        when size is None (shadows off). Runs inside a paint, where
        the context is current. A rebuild that fails leaves the
        scene shadowless with _shadow_error set, the same degraded
        path as the first build."""
        if self._shadow_fbo is not None:
            try:
                self._shadow_fbo.destroy()
            except Exception:
                pass
            self._shadow_fbo = None
        if self._shadow_tex is not None:
            try:
                self._shadow_tex.destroy()
            except Exception:
                pass
            self._shadow_tex = None
        self._shadow_size = size
        if size is None:
            self._shadow_error = None
            return
        self._build_shadow_map(size)

    def _key_light_position(self):
        """The key light's position, or the default key position when
        the scene has no key light."""
        if self._scene:
            for light in self._scene.get("lights", []):
                if light.get("kind") == "key":
                    p = light.get("position", [0, 0, 0])
                    return (float(p[0]), float(p[1]), float(p[2]))
        return (-300.0, -300.0, 400.0)

    def _scene_aabb(self):
        """World-space bounds of the scene at the current frame:
        every object's local bounds corners through its model matrix
        (which samples the animation). Falls back to a 200 cm box
        for an empty scene."""
        mn = [float("inf")] * 3
        mx = [float("-inf")] * 3
        if self._scene:
            for obj in self._scene["objects"]:
                lo, hi = self._object_bounds(obj)
                model = self._model_matrix(obj)
                for cx in (lo[0], hi[0]):
                    for cy in (lo[1], hi[1]):
                        for cz in (lo[2], hi[2]):
                            w = apply_matrix(model, (cx, cy, cz))
                            for i in range(3):
                                mn[i] = min(mn[i], w[i])
                                mx[i] = max(mx[i], w[i])
        if mn[0] == float("inf"):
            return (-100.0, -100.0, -100.0), (100.0, 100.0, 100.0)
        return tuple(mn), tuple(mx)

    def _light_space_matrix(self):
        """Ortho projection * view from the key light, covering the
        scene's bounds. The box is computed from the scene AABB in
        light space each paint, so it fits any scene — nothing here
        is sized for the tavern example. The key light is treated as
        directional (the design's "one directional shadow"), with its
        direction from the light position toward the scene centre."""
        mn, mx = self._scene_aabb()
        center = tuple((mn[i] + mx[i]) / 2.0 for i in range(3))
        light_pos = self._key_light_position()
        direction = tuple(center[i] - light_pos[i] for i in range(3))
        length = math.sqrt(sum(v * v for v in direction)) or 1.0
        direction = tuple(v / length for v in direction)
        # lookAt breaks when the view direction is parallel to the up
        # vector (a light straight overhead); fall back to +Y then.
        up = (0.0, 0.0, 1.0) if abs(direction[2]) < 0.99 else (0.0, 1.0, 0.0)
        view = QtGui.QMatrix4x4()
        view.lookAt(QtGui.QVector3D(*light_pos),
                    QtGui.QVector3D(*center),
                    QtGui.QVector3D(*up))
        # The AABB corners in light space give the ortho box.
        xs, ys, zs = [], [], []
        for cx in (mn[0], mx[0]):
            for cy in (mn[1], mx[1]):
                for cz in (mn[2], mx[2]):
                    v = view.map(QtGui.QVector3D(cx, cy, cz))
                    xs.append(v.x())
                    ys.append(v.y())
                    zs.append(v.z())
        span = max(max(xs) - min(xs), max(ys) - min(ys),
                   max(zs) - min(zs))
        margin = max(10.0, span * 0.1)
        # In view space the light looks down -Z: the near plane is
        # the closest corner's distance, the far plane the farthest.
        near = max(1.0, -max(zs) - margin)
        far = -min(zs) + margin
        proj = QtGui.QMatrix4x4()
        proj.ortho(min(xs) - margin, max(xs) + margin,
                   min(ys) - margin, max(ys) + margin, near, far)
        return proj * view

    def _render_shadow_map(self, gl, light_space):
        """Shadow pass: draw every casting object from the key light
        into the depth map. Slope-scale polygon offset (plus the
        fragment shader's bias) keeps surfaces from shadowing
        themselves. The default framebuffer is the widget's own FBO,
        restored by id — binding 0 here would draw to nowhere.
        """
        if self._shadow_fbo is None:
            return
        gl.glBindFramebuffer(GL_FRAMEBUFFER, self._shadow_fbo.handle())
        gl.glViewport(0, 0, self._shadow_size, self._shadow_size)
        gl.glClear(GL_DEPTH_BUFFER_BIT)
        gl.glEnable(GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(2.0, 2.0)
        prog = self._programs["depth"]
        prog.bind()
        prog.setUniformValue("uLightSpace", light_space)
        for obj in self._scene["objects"]:
            if not obj.get("cast_shadow", True):
                continue
            _buf, vao, _ibo, count, indexed, _key = self._ensure_vbo(obj)
            model = self._model_matrix(obj)
            qm = QtGui.QMatrix4x4(*[v for row in model for v in row])
            prog.setUniformValue("uModel", qm)
            vao.bind()
            if indexed:
                gl.glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT,
                                  shiboken6.VoidPtr(0))
            else:
                gl.glDrawArrays(GL_TRIANGLES, 0, count)
            vao.release()
        prog.release()
        gl.glDisable(GL_POLYGON_OFFSET_FILL)
        # Restore the paint's target (UI11): the widget's own FBO at
        # full scale, the scaled FBO below it. _paint_scene sets the
        # target before the shadow pass runs.
        gl.glBindFramebuffer(GL_FRAMEBUFFER, self._target_fbo)
        gl.glViewport(0, 0, *self._target_size)

    # -- render scale (UI11) ---------------------------------------------
    def _ensure_scale_fbo(self, scale):
        """The framebuffer the scene renders into at a lower render
        scale: (width*scale, height*scale), color + depth. Rebuilt
        when the widget size or the scale changes; None when it
        cannot be built (the paint falls back to full size)."""
        width = max(1, int(self.width() * scale))
        height = max(1, int(self.height() * scale))
        if (self._scale_fbo is not None
                and self._scale_fbo_size == (width, height)):
            return self._scale_fbo
        self._destroy_scale_fbo()
        try:
            fbo = QOpenGLFramebufferObject(
                QtCore.QSize(width, height),
                QOpenGLFramebufferObject.CombinedDepthStencil)
            if not fbo.isValid():
                return None
            # Linear filtering on the color texture, so the upscale
            # blit is smooth instead of blocky.
            gl = self._gl
            gl.glBindTexture(GL_TEXTURE_2D, fbo.texture())
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                               GL_LINEAR)
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                               GL_LINEAR)
            gl.glBindTexture(GL_TEXTURE_2D, 0)
        except Exception:
            return None
        self._scale_fbo = fbo
        self._scale_fbo_size = (width, height)
        return fbo

    def _destroy_scale_fbo(self):
        # QOpenGLFramebufferObject has no destroy() in this PySide6
        # build — dropping the reference frees it, the same as the
        # shadow FBO's cleanup path.
        self._scale_fbo = None
        self._scale_fbo_size = None

    def _ensure_blit_program(self):
        """The render-scale blit program, compiled once from the
        inline GLSL above. None when it cannot compile — a compile
        failure is deterministic, so it is cached (as False) and
        not retried; the paint falls back to full size."""
        if self._blit_program is not None:
            return self._blit_program or None
        prog = QOpenGLShaderProgram()
        ok = prog.addShaderFromSourceCode(QOpenGLShader.Vertex,
                                          _BLIT_VERT)
        ok = prog.addShaderFromSourceCode(QOpenGLShader.Fragment,
                                          _BLIT_FRAG) and ok
        ok = prog.link() and ok
        if not ok:
            self._blit_program = False
            return None
        self._blit_program = prog
        return prog

    def _blit_to_widget(self, gl, fbo):
        """Upscale the scaled framebuffer onto the widget: bind the
        widget's FBO, draw the scaled color texture as a fullscreen
        triangle. Depth test off (the widget FBO's depth buffer is
        stale — it was never cleared this paint) and blending off
        (the blit replaces, never blends)."""
        prog = self._ensure_blit_program()
        if prog is None:
            return
        if self._blit_vao is None:
            vao = QOpenGLVertexArrayObject()
            if not vao.create():
                return
            self._blit_vao = vao
        gl.glBindFramebuffer(GL_FRAMEBUFFER,
                             self.defaultFramebufferObject())
        gl.glViewport(0, 0, max(1, self.width()),
                      max(1, self.height()))
        gl.glDisable(GL_DEPTH_TEST)
        gl.glDisable(GL_BLEND)
        try:
            gl.glActiveTexture(GL_TEXTURE0)
            gl.glBindTexture(GL_TEXTURE_2D, fbo.texture())
            prog.bind()
            _set_scalar(gl, prog, "uTex", 0)
            self._blit_vao.bind()
            gl.glDrawArrays(GL_TRIANGLES, 0, 3)
            self._blit_vao.release()
            prog.release()
            gl.glBindTexture(GL_TEXTURE_2D, 0)
        finally:
            gl.glEnable(GL_DEPTH_TEST)

    def _ensure_grid_vao(self, span):
        """Grid + axes VAO, rebuilt only when the scene span changes."""
        if self._grid is not None and self._grid[3] == span:
            return self._grid
        if self._grid is not None:
            try:
                self._grid[0].destroy()
            except Exception:
                pass
            try:
                self._grid[1].destroy()
            except Exception:
                pass
        step = span / 10.0
        half = span * 1.2
        grid_rgb = (0.25, 0.25, 0.28)
        verts = []
        n = int(half / step)
        for i in range(-n, n + 1):
            verts += [i * step, -half, GRID_Z, *grid_rgb,
                      i * step, half, GRID_Z, *grid_rgb,
                      -half, i * step, GRID_Z, *grid_rgb,
                      half, i * step, GRID_Z, *grid_rgb]
        # Axes: X red, Y green, Z blue.
        for color, end in (((0.85, 0.25, 0.25), (span * 0.6, 0.0, GRID_Z)),
                           ((0.25, 0.75, 0.30), (0.0, span * 0.6, GRID_Z)),
                           ((0.30, 0.45, 0.9), (0.0, 0.0, span * 0.6))):
            verts += [0.0, 0.0, GRID_Z, *color,
                      end[0], end[1], end[2], *color]
        vao, buf, count = _line_vao(self._gl, verts)
        self._grid = (vao, buf, count, span)
        return self._grid

    def _draw_grid(self, gl, proj, view):
        vao, _buf, count, _span = self._ensure_grid_vao(self.span())
        prog = self._programs["lines"]
        prog.bind()
        prog.setUniformValue("uProj", proj)
        prog.setUniformValue("uView", view)
        prog.setUniformValue("uModel", QtGui.QMatrix4x4())
        vao.bind()
        gl.glDrawArrays(GL_LINES, 0, count)
        vao.release()
        prog.release()

    def _build_selection_vao(self):
        """One static VAO: unit-cube edges (side 1, centred on origin),
        yellow per-vertex color. The selection draw scales it to the
        object's local bounds via the model matrix."""
        yellow = (1.0, 0.85, 0.2)
        corners = [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
                   (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
                   (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
                   (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                 (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        verts = []
        for a, b in edges:
            verts += [*corners[a], *yellow, *corners[b], *yellow]
        self._sel_vao, self._sel_vbo, self._sel_count = _line_vao(
            self._gl, verts)

    def _model_matrix(self, obj):
        sampled = anim_mod.sample_object(self._scene, obj, self._frame)
        return compose_matrix(sampled["position"], sampled["rotation"],
                              sampled["scale"])

    def _draw_objects(self, gl, proj, view, light_space):
        self._prune_vbos()
        prog = self._programs["pbr_lite"]
        prog.bind()
        prog.setUniformValue("uProj", proj)
        prog.setUniformValue("uView", view)
        _set_scalar(gl, prog, "uTex", 0)
        _set_scalar(gl, prog, "uExposure", self._exposure)
        self._light_uniforms(prog)
        # Shadow map on texture unit 1 (T5); material textures stay
        # on unit 0. With no shadow map (build failed), the light
        # space is the identity and every object renders unshadowed.
        if light_space is not None and self._shadow_tex is not None:
            prog.setUniformValue("uLightSpace", light_space)
            gl.glActiveTexture(GL_TEXTURE1)
            self._shadow_tex.bind()
            _set_scalar(gl, prog, "uShadowMap", 1)
            gl.glActiveTexture(GL_TEXTURE0)
        else:
            prog.setUniformValue("uLightSpace", QtGui.QMatrix4x4())
        for obj in self._scene["objects"]:
            _buf, vao, _ibo, count, indexed, _key = self._ensure_vbo(obj)
            mat = obj.get("material", {})
            # Hover-preview (UI12): the previewed object renders
            # with the overrides instead of its stored values.
            # The copy keeps the scene's dict untouched, and the
            # VBO is keyed on geometry, not material, so a preview
            # costs no rebuild — it is instant by construction.
            if (self._material_preview is not None
                    and obj.get("id") == self._material_preview[0]):
                mat = {**mat, **self._material_preview[1]}
            r, g, b = hex_to_rgb(mat.get("base_color", "#b0b0b0"))
            roughness = float(mat.get("roughness", 0.6))
            metalness = float(mat.get("metalness", 0.0))
            texture = mat.get("texture", "none")
            # Layout (PO4): the shader mode + piece sizes for this
            # object, from the (possibly preview-merged) material.
            # Set on every draw — uniforms persist across draws,
            # so a textured object after an untextured one must not
            # inherit the previous object's mode. Untextured
            # objects resolve to mode 0, which the shader ignores.
            style = resolve_style(texture)
            layout = resolve_layout(style, mat.get("layout"))
            _set_scalar(gl, prog, "uLayoutMode",
                        layout_shader_mode(style, layout))
            width, length = merged_layout_size(
                layout, mat.get("layout_size"))
            prog.setUniformValue("uLayoutSize",
                                 QtGui.QVector2D(width, length))
            # Wear (PO5): the weather/age overlay, from the
            # (possibly preview-merged) material. Set on every
            # draw — uniforms persist across draws, so a worn
            # object after a clean one must not inherit its wear.
            # The packing lives in textures.wear_uniforms, so the
            # viewport never packs by hand.
            wear_a, wear_b = wear_uniforms(style, mat.get("wear"))
            prog.setUniformValue("uWearA", QtGui.QVector4D(*wear_a))
            prog.setUniformValue("uWearB", QtGui.QVector3D(*wear_b))
            model = self._model_matrix(obj)
            qm = QtGui.QMatrix4x4(*[v for row in model for v in row])
            prog.setUniformValue("uModel", qm)
            prog.setUniformValue("uNormalMatrix", qm.normalMatrix())
            prog.setUniformValue("uBaseColor", QtGui.QVector3D(r, g, b))
            _set_scalar(gl, prog, "uRoughness", roughness)
            _set_scalar(gl, prog, "uMetalness", metalness)
            # Objects with receive_shadow off skip the shadow test
            # entirely; with no shadow map every object is unshadowed.
            receive = bool(obj.get("receive_shadow", True)) \
                and light_space is not None \
                and self._shadow_tex is not None
            _set_scalar(gl, prog, "uUseShadow", 1 if receive else 0)
            if texture and texture != "none":
                self._ensure_texture(
                    texture, mat.get("base_color", "#b0b0b0"),
                    mat.get("params"), mat.get("layout")).bind()
                _set_scalar(gl, prog, "uUseTexture", 1)
            else:
                _set_scalar(gl, prog, "uUseTexture", 0)
            vao.bind()
            if indexed:
                # The IBO binding rides with the VAO; the offset is 0.
                gl.glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT,
                                  shiboken6.VoidPtr(0))
            else:
                gl.glDrawArrays(GL_TRIANGLES, 0, count)
            vao.release()
        prog.release()

    # -- particles (T17) ---------------------------------------------------
    def _ensure_particle_vao(self):
        """The particle VAO + VBO, created once and reused: attribute
        locations match the points program (0 position, 1 color,
        2 size, 3 alpha) over the 32-byte interleaved stride. The
        VBO's storage is re-uploaded every frame by _draw_particles;
        the VAO's bindings reference the buffer object, not its
        storage, so they survive the re-upload. See _line_vao for
        the PySide6 attribute-pointer arg quirks."""
        if self._particle_vao is not None:
            return self._particle_vao, self._particle_vbo
        gl = self._gl
        vao = QOpenGLVertexArrayObject()
        vao.create()
        vao.bind()
        buf = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        buf.create()
        buf.setUsagePattern(QOpenGLBuffer.DynamicDraw)
        buf.bind()
        for loc, size, offset in ((0, 3, 0), (1, 3, 12), (2, 1, 24),
                                  (3, 1, 28)):
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, size, GL_FLOAT, 0,
                                     PARTICLE_STRIDE,
                                     shiboken6.VoidPtr(offset))
        buf.release()
        vao.release()
        self._particle_vao = vao
        self._particle_vbo = buf
        return vao, buf

    def _draw_particles(self, gl, proj, view):
        """Draw every live particle of every emitter (T17): one merged
        VBO upload per frame, two draw calls by blend mode. Alpha
        particles (smoke, dust, snow, rain, water) blend normally;
        additive particles (fire, sparks) add their light. Depth
        test stays on (particles hide behind solid objects) and
        depth write is off (they never occlude each other). With no
        live particles this returns before any GL call — a scene
        without emitters renders exactly as it did in T16."""
        data = particles_mod.build_render_data(self._scene, self._frame)
        self._particle_count = data["total"]
        if data["total"] == 0:
            return
        vao, buf = self._ensure_particle_vao()
        raw = data["data"].tobytes()
        buf.bind()
        buf.allocate(raw, len(raw))
        buf.release()
        prog = self._programs["points"]
        prog.bind()
        prog.setUniformValue("uProj", proj)
        prog.setUniformValue("uView", view)
        # Point size in cm becomes pixels by distance attenuation:
        # the vertex shader multiplies size by uPointScale / distance,
        # where uPointScale is viewport_height / (2 * tan(fov / 2)).
        # The height is the paint target's (UI11): at full scale it
        # is the widget height, as before; at a lower render scale
        # the points are sized for the scaled buffer, then upscaled
        # to the same on-screen size.
        point_scale = max(1, self._target_size[1]) / (
            2.0 * math.tan(math.radians(POINTS_FOV_DEGREES / 2.0)))
        _set_scalar(gl, prog, "uPointScale", point_scale)
        gl.glDepthMask(False)
        gl.glEnable(GL_BLEND)
        vao.bind()
        if data["alpha_count"] > 0:
            gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            gl.glDrawArrays(GL_POINTS, 0, data["alpha_count"])
        if data["additive_count"] > 0:
            gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            gl.glDrawArrays(GL_POINTS, data["alpha_count"],
                            data["additive_count"])
        vao.release()
        gl.glDisable(GL_BLEND)
        gl.glDepthMask(True)
        prog.release()

    def _draw_selection(self, gl, proj, view):
        obj = next((o for o in self._scene["objects"]
                    if o["id"] == self.selected_id), None)
        if obj is None or self._sel_vao is None:
            return
        mn, mx = self._object_bounds(obj)
        model = self._model_matrix(obj)
        qm = QtGui.QMatrix4x4(*[v for row in model for v in row])
        cx = (mn[0] + mx[0]) / 2.0
        cy = (mn[1] + mx[1]) / 2.0
        cz = (mn[2] + mx[2]) / 2.0
        qm.translate(cx, cy, cz)
        qm.scale(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
        prog = self._programs["lines"]
        prog.bind()
        prog.setUniformValue("uProj", proj)
        prog.setUniformValue("uView", view)
        prog.setUniformValue("uModel", qm)
        self._sel_vao.bind()
        gl.glDrawArrays(GL_LINES, 0, self._sel_count)
        self._sel_vao.release()
        prog.release()

    def _ensure_gizmo_vao(self, mode, highlight=None):
        """One cached VAO per (mode, highlight) pair (T12/T13), built
        lazily from the unit-space geometry. The draw scales it via
        the model matrix, so the VAO never needs rebuilding; the
        highlight only changes the axis colors, so the plain VAO
        (highlight None) is byte-identical to T12's."""
        key = (mode, highlight)
        entry = self._gizmo_vaos.get(key)
        if entry is None:
            entry = _line_vao(self._gl, gizmo_lines(mode, highlight))
            self._gizmo_vaos[key] = entry
        return entry

    def _draw_gizmo(self, gl, proj, view):
        """The transform gizmo (T12) on the selected object: unit
        geometry scaled by gizmo_size(distance) — constant screen
        size — translated to the object's sampled world position
        (the translation column of its model matrix, so the gizmo
        follows the animation). Axes are world-aligned in T12; a
        local-orientation gizmo is a later item, not in the T12
        spec. Depth test off for the gizmo only, so it is never
        hidden inside the object; the test is back on before this
        returns. The hovered handle (T13) draws from the brightened
        VAO; with no hover the draw is exactly T12's."""
        center = self._gizmo_center()
        if center is None:
            return
        cx, cy, cz = center
        size = gizmo_size(self.distance)
        qm = QtGui.QMatrix4x4()
        qm.translate(cx, cy, cz)
        qm.scale(size, size, size)
        vao, _buf, count = self._ensure_gizmo_vao(self.gizmo_mode,
                                                  self._gizmo_hover)
        prog = self._programs["lines"]
        gl.glDisable(GL_DEPTH_TEST)
        prog.bind()
        prog.setUniformValue("uProj", proj)
        prog.setUniformValue("uView", view)
        prog.setUniformValue("uModel", qm)
        vao.bind()
        gl.glDrawArrays(GL_LINES, 0, count)
        vao.release()
        prog.release()
        gl.glEnable(GL_DEPTH_TEST)

    # -- picking ---------------------------------------------------------
    def _pick_ray(self, x, y):
        """(origin, unit direction) of the ray through pixel (x, y),
        or None when the projection cannot be inverted. Factored out
        of pick() in T13 so object picking, gizmo picking, and the
        drag math all shoot the same ray."""
        proj, view = self._matrices()
        combined = proj * view
        inv, ok = combined.inverted()
        if not ok:
            return None
        ndc_x = 2.0 * x / max(1, self.width()) - 1.0
        ndc_y = 1.0 - 2.0 * y / max(1, self.height())
        near = inv.map(QtGui.QVector3D(ndc_x, ndc_y, -1.0))
        far = inv.map(QtGui.QVector3D(ndc_x, ndc_y, 1.0))
        origin = (near.x(), near.y(), near.z())
        direction = (far.x() - near.x(), far.y() - near.y(),
                     far.z() - near.z())
        length = math.sqrt(sum(v * v for v in direction)) or 1.0
        return origin, tuple(v / length for v in direction)

    def pick(self, x, y):
        """Object id under pixel (x, y), or None. Ray vs per-object box."""
        if not self._scene or not self._scene["objects"]:
            return None
        ray = self._pick_ray(x, y)
        if ray is None:
            return None
        origin, direction = ray
        eye = self.camera_position()
        best, best_dist = None, float("inf")
        for obj in self._scene["objects"]:
            model = self._model_matrix(obj)
            inv_model = mat_inverse(model)
            local_o = apply_matrix(inv_model, origin)
            local_d_end = apply_matrix(
                inv_model,
                (origin[0] + direction[0], origin[1] + direction[1],
                 origin[2] + direction[2]))
            local_d = tuple(local_d_end[i] - local_o[i] for i in range(3))
            mn, mx = self._object_bounds(obj)
            t = ray_hits_aabb(local_o, local_d, mn, mx)
            if t is None:
                continue
            hit_local = tuple(local_o[i] + local_d[i] * t for i in range(3))
            hit_world = apply_matrix(model, hit_local)
            dist = math.sqrt(sum((hit_world[i] - eye[i]) ** 2
                                 for i in range(3)))
            if dist < best_dist:
                best, best_dist = obj["id"], dist
        return best

    def pick_gizmo(self, x, y):
        """(axis, mode) of the gizmo handle under pixel (x, y), or
        None. The handle geometry comes from the same constants the
        draw uses — GIZMO_MOVE_TIP for the arrow cylinder, GIZMO_RING_
        RADIUS for the rotate ring, GIZMO_SCALE_CUBE_CENTER/HALF for
        the scale cube — so the pick can never drift from what is
        drawn; GIZMO_PICK_TOLERANCE fattens the zero-width lines into
        grabbable handles. The closest hit wins, which is the handle
        facing the camera when two overlap. Nothing is pickable with
        no selection, or while the gizmo is suppressed for a render."""
        if not self._scene or not self.selected_id \
                or self._gizmo_suppressed:
            return None
        center = self._gizmo_center()
        if center is None:
            return None
        ray = self._pick_ray(x, y)
        if ray is None:
            return None
        origin, direction = ray
        size = gizmo_size(self.distance)
        tol = GIZMO_PICK_TOLERANCE * size
        best, best_t = None, float("inf")
        for axis in ("x", "y", "z"):
            u = _gizmo_axis_frame(axis)[0]
            if self.gizmo_mode == "move":
                # One cylinder per arrow, centre to tip: it covers
                # the shaft and the cone (base radius 0.09 < 0.12).
                end = tuple(center[i] + u[i] * GIZMO_MOVE_TIP * size
                            for i in range(3))
                t = ray_vs_cylinder_segment(origin, direction, center,
                                            end, tol)
            elif self.gizmo_mode == "rotate":
                t = ray_vs_ring(origin, direction, center, u,
                                GIZMO_RING_RADIUS * size, tol)
            else:  # scale: the cube only, fattened by the tolerance.
                c = tuple(center[i]
                          + u[i] * GIZMO_SCALE_CUBE_CENTER * size
                          for i in range(3))
                half = (GIZMO_SCALE_CUBE_HALF
                        + GIZMO_PICK_TOLERANCE) * size
                t = ray_hits_aabb(
                    origin, direction,
                    tuple(c[i] - half for i in range(3)),
                    tuple(c[i] + half for i in range(3)))
            if t is not None and t < best_t:
                best, best_t = axis, t
        if best is None:
            return None
        return best, self.gizmo_mode

    # -- gizmo dragging (T13) ------------------------------------------
    def _start_gizmo_drag(self, axis, x, y):
        """Arm a drag of `axis` in the current mode from pixel (x, y).
        Returns False when the drag plane cannot be built (the press
        ray runs parallel to it) — the caller then falls through to
        the orbit path. The drag plane is the ring's own plane for
        Rotate, and the plane through the gizmo centre perpendicular
        to the camera for Move and Scale. The start transform is the
        SAMPLED one at the current frame: that is what the user sees
        and what the gizmo sits on."""
        ray = self._pick_ray(x, y)
        center = self._gizmo_center()
        if ray is None or center is None:
            return False
        obj = next((o for o in self._scene["objects"]
                    if o["id"] == self.selected_id), None)
        if obj is None:
            return False
        origin, direction = ray
        u = _gizmo_axis_frame(axis)[0]
        if self.gizmo_mode == "rotate":
            plane_point, plane_normal = center, u
        else:
            plane_point, plane_normal = center, self._view_dir()
        t = ray_vs_plane(origin, direction, plane_point, plane_normal)
        if t is None:
            return False
        grab = tuple(origin[i] + direction[i] * t for i in range(3))
        state = {"mode": self.gizmo_mode, "axis": axis,
                 "object_id": obj["id"],
                 "start": anim_mod.sample_object(self._scene, obj,
                                                 self._frame),
                 "center": center,
                 "plane_point": plane_point,
                 "plane_normal": plane_normal,
                 "grab_point": grab}
        if self.gizmo_mode == "rotate":
            _u, p, q = _gizmo_axis_frame(axis)
            rel = tuple(grab[i] - center[i] for i in range(3))
            # Rotation accumulates wrapped per-step increments (not a
            # single wrapped total), so a drag past 180 degrees keeps
            # turning instead of snapping back.
            state["last_angle"] = math.atan2(
                sum(rel[i] * q[i] for i in range(3)),
                sum(rel[i] * p[i] for i in range(3)))
            state["total_deg"] = 0.0
        elif self.gizmo_mode == "scale":
            state["grab_distance"] = sum(
                (grab[i] - center[i]) * u[i] for i in range(3))
        self._gizmo_drag = state
        # The dragged handle stays bright for the whole drag.
        self._gizmo_hover = axis
        # T14: the window snapshots the pre-drag scene now, so the
        # whole drag commits as one undo step on release.
        self.gizmo_drag_started.emit()
        self.update()
        return True

    def _apply_drag_value(self, group, idx, value):
        """Write one drag result into the object's base transform —
        and into the key on this frame when the track has one there.
        Keys override the base at their frames, so without the key
        write a drag on a keyed object would move the inspector
        numbers but not the object on screen. A drag never CREATES a
        key: at an interpolated frame it edits the base only, the
        same as the inspector does today."""
        state = self._gizmo_drag
        obj = next((o for o in self._scene["objects"]
                    if o["id"] == state["object_id"]), None)
        if obj is None:
            return
        obj[group][idx] = float(value)
        keys = anim_mod.get_keys(self._scene, obj["id"], group)
        if any(k[0] == self._frame for k in keys):
            anim_mod.set_key(self._scene, obj["id"], group, self._frame,
                             obj[group])

    def _apply_gizmo_drag(self, x, y, shift=False):
        """One drag step for the pointer at pixel (x, y). Move: the
        ray meets the drag plane, the movement from the grab point
        projects onto the axis. Rotate: the angle around the axis
        from the grab point's position on the ring plane. Scale: the
        pointer's distance from the centre along the axis, as a ratio
        of the grab distance. Shift snaps the result (10 cm / 15
        degrees / 0.1). A ray parallel to the drag plane skips the
        step, keeping the last good value. Every step fires
        transform_changed, so the inspector follows live."""
        state = self._gizmo_drag
        if state is None:
            return
        ray = self._pick_ray(x, y)
        if ray is None:
            return
        origin, direction = ray
        t = ray_vs_plane(origin, direction, state["plane_point"],
                         state["plane_normal"])
        if t is None:
            return
        point = tuple(origin[i] + direction[i] * t for i in range(3))
        mode, axis = state["mode"], state["axis"]
        idx = ("x", "y", "z").index(axis)
        u = _gizmo_axis_frame(axis)[0]
        if mode == "move":
            delta = sum((point[i] - state["grab_point"][i]) * u[i]
                        for i in range(3))
            if shift:
                delta = round(delta / GIZMO_SNAP_MOVE) * GIZMO_SNAP_MOVE
            self._apply_drag_value(
                "position", idx, state["start"]["position"][idx] + delta)
        elif mode == "rotate":
            _u, p, q = _gizmo_axis_frame(axis)
            rel = tuple(point[i] - state["center"][i] for i in range(3))
            angle = math.atan2(sum(rel[i] * q[i] for i in range(3)),
                               sum(rel[i] * p[i] for i in range(3)))
            inc = angle - state["last_angle"]
            inc = (inc + math.pi) % (2.0 * math.pi) - math.pi
            state["last_angle"] = angle
            state["total_deg"] += math.degrees(inc)
            delta_deg = state["total_deg"]
            if shift:
                delta_deg = (round(delta_deg / GIZMO_SNAP_ROTATE)
                             * GIZMO_SNAP_ROTATE)
            self._apply_drag_value(
                "rotation", idx,
                state["start"]["rotation"][idx] + delta_deg)
        else:  # scale
            dist = sum((point[i] - state["center"][i]) * u[i]
                       for i in range(3))
            grab_dist = state["grab_distance"]
            ratio = (dist / grab_dist if abs(grab_dist) > 1e-6 else 1.0)
            value = state["start"]["scale"][idx] * ratio
            if shift:
                value = round(value / GIZMO_SNAP_SCALE) * GIZMO_SNAP_SCALE
            value = max(value, GIZMO_MIN_SCALE)
            self._apply_drag_value("scale", idx, value)
        self.transform_changed.emit()
        self.update()

    # -- mouse ------------------------------------------------------------
    def mousePressEvent(self, event):
        # A left press on a gizmo handle (T13) starts a drag — the
        # handle wins over object picking and over orbit, and no
        # press state is recorded, so the release cannot re-pick.
        if event.button() == QtCore.Qt.LeftButton:
            pos = event.position().toPoint()
            hit = self.pick_gizmo(pos.x(), pos.y())
            if hit is not None and self._start_gizmo_drag(hit[0], pos.x(),
                                                          pos.y()):
                super().mousePressEvent(event)
                return
        self._drag_button = event.button()
        self._drag_pos = event.position().toPoint()
        self._press_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._gizmo_drag is not None:
            shift = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
            self._apply_gizmo_drag(pos.x(), pos.y(), shift)
            super().mouseMoveEvent(event)
            return
        if self._drag_pos is None:
            # No button held: hover-pick the gizmo (T13) so the
            # handle under the pointer brightens.
            hit = self.pick_gizmo(pos.x(), pos.y())
            hover = hit[0] if hit is not None else None
            if hover != self._gizmo_hover:
                self._gizmo_hover = hover
                self.update()
            super().mouseMoveEvent(event)
            return
        dx, dy = pos.x() - self._drag_pos.x(), pos.y() - self._drag_pos.y()
        self._drag_pos = pos
        if self._drag_button == QtCore.Qt.LeftButton:
            self.azimuth = (self.azimuth - dx * 0.5) % 360.0
            self.elevation = max(-89.0, min(89.0, self.elevation + dy * 0.5))
            self.update()
        elif self._drag_button in (QtCore.Qt.RightButton,
                                   QtCore.Qt.MiddleButton):
            self._pan(dx, dy)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._gizmo_drag is not None \
                and event.button() == QtCore.Qt.LeftButton:
            # End of a gizmo drag (T13): no re-pick — the press never
            # selected anything, and the object under the handle is
            # not the user's target. T14: the window commits the
            # live-edit snapshot as one undo step (skipped when the
            # drag changed nothing — a press with no movement).
            self._gizmo_drag = None
            self._gizmo_hover = None
            self.gizmo_drag_finished.emit()
            self.update()
            super().mouseReleaseEvent(event)
            return
        if (self._press_pos is not None
                and event.button() == QtCore.Qt.LeftButton):
            moved = (event.position().toPoint() - self._press_pos).manhattanLength()
            if moved < 5:
                picked = self.pick(event.position().toPoint().x(),
                                   event.position().toPoint().y())
                self.selected_id = picked
                self.object_selected.emit(picked or "")
                self.update()
        self._drag_button = None
        self._drag_pos = None
        self._press_pos = None
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        # The pointer left the viewport: drop the hover highlight.
        if self._gizmo_hover is not None:
            self._gizmo_hover = None
            self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.distance *= 1.1 ** (-steps)
            self.distance = max(self.span() * 0.15,
                                min(self.span() * 30.0, self.distance))
            self.update()
        super().wheelEvent(event)
