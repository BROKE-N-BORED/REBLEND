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
flags decide who casts and who receives.

Units are cm, z-up.
"""
from __future__ import annotations

import math
import os
from array import array

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject,
                              QOpenGLFunctions_3_3_Core, QOpenGLTexture,
                              QOpenGLVertexArrayObject)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
import shiboken6

from . import animation as anim_mod
from . import meshfile as meshfile_mod
from . import shaders as shader_mod
from .geometry import (apply_matrix, compose_matrix, local_bounds,
                       mat_inverse, ray_hits_aabb, tessellate,
                       tessellate_indexed)
from .scene import default_smooth_shading, hex_to_rgb
from .textures import build_qimage

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
# Shadow map (T5) enums.
GL_TEXTURE0 = 0x84C0
GL_TEXTURE1 = 0x84C1
GL_TEXTURE_2D = 0x0DE1
GL_FRAMEBUFFER = 0x8D40
GL_DEPTH_ATTACHMENT = 0x8D00
GL_FRAMEBUFFER_COMPLETE = 0x8CD5
GL_NONE = 0x0
GL_POLYGON_OFFSET_FILL = 0x8037

# Interleaved object VBO layout: position (3) + normal (3) + uv (2).
VERTEX_STRIDE = 32

# One shadow map, key light only, 1024x1024 (T5).
SHADOW_MAP_SIZE = 1024

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
        self._textures = {}  # (texture name, base color) -> QOpenGLTexture
        # Shadow map (T5): one 1024x1024 depth map from the key light.
        # _shadow_fbo is a QOpenGLFramebufferObject (NoAttachment) whose
        # handle carries the depth texture; _shadow_tex is the depth
        # texture itself. _shadow_error is set (plain words) when the
        # map could not be built — the scene still renders, shadowless.
        self._shadow_fbo = None
        self._shadow_tex = None
        self._shadow_error = None
        self._grid = None    # (vao, vbo, vertex count, span key)
        self._sel_vao = None  # unit-cube edge VAO for the selection box
        self._sel_vbo = None
        self._sel_count = 0
        self._frame = 0
        self.selected_id = None
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
        self._drop_vbos()
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

    def _ensure_texture(self, name, base_color):
        key = (name, base_color)
        tex = self._textures.get(key)
        if tex is not None:
            return tex
        img = build_qimage(name, hex_to_rgb(base_color), size=128)
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
            if self.selected_id:
                self._draw_selection(gl, proj, view)

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
    def _build_shadow_map(self):
        """Create the depth-only FBO + 1024x1024 depth texture.

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
            tex.setSize(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
            tex.setMinificationFilter(QOpenGLTexture.Nearest)
            tex.setMagnificationFilter(QOpenGLTexture.Nearest)
            tex.setWrapMode(QOpenGLTexture.ClampToEdge)
            tex.allocateStorage()
            if not tex.isCreated():
                raise RuntimeError("the depth texture did not allocate")
            fbo = QOpenGLFramebufferObject(
                QtCore.QSize(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE),
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
        gl.glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE)
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
        gl.glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
        gl.glViewport(0, 0, max(1, self.width()), max(1, self.height()))

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
            r, g, b = hex_to_rgb(mat.get("base_color", "#b0b0b0"))
            roughness = float(mat.get("roughness", 0.6))
            metalness = float(mat.get("metalness", 0.0))
            texture = mat.get("texture", "none")
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
                    texture, mat.get("base_color", "#b0b0b0")).bind()
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

    # -- picking ---------------------------------------------------------
    def pick(self, x, y):
        """Object id under pixel (x, y), or None. Ray vs per-object box."""
        if not self._scene or not self._scene["objects"]:
            return None
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
        direction = tuple(v / length for v in direction)
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

    # -- mouse ------------------------------------------------------------
    def mousePressEvent(self, event):
        self._drag_button = event.button()
        self._drag_pos = event.position().toPoint()
        self._press_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos is None:
            super().mouseMoveEvent(event)
            return
        pos = event.position().toPoint()
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

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.distance *= 1.1 ** (-steps)
            self.distance = max(self.span() * 0.15,
                                min(self.span() * 30.0, self.distance))
            self.update()
        super().wheelEvent(event)
