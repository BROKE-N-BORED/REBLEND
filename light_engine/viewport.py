"""The live 3D viewport: a real OpenGL view in a Qt window. No web tech.

Pattern note: the orbit camera (drag orbit, right-drag pan, scroll zoom),
the fixed-function lighting setup, and the grid + axes follow the pattern
in the Design Shell v1.0 Bench viewport (design-shell/designshell/
qtbench.py). This is new code written for this engine; nothing is imported
from Design Shell. What differs here: geometry lives in GPU vertex buffers
(VBOs) that are rebuilt only when an object changes, so paintGL never
re-tessellates the scene.

Units are cm, z-up.
"""
from __future__ import annotations

import math
from array import array

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFunctions_2_1,
                              QOpenGLTexture)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
import shiboken6

from . import animation as anim_mod
from .geometry import (apply_matrix, compose_matrix, local_bounds,
                       mat_inverse, ray_hits_aabb, tessellate)
from .scene import hex_to_rgb
from .textures import build_qimage

# GL enum values (kept numeric, as in the Bench pattern, so the calls read
# the same as the OpenGL docs).
GL_DEPTH_TEST = 0x0B71
GL_LIGHTING = 0x0B50
GL_LIGHT0 = 0x4000
GL_LIGHT1 = 0x4001
GL_LIGHT2 = 0x4002
GL_COLOR_MATERIAL = 0x0B57
GL_TEXTURE_2D = 0x0DE1
GL_PROJECTION = 0x1701
GL_MODELVIEW = 0x1700
GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_TRIANGLES = 0x0004
GL_LINES = 0x0001
GL_FLOAT = 0x1406
GL_VERTEX_ARRAY = 0x8074
GL_NORMAL_ARRAY = 0x8075
GL_TEXTURE_COORD_ARRAY = 0x8078
GL_FRONT = 0x0408
GL_AMBIENT = 0x1200
GL_DIFFUSE = 0x1201
GL_SPECULAR = 0x1202
GL_POSITION = 0x1203
GL_SHININESS = 0x1601
GL_LIGHT_MODEL_AMBIENT = 0x0B53
GL_LIGHT_MODEL_TWO_SIDE = 0x0B52


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


class Viewport(QOpenGLWidget):
    object_selected = QtCore.Signal(str)  # object id, or "" for none

    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = QtGui.QSurfaceFormat()
        fmt.setVersion(2, 1)
        fmt.setProfile(QtGui.QSurfaceFormat.CompatibilityProfile)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        self.setFormat(fmt)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._scene = None
        self._gl = None
        self._vbos = {}      # object id -> (QOpenGLBuffer, vertex count, key)
        self._textures = {}  # (texture name, base color) -> QOpenGLTexture
        self._frame = 0
        self.selected_id = None
        # Orbit camera: target + distance + azimuth/elevation (degrees).
        self.target = (0.0, 0.0, 0.0)
        self.distance = 400.0
        self.azimuth = -126.0
        self.elevation = 29.0
        self._drag_button = None
        self._drag_pos = None
        self._press_pos = None

    # -- scene ----------------------------------------------------------
    def set_scene(self, scene):
        self._scene = scene
        self._frame = 0
        self.selected_id = None
        self._drop_vbos()
        self.frame()
        self.update()

    def refresh(self):
        """Rebuild changed buffers on the next paint. Cheap: only objects
        whose shape/params changed get re-tessellated."""
        self.update()

    def set_frame(self, frame):
        self._frame = int(frame)
        self.update()

    def current_frame(self):
        return self._frame

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
        for buf, _count, _key in self._vbos.values():
            try:
                buf.destroy()
            except Exception:
                pass
        self._vbos = {}
        if made_current:
            try:
                self.doneCurrent()
            except Exception:
                pass

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
        for buf, _count, _key in self._vbos.values():
            try:
                buf.destroy()
            except Exception:
                pass
        self._vbos = {}
        for tex in self._textures.values():
            try:
                tex.destroy()
            except Exception:
                pass
        self._textures = {}
        if made_current:
            try:
                self.doneCurrent()
            except Exception:
                pass

    def _mesh_key(self, obj):
        return (obj["shape"], repr(sorted(obj.get("params", {}).items())))

    def _ensure_vbo(self, obj):
        key = self._mesh_key(obj)
        entry = self._vbos.get(obj["id"])
        if entry is not None and entry[2] == key:
            return entry
        if entry is not None:
            try:
                entry[0].destroy()
            except Exception:
                pass
        mesh = tessellate(obj["shape"], obj.get("params", {}))
        flat, verts = _interleaved(mesh)
        buf = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        buf.create()
        buf.bind()
        buf.allocate(flat.tobytes(), len(flat.tobytes()))
        buf.release()
        entry = (buf, verts, key)
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
            mn, mx = local_bounds(obj["shape"], obj.get("params", {}))
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
        self._gl = QOpenGLFunctions_2_1()
        self._gl.initializeOpenGLFunctions()
        gl = self._gl
        gl.glEnable(GL_DEPTH_TEST)
        gl.glEnable(GL_LIGHTING)
        gl.glEnable(GL_LIGHT0)
        gl.glEnable(GL_LIGHT1)
        gl.glEnable(GL_LIGHT2)
        gl.glEnable(GL_COLOR_MATERIAL)
        gl.glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, 1.0)
        gl.glClearColor(0.09, 0.09, 0.10, 1.0)

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
        if gl is None:
            return
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        proj, view = self._matrices()
        gl.glMatrixMode(GL_PROJECTION)
        gl.glLoadMatrixf(list(proj.data()))
        gl.glMatrixMode(GL_MODELVIEW)
        gl.glLoadMatrixf(list(view.data()))
        self._setup_lights(gl)
        self._draw_grid(gl)
        if self._scene:
            self._draw_objects(gl)
            if self.selected_id:
                self._draw_selection(gl)

    def _setup_lights(self, gl):
        lights = (self._scene or {}).get("lights", []) if self._scene else []
        ambient = 0.35
        slots = {}
        for light in lights:
            kind = light.get("kind")
            if kind == "ambient":
                ambient = float(light.get("intensity", 0.35))
            elif kind in ("key", "fill", "rim"):
                slots[kind] = light
        gl.glLightModelfv(GL_LIGHT_MODEL_AMBIENT,
                          (ambient, ambient, ambient, 1.0))
        for slot, light_id in (("key", GL_LIGHT0), ("fill", GL_LIGHT1),
                               ("rim", GL_LIGHT2)):
            light = slots.get(slot)
            if light is None:
                continue
            r, g, b = hex_to_rgb(light.get("color", "#ffffff"))
            inten = float(light.get("intensity", 0.8))
            gl.glLightfv(light_id, GL_DIFFUSE, (r * inten, g * inten,
                                                b * inten, 1.0))
            gl.glLightfv(light_id, GL_SPECULAR, (r * inten, g * inten,
                                                 b * inten, 1.0))
            gl.glLightfv(light_id, GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))
            pos = light.get("position", [0, 0, 0])
            gl.glLightfv(light_id, GL_POSITION,
                         (float(pos[0]), float(pos[1]), float(pos[2]), 1.0))

    def _draw_grid(self, gl):
        span = self.span()
        step = span / 10.0
        half = span * 1.2
        gl.glDisable(GL_LIGHTING)
        gl.glDisable(GL_TEXTURE_2D)
        gl.glColor3f(0.25, 0.25, 0.28)
        gl.glBegin(GL_LINES)
        n = int(half / step)
        for i in range(-n, n + 1):
            gl.glVertex3f(i * step, -half, 0.0)
            gl.glVertex3f(i * step, half, 0.0)
            gl.glVertex3f(-half, i * step, 0.0)
            gl.glVertex3f(half, i * step, 0.0)
        gl.glEnd()
        gl.glBegin(GL_LINES)
        gl.glColor3f(0.85, 0.25, 0.25)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(span * 0.6, 0.0, 0.0)
        gl.glColor3f(0.25, 0.75, 0.30)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, span * 0.6, 0.0)
        gl.glColor3f(0.30, 0.45, 0.9)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, span * 0.6)
        gl.glEnd()
        gl.glEnable(GL_LIGHTING)

    def _model_matrix(self, obj):
        sampled = anim_mod.sample_object(self._scene, obj, self._frame)
        return compose_matrix(sampled["position"], sampled["rotation"],
                              sampled["scale"])

    def _draw_objects(self, gl):
        for obj in self._scene["objects"]:
            buf, verts, _key = self._ensure_vbo(obj)
            mat = obj.get("material", {})
            r, g, b = hex_to_rgb(mat.get("base_color", "#b0b0b0"))
            roughness = float(mat.get("roughness", 0.6))
            metalness = float(mat.get("metalness", 0.0))
            texture = mat.get("texture", "none")
            model = self._model_matrix(obj)
            qm = QtGui.QMatrix4x4(*[v for row in model for v in row])
            gl.glPushMatrix()
            gl.glMultMatrixf(list(qm.data()))
            gl.glMaterialfv(GL_FRONT, GL_SPECULAR,
                            (metalness, metalness, metalness, 1.0))
            gl.glMaterialf(GL_FRONT, GL_SHININESS,
                           (1.0 - roughness) * 64.0 + 2.0)
            if texture and texture != "none":
                self._ensure_texture(texture, mat.get("base_color", "#b0b0b0")).bind()
                gl.glEnable(GL_TEXTURE_2D)
                gl.glColor3f(1.0, 1.0, 1.0)
            else:
                gl.glDisable(GL_TEXTURE_2D)
                gl.glColor3f(r, g, b)
            buf.bind()
            gl.glEnableClientState(GL_VERTEX_ARRAY)
            gl.glEnableClientState(GL_NORMAL_ARRAY)
            gl.glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            stride = 32
            gl.glVertexPointer(3, GL_FLOAT, stride, shiboken6.VoidPtr(0))
            gl.glNormalPointer(GL_FLOAT, stride, shiboken6.VoidPtr(12))
            gl.glTexCoordPointer(2, GL_FLOAT, stride, shiboken6.VoidPtr(24))
            gl.glDrawArrays(GL_TRIANGLES, 0, verts)
            gl.glDisableClientState(GL_VERTEX_ARRAY)
            gl.glDisableClientState(GL_NORMAL_ARRAY)
            gl.glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            buf.release()
            if texture and texture != "none":
                gl.glDisable(GL_TEXTURE_2D)
            gl.glPopMatrix()

    def _draw_selection(self, gl):
        obj = next((o for o in self._scene["objects"]
                    if o["id"] == self.selected_id), None)
        if obj is None:
            return
        mn, mx = local_bounds(obj["shape"], obj.get("params", {}))
        model = self._model_matrix(obj)
        corners = [(mn[0], mn[1], mn[2]), (mx[0], mn[1], mn[2]),
                   (mx[0], mx[1], mn[2]), (mn[0], mx[1], mn[2]),
                   (mn[0], mn[1], mx[2]), (mx[0], mn[1], mx[2]),
                   (mx[0], mx[1], mx[2]), (mn[0], mx[1], mx[2])]
        world = [apply_matrix(model, c) for c in corners]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                 (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        gl.glDisable(GL_LIGHTING)
        gl.glDisable(GL_TEXTURE_2D)
        gl.glColor3f(1.0, 0.85, 0.2)
        gl.glBegin(GL_LINES)
        for a, b in edges:
            gl.glVertex3f(*world[a])
            gl.glVertex3f(*world[b])
        gl.glEnd()
        gl.glEnable(GL_LIGHTING)

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
            mn, mx = local_bounds(obj["shape"], obj.get("params", {}))
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
