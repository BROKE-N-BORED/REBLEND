"""The main window: outliner left, viewport center, inspector right,
timeline along the bottom. Plain labels, no marketing copy.

Budgets warn, never refuse: the triangle readout turns to a warning past
the soft budget, and nothing is ever blocked for size.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from . import animation as anim_mod
from . import meshfile as meshfile_mod
from . import meshio as meshio_mod
from . import meshio_glb as meshio_glb_mod
from . import meshio_obj as meshio_obj_mod
from .scene import (default_smooth_shading, find_object, load_scene,
                    new_object, new_scene, save_scene, scene_copy,
                    total_triangles)
from .viewport import Viewport

SOFT_TRI_BUDGET = 100000


class ImportDialog(QtWidgets.QDialog):
    """Scale + report preview before an import is confirmed (T8).
    The report comes from parsing the file; the scale starts at the
    meters-to-cm default and is editable here, before anything is
    written. scale() returns the chosen value after accept()."""

    def __init__(self, parent, report, scale=meshio_mod.DEFAULT_SCALE):
        super().__init__(parent)
        self.setWindowTitle("Import mesh")
        layout = QtWidgets.QVBoxLayout(self)
        summary = (f"{report['objects']} object(s) · "
                   f"{report['triangles']:,} triangles")
        layout.addWidget(QtWidgets.QLabel(summary))
        warnings = report.get("warnings") or []
        if warnings:
            box = QtWidgets.QLabel("\n".join(warnings))
            box.setWordWrap(True)
            layout.addWidget(box)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Scale (file units to cm):"))
        self._scale_spin = QtWidgets.QDoubleSpinBox()
        self._scale_spin.setRange(0.01, 10000.0)
        self._scale_spin.setDecimals(2)
        self._scale_spin.setValue(float(scale))
        row.addWidget(self._scale_spin)
        layout.addLayout(row)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def scale(self):
        return self._scale_spin.value()


class ExportDialog(QtWidgets.QDialog):
    """Scale + report preview before an export is confirmed (T10
    .obj, T11 .glb). Mirrors ImportDialog: the report comes from
    planning the export (counts of what will actually be written,
    warnings for skipped objects); the scale starts at the
    cm-to-meters default and is editable here, before anything is
    written. scale() returns the chosen value after accept(). The
    title names the format; it defaults to .obj so the T10 callers
    are untouched."""

    def __init__(self, parent, report,
                 scale=meshio_obj_mod.DEFAULT_EXPORT_SCALE,
                 title="Export .obj"):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(self)
        summary = (f"{report['objects']} object(s) · "
                   f"{report['triangles']:,} triangles")
        layout.addWidget(QtWidgets.QLabel(summary))
        warnings = report.get("warnings") or []
        if warnings:
            box = QtWidgets.QLabel("\n".join(warnings))
            box.setWordWrap(True)
            layout.addWidget(box)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Scale (cm to file units):"))
        self._scale_spin = QtWidgets.QDoubleSpinBox()
        self._scale_spin.setRange(0.0001, 10000.0)
        self._scale_spin.setDecimals(4)
        self._scale_spin.setSingleStep(0.01)
        self._scale_spin.setValue(float(scale))
        row.addWidget(self._scale_spin)
        layout.addLayout(row)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def scale(self):
        return self._scale_spin.value()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, scene=None, scene_path=None):
        super().__init__()
        self.setWindowTitle("Reblend — v0.2")
        self.resize(1280, 800)
        self.scene = scene or new_scene()
        self.scene_path = scene_path
        self.selected_id = None
        self._undo_stack = []
        self._live_edit_snapshot = None  # pre-edit scene for live controls
        self._playing = False
        self._frame = 0
        self._refreshing = False
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)
        self._build_ui()
        # A scene opened on the command line can carry mesh objects
        # whose files are missing — resolve them before the viewport
        # ever sees the scene. Warnings go to the status bar here
        # (the window is not shown yet, so no dialog); _open_scene
        # shows the dialog as well.
        missing = self._resolve_meshes()
        self._apply_scene_to_ui()
        if missing:
            self.statusBar().showMessage(missing[0])
        self._update_play_timer()

    def closeEvent(self, event):
        self._play_timer.stop()
        self._viewport.cleanup_gl()
        super().closeEvent(event)

    # -- layout ----------------------------------------------------------
    def _build_ui(self):
        self._build_menu()
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_center())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        main = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._build_timeline())
        self.setCentralWidget(main)
        self.statusBar().showMessage("Ready")

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("File")
        for label, slot in (("New", self._new_scene),
                            ("Open…", self._open_scene),
                            ("Save", self._save_scene),
                            ("Save As…", self._save_scene_as),
                            ("Import mesh…", self._import_mesh),
                            ("Export mesh…", self._export_mesh),
                            ("Export scene as .obj…",
                             self._export_scene_obj),
                            ("Export mesh as .glb…",
                             self._export_mesh_glb),
                            ("Export scene as .glb…",
                             self._export_scene_glb)):
            action = QtGui.QAction(label, self)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        file_menu.addSeparator()
        render_action = QtGui.QAction("Render still to PNG…", self)
        render_action.triggered.connect(self._render_still)
        file_menu.addAction(render_action)
        export_action = QtGui.QAction("Export frames to folder…", self)
        export_action.triggered.connect(self._export_frames)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        quit_action = QtGui.QAction("Quit", self)
        quit_action.setShortcut(QtGui.QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        edit_menu = self.menuBar().addMenu("Edit")
        undo_action = QtGui.QAction("Undo", self)
        undo_action.setShortcut(QtGui.QKeySequence.Undo)
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)

    def _build_left(self):
        pane = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(pane)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(QtWidgets.QLabel("Objects"))
        self._outliner = QtWidgets.QListWidget()
        self._outliner.currentItemChanged.connect(self._on_outliner_changed)
        layout.addWidget(self._outliner, 1)
        layout.addWidget(QtWidgets.QLabel("Add"))
        grid = QtWidgets.QGridLayout()
        for i, shape in enumerate(("box", "sphere", "cylinder", "cone",
                                   "torus", "plane")):
            btn = QtWidgets.QPushButton(shape.capitalize())
            btn.clicked.connect(lambda _c=False, s=shape: self._add_object(s))
            grid.addWidget(btn, i // 2, i % 2)
        layout.addLayout(grid)
        row = QtWidgets.QHBoxLayout()
        dup_btn = QtWidgets.QPushButton("Duplicate")
        dup_btn.clicked.connect(self._duplicate_selected)
        del_btn = QtWidgets.QPushButton("Delete")
        del_btn.clicked.connect(self._delete_selected)
        row.addWidget(dup_btn)
        row.addWidget(del_btn)
        layout.addLayout(row)
        return pane

    def _build_center(self):
        pane = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(pane)
        layout.setContentsMargins(6, 6, 6, 6)
        self._viewport = Viewport()
        self._viewport.object_selected.connect(self._on_viewport_selected)
        # Gizmo toolbar (T12): mode buttons above the viewport + W/E/R
        # shortcuts (Blender convention). The shortcuts live on the
        # centre pane with WidgetWithChildrenShortcut, so they fire
        # when the viewport or the camera row has focus — and never
        # while the user is typing a name or a number in another pane.
        gizmo_row = QtWidgets.QHBoxLayout()
        gizmo_row.addWidget(QtWidgets.QLabel("Gizmo:"))
        self._gizmo_buttons = {}
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for mode, label, key in (("move", "Move (W)", "W"),
                                 ("rotate", "Rotate (E)", "E"),
                                 ("scale", "Scale (R)", "R")):
            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(mode == "move")
            btn.clicked.connect(
                lambda _c=False, m=mode: self._set_gizmo_mode(m))
            group.addButton(btn)
            self._gizmo_buttons[mode] = btn
            gizmo_row.addWidget(btn)
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), pane)
            shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(
                lambda m=mode: self._set_gizmo_mode(m))
        gizmo_row.addStretch(1)
        layout.addLayout(gizmo_row)
        layout.addWidget(self._viewport, 1)
        cam_row = QtWidgets.QHBoxLayout()
        frame_btn = QtWidgets.QPushButton("Frame")
        frame_btn.clicked.connect(self._viewport.frame)
        cam_row.addWidget(frame_btn)
        self._camera_combo = QtWidgets.QComboBox()
        self._camera_combo.currentIndexChanged.connect(self._on_camera_chosen)
        cam_row.addWidget(self._camera_combo, 1)
        save_cam_btn = QtWidgets.QPushButton("Save camera")
        save_cam_btn.clicked.connect(self._save_camera)
        cam_row.addWidget(save_cam_btn)
        layout.addLayout(cam_row)
        info_row = QtWidgets.QHBoxLayout()
        self._tri_label = QtWidgets.QLabel("")
        info_row.addWidget(self._tri_label, 1)
        self._budget_label = QtWidgets.QLabel("")
        info_row.addWidget(self._budget_label)
        layout.addLayout(info_row)
        return pane

    def _build_right(self):
        pane = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(pane)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(QtWidgets.QLabel("Inspector"))
        self._name_edit = QtWidgets.QLineEdit()
        self._name_edit.editingFinished.connect(self._on_name_edited)
        layout.addWidget(self._name_edit)
        self._spins = {}
        for group in ("position", "rotation", "scale"):
            layout.addWidget(QtWidgets.QLabel(group.capitalize()))
            row = QtWidgets.QHBoxLayout()
            for axis in ("x", "y", "z"):
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(-100000.0, 100000.0)
                spin.setDecimals(1)
                spin.setSingleStep(10.0 if group != "scale" else 0.1)
                spin.valueChanged.connect(
                    lambda _v, g=group, a=axis: self._on_transform_edited(g, a))
                spin.editingFinished.connect(self._commit_live_edit)
                self._spins[(group, axis)] = spin
                row.addWidget(spin)
            layout.addLayout(row)
        layout.addWidget(QtWidgets.QLabel("Material"))
        color_row = QtWidgets.QHBoxLayout()
        self._color_btn = QtWidgets.QPushButton("Color…")
        self._color_btn.clicked.connect(self._pick_color)
        self._color_swatch = QtWidgets.QLabel("      ")
        color_row.addWidget(self._color_btn)
        color_row.addWidget(self._color_swatch)
        color_row.addStretch(1)
        layout.addLayout(color_row)
        for label, key in (("Roughness", "roughness"),
                           ("Metalness", "metalness")):
            layout.addWidget(QtWidgets.QLabel(label))
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 100)
            slider.valueChanged.connect(
                lambda v, k=key: self._on_material_slider(k, v / 100.0))
            slider.sliderReleased.connect(self._commit_live_edit)
            setattr(self, f"_{key}_slider", slider)
            layout.addWidget(slider)
        layout.addWidget(QtWidgets.QLabel("Texture"))
        self._texture_combo = QtWidgets.QComboBox()
        self._texture_combo.addItems(("none", "wood_grain", "brushed_metal",
                                      "worn_leather"))
        self._texture_combo.currentTextChanged.connect(self._on_texture_changed)
        layout.addWidget(self._texture_combo)
        self._smooth_check = QtWidgets.QCheckBox("Smooth shading")
        self._smooth_check.toggled.connect(self._on_smooth_shading_changed)
        layout.addWidget(self._smooth_check)
        self._cast_check = QtWidgets.QCheckBox("Cast shadow")
        self._cast_check.toggled.connect(self._on_cast_shadow_changed)
        layout.addWidget(self._cast_check)
        self._receive_check = QtWidgets.QCheckBox("Receive shadow")
        self._receive_check.toggled.connect(self._on_receive_shadow_changed)
        layout.addWidget(self._receive_check)
        layout.addStretch(1)
        return pane

    def _build_timeline(self):
        bar = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(6, 6, 6, 6)
        self._play_btn = QtWidgets.QPushButton("Play")
        self._play_btn.clicked.connect(self._toggle_play)
        layout.addWidget(self._play_btn)
        self._frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._frame_slider.setRange(0, self.scene["animation"]["end_frame"])
        self._frame_slider.valueChanged.connect(self._on_slider_frame)
        layout.addWidget(self._frame_slider, 1)
        self._frame_label = QtWidgets.QLabel("Frame 0 / 48")
        layout.addWidget(self._frame_label)
        self._fps_label = QtWidgets.QLabel("24 fps")
        layout.addWidget(self._fps_label)
        add_key_btn = QtWidgets.QPushButton("Add key")
        add_key_btn.clicked.connect(self._add_key)
        layout.addWidget(add_key_btn)
        del_key_btn = QtWidgets.QPushButton("Delete key")
        del_key_btn.clicked.connect(self._delete_key)
        layout.addWidget(del_key_btn)
        end_spin = QtWidgets.QSpinBox()
        end_spin.setRange(1, 10000)
        end_spin.setValue(self.scene["animation"]["end_frame"])
        end_spin.setPrefix("End ")
        end_spin.valueChanged.connect(self._on_end_frame_changed)
        layout.addWidget(end_spin)
        return bar

    # -- scene <-> UI -----------------------------------------------------
    def _scene_base_path(self):
        """Folder the scene was loaded from, or None for a fresh
        unsaved scene. Mesh files resolve against this (T7)."""
        if not self.scene_path:
            return None
        return os.path.dirname(os.path.abspath(self.scene_path))

    def _resolve_meshes(self):
        """Drop mesh objects whose asset file is missing or unreadable,
        and return a list of plain warnings — one per dropped object,
        each naming the file. The rest of the scene loads untouched;
        a missing mesh is skipped, never faked. Pure scene work: it
        does not show anything itself, so the caller decides how the
        warnings are presented (status bar always, dialog on Open).
        Saving never moves mesh files, so a scene saved into another
        folder warns about its meshes on the next load — that is the
        design, not a bug."""
        warnings = []
        base = self._scene_base_path()
        kept = []
        for obj in self.scene["objects"]:
            if obj.get("shape") != "mesh":
                kept.append(obj)
                continue
            name = obj.get("name") or obj["id"]
            mesh_file = obj.get("mesh_file")
            if not mesh_file:
                warnings.append(
                    f"Mesh object '{name}' has no mesh file, so it was "
                    "skipped.")
                continue
            path = mesh_file if os.path.isabs(mesh_file) else (
                os.path.join(base, mesh_file) if base else None)
            if path is None or not os.path.isfile(path):
                warnings.append(
                    f"Mesh file not found: {mesh_file} — object "
                    f"'{name}' was skipped.")
                continue
            try:
                meshfile_mod.load_mesh(path)
            except (OSError, ValueError) as exc:
                warnings.append(
                    f"Mesh file could not be read: {mesh_file} ({exc}) "
                    f"— object '{name}' was skipped.")
                continue
            kept.append(obj)
        if warnings:
            self.scene["objects"] = kept
            if self.selected_id and not find_object(self.scene,
                                                    self.selected_id):
                self.selected_id = None
        return warnings

    def _apply_scene_to_ui(self):
        self._viewport.set_scene(self.scene,
                                 base_path=self._scene_base_path())
        self._viewport.set_frame(self._frame)
        self._refresh_outliner()
        self._refresh_cameras()
        self._refresh_inspector()
        self._refresh_counts()
        self._frame_slider.setRange(0, self.scene["animation"]["end_frame"])
        self._frame_slider.setValue(self._frame)
        self._update_frame_label()

    def _refresh_outliner(self):
        self._refreshing = True
        self._outliner.clear()
        for obj in self.scene["objects"]:
            item = QtWidgets.QListWidgetItem(
                f"{obj['name']} ({obj['shape']})")
            item.setData(QtCore.Qt.UserRole, obj["id"])
            self._outliner.addItem(item)
            if obj["id"] == self.selected_id:
                self._outliner.setCurrentItem(item)
        self._refreshing = False

    def _refresh_cameras(self):
        self._refreshing = True
        self._camera_combo.clear()
        self._camera_combo.addItem("Free view")
        for cam in self.scene.get("cameras", []):
            self._camera_combo.addItem(cam["name"])
        self._refreshing = False

    def _refresh_inspector(self):
        self._refreshing = True
        obj = find_object(self.scene, self.selected_id) if self.selected_id else None
        enabled = obj is not None
        self._name_edit.setEnabled(enabled)
        for spin in self._spins.values():
            spin.setEnabled(enabled)
        self._color_btn.setEnabled(enabled)
        self._roughness_slider.setEnabled(enabled)
        self._metalness_slider.setEnabled(enabled)
        self._texture_combo.setEnabled(enabled)
        # Smooth shading has no effect on a mesh object (T7): the
        # asset's stored normals are used as-is, so the checkbox is
        # disabled for meshes rather than silently ignored.
        self._smooth_check.setEnabled(
            enabled and (obj is None or obj.get("shape") != "mesh"))
        self._cast_check.setEnabled(enabled)
        self._receive_check.setEnabled(enabled)
        if obj is None:
            self._name_edit.setText("")
            self._refreshing = False
            return
        self._name_edit.setText(obj["name"])
        for group in ("position", "rotation", "scale"):
            for i, axis in enumerate(("x", "y", "z")):
                self._spins[(group, axis)].setValue(obj[group][i])
        mat = obj.get("material", {})
        self._roughness_slider.setValue(int(float(mat.get("roughness", 0.6)) * 100))
        self._metalness_slider.setValue(int(float(mat.get("metalness", 0.0)) * 100))
        tex = mat.get("texture", "none")
        idx = self._texture_combo.findText(tex)
        self._texture_combo.setCurrentIndex(max(0, idx))
        color = mat.get("base_color", "#b0b0b0")
        self._color_swatch.setStyleSheet(f"background: {color};")
        smooth = obj.get("smooth_shading",
                         default_smooth_shading(obj.get("shape", "box")))
        self._smooth_check.setChecked(bool(smooth))
        self._cast_check.setChecked(bool(obj.get("cast_shadow", True)))
        self._receive_check.setChecked(bool(obj.get("receive_shadow", True)))
        self._refreshing = False

    def _refresh_counts(self):
        # Mesh triangles come from the assets' recorded counts via
        # the viewport cache (T7) — no vertex data is built to count.
        tris = total_triangles(
            self.scene,
            mesh_counts=self._viewport.mesh_triangle_counts())
        self._tri_label.setText(f"{len(self.scene['objects'])} objects · "
                                f"{tris:,} triangles")
        if tris > SOFT_TRI_BUDGET:
            self._budget_label.setText(
                f"Over the {SOFT_TRI_BUDGET:,} soft budget — still fine, "
                "just heavier")
            self._budget_label.setStyleSheet("color: #d8a020;")
        else:
            self._budget_label.setText("")
            self._budget_label.setStyleSheet("")

    def _update_frame_label(self):
        end = self.scene["animation"]["end_frame"]
        fps = self.scene["animation"]["fps"]
        self._frame_label.setText(f"Frame {self._frame} / {end}")
        self._fps_label.setText(f"{fps} fps")

    def _update_play_timer(self):
        fps = max(1, self.scene["animation"]["fps"])
        self._play_timer.setInterval(int(1000 / fps))

    # -- undo ---------------------------------------------------------------
    def _push_undo(self):
        # A live edit still in progress commits first, so the stack stays
        # in time order (live snapshot before this discrete snapshot).
        self._commit_live_edit()
        self._undo_stack.append(scene_copy(self.scene))
        del self._undo_stack[:-50]

    def undo(self):
        self._live_edit_snapshot = None
        if not self._undo_stack:
            self.statusBar().showMessage("Nothing to undo")
            return
        self.scene = self._undo_stack.pop()
        if self.selected_id and not find_object(self.scene, self.selected_id):
            self.selected_id = None
        self._apply_scene_to_ui()
        self.statusBar().showMessage("Undone")

    # -- selection ------------------------------------------------------------
    def _select(self, object_id):
        self.selected_id = object_id
        self._viewport.selected_id = object_id
        self._viewport.update()
        self._refresh_outliner()
        self._refresh_inspector()

    def _set_gizmo_mode(self, mode):
        """Switch the gizmo mode (T12) from a toolbar button or a
        W/E/R shortcut. The viewport owns the mode; the buttons just
        reflect it. setChecked emits toggled, not clicked, so syncing
        the buttons here cannot re-enter this method."""
        self._viewport.set_gizmo_mode(mode)
        self._gizmo_buttons[mode].setChecked(True)

    def _on_outliner_changed(self, current, _previous):
        if self._refreshing or current is None:
            return
        self._select(current.data(QtCore.Qt.UserRole))

    def _on_viewport_selected(self, object_id):
        self._select(object_id or None)

    # -- modeling --------------------------------------------------------------
    def _add_object(self, shape):
        self._push_undo()
        obj = new_object(shape)
        obj["name"] = f"{shape.capitalize()} {len(self.scene['objects']) + 1}"
        self.scene["objects"].append(obj)
        self._select(obj["id"])
        self._refresh_counts()
        self._viewport.refresh()
        self.statusBar().showMessage(f"Added {obj['name']}")

    def _duplicate_selected(self):
        obj = find_object(self.scene, self.selected_id) if self.selected_id else None
        if obj is None:
            return
        self._push_undo()
        copy_obj = scene_copy(obj)
        copy_obj["id"] = new_object(obj["shape"])["id"]
        copy_obj["name"] = obj["name"] + " copy"
        copy_obj["position"] = [obj["position"][0] + 30.0,
                                obj["position"][1] + 30.0,
                                obj["position"][2]]
        self.scene["objects"].append(copy_obj)
        self._select(copy_obj["id"])
        self._refresh_counts()
        self._viewport.refresh()

    def _delete_selected(self):
        if not self.selected_id:
            return
        self._push_undo()
        self.scene["objects"] = [o for o in self.scene["objects"]
                                 if o["id"] != self.selected_id]
        anim_keys = self.scene["animation"]["keys"]
        anim_keys.pop(self.selected_id, None)
        self._select(None)
        self._refresh_counts()
        self._viewport.refresh()

    def _on_name_edited(self):
        obj = find_object(self.scene, self.selected_id) if self.selected_id else None
        if obj is None or self._refreshing:
            return
        name = self._name_edit.text().strip()
        if name and name != obj["name"]:
            self._push_undo()
            obj["name"] = name
            self._refresh_outliner()

    def _arm_live_edit(self):
        # Snapshot the pre-edit scene once per edit gesture. Called from
        # the live handlers BEFORE they apply their change, so the
        # snapshot holds the old values; _commit_live_edit pushes it onto
        # the undo stack when the gesture ends.
        if self._live_edit_snapshot is None:
            self._live_edit_snapshot = scene_copy(self.scene)

    def _commit_live_edit(self):
        if self._live_edit_snapshot is not None:
            self._undo_stack.append(self._live_edit_snapshot)
            del self._undo_stack[:-50]
            self._live_edit_snapshot = None

    def _on_transform_edited(self, group, axis):
        if self._refreshing:
            return
        obj = find_object(self.scene, self.selected_id) if self.selected_id else None
        if obj is None:
            return
        idx = ("x", "y", "z").index(axis)
        value = self._spins[(group, axis)].value()
        if obj[group][idx] != value:
            self._arm_live_edit()
            obj[group][idx] = value
            self._viewport.update()

    def _on_material_slider(self, key, value):
        if self._refreshing:
            return
        obj = find_object(self.scene, self.selected_id) if self.selected_id else None
        if obj is None:
            return
        if obj.get("material", {}).get(key) != round(value, 2):
            self._arm_live_edit()
            obj.setdefault("material", {})[key] = round(value, 2)
            self._viewport.update()

    def _on_texture_changed(self, text):
        if self._refreshing:
            return
        obj = find_object(self.scene, self.selected_id) if self.selected_id else None
        if obj is None:
            return
        if obj.get("material", {}).get("texture") != text:
            self._push_undo()
            obj.setdefault("material", {})["texture"] = text
            self._viewport.update()

    def _on_smooth_shading_changed(self, checked):
        if self._refreshing:
            return
        obj = find_object(self.scene, self.selected_id) if self.selected_id else None
        if obj is None:
            return
        if obj.get("smooth_shading") != checked:
            self._push_undo()
            obj["smooth_shading"] = checked
            self._viewport.update()

    def _on_cast_shadow_changed(self, checked):
        if self._refreshing:
            return
        obj = find_object(self.scene, self.selected_id) if self.selected_id else None
        if obj is None:
            return
        if obj.get("cast_shadow", True) != checked:
            self._push_undo()
            obj["cast_shadow"] = checked
            self._viewport.update()

    def _on_receive_shadow_changed(self, checked):
        if self._refreshing:
            return
        obj = find_object(self.scene, self.selected_id) if self.selected_id else None
        if obj is None:
            return
        if obj.get("receive_shadow", True) != checked:
            self._push_undo()
            obj["receive_shadow"] = checked
            self._viewport.update()

    def _pick_color(self):
        obj = find_object(self.scene, self.selected_id) if self.selected_id else None
        if obj is None:
            return
        current = QtGui.QColor(obj.get("material", {}).get("base_color", "#b0b0b0"))
        color = QtWidgets.QColorDialog.getColor(current, self, "Base color")
        if color.isValid():
            self._push_undo()
            obj.setdefault("material", {})["base_color"] = color.name()
            self._color_swatch.setStyleSheet(f"background: {color.name()};")
            self._viewport.update()

    # -- cameras ---------------------------------------------------------------
    def _on_camera_chosen(self, index):
        if self._refreshing or index <= 0:
            return
        name = self._camera_combo.currentText()
        for cam in self.scene.get("cameras", []):
            if cam["name"] == name:
                self._viewport.set_camera_position(cam["position"],
                                                   cam["look_at"])
                return

    def _save_camera(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save camera",
                                                  "Camera name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        self._push_undo()
        cams = self.scene.setdefault("cameras", [])
        cam = self._viewport.current_camera(name)
        for i, existing in enumerate(cams):
            if existing["name"] == name:
                cams[i] = cam
                break
        else:
            cams.append(cam)
        self._refresh_cameras()
        idx = self._camera_combo.findText(name)
        if idx >= 0:
            self._camera_combo.setCurrentIndex(idx)
        self.statusBar().showMessage(f"Camera '{name}' saved")

    # -- file ------------------------------------------------------------------
    def _new_scene(self):
        self._push_undo()
        self.scene = new_scene()
        self.scene_path = None
        self.selected_id = None
        self._frame = 0
        self._apply_scene_to_ui()

    def _open_scene(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open scene", "", "Scene files (*.json)")
        if not path:
            return
        try:
            self.scene = load_scene(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open failed", str(exc))
            return
        self.scene_path = path
        self.selected_id = None
        self._frame = 0
        missing = self._resolve_meshes()
        self._apply_scene_to_ui()
        if missing:
            self.statusBar().showMessage(missing[0])
            QtWidgets.QMessageBox.warning(self, "Mesh files missing",
                                          "\n".join(missing))
        else:
            self.statusBar().showMessage(f"Opened {os.path.basename(path)}")

    def _save_scene(self):
        if self.scene_path:
            save_scene(self.scene, self.scene_path)
            self.statusBar().showMessage("Saved")
        else:
            self._save_scene_as()

    def _save_scene_as(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save scene as", "", "Scene files (*.json)")
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        save_scene(self.scene, path)
        self.scene_path = path
        self.statusBar().showMessage(f"Saved to {os.path.basename(path)}")

    # -- mesh import (T8 .obj, T9 .glb) --------------------------------------
    def import_mesh_objects(self, parsed, scale):
        """Write a parsed import's assets and add its objects to the
        scene. Dialog-free, so tests can drive it directly. One undo
        step for the whole import (undo removes the objects; the
        .mesh.json files stay on disk, which is fine — they are
        harmless until a scene references them). Returns the created
        objects, empty when the scene has no folder yet (assets live
        next to the scene file) or when no asset could be written."""
        base = self._scene_base_path()
        if base is None:
            return []
        assets_dir = os.path.join(base, "assets")
        try:
            os.makedirs(assets_dir, exist_ok=True)
        except OSError as exc:
            self.statusBar().showMessage(
                f"Could not make the assets folder: {exc}")
            return []
        objects = meshio_mod.build(parsed, scale)
        stem = os.path.splitext(os.path.basename(parsed["source"]))[0]
        used = set()
        written = []
        for item in objects:
            filename = meshio_mod.unique_asset_name(
                stem, item["name"], assets_dir, used)
            path = os.path.join(assets_dir, filename)
            try:
                meshfile_mod.save_mesh(
                    path, item["mesh"], source=parsed["source"],
                    imported=datetime.now().isoformat(timespec="seconds"))
            except OSError as exc:
                self.statusBar().showMessage(
                    f"Could not write {filename}: {exc}")
                continue
            written.append((item, filename))
        if not written:
            return []
        self._push_undo()
        created = []
        for item, filename in written:
            obj = new_object("mesh", name=item["name"],
                             material=item["material"])
            obj["mesh_file"] = f"assets/{filename}"
            # .glb items carry their node transform (T9); .obj items
            # do not, and keep the origin default.
            for key in ("position", "rotation", "scale"):
                if key in item:
                    obj[key] = [float(v) for v in item[key]]
            self.scene["objects"].append(obj)
            created.append(obj)
        self._refresh_outliner()
        # The soft triangle budget is checked here (T8): the readout
        # warns past it, never refuses — _refresh_counts does that
        # for every scene change, imports included.
        self._refresh_counts()
        self._viewport.frame()
        self._viewport.refresh()
        self._select(created[0]["id"])
        return created

    def _import_mesh(self):
        if self.scene_path is None:
            QtWidgets.QMessageBox.information(
                self, "Save the scene first",
                "Imported meshes live in an assets folder next to the "
                "scene file, so please save the scene first.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import mesh", "", "Mesh files (*.obj *.glb)")
        if not path:
            return
        try:
            parsed = meshio_mod.parse(path)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Import failed", str(exc))
            return
        report = parsed["report"]
        if meshio_mod.is_heavy(report["triangles"]):
            answer = QtWidgets.QMessageBox.question(
                self, "Heavy mesh",
                f"This mesh has {report['triangles']:,} triangles — "
                "this will be heavy on this machine. Import it anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
        dialog = ImportDialog(self, report)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        created = self.import_mesh_objects(parsed, dialog.scale())
        if created:
            self.statusBar().showMessage(
                f"Imported {len(created)} object(s), "
                f"{report['triangles']:,} triangles from "
                f"{os.path.basename(path)}")
            if report["warnings"]:
                QtWidgets.QMessageBox.information(
                    self, "Import warnings", "\n".join(report["warnings"]))
        else:
            self.statusBar().showMessage("Nothing was imported")

    # -- mesh export (.obj T10, .glb T11) --------------------------------
    def export_mesh_objects(self, objects, path, scale):
        """Dialog-free export core: write the objects to path (.obj)
        + the .mtl next to it, and return the report {objects,
        triangles, warnings}. Tests drive it directly. Raises
        OSError for an unwritable path — the menu slots turn that
        into a plain warning dialog."""
        return meshio_obj_mod.export_obj(
            objects, path, scale=scale,
            base_path=self._scene_base_path())

    def export_glb_objects(self, objects, path, scale):
        """Dialog-free export core: write the objects to path (.glb)
        and return the report {objects, triangles, warnings}. The
        .glb twin of export_mesh_objects; tests drive it directly.
        Raises OSError for an unwritable path."""
        return meshio_glb_mod.export_glb(
            objects, path, scale=scale,
            base_path=self._scene_base_path())

    def _export_objects(self, objects, title, kind="obj"):
        """File dialog + ExportDialog + write for one export. The
        plan step resolves the geometry first, so the dialog shows
        the real object/triangle counts and names any objects that
        will be skipped (missing mesh files) before anything is
        written. `kind` picks the format: "obj" (T10) or "glb"
        (T11) — the file filter, the plan/write module, and the
        dialog title follow it."""
        if kind == "glb":
            file_filter = "glTF Binary (*.glb)"
            ext = ".glb"
            dialog_title = "Export .glb"
            plan_mod = meshio_glb_mod
            write = self.export_glb_objects
        else:
            file_filter = "Wavefront OBJ (*.obj)"
            ext = ".obj"
            dialog_title = "Export .obj"
            plan_mod = meshio_obj_mod
            write = self.export_mesh_objects
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, title, "", file_filter)
        if not path:
            return
        if not path.endswith(ext):
            path += ext
        _items, plan = plan_mod.plan_export(
            objects, base_path=self._scene_base_path())
        dialog = ExportDialog(self, plan, title=dialog_title)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        try:
            report = write(objects, path, dialog.scale())
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(
            f"Exported {report['objects']} object(s), "
            f"{report['triangles']:,} triangles to "
            f"{os.path.basename(path)}")
        # The plan's warnings were already shown in the dialog; only
        # new ones (something changed between the dialog and the
        # write) get a second dialog.
        shown = set(plan["warnings"])
        new = [w for w in report["warnings"] if w not in shown]
        if new:
            QtWidgets.QMessageBox.information(self, "Export warnings",
                                              "\n".join(new))

    def _export_mesh(self):
        obj = find_object(self.scene, self.selected_id) \
            if self.selected_id else None
        if obj is None:
            self.statusBar().showMessage("Select an object first")
            return
        self._export_objects([obj], "Export mesh")

    def _export_scene_obj(self):
        if not self.scene["objects"]:
            self.statusBar().showMessage(
                "Nothing to export — the scene is empty")
            return
        self._export_objects(self.scene["objects"],
                             "Export scene as .obj")

    def _export_mesh_glb(self):
        obj = find_object(self.scene, self.selected_id) \
            if self.selected_id else None
        if obj is None:
            self.statusBar().showMessage("Select an object first")
            return
        self._export_objects([obj], "Export mesh as .glb", kind="glb")

    def _export_scene_glb(self):
        if not self.scene["objects"]:
            self.statusBar().showMessage(
                "Nothing to export — the scene is empty")
            return
        self._export_objects(self.scene["objects"],
                             "Export scene as .glb", kind="glb")

    # -- render / export --------------------------------------------------------
    def _render_still(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Render still", "", "PNG files (*.png)")
        if not path:
            return
        if not path.endswith(".png"):
            path += ".png"
        self._viewport.update()
        QtWidgets.QApplication.processEvents()
        image = self._viewport.grabFramebuffer()
        if image.save(path):
            self.statusBar().showMessage(f"Rendered {os.path.basename(path)}")
        else:
            QtWidgets.QMessageBox.warning(self, "Render failed",
                                          f"Could not write {path}")

    def render_frame_to(self, frame, path):
        """Set the frame, grab the viewport, save a PNG. Used by export
        and by the verification script. The gizmo never appears in a
        render (T12): it is interactive-viewport only, so it is
        suppressed for the grab and restored after — even if the
        save fails. (Render still, the File-menu still, grabs the
        live viewport as-is including the gizmo, the same way it
        already includes the selection box: what you see is what
        you get there. Only the frame-export path suppresses it.)"""
        self.set_frame(frame)
        self._viewport.set_gizmo_suppressed(True)
        try:
            self._viewport.update()
            QtWidgets.QApplication.processEvents()
            QtCore.QCoreApplication.sendPostedEvents()
            image = self._viewport.grabFramebuffer()
            return image.save(path)
        finally:
            self._viewport.set_gizmo_suppressed(False)

    def _export_frames(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Export frames to folder")
        if not folder:
            return
        end = self.scene["animation"]["end_frame"]
        saved = 0
        was_playing = self._playing
        if was_playing:
            self._toggle_play()
        for frame in range(end + 1):
            path = os.path.join(folder, f"frame_{frame:04d}.png")
            if self.render_frame_to(frame, path):
                saved += 1
        self.statusBar().showMessage(
            f"Exported {saved} of {end + 1} frames to {folder}")

    # -- animation ----------------------------------------------------------------
    def set_frame(self, frame):
        self._frame = max(0, min(self.scene["animation"]["end_frame"],
                                 int(frame)))
        self._viewport.set_frame(self._frame)
        self._refreshing = True
        self._frame_slider.setValue(self._frame)
        self._refreshing = False
        self._update_frame_label()

    def _on_slider_frame(self, value):
        if self._refreshing:
            return
        self.set_frame(value)

    def _on_end_frame_changed(self, value):
        self.scene["animation"]["end_frame"] = int(value)
        self._frame_slider.setRange(0, int(value))
        if self._frame > int(value):
            self.set_frame(int(value))
        self._update_frame_label()

    def _toggle_play(self):
        self._playing = not self._playing
        if self._playing:
            self._update_play_timer()
            self._play_timer.start()
            self._play_btn.setText("Pause")
        else:
            self._play_timer.stop()
            self._play_btn.setText("Play")

    def _advance_frame(self):
        end = self.scene["animation"]["end_frame"]
        frame = self._frame + 1
        if frame > end:
            frame = 0
        self.set_frame(frame)

    def _add_key(self):
        obj = find_object(self.scene, self.selected_id) if self.selected_id else None
        if obj is None:
            self.statusBar().showMessage("Select an object first")
            return
        self._push_undo()
        for track in ("position", "rotation", "scale"):
            anim_mod.set_key(self.scene, obj["id"], track, self._frame,
                             obj[track])
        self.statusBar().showMessage(
            f"Key added for {obj['name']} at frame {self._frame}")

    def _delete_key(self):
        obj = find_object(self.scene, self.selected_id) if self.selected_id else None
        if obj is None:
            return
        self._push_undo()
        for track in ("position", "rotation", "scale"):
            anim_mod.delete_key(self.scene, obj["id"], track, self._frame)
        self.statusBar().showMessage(
            f"Key removed for {obj['name']} at frame {self._frame}")
