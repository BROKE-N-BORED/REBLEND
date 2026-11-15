"""The main window: outliner left, viewport center, inspector right,
timeline along the bottom. Plain labels, no marketing copy.

Budgets warn, never refuse: the triangle readout turns to a warning past
the soft budget, and nothing is ever blocked for size.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

from . import animation as anim_mod
from . import meshfile as meshfile_mod
from . import meshio as meshio_mod
from . import meshio_glb as meshio_glb_mod
from . import meshio_obj as meshio_obj_mod
from . import particles as particles_mod
from . import style as style_mod
from . import video as video_mod
from .inspector_data import FIELD_HELP, _physics_enabled
from .inspector_data import param_knobs_for
from .scene import (EMITTER_COLLIDE, EMITTER_PRESETS, EMITTER_SHAPES,
                    MATERIAL_PRESETS, default_smooth_shading,
                    find_object, load_scene, material_preset_for,
                    new_emitter, new_object, new_scene,
                    resolve_material_style, save_scene, scene_copy,
                    total_triangles)
from .textures import (FAMILIES, STYLES, count_to_spacing,
                       merged_params, spacing_to_count, style_params)
from .viewport import Viewport

SOFT_TRI_BUDGET = 100000

# Section open/close animation (UI1, library entry 002): 250 ms with
# an OutBack settle — spring physics on user-caused panel motion.
SECTION_ANIMATION_MS = 250

# Emitter editor morph (UI2, library entry 009): the window grows
# from the control that opened it in 150 ms — under the 400 ms
# ceiling with room to spare (a 250 ms morph measured 300 ms
# wall-clock idle and 448 ms under sandbox load, too close to the
# ceiling; 150 ms is a standard fast transition and still reads
# as a morph, not a pop). The animation can be interrupted
# (opening again mid-flight retargets from the current geometry).
EMITTER_MORPH_MS = 150

# Per-material knobs (PO2): the re-bake debounce. A knob drag
# re-bakes the texture, and a bake costs 100-300 ms in the sandbox
# on every style — re-baking on every value change would stall
# the drag (the spin itself would lag behind the mouse). The
# timer waits for a 150 ms pause, then repaints once: the spin
# stays responsive and the texture follows within a frame or two
# of the last change. 150 ms is under the 300 ms ceiling the
# morph and the done-moment already use, and the value is already
# in the scene, so an export mid-drag still bakes the latest
# value (render_frame_to renders the current scene, not the
# debounced viewport).
PARAM_REBAKE_MS = 150

# UI3 (U5): the default window size — 1280x700 fits 1366x768 with
# room for window decorations and the taskbar.
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 700

# UI7 (U16, entry 008): the video export's one-shot completion
# moment — how long the done state shows in the status bar. 200 ms
# is under the 300 ms ceiling with room for event-loop latency
# (the UI2 morph lesson: wall-clock stretches under sandbox load,
# so the margin matters). The dot is amber while work runs, green
# for the done moment.
VIDEO_DONE_MS = 200
# The dot colors live in style.py now (UI8): one look, one place —
# style_mod.BUSY_DOT_COLOR (amber while work runs) and
# style_mod.DONE_DOT_COLOR (green done moment).


def app_icon_path():
    """The app icon file (assets/reblend.png, the provisional cube
    until the logo is picked), resolved relative to this package so
    it works from the installed location too."""
    return Path(__file__).resolve().parent.parent / "assets" / "reblend.png"

# Plain-words explanations under each emitter editor field (U20),
# in the UI1 inspector style. No em-dashes, no bare "&" — both are
# swept by the suite. Each line ends with a period.
EMITTER_HELP = {
    "preset": "A starting point. Picking one fills every setting below.",
    "shape": "Where new particles appear. Point is one spot, box and "
             "sphere fill a volume, cone sprays from a tip.",
    "box_size": "How wide, deep, and tall the box is, in centimeters.",
    "sphere_radius": "How big the sphere is, in centimeters.",
    "rate": "How many particles start each second.",
    "lifetime": "How long each particle lives, in seconds.",
    "lifetime_spread": "How much lifetimes vary. 0 means all the same, "
                       "1 means fully mixed.",
    "speed": "How fast new particles leave, in centimeters per second.",
    "speed_spread": "How much speeds vary. 0 means all the same, "
                    "1 means fully mixed.",
    "spread": "How wide the spray cone opens, in degrees. "
              "0 is a straight jet.",
    "size": "How big a particle starts, in centimeters.",
    "size_over_life": "How big a particle ends, compared to its start. "
                      "1 means unchanged, 3 means three times bigger, "
                      "0 means it shrinks away.",
    "color_start": "The color a particle starts with.",
    "color_end": "The color a particle ends with. "
                 "Colors blend over its life.",
    "fade_in": "How much of its life a particle spends fading in. "
               "0 means it appears at full strength.",
    "fade_out": "How much of its life a particle spends fading out. "
                "0 means it vanishes at the end.",
    "gravity_factor": "How much gravity pulls these particles. 1 is "
                      "normal, 0 floats, a negative value rises.",
    "drag": "How fast the air slows particles down. 0 means no drag.",
    "wind_response": "How much the scene wind pushes these particles. "
                     "0 ignores the wind, 1 follows it fully.",
    "turbulence": "How much particles wobble as they fly, in "
                  "centimeters. 0 is a straight path.",
    "attract": "Pull particles toward another object. None means no "
               "pull. The pull bends how particles fly.",
    "attract_strength": "How hard the pull is. A negative value pushes "
                        "away instead.",
    "collide": "What happens when a particle reaches the floor. None "
               "lets it fall through, Kill removes it on impact, "
               "Bounce bounces it back up.",
    "seed": "The same seed and settings always make the same "
            "particles. Change it for a different pattern.",
}

# Qt's QWIDGETSIZE_MAX (no maximum height): this PySide6 build does
# not expose the constant, so the value is written out.
NO_MAX_HEIGHT = 16777215


class _HeightDriver(QtCore.QObject):
    """Drives a widget's maximumHeight through an animation, clamped
    at zero — OutBack overshoots below 0 on a collapse, and Qt warns
    on a negative maximum size. The animation targets this driver's
    height property, never the widget's maximumHeight directly."""

    def __init__(self, widget):
        super().__init__(widget)
        self._widget = widget
        self._height = 0

    @QtCore.Property(int)
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value
        self._widget.setMaximumHeight(max(0, value))


class CollapsibleSection(QtWidgets.QWidget):
    """One inspector section (UI1): a header that opens and closes
    the body with a 250 ms OutBack settle (library entry 002 —
    spring physics on user-caused panel motion). The logical
    expanded state changes at once; only the height animates, so
    callers can read is_expanded() without waiting for the
    animation. The animation caps the body's maximumHeight while it
    runs and releases the cap when it finishes, so a later window
    resize never clips the body. Toggling mid-flight retargets
    from the current height."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._title = title
        self._expanded = True
        self._header = QtWidgets.QPushButton()
        self._header.setFlat(True)
        self._header.setStyleSheet("text-align: left; font-weight: bold;")
        self._header.setText("\u25be " + title)
        self._header.clicked.connect(self.toggle)
        self.body = QtWidgets.QWidget()
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        # A small indent under the header — the pane is narrow, so
        # the indent stays small enough that the spin rows still fit.
        self.body_layout.setContentsMargins(6, 2, 6, 6)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._header)
        layout.addWidget(self.body)
        self._driver = _HeightDriver(self.body)
        self._anim = QtCore.QPropertyAnimation(
            self._driver, b"height", self)
        self._anim.setDuration(SECTION_ANIMATION_MS)
        self._anim.setEasingCurve(QtCore.QEasingCurve.OutBack)
        self._anim.finished.connect(self._on_animation_finished)

    def is_expanded(self):
        return self._expanded

    def toggle(self):
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded, animate=True):
        if expanded == self._expanded:
            return
        # UI8 (reduced motion): animations off, state changes stay —
        # the section jumps to its final state instead of settling.
        if style_mod.reduced_motion():
            animate = False
        self._expanded = expanded
        self._header.setText(("\u25be " if expanded else "\u25b8 ")
                             + self._title)
        self._anim.stop()
        if expanded:
            was_hidden = self.body.isHidden()
            self.body.setVisible(True)
            target = self._target_height()
            if animate:
                # From zero when the body was hidden (its height is
                # stale then), from the current height mid-flight.
                start = 0 if was_hidden else self.body.height()
                self.body.setMaximumHeight(start)
                self._anim.setStartValue(start)
                self._anim.setEndValue(target)
                self._anim.start()
            else:
                self.body.setMaximumHeight(NO_MAX_HEIGHT)
        else:
            if animate:
                start = self.body.height()
                self.body.setMaximumHeight(start)
                self._anim.setStartValue(start)
                self._anim.setEndValue(0)
                self._anim.start()
            else:
                self.body.setMaximumHeight(0)
                self.body.setVisible(False)

    def _target_height(self):
        # The body holds word-wrapped help lines, so its height
        # depends on its width — totalHeightForWidth accounts for
        # that; sizeHint alone would clip the wrapped lines.
        layout = self.body.layout()
        width = self.body.width()
        if layout is not None and width > 0:
            height = layout.totalHeightForWidth(width)
            if height > 0:
                return height
        return self.body.sizeHint().height()

    def _on_animation_finished(self):
        if self._expanded:
            self.body.setMaximumHeight(NO_MAX_HEIGHT)
        else:
            self.body.setVisible(False)


class _PopupHideFilter(QtCore.QObject):
    """Calls back when a combo's popup closes (UI12). The popup
    view gets a Hide event on every close — a pick, Esc, or a
    click outside — so a hover-preview ends exactly when the
    browsing ends. Probed, not assumed: both the view and its
    top-level container get the Hide; watching the view alone
    is enough. Never consumes the event. Parented to the window
    so Qt destroys it with the window (a dangling filter must
    never call back into a half-destroyed window)."""

    def __init__(self, parent, on_hide):
        super().__init__(parent)
        self._on_hide = on_hide

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Hide:
            self._on_hide()
        return False


def _help_label(text):
    # A plain-words line under a dialog field (U20), in the UI1
    # inspector style: muted grey, word-wrapped, so it reads as
    # help, not as another control.
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {style_mod.MUTED};")
    return label


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
        layout.addWidget(_help_label(
            "One file unit becomes this many centimeters. "
            "100 fits files measured in meters."))
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
        layout.addWidget(_help_label(
            "One centimeter becomes this much in the file. "
            "0.01 fits files measured in meters."))
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def scale(self):
        return self._scale_spin.value()


class VideoExportDialog(QtWidgets.QDialog):
    """Format / resolution / frame range before a video export (T15).
    Mirrors ExportDialog: plain rows, Ok/Cancel, the choices read back
    after accept(). The resolution options are fractions of the
    viewport size (aspect preserved); the sizes shown are the
    even-rounded export sizes ffmpeg will actually get. The frame
    range defaults to the scene's full animation range. fps comes
    from the scene and is shown, not edited. validate() returns an
    error string or None — accept() shows it and stays open, and
    tests call validate() directly so no message box ever opens
    under test."""

    def __init__(self, parent, viewport_width, viewport_height,
                 end_frame, fps=24):
        super().__init__(parent)
        self.setWindowTitle("Export video")
        self._end_frame = int(end_frame)
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Format:"))
        self._format_combo = QtWidgets.QComboBox()
        for key in ("mp4", "webm"):
            self._format_combo.addItem(video_mod.FORMATS[key]["label"],
                                       key)
        row.addWidget(self._format_combo)
        layout.addLayout(row)
        layout.addWidget(_help_label(
            "MP4 plays almost everywhere. WebM is smaller and plays "
            "in browsers."))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Resolution:"))
        self._size_combo = QtWidgets.QComboBox()
        for scale, label in ((1.0, "Full size"),
                             (0.5, "Half size"),
                             (0.25, "Quarter size")):
            width, height = video_mod.export_size(viewport_width,
                                                  viewport_height,
                                                  scale)
            self._size_combo.addItem(f"{label} ({width}×{height})",
                                     (width, height))
        row.addWidget(self._size_combo)
        layout.addLayout(row)
        layout.addWidget(_help_label(
            "Smaller sizes export faster and make smaller files."))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Frames:"))
        self._start_spin = QtWidgets.QSpinBox()
        self._start_spin.setRange(0, self._end_frame)
        self._start_spin.setValue(0)
        row.addWidget(self._start_spin)
        row.addWidget(QtWidgets.QLabel("to"))
        self._end_spin = QtWidgets.QSpinBox()
        self._end_spin.setRange(0, self._end_frame)
        self._end_spin.setValue(self._end_frame)
        row.addWidget(self._end_spin)
        layout.addLayout(row)
        layout.addWidget(_help_label(
            "Which frames go into the video, first to last."))
        layout.addWidget(
            QtWidgets.QLabel(f"{fps} frames per second (from the scene)"))
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def validate(self):
        if self._start_spin.value() > self._end_spin.value():
            return "The start frame must not be after the end frame."
        return None

    def accept(self):
        error = self.validate()
        if error is not None:
            QtWidgets.QMessageBox.warning(self, "Export video", error)
            return
        super().accept()

    def format(self):
        return self._format_combo.currentData()

    def size(self):
        return self._size_combo.currentData()

    def frame_range(self):
        return self._start_spin.value(), self._end_spin.value()


class EmitterEditorWindow(QtWidgets.QWidget):
    """The dedicated emitter editor (UI2): a closable window that
    edits one emitter object's full T16 parameter set — preset,
    shape, emission, appearance over life, forces, seed — grouped
    and explained in plain words, in the UI1 inspector style.

    The window edits the live emitter dict: every change lands in
    the scene at once, so the viewport shows it immediately. Undo
    follows the inspector's pattern — a spin edit arms the live
    snapshot on the first change and commits on editingFinished,
    a combo pick or a color pick pushes one discrete step — so
    each parameter change, and each preset pick, is one undo
    step. Hand-editing any parameter flips the preset picker to
    custom (the preset rule); picking a preset fills every
    parameter from that preset.

    The window opens with a morph (library entry 009): it grows
    from the control that opened it to its full size in
    EMITTER_MORPH_MS, under the 400 ms ceiling. The logical state
    flips at once — the window is shown immediately — and only
    the geometry animates, following the UI1 section pattern.
    Opening again mid-flight stops the animation and retargets
    from the current geometry, so the morph is interruptible.

    The editor never holds an object reference: it re-finds the
    emitter by id on every access, so an undo or redo (which
    replaces the whole scene) cannot leave it editing a stale
    copy. When the emitter is gone, the main window closes this
    window (see _sync_emitter_editor)."""

    # Full size the morph ends at.
    _WIDTH = 440
    _HEIGHT = 620

    def __init__(self, main):
        super().__init__(main, QtCore.Qt.Window)
        self._main = main
        self.object_id = None
        self._refreshing = False
        self.setWindowTitle("Emitter editor")
        # No minimum size on the window itself: the morph starts
        # from a small rect, and a minimum would clamp it to full
        # size before the animation ever ran.
        self._anim = QtCore.QPropertyAnimation(self, b"geometry", self)
        self._anim.setDuration(EMITTER_MORPH_MS)
        self._anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._spins = {}
        self._color_btns = {}
        self._color_swatches = {}
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        inner = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(inner)
        layout.setContentsMargins(10, 10, 10, 10)
        self._build_preset(layout)
        self._build_shape(layout)
        self._build_emission(layout)
        self._build_appearance(layout)
        self._build_forces(layout)
        self._build_seed(layout)
        layout.addStretch(1)
        scroll.setWidget(inner)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # -- construction ------------------------------------------------
    @staticmethod
    def _help(key):
        # A plain-words line under a field (U20), in the UI1 style:
        # muted, word-wrapped, a sentence, not another control.
        label = QtWidgets.QLabel(EMITTER_HELP[key])
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {style_mod.MUTED};")
        return label

    @staticmethod
    def _group(layout, title):
        header = QtWidgets.QLabel(title)
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

    def _add_spin(self, layout, key, label, low, high, decimals, step):
        layout.addWidget(QtWidgets.QLabel(label))
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.valueChanged.connect(
            lambda _v, k=key: self._on_spin_changed(k))
        spin.editingFinished.connect(self._on_spin_finished)
        self._spins[key] = spin
        layout.addWidget(spin)
        return spin

    def _build_preset(self, layout):
        self._group(layout, "Preset")
        self._preset_combo = QtWidgets.QComboBox()
        for name in particles_mod.preset_names():
            self._preset_combo.addItem(name.capitalize(), name)
        self._preset_combo.currentIndexChanged.connect(
            self._on_preset_changed)
        layout.addWidget(self._preset_combo)
        layout.addWidget(self._help("preset"))

    def _build_shape(self, layout):
        self._group(layout, "Shape")
        layout.addWidget(QtWidgets.QLabel("Shape"))
        self._shape_combo = QtWidgets.QComboBox()
        for shape in EMITTER_SHAPES:
            self._shape_combo.addItem(shape.capitalize(), shape)
        self._shape_combo.currentIndexChanged.connect(
            self._on_shape_changed)
        layout.addWidget(self._shape_combo)
        layout.addWidget(self._help("shape"))
        layout.addWidget(QtWidgets.QLabel("Box size"))
        row = QtWidgets.QHBoxLayout()
        for i, axis in enumerate(("x", "y", "z")):
            row.addWidget(QtWidgets.QLabel(axis.upper()))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.0, 100000.0)
            spin.setDecimals(1)
            spin.setSingleStep(10.0)
            spin.valueChanged.connect(
                lambda _v, k=("box_size", i): self._on_spin_changed(k))
            spin.editingFinished.connect(self._on_spin_finished)
            self._spins[("box_size", i)] = spin
            row.addWidget(spin)
        layout.addLayout(row)
        layout.addWidget(self._help("box_size"))
        self._add_spin(layout, "sphere_radius", "Sphere radius",
                       0.0, 100000.0, 1, 5.0)
        layout.addWidget(self._help("sphere_radius"))

    def _build_emission(self, layout):
        self._group(layout, "Emission")
        for key, label, low, high, decimals, step in (
                ("rate", "Rate", 0.0, 100000.0, 1, 10.0),
                ("lifetime", "Lifetime", 0.01, 10000.0, 2, 0.1),
                ("lifetime_spread", "Lifetime spread", 0.0, 1.0, 2, 0.1),
                ("speed", "Speed", 0.0, 100000.0, 1, 10.0),
                ("speed_spread", "Speed spread", 0.0, 1.0, 2, 0.1),
                ("spread", "Spread", 0.0, 180.0, 1, 5.0)):
            self._add_spin(layout, key, label, low, high, decimals, step)
            layout.addWidget(self._help(key))

    def _build_appearance(self, layout):
        self._group(layout, "Appearance over life")
        for key, label, low, high, decimals, step in (
                ("size", "Size", 0.01, 10000.0, 2, 1.0),
                ("size_over_life", "Size over life", 0.0, 100.0, 2, 0.1)):
            self._add_spin(layout, key, label, low, high, decimals, step)
            layout.addWidget(self._help(key))
        for key, label in (("color_start", "Start color"),
                           ("color_end", "End color")):
            layout.addWidget(QtWidgets.QLabel(label))
            row = QtWidgets.QHBoxLayout()
            btn = QtWidgets.QPushButton("Pick…")
            btn.clicked.connect(
                lambda _c=False, k=key: self._on_color_clicked(k))
            self._color_btns[key] = btn
            swatch = QtWidgets.QLabel("      ")
            self._color_swatches[key] = swatch
            row.addWidget(btn)
            row.addWidget(swatch)
            row.addStretch(1)
            layout.addLayout(row)
            layout.addWidget(self._help(key))
        for key, label in (("fade_in", "Fade in"),
                           ("fade_out", "Fade out")):
            self._add_spin(layout, ("opacity", key), label,
                           0.0, 1.0, 2, 0.05)
            layout.addWidget(self._help(key))

    def _build_forces(self, layout):
        self._group(layout, "Forces")
        for key, label, low, high, decimals, step in (
                ("gravity_factor", "Gravity factor",
                 -100.0, 100.0, 2, 0.1),
                ("drag", "Drag", 0.0, 100.0, 2, 0.1),
                ("wind_response", "Wind response", 0.0, 1.0, 2, 0.1),
                ("turbulence", "Turbulence", 0.0, 1000.0, 1, 1.0)):
            self._add_spin(layout, key, label, low, high, decimals, step)
            layout.addWidget(self._help(key))
        layout.addWidget(QtWidgets.QLabel("Attract to"))
        self._attract_combo = QtWidgets.QComboBox()
        self._attract_combo.currentIndexChanged.connect(
            self._on_attract_changed)
        layout.addWidget(self._attract_combo)
        layout.addWidget(self._help("attract"))
        self._add_spin(layout, "attract_strength", "Attract strength",
                       -100000.0, 100000.0, 1, 10.0)
        layout.addWidget(self._help("attract_strength"))
        layout.addWidget(QtWidgets.QLabel("At the floor"))
        self._collide_combo = QtWidgets.QComboBox()
        for mode, display in (("none", "None"),
                              ("kill", "Kill at the floor"),
                              ("bounce", "Bounce off the floor")):
            self._collide_combo.addItem(display, mode)
        self._collide_combo.currentIndexChanged.connect(
            self._on_collide_changed)
        layout.addWidget(self._collide_combo)
        layout.addWidget(self._help("collide"))

    def _build_seed(self, layout):
        self._group(layout, "Seed")
        layout.addWidget(QtWidgets.QLabel("Seed"))
        spin = QtWidgets.QSpinBox()
        spin.setRange(0, 999999999)
        spin.setSingleStep(1)
        spin.valueChanged.connect(
            lambda _v: self._on_spin_changed("seed"))
        spin.editingFinished.connect(self._on_spin_finished)
        self._spins["seed"] = spin
        layout.addWidget(spin)
        layout.addWidget(self._help("seed"))

    # -- open / close / morph -----------------------------------------
    def _end_rect(self):
        # Full size, centered on the main window.
        center = self._main.geometry().center()
        return QtCore.QRect(center.x() - self._WIDTH // 2,
                            center.y() - self._HEIGHT // 2,
                            self._WIDTH, self._HEIGHT)

    def open_for(self, object_id, anchor_rect=None):
        """Open the editor for an emitter, morphing from the control
        that opened it (anchor_rect, in global coordinates) to the
        full window. The window shows at once — only the geometry
        animates — and a mid-flight open retargets from the current
        geometry, so the morph is interruptible."""
        self.object_id = object_id
        self._load_params()
        end = self._end_rect()
        if self._anim.state() != QtCore.QAbstractAnimation.Stopped:
            self._anim.stop()
            start = self.geometry()
        elif anchor_rect is not None:
            start = QtCore.QRect(anchor_rect)
        else:
            start = QtCore.QRect(end.center().x() - 20,
                                 end.center().y() - 20, 40, 40)
        if style_mod.reduced_motion():
            # UI8 (reduced motion): no morph — the editor appears
            # at its full size at once.
            self.setGeometry(end)
            self.show()
            self.raise_()
            return
        self.setGeometry(start)
        self.show()
        self.raise_()
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._anim.start()

    def refresh(self):
        """Re-read the emitter's parameters into the controls (after
        an undo, a redo, or a scene change). If the emitter is gone,
        the window closes instead."""
        if self._params() is None:
            self.close()
            return
        self._load_params()

    def closeEvent(self, event):
        self._anim.stop()
        super().closeEvent(event)

    # -- scene access ---------------------------------------------------
    def _object(self):
        if self.object_id is None:
            return None
        obj = find_object(self._main.scene, self.object_id)
        if obj is None or obj.get("kind") != "emitter":
            return None
        return obj

    def _params(self):
        obj = self._object()
        if obj is None:
            return None
        return obj.get("emitter")

    @staticmethod
    def _param_value(params, key):
        # A tuple key reads a nested value (box_size components,
        # opacity fades); a plain key reads a top-level parameter.
        if isinstance(key, tuple):
            return params[key[0]][key[1]]
        if key == "attract_strength":
            attract = params.get("attract") or {}
            return attract.get("strength", 0.0)
        return params[key]

    @staticmethod
    def _set_param_value(params, key, value):
        if isinstance(key, tuple):
            params[key[0]][key[1]] = value
        elif key == "attract_strength":
            attract = params.get("attract")
            if isinstance(attract, dict):
                attract["strength"] = value
        else:
            params[key] = value

    def _load_params(self):
        """Fill every control from the emitter's stored parameters.
        The attract combo is rebuilt from the scene's objects each
        time, so a renamed or deleted object cannot leave a stale
        entry behind."""
        obj = self._object()
        params = obj.get("emitter") if obj else None
        self._refreshing = True
        if obj is not None:
            self.setWindowTitle(f"Emitter editor - {obj['name']}")
        if params is not None:
            idx = self._preset_combo.findData(params.get("preset",
                                                         "custom"))
            self._preset_combo.setCurrentIndex(max(0, idx))
            idx = self._shape_combo.findData(params.get("shape",
                                                        "point"))
            self._shape_combo.setCurrentIndex(max(0, idx))
            for key, spin in self._spins.items():
                spin.setValue(self._param_value(params, key))
            for key, swatch in self._color_swatches.items():
                swatch.setStyleSheet(
                    f"background: {params.get(key, '#ffffff')};")
            idx = self._collide_combo.findData(params.get("collide",
                                                          "none"))
            self._collide_combo.setCurrentIndex(max(0, idx))
            self._attract_combo.clear()
            self._attract_combo.addItem("None", None)
            for other in self._main.scene["objects"]:
                if other["id"] != self.object_id:
                    self._attract_combo.addItem(other["name"],
                                                other["id"])
            attract = params.get("attract") or {}
            target = attract.get("target")
            idx = self._attract_combo.findData(target) if target else -1
            # A stored target that is no longer in the scene (its
            # object was deleted) shows as None; the stored value
            # is left alone until the user picks something.
            self._attract_combo.setCurrentIndex(max(0, idx))
            self._update_shape_rows(params)
            self._update_attract_rows(params)
        self._refreshing = False

    def _update_shape_rows(self, params):
        # Box size matters only for a box, sphere radius only for a
        # sphere — the other row stays visible but disabled, so the
        # layout never jumps when the shape changes.
        shape = params.get("shape", "point")
        for i in range(3):
            self._spins[("box_size", i)].setEnabled(shape == "box")
        self._spins["sphere_radius"].setEnabled(shape == "sphere")

    def _update_attract_rows(self, params):
        # Strength only applies while a target is picked.
        has_target = isinstance(params.get("attract"), dict)
        self._spins["attract_strength"].setEnabled(has_target)

    # -- edits ------------------------------------------------------------
    def _flip_custom(self, params):
        # Hand-editing any parameter flips the picker to custom (the
        # preset rule). The flip rides inside the same undo step as
        # the edit itself — the snapshot was taken before either.
        if params.get("preset") != "custom":
            params["preset"] = "custom"
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentIndex(
                self._preset_combo.findData("custom"))
            self._preset_combo.blockSignals(False)

    def _after_edit(self):
        # The viewport shows the change at once, and the inspector's
        # Emitter facts follow the edit.
        self._main._viewport.update()
        self._main._refresh_inspector()

    def _on_spin_changed(self, key):
        if self._refreshing:
            return
        params = self._params()
        if params is None:
            return
        # Strength has no stored value without a target (the spin
        # is disabled then); a programmatic set must not flip the
        # preset to custom for a change that stores nothing.
        if key == "attract_strength" and params.get("attract") is None:
            return
        value = self._spins[key].value()
        if self._param_value(params, key) != value:
            self._main._arm_live_edit()
            self._set_param_value(params, key, value)
            self._flip_custom(params)
            self._after_edit()

    def _on_spin_finished(self):
        # The gesture ends: commit the armed snapshot as one undo
        # step (the inspector pattern — a change that ends where it
        # started pushes no step).
        self._main._commit_live_edit()

    def _on_preset_changed(self, index):
        if self._refreshing:
            return
        params = self._params()
        if params is None:
            return
        name = self._preset_combo.itemData(index)
        if name is None or params.get("preset") == name:
            return
        # Picking a preset fills EVERY parameter (one undo step).
        # "Custom" keeps the current values and just renames them.
        self._main._push_undo()
        new_params = particles_mod.apply_preset(params, name)
        params.clear()
        params.update(new_params)
        self._load_params()
        self._after_edit()

    def _on_shape_changed(self, index):
        if self._refreshing:
            return
        params = self._params()
        if params is None:
            return
        shape = self._shape_combo.itemData(index)
        if shape is None or params.get("shape") == shape:
            return
        self._main._push_undo()
        params["shape"] = shape
        self._flip_custom(params)
        self._update_shape_rows(params)
        self._after_edit()

    def _on_collide_changed(self, index):
        if self._refreshing:
            return
        params = self._params()
        if params is None:
            return
        mode = self._collide_combo.itemData(index)
        if mode is None or params.get("collide") == mode:
            return
        self._main._push_undo()
        params["collide"] = mode
        self._flip_custom(params)
        self._after_edit()

    def _on_attract_changed(self, index):
        if self._refreshing:
            return
        params = self._params()
        if params is None:
            return
        target = self._attract_combo.itemData(index)
        current = params.get("attract") or {}
        if target is None:
            if params.get("attract") is None:
                return
            self._main._push_undo()
            params["attract"] = None
        else:
            if current.get("target") == target:
                return
            self._main._push_undo()
            params["attract"] = {"target": target,
                                 "strength": current.get("strength", 0.0)}
        self._flip_custom(params)
        self._update_attract_rows(params)
        self._after_edit()

    def _on_color_clicked(self, key):
        params = self._params()
        if params is None:
            return
        current = QtGui.QColor(params.get(key, "#ffffff"))
        color = QtWidgets.QColorDialog.getColor(current, self,
                                               "Pick a color")
        if color.isValid():
            self._set_color(key, color.name())

    def _set_color(self, key, hex_value):
        """Set a color parameter (the dialog handler's path, and the
        testable path — no modal dialog involved). One undo step."""
        params = self._params()
        if params is None or params.get(key) == hex_value:
            return
        self._main._push_undo()
        params[key] = hex_value
        self._flip_custom(params)
        self._color_swatches[key].setStyleSheet(
            f"background: {hex_value};")
        self._after_edit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, scene=None, scene_path=None, settings=None):
        super().__init__()
        # UI5 (U31): the [*] placeholder is Qt's modified marker —
        # the window manager shows a * here when isWindowModified()
        # is true. (The em-dash is pre-existing; UI6 owns the
        # visible-text sweep.)
        self.setWindowTitle("Reblend v0.2[*]")
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        # UI3 (U27): the app icon in the window chrome. Missing file
        # is not an error — the window just has no icon.
        icon_path = app_icon_path()
        if icon_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))
        # UI3 (U5): the QSettings object that remembers the window
        # geometry + splitter positions. main.py passes the app's
        # settings; None means no remembering (tests and verify
        # scripts construct windows without settings, and tests
        # that do care pass an isolated QSettings file).
        self._settings = settings
        self.scene = scene or new_scene()
        self.scene_path = scene_path
        self.selected_id = None
        self._undo_stack = []
        # Redo (T14): the scene as it was before an undo, so redo
        # restores it. Same scene-copy model as undo, same 50 cap.
        # Any new edit clears it — redo only replays undone edits.
        self._redo_stack = []
        self._live_edit_snapshot = None  # pre-edit scene for live controls
        self._playing = False
        self._frame = 0
        # The progress dialog while a video export runs (T15); None
        # otherwise. Held as an attribute so tests and the verify
        # script can press Cancel on it mid-export.
        self._video_progress = None
        # The frame-export progress dialog (UI7); None otherwise.
        # The same pattern as _video_progress.
        self._frame_progress = None
        # UI7 (U16/entry 008): the activity indicator's generation.
        # Every busy-state change bumps it, so the done moment's
        # hide only clears the state it scheduled for — never a
        # newer busy state that started while the moment showed.
        self._busy_generation = 0
        self._done_generation = 0
        # UI7 (entry 008): how many times the video done moment has
        # fired — one per completed export; tests + the verify
        # script count it to prove the moment is one-shot.
        self._video_done_count = 0
        self._refreshing = False
        # The emitter editor window (UI2); None until first opened,
        # then one shared instance, re-targeted per emitter.
        self._emitter_editor = None
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)
        self._build_ui()
        # UI3 (U5): reopen where the last run left off, when
        # settings were passed in. No saved values means the
        # default size above stands.
        self._restore_window_state()
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
        # UI5 (U31): a close from the window system (the X button,
        # Alt+F4) asks about unsaved changes first. A programmatic
        # close() never prompts — there is no user to answer, and
        # the test suite + verify scripts close windows freely
        # (probed: close() gives spontaneous() False). File > Quit
        # prompts through _quit before it ever calls close(). The
        # prompt runs before any cleanup, and a Cancel leaves the
        # window exactly as it was.
        if event.spontaneous() and self.isWindowModified():
            if not self._confirm_unsaved():
                event.ignore()
                return
        self._play_timer.stop()
        if self._emitter_editor is not None:
            self._emitter_editor.close()
        self._viewport.cleanup_gl()
        # UI3 (U5): remember where the window was for the next run.
        self._save_window_state()
        super().closeEvent(event)

    def _restore_window_state(self):
        """UI3 (U5): restore the saved geometry + splitter positions.
        Nothing saved (or no settings at all) leaves the default
        size and stretch factors in place."""
        if self._settings is None:
            return
        geometry = self._settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self._settings.value("window/splitter")
        if state is not None:
            self._splitter.restoreState(state)

    def _save_window_state(self):
        """UI3 (U5): save the geometry + splitter positions now, and
        flush them to disk so a relaunch reads them back."""
        if self._settings is None:
            return
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.setValue("window/splitter",
                                self._splitter.saveState())
        self._settings.sync()

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
        # UI3 (U5): kept so the splitter positions can be saved and
        # restored with the window geometry.
        self._splitter = splitter
        main = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._build_timeline())
        self.setCentralWidget(main)
        self.statusBar().showMessage("Ready")
        self._build_activity_indicator()

    def _build_activity_indicator(self):
        """UI7 (U16): the busy dot + status text in the status bar.
        Both are permanent widgets, hidden while idle — a hidden
        widget takes no space, so the idle window looks exactly as
        before (the full-window shot stays byte-identical). The
        done timer is a child of the window, so closing the window
        mid-moment destroys it and the hide slot never fires on a
        deleted window."""
        dot = QtWidgets.QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(
            f"background-color: {style_mod.BUSY_DOT_COLOR}; border-radius: 5px;")
        dot.hide()
        self._busy_dot = dot
        text = QtWidgets.QLabel()
        text.hide()
        self._busy_text = text
        bar = self.statusBar()
        bar.addPermanentWidget(dot)
        bar.addPermanentWidget(text)
        self._done_timer = QtCore.QTimer(self)
        self._done_timer.setSingleShot(True)
        self._done_timer.timeout.connect(self._hide_done_moment)

    def _set_busy(self, text):
        """UI7 (U16): show the busy dot + status text (plain words).
        Call before a long operation; _clear_busy hides them
        again. The dot is amber while work runs."""
        self._busy_generation += 1
        self._busy_dot.setStyleSheet(
            f"background-color: {style_mod.BUSY_DOT_COLOR}; border-radius: 5px;")
        self._busy_text.setText(text)
        self._busy_dot.show()
        self._busy_text.show()

    def _clear_busy(self):
        """UI7 (U16): hide the busy dot + status text."""
        self._busy_generation += 1
        self._busy_dot.hide()
        self._busy_text.hide()

    def _show_video_done_moment(self):
        """UI7 (entry 008): the video export's one-shot completion
        moment — the dot turns green and the text reads "Video
        saved" for VIDEO_DONE_MS, then hides. One-shot by
        construction: one single-shot timer per call, and the hide
        only clears the state this call scheduled (the generation
        guard), so a busy state that starts while the moment shows
        is never cleared by the moment's timer."""
        self._busy_generation += 1
        self._done_generation = self._busy_generation
        self._video_done_count += 1
        self._busy_dot.setStyleSheet(
            f"background-color: {style_mod.DONE_DOT_COLOR}; border-radius: 5px;")
        self._busy_text.setText("Video saved")
        self._busy_dot.show()
        self._busy_text.show()
        self._done_timer.start(VIDEO_DONE_MS)

    def _hide_done_moment(self):
        """Hide the done moment — only if no newer busy state has
        started since it was scheduled."""
        if self._done_generation != self._busy_generation:
            return
        self._busy_dot.hide()
        self._busy_text.hide()

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
        video_action = QtGui.QAction("Export video…", self)
        video_action.triggered.connect(self._export_video)
        file_menu.addAction(video_action)
        file_menu.addSeparator()
        quit_action = QtGui.QAction("Quit", self)
        quit_action.setShortcut(QtGui.QKeySequence.Quit)
        quit_action.triggered.connect(self._quit)
        file_menu.addAction(quit_action)
        edit_menu = self.menuBar().addMenu("Edit")
        undo_action = QtGui.QAction("Undo", self)
        undo_action.setShortcut(QtGui.QKeySequence.Undo)
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)
        redo_action = QtGui.QAction("Redo", self)
        redo_action.setShortcut(QtGui.QKeySequence.Redo)
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)

    def _build_left(self):
        pane = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(pane)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(QtWidgets.QLabel("Objects"))
        self._outliner = QtWidgets.QListWidget()
        self._outliner.currentItemChanged.connect(self._on_outliner_changed)
        # Double-clicking an emitter opens its editor (UI2); other
        # objects ignore a double-click.
        self._outliner.itemDoubleClicked.connect(
            self._on_outliner_double_clicked)
        layout.addWidget(self._outliner, 1)
        layout.addWidget(QtWidgets.QLabel("Add"))
        grid = QtWidgets.QGridLayout()
        for i, shape in enumerate(("box", "sphere", "cylinder", "cone",
                                   "torus", "plane")):
            btn = QtWidgets.QPushButton(shape.capitalize())
            btn.clicked.connect(lambda _c=False, s=shape: self._add_object(s))
            grid.addWidget(btn, i // 2, i % 2)
        # Emitter (UI2): without this, emitters cannot be created in
        # the app at all — they only arrived from a loaded scene.
        self._add_emitter_btn = QtWidgets.QPushButton("Emitter")
        self._add_emitter_btn.clicked.connect(self._add_emitter)
        grid.addWidget(self._add_emitter_btn, 3, 0)
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
        # Live measurements (UI10): the FPS readout follows the
        # viewport's rolling average, one label update per paint.
        self._viewport.frame_stats_changed.connect(
            self._refresh_frame_stats)
        # Live inspector during a gizmo drag (T13): the viewport fires
        # transform_changed on every drag step, and the inspector
        # re-reads the object. _refresh_inspector sets _refreshing
        # while it writes the spin boxes, so the values cannot feed
        # back into _on_transform_edited.
        self._viewport.transform_changed.connect(self._refresh_inspector)
        # Gizmo-drag undo (T14): one undo step per drag gesture, via
        # the existing live-edit pattern — arm on drag start (the
        # snapshot holds the pre-drag scene), commit on release.
        self._viewport.gizmo_drag_started.connect(self._arm_live_edit)
        self._viewport.gizmo_drag_finished.connect(self._commit_live_edit)
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
        for mode, label, key, tip in (
                ("move", "Move (W)", "W",
                 "Move the selected object. Press W."),
                ("rotate", "Rotate (E)", "E",
                 "Rotate the selected object. Press E."),
                ("scale", "Scale (R)", "R",
                 "Scale the selected object. Press R.")):
            btn = QtWidgets.QPushButton(label)
            btn.setToolTip(tip)
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
        # Live measurements (UI10): the measured FPS + ms/frame
        # readout, next to the object/triangle counts. Hidden until
        # the viewport has measured enough paints — a hidden widget
        # takes no space, so the row looks exactly as before until
        # the first real numbers arrive.
        self._frame_stats_label = QtWidgets.QLabel("")
        self._frame_stats_label.setVisible(False)
        info_row.addWidget(self._frame_stats_label)
        self._budget_label = QtWidgets.QLabel("")
        info_row.addWidget(self._budget_label)
        layout.addLayout(info_row)
        return pane

    @staticmethod
    def _help(key):
        # A plain-words line under a field (U20), muted so it reads
        # as help, not as another control. Word-wrapped: the
        # inspector is narrow and the lines are sentences.
        label = QtWidgets.QLabel(FIELD_HELP[key])
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {style_mod.MUTED};")
        return label

    def _build_right(self):
        # Inspector (UI1): collapsible sections inside a scroll area,
        # so the pane can grow (T19's material dropdown and T20's
        # physics controls land in these sections) without crowding
        # the window. Always-visible sections: Object, Transform,
        # Material. Just-in-time sections (entry 004): Emitter
        # (emitter objects only), Physics (objects with physics
        # enabled — nothing yet, T20 fills it).
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # The pane must never squeeze below the width its content
        # needs: with no horizontal scrollbar, a narrower pane would
        # clip the spin boxes and the help lines instead of
        # scrolling. 250 px holds the three-spin rows at their
        # minimum width with room to spare.
        scroll.setMinimumWidth(250)
        self._inspector_scroll = scroll
        inner = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(inner)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(QtWidgets.QLabel("Inspector"))
        self._sections = {}
        self._section_conditions = {}
        self._build_object_section(layout)
        self._build_transform_section(layout)
        self._build_material_section(layout)
        self._build_emitter_section(layout)
        self._build_physics_section(layout)
        layout.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def _build_object_section(self, layout):
        section = CollapsibleSection("Object")
        self._sections["object"] = section
        layout.addWidget(section)
        body = section.body_layout
        body.addWidget(QtWidgets.QLabel("Name"))
        self._name_edit = QtWidgets.QLineEdit()
        self._name_edit.editingFinished.connect(self._on_name_edited)
        body.addWidget(self._name_edit)
        body.addWidget(self._help("name"))
        self._cast_check = QtWidgets.QCheckBox("Cast shadow")
        self._cast_check.toggled.connect(self._on_cast_shadow_changed)
        body.addWidget(self._cast_check)
        body.addWidget(self._help("cast_shadow"))
        self._receive_check = QtWidgets.QCheckBox("Receive shadow")
        self._receive_check.toggled.connect(self._on_receive_shadow_changed)
        body.addWidget(self._receive_check)
        body.addWidget(self._help("receive_shadow"))

    def _build_transform_section(self, layout):
        section = CollapsibleSection("Transform")
        self._sections["transform"] = section
        layout.addWidget(section)
        body = section.body_layout
        self._spins = {}
        for group in ("position", "rotation", "scale"):
            body.addWidget(QtWidgets.QLabel(group.capitalize()))
            row = QtWidgets.QHBoxLayout()
            for axis in ("x", "y", "z"):
                row.addWidget(QtWidgets.QLabel(axis.upper()))
                spin = QtWidgets.QDoubleSpinBox()
                # Narrow by design: three spins plus their axis
                # labels share one row in a ~250 px pane. At 48 px
                # each the row fits with room to spare; a wider
                # minimum would push the inner widget past the
                # pane and clip the Z spin (no horizontal scrollbar).
                spin.setMinimumWidth(48)
                spin.setRange(-100000.0, 100000.0)
                spin.setDecimals(1)
                spin.setSingleStep(10.0 if group != "scale" else 0.1)
                spin.valueChanged.connect(
                    lambda _v, g=group, a=axis: self._on_transform_edited(g, a))
                spin.editingFinished.connect(self._commit_live_edit)
                self._spins[(group, axis)] = spin
                row.addWidget(spin)
            body.addLayout(row)
            body.addWidget(self._help(group))

    def _build_material_section(self, layout):
        section = CollapsibleSection("Material")
        self._sections["material"] = section
        layout.addWidget(section)
        body = section.body_layout
        body.addWidget(QtWidgets.QLabel("Color"))
        color_row = QtWidgets.QHBoxLayout()
        self._color_btn = QtWidgets.QPushButton("Color…")
        self._color_btn.clicked.connect(self._pick_color)
        self._color_swatch = QtWidgets.QLabel("      ")
        color_row.addWidget(self._color_btn)
        color_row.addWidget(self._color_swatch)
        color_row.addStretch(1)
        body.addLayout(color_row)
        body.addWidget(self._help("color"))
        for label, key in (("Roughness", "roughness"),
                           ("Metalness", "metalness")):
            body.addWidget(QtWidgets.QLabel(label))
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 100)
            slider.valueChanged.connect(
                lambda v, k=key: self._on_material_slider(k, v / 100.0))
            slider.sliderReleased.connect(self._commit_live_edit)
            setattr(self, f"_{key}_slider", slider)
            body.addWidget(slider)
            body.addWidget(self._help(key))
        body.addWidget(QtWidgets.QLabel("Texture"))
        # Grouped by family (PO1): None first, then one disabled
        # header row per family (no data, so a click or a hover
        # on it does nothing — the handlers below already return
        # early on a None item), then that family's styles with
        # their plain-words names and style keys as item data.
        self._texture_combo = QtWidgets.QComboBox()
        self._texture_combo.addItem("None", "none")
        for family_key, family_display in FAMILIES:
            header_index = self._texture_combo.count()
            self._texture_combo.addItem(family_display, None)
            self._texture_combo.model().item(header_index).setEnabled(
                False)
            for style_key, style in STYLES.items():
                if style["family"] != family_key:
                    continue
                self._texture_combo.addItem(style["display"], style_key)
        self._texture_combo.currentIndexChanged.connect(
            self._on_texture_changed)
        # Hover-preview (UI12): hovering an item previews it,
        # and the popup closing ends the preview — see
        # _on_texture_highlighted and _PopupHideFilter.
        self._texture_combo.highlighted.connect(
            self._on_texture_highlighted)
        body.addWidget(self._texture_combo)
        body.addWidget(self._help("texture"))
        # Simple / Advanced (PO3): Simple shows only the knobs
        # that change the look most; Advanced shows every knob.
        # Session-only, like the "Also set look" tick below —
        # the checkbox lives on the window, never in the scene,
        # and toggling it rebuilds the rows without touching the
        # material (no undo step, no modified flag). Default is
        # Simple (unchecked): the spec's default mode. The
        # toggle is always visible, like the look tick — PO4's
        # layout rows and PO5's wear rows will make it meaningful
        # for nearly every style, so it stays put instead of
        # appearing and vanishing with the style.
        self._material_advanced_check = QtWidgets.QCheckBox(
            "Show all settings")
        self._material_advanced_check.toggled.connect(
            self._on_material_advanced_toggled)
        body.addWidget(self._material_advanced_check)
        body.addWidget(self._help("material_advanced"))
        # Per-material knobs (PO2): a container the rows are
        # rebuilt into whenever the style changes — each style
        # shows its own knobs (see _sync_param_rows), and "none"
        # shows none. The container stays HIDDEN while it has no
        # rows: an empty widget in the body layout still takes
        # the layout's spacing around it, which lengthened the
        # inspector by a few pixels and moved the scrollbar thumb
        # by 1 px (37 px of diff in the 00 shot). Hidden, it
        # takes no space at all, so the style-less inspector is
        # pixel-identical to before PO2.
        self._param_rows = QtWidgets.QWidget()
        self._param_layout = QtWidgets.QVBoxLayout(self._param_rows)
        self._param_layout.setContentsMargins(0, 0, 0, 0)
        self._param_rows.setVisible(False)
        self._param_spins = {}
        body.addWidget(self._param_rows)
        # The re-bake debounce (see PARAM_REBAKE_MS): a knob
        # change restarts it, and the repaint fires once, 150 ms
        # after the last change.
        self._param_rebake_timer = QtCore.QTimer(self)
        self._param_rebake_timer.setSingleShot(True)
        self._param_rebake_timer.setInterval(PARAM_REBAKE_MS)
        self._param_rebake_timer.timeout.connect(self._viewport.update)
        self._smooth_check = QtWidgets.QCheckBox("Smooth shading")
        self._smooth_check.toggled.connect(self._on_smooth_shading_changed)
        body.addWidget(self._smooth_check)
        body.addWidget(self._help("smooth"))
        # Physics (T19, Design 8): the preset picker and the three
        # numbers it fills. The preset name is never stored — the
        # picker derives it from the numbers on every refresh, so a
        # hand-edit flips it to custom by itself.
        body.addWidget(QtWidgets.QLabel("Preset"))
        self._material_preset_combo = QtWidgets.QComboBox()
        self._material_preset_combo.addItem("Custom", "custom")
        for name, preset in MATERIAL_PRESETS.items():
            self._material_preset_combo.addItem(preset["display"], name)
        self._material_preset_combo.currentIndexChanged.connect(
            self._on_material_preset_changed)
        # Hover-preview (UI12): hovering a preset previews its
        # look on the selected object (see
        # _on_material_preset_highlighted).
        self._material_preset_combo.highlighted.connect(
            self._on_material_preset_highlighted)
        # One hide filter for both popups: either popup closing
        # ends the preview, whichever one opened it.
        self._preview_hide_filter = _PopupHideFilter(
            self, self._clear_material_preview)
        self._texture_combo.view().installEventFilter(
            self._preview_hide_filter)
        self._material_preset_combo.view().installEventFilter(
            self._preview_hide_filter)
        body.addWidget(self._material_preset_combo)
        body.addWidget(self._help("material_preset"))
        self._material_look_check = QtWidgets.QCheckBox("Also set look")
        body.addWidget(self._material_look_check)
        body.addWidget(self._help("material_look"))
        self._material_spins = {}
        for label, key, low, high, step in (
                ("Density", "density", 0.0, 100.0, 0.1),
                ("Friction", "friction", 0.0, 5.0, 0.05),
                ("Restitution", "restitution", 0.0, 1.0, 0.05)):
            body.addWidget(QtWidgets.QLabel(label))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(2)
            spin.setSingleStep(step)
            spin.setMinimumWidth(48)
            spin.valueChanged.connect(
                lambda v, k=key: self._on_material_physics_changed(k, v))
            spin.editingFinished.connect(self._commit_live_edit)
            self._material_spins[key] = spin
            body.addWidget(spin)
            body.addWidget(self._help(key))

    def _build_emitter_section(self, layout):
        # Just-in-time (entry 004): only emitter objects have one.
        # Read-only facts plus a live button into the dedicated
        # editor window (UI2) — the button opens the window, so
        # nothing here is dead.
        section = CollapsibleSection("Emitter")
        self._sections["emitter"] = section
        self._section_conditions["emitter"] = (
            lambda obj: obj is not None
            and obj.get("kind") == "emitter")
        layout.addWidget(section)
        body = section.body_layout
        body.addWidget(self._help("emitter"))
        self._emitter_facts = QtWidgets.QLabel("")
        body.addWidget(self._emitter_facts)
        self._edit_emitter_btn = QtWidgets.QPushButton("Edit emitter")
        self._edit_emitter_btn.clicked.connect(self._open_emitter_editor)
        body.addWidget(self._edit_emitter_btn)

    def _build_physics_section(self, layout):
        # Just-in-time (entry 004): only objects with physics
        # enabled. No physics exists yet (T20), so this shows for
        # nothing today — the section and its condition are the home
        # T20's controls land in. No controls are invented here.
        section = CollapsibleSection("Physics")
        self._sections["physics"] = section
        self._section_conditions["physics"] = _physics_enabled
        layout.addWidget(section)

    def _build_timeline(self):
        bar = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(6, 6, 6, 6)
        self._play_btn = QtWidgets.QPushButton("Play")
        self._play_btn.setToolTip("Play the animation from the current frame.")
        self._play_btn.clicked.connect(self._toggle_play)
        layout.addWidget(self._play_btn)
        self._frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._frame_slider.setToolTip(
            "Scrub through the animation. Drag to move between frames.")
        self._frame_slider.setRange(0, self.scene["animation"]["end_frame"])
        self._frame_slider.valueChanged.connect(self._on_slider_frame)
        layout.addWidget(self._frame_slider, 1)
        self._frame_label = QtWidgets.QLabel("Frame 0 / 48")
        layout.addWidget(self._frame_label)
        self._fps_label = QtWidgets.QLabel("24 fps")
        layout.addWidget(self._fps_label)
        add_key_btn = QtWidgets.QPushButton("Add key")
        add_key_btn.setToolTip("Save the current position, rotation, and "
                               "scale as a keyframe.")
        add_key_btn.clicked.connect(self._add_key)
        layout.addWidget(add_key_btn)
        self._add_key_btn = add_key_btn
        del_key_btn = QtWidgets.QPushButton("Delete key")
        del_key_btn.setToolTip("Remove the keyframe at the current frame.")
        del_key_btn.clicked.connect(self._delete_key)
        layout.addWidget(del_key_btn)
        self._del_key_btn = del_key_btn
        end_spin = QtWidgets.QSpinBox()
        end_spin.setToolTip("The last frame of the animation.")
        end_spin.setRange(1, 10000)
        end_spin.setValue(self.scene["animation"]["end_frame"])
        end_spin.setPrefix("End ")
        end_spin.valueChanged.connect(self._on_end_frame_changed)
        layout.addWidget(end_spin)
        self._end_spin = end_spin
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
                    f"Mesh file not found: {mesh_file}, so object "
                    f"'{name}' was skipped.")
                continue
            try:
                meshfile_mod.load_mesh(path)
            except (OSError, ValueError) as exc:
                warnings.append(
                    f"Mesh file could not be read: {mesh_file} ({exc}), "
                    f"so object '{name}' was skipped.")
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
        self._sync_emitter_editor()

    def _sync_emitter_editor(self):
        # After a scene-level change (undo, redo, new, open,
        # delete): close the editor if its emitter is gone, else
        # reload it from the restored scene.
        editor = self._emitter_editor
        if editor is None:
            return
        obj = (find_object(self.scene, editor.object_id)
               if editor.object_id else None)
        if obj is None or obj.get("kind") != "emitter":
            editor.close()
        else:
            editor.refresh()

    def _refresh_outliner(self):
        # Update items in place (U28): a clear + rebuild resets the
        # scroll position and drops the selection, so an edit in a
        # long list used to yank the view back to the top. Rows are
        # matched by index and replaced only when the object id at
        # that row changed; the scroll value is saved and restored
        # regardless, because setting the current row can scroll the
        # list to the selection.
        self._refreshing = True
        scroll = self._outliner.verticalScrollBar().value()
        objects = self.scene["objects"]
        for i, obj in enumerate(objects):
            text = f"{obj['name']} ({obj['shape']})"
            if i < self._outliner.count():
                item = self._outliner.item(i)
                if item.data(QtCore.Qt.UserRole) != obj["id"]:
                    self._outliner.takeItem(i)
                    item = QtWidgets.QListWidgetItem(text)
                    item.setData(QtCore.Qt.UserRole, obj["id"])
                    self._outliner.insertItem(i, item)
                elif item.text() != text:
                    item.setText(text)
            else:
                item = QtWidgets.QListWidgetItem(text)
                item.setData(QtCore.Qt.UserRole, obj["id"])
                self._outliner.addItem(item)
        while self._outliner.count() > len(objects):
            self._outliner.takeItem(self._outliner.count() - 1)
        target = -1
        for i in range(self._outliner.count()):
            if self._outliner.item(i).data(
                    QtCore.Qt.UserRole) == self.selected_id:
                target = i
                break
        if self._outliner.currentRow() != target:
            self._outliner.setCurrentRow(target)
        self._outliner.verticalScrollBar().setValue(scroll)
        self._refreshing = False

    def _refresh_cameras(self):
        self._refreshing = True
        self._camera_combo.clear()
        self._camera_combo.addItem("Free view")
        for cam in self.scene.get("cameras", []):
            self._camera_combo.addItem(cam["name"])
        self._refreshing = False

    def _refresh_inspector(self):
        # Any scene or selection change invalidates a preview
        # (UI12): the previewed object may be gone, or its stored
        # material may have changed under the preview.
        self._clear_material_preview()
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
        self._material_preset_combo.setEnabled(enabled)
        self._material_look_check.setEnabled(enabled)
        # The Advanced toggle is session state like the look
        # tick: enabled with a selection, never set here.
        self._material_advanced_check.setEnabled(enabled)
        for spin in self._material_spins.values():
            spin.setEnabled(enabled)
        # Smooth shading has no effect on a mesh object (T7): the
        # asset's stored normals are used as-is, so the checkbox is
        # disabled for meshes rather than silently ignored.
        self._smooth_check.setEnabled(
            enabled and (obj is None or obj.get("shape") != "mesh"))
        self._cast_check.setEnabled(enabled)
        self._receive_check.setEnabled(enabled)
        # Just-in-time sections (UI1, entry 004): a section shows
        # only where its condition holds — Emitter for emitter
        # objects, Physics for physics-enabled objects.
        for name, condition in self._section_conditions.items():
            self._sections[name].setVisible(condition(obj))
        if obj is None:
            self._name_edit.setText("")
            # No selection, no knobs (PO2): the rows from the
            # previous selection must not linger.
            self._sync_param_rows({})
            self._refreshing = False
            return
        self._name_edit.setText(obj["name"])
        for group in ("position", "rotation", "scale"):
            for i, axis in enumerate(("x", "y", "z")):
                self._spins[(group, axis)].setValue(obj[group][i])
        mat = obj.get("material", {})
        self._roughness_slider.setValue(int(float(mat.get("roughness", 0.6)) * 100))
        self._metalness_slider.setValue(int(float(mat.get("metalness", 0.0)) * 100))
        # "style" is canonical (PO1); fall back to "texture" for a
        # material built by hand without a style key.
        tex = mat.get("style", mat.get("texture", "none"))
        # The combo shows plain words; the internal name is the item
        # data (UI1), so look the row up by data, not by text.
        idx = self._texture_combo.findData(tex)
        self._texture_combo.setCurrentIndex(max(0, idx))
        # The knobs follow the style (PO2): a selection change,
        # an undo, or an open rebuilds them from the stored
        # material, so they never show another object's values.
        self._sync_param_rows(mat)
        color = mat.get("base_color", "#b0b0b0")
        self._color_swatch.setStyleSheet(f"background: {color};")
        # Physics (T19): the spins show the stored numbers; the
        # picker derives its preset from them (the name is never
        # stored). The look checkbox is a pick-time mode, not scene
        # state — it keeps whatever the user last set it to.
        for key, default in (("density", 1.0), ("friction", 0.5),
                             ("restitution", 0.3)):
            self._material_spins[key].setValue(
                float(mat.get(key, default)))
        self._sync_material_preset(mat)
        smooth = obj.get("smooth_shading",
                         default_smooth_shading(obj.get("shape", "box")))
        self._smooth_check.setChecked(bool(smooth))
        self._cast_check.setChecked(bool(obj.get("cast_shadow", True)))
        self._receive_check.setChecked(bool(obj.get("receive_shadow", True)))
        if obj.get("kind") == "emitter":
            # Emitter facts (UI1): read-only, refreshed with the
            # selection. The values come from the validated T16
            # parameter set, so they are numbers by construction.
            params = obj.get("emitter", {})
            self._emitter_facts.setText(
                f"Preset: {str(params.get('preset', 'custom')).capitalize()}\n"
                f"Rate: {params.get('rate', 0):g} per second\n"
                f"Lifetime: {params.get('lifetime', 0):g} seconds\n"
                f"Shape: {params.get('shape', 'point')}")
        self._refreshing = False

    def _sync_param_rows(self, mat):
        # Rebuild the contextual knob rows for a material's style
        # (PO2). Values come from the merged params (style
        # defaults + stored overrides), converted to the knob's
        # display units: spacing knobs show centimeters (the tile
        # is one metre of material — see textures.py), value
        # knobs show the param itself. The display value is
        # rounded to the spin's decimals, so typing the value
        # shown is a no-op instead of a tiny edit. Simple /
        # Advanced (PO3): the rows shown are the mode's set —
        # only the simple knobs in Simple, every knob in
        # Advanced — so a hidden knob keeps its stored override
        # and shows it again when the mode changes. PO4's layout
        # rows and PO5's wear rows slot in here, after the param
        # knobs, Advanced-only, when their tasks land. The old
        # rows are taken out of the layout and deleted at once,
        # so a style with fewer knobs leaves nothing behind; the
        # spins are built with their values set before the
        # signals are connected, so syncing never fires a
        # change. The deletion is shiboken6.delete, not
        # deleteLater: probed on this PySide6 build,
        # deleteLater() never deletes the widget (it stays
        # parented and visible, and every style switch stacked
        # another generation of rows on top of the last).
        # Immediate deletion is safe here — the old spins are
        # never the signal sender when this runs (the callers
        # are the style pick, the inspector refresh, and the
        # Advanced toggle, none of which is a knob spin).
        while self._param_layout.count():
            item = self._param_layout.takeAt(0)
            if item.widget() is not None:
                shiboken6.delete(item.widget())
        self._param_spins = {}
        style = resolve_material_style(mat)
        params = merged_params(style, mat.get("params"))
        for knob in param_knobs_for(
                style, self._material_advanced_check.isChecked()):
            self._param_layout.addWidget(
                QtWidgets.QLabel(knob["label"]))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(knob["low"], knob["high"])
            spin.setDecimals(knob["decimals"])
            spin.setSingleStep(knob["step"])
            spin.setMinimumWidth(48)
            value = params.get(knob["param"])
            if knob["kind"] == "spacing_cm":
                value = count_to_spacing(value)
            spin.setValue(round(value, knob["decimals"]))
            spin.valueChanged.connect(
                lambda v, k=knob: self._on_material_param_changed(k, v))
            spin.editingFinished.connect(self._commit_live_edit)
            self._param_spins[knob["param"]] = spin
            self._param_layout.addWidget(spin)
            self._param_layout.addWidget(self._help(knob["help"]))
        # Hidden while empty, so a style-less material leaves
        # the inspector exactly as it was before PO2.
        self._param_rows.setVisible(bool(self._param_spins))

    def _on_material_advanced_toggled(self, _checked):
        # Simple / Advanced (PO3): rebuild the knob rows for the
        # current material under the new mode. The rows rebuild
        # from the stored material, which holds every param
        # (PO2's overrides-only model), so toggling never loses
        # a value — a knob hidden in Simple keeps its override
        # and shows it again in Advanced. No undo step, no
        # modified flag: the scene is untouched.
        if self._refreshing:
            return
        obj = find_object(self.scene, self.selected_id) \
            if self.selected_id else None
        self._sync_param_rows(obj.get("material", {}) if obj else {})

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
                f"Over the {SOFT_TRI_BUDGET:,} soft budget. "
                "Still fine, just heavier")
            self._budget_label.setStyleSheet("color: #d8a020;")
        else:
            self._budget_label.setText("")
            self._budget_label.setStyleSheet("")

    def _refresh_frame_stats(self):
        # Live measurements (UI10): the viewport's rolling average,
        # display only. Nothing shows until enough paints are
        # measured — a half-formed number would mislead.
        fps, avg_ms, _n = self._viewport.frame_stats()
        if fps is None:
            self._frame_stats_label.setVisible(False)
            return
        self._frame_stats_label.setText(
            f"{fps:.0f} fps · {avg_ms:.1f} ms")
        self._frame_stats_label.setVisible(True)

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
        # A new edit invalidates everything undone before it (T14).
        self._redo_stack.clear()
        # UI5 (U31): every discrete edit goes through here, so the
        # modified flag lives here too — one choke point, not twenty
        # call sites. Every caller guards against no-ops before
        # pushing, so a push always precedes a real mutation.
        # (_new_scene / _open_scene clear it again after replacing
        # the scene.)
        self.setWindowModified(True)

    def undo(self):
        self._live_edit_snapshot = None
        if not self._undo_stack:
            self.statusBar().showMessage("Nothing to undo")
            return
        # T14: the current scene goes on the redo stack, so redo
        # restores exactly the state this undo leaves behind.
        self._redo_stack.append(scene_copy(self.scene))
        del self._redo_stack[:-50]
        self.scene = self._undo_stack.pop()
        if self.selected_id and not find_object(self.scene, self.selected_id):
            self.selected_id = None
        self._apply_scene_keep_camera()
        # _apply_scene_to_ui re-points the viewport at the restored
        # scene, and set_scene clears the viewport's selection — so
        # re-sync it, or the gizmo vanishes after an undo even though
        # the window still shows the object selected (T14: found by
        # the redo-cleared test, which drags again after an undo).
        self._viewport.selected_id = self.selected_id
        self._viewport.update()
        # UI5 (U31): an undo is a state change, so the scene counts
        # as modified even if it lands back on the saved state.
        self.setWindowModified(True)
        self.statusBar().showMessage("Undone")

    def _apply_scene_keep_camera(self):
        """Re-apply the scene to the UI without moving the camera
        (T14). set_scene re-fits the view — frame() re-centers on
        the scene centroid and resets the orbit angles — which is
        right for a new or opened scene and wrong for undo/redo:
        undoing a drag must not yank the user's view back to the
        default angles, and a moved object shifts the centroid, so
        without this the view jumps a pixel or two on every undo of
        a move. The camera is the viewer's state, not the scene's,
        so it is saved before the re-apply and restored after."""
        vp = self._viewport
        camera = (vp.target, vp.distance, vp.azimuth, vp.elevation)
        self._apply_scene_to_ui()
        vp.target, vp.distance, vp.azimuth, vp.elevation = camera
        vp.update()

    def redo(self):
        """Re-apply the last undone edit (T14). The mirror of undo:
        the current scene goes back on the undo stack, the redo
        stack's top becomes the scene. Added with T14 because drag
        undo is only half the story without it — before T14 there
        was no redo at all."""
        self._live_edit_snapshot = None
        if not self._redo_stack:
            self.statusBar().showMessage("Nothing to redo")
            return
        self._undo_stack.append(scene_copy(self.scene))
        del self._undo_stack[:-50]
        self.scene = self._redo_stack.pop()
        if self.selected_id and not find_object(self.scene, self.selected_id):
            self.selected_id = None
        self._apply_scene_keep_camera()
        # Same re-sync as undo: set_scene clears the viewport's
        # selection, so restore it from the window's.
        self._viewport.selected_id = self.selected_id
        self._viewport.update()
        # UI5 (U31): a redo is a state change, same as an undo.
        self.setWindowModified(True)
        self.statusBar().showMessage("Redone")

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

    def _on_outliner_double_clicked(self, item):
        # Double-clicking an emitter opens its editor (UI2); other
        # objects ignore a double-click. The morph starts from the
        # clicked row. QListWidget's signal carries the item only
        # (no column — that is QTreeWidget's signal).
        obj_id = item.data(QtCore.Qt.UserRole)
        obj = find_object(self.scene, obj_id)
        if obj is None or obj.get("kind") != "emitter":
            return
        rect = self._outliner.visualItemRect(item)
        rect.moveTopLeft(self._outliner.mapToGlobal(rect.topLeft()))
        self._open_emitter_editor(object_id=obj_id, anchor=rect)

    def _open_emitter_editor(self, _checked=False, object_id=None,
                             anchor=None):
        """Open the emitter editor for an emitter (UI2): from the
        inspector's Edit button (the selected emitter, morphing
        from the button) or for a given object id (the outliner
        double-click passes its own anchor)."""
        target = object_id or self.selected_id
        obj = find_object(self.scene, target) if target else None
        if obj is None or obj.get("kind") != "emitter":
            return
        if self._emitter_editor is None:
            self._emitter_editor = EmitterEditorWindow(self)
        if anchor is None:
            btn = self._edit_emitter_btn
            anchor = QtCore.QRect(
                btn.mapToGlobal(QtCore.QPoint(0, 0)), btn.size())
        self._emitter_editor.open_for(target, anchor)

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

    def _add_emitter(self):
        # Fire by default: the most recognizable preset, and the
        # one the particles demo leads with.
        self._push_undo()
        emitter = new_emitter(
            f"Emitter {len(self.scene['objects']) + 1}")
        emitter["emitter"] = particles_mod.apply_preset(None, "fire")
        self.scene["objects"].append(emitter)
        self._select(emitter["id"])
        self._refresh_counts()
        self._viewport.refresh()
        self.statusBar().showMessage(f"Added {emitter['name']}")

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
        # The deleted object may have been the editor's emitter, or
        # its attract target — close or reload the editor.
        self._sync_emitter_editor()

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
            # A gesture that changed nothing pushes no step (T14):
            # a press on a gizmo handle with no movement, or a drag
            # that ends exactly where it started, leaves the undo
            # stack untouched. Same for a slider grabbed and released
            # without moving. The snapshot is a full scene copy, so
            # the comparison covers keys as well as base transforms.
            if self._live_edit_snapshot != self.scene:
                self._undo_stack.append(self._live_edit_snapshot)
                del self._undo_stack[:-50]
                # A new edit invalidates everything undone before it.
                self._redo_stack.clear()
                # UI5 (U31): a live edit that actually changed
                # something marks the scene modified; a no-op
                # gesture does not.
                self.setWindowModified(True)
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

    def _on_texture_changed(self, index):
        # The combo shows plain words; the internal name is the item
        # data (UI1) — that is what the material stores.
        if self._refreshing:
            return
        obj = find_object(self.scene, self.selected_id) if self.selected_id else None
        if obj is None:
            return
        text = self._texture_combo.itemData(index)
        if text is None:
            return
        if obj.get("material", {}).get("texture") != text:
            self._clear_material_preview()
            self._push_undo()
            # "style" is canonical (PO1); "texture" mirrors it for
            # the viewport — the two stay equal.
            mat = obj.setdefault("material", {})
            mat["style"] = text
            mat["texture"] = text
            # A new style starts from its own defaults (PO2):
            # params holds overrides only, so an empty dict IS
            # the new style's defaults. The reset rides in the
            # same undo step as the pick, and the knob rows
            # rebuild for the new style at once.
            mat["params"] = {}
            self._sync_param_rows(mat)
            self._viewport.update()

    def _preview_material(self, overrides):
        """Hover-preview (UI12): show these material values on the
        selected object without touching the scene — the preview
        lives only in the viewport until the popup closes or a
        pick applies. No undo step, no modified flag. Nothing
        happens with no selection."""
        if not self.selected_id:
            return
        self._viewport.set_material_preview(self.selected_id, overrides)

    def _clear_material_preview(self):
        """End a hover-preview (UI12): the object renders from
        its stored material again. Idempotent — safe to call
        from every exit path (popup close, pick, refresh)."""
        self._viewport.clear_material_preview()

    def _on_texture_highlighted(self, index):
        # Hovering a texture previews it on the selected object
        # (UI12, entry 017): see it before you commit. The
        # highlighted signal fires on mouse hover and on arrow-key
        # moves alike, so keyboard browsing previews too.
        if self._refreshing:
            return
        name = self._texture_combo.itemData(index)
        if name is None:
            self._clear_material_preview()
            return
        # The previewed style shows with ITS defaults (PO2): the
        # override pins params to empty, so the stored params of
        # the current style — which may share keys like
        # "strength" — never leak into the preview. Restoring
        # returns the stored params, untouched in the scene.
        self._preview_material({"texture": name, "style": name,
                                "params": {}})

    def _on_material_preset_highlighted(self, index):
        # Hovering a preset previews its look (UI12, entry 017):
        # the roughness and metalness from the visual hint. The
        # physics numbers change no pixel, so the look is what
        # there is to see. Custom has no look: hovering it
        # restores. The preview shows the hint whether or not
        # "Also set look" is ticked — the tick governs what a
        # click applies, not what can be seen (a tick-off click
        # applies the numbers only, so the look snaps back).
        if self._refreshing:
            return
        name = self._material_preset_combo.itemData(index)
        if name is None or name == "custom":
            self._clear_material_preview()
            return
        preset = MATERIAL_PRESETS[name]
        self._preview_material({"roughness": preset["roughness"],
                                "metalness": preset["metalness"]})

    def _sync_material_preset(self, mat):
        # The picker shows the preset the stored numbers match, or
        # Custom. Signals blocked: this is a sync, not a pick.
        name = material_preset_for(mat)
        self._material_preset_combo.blockSignals(True)
        self._material_preset_combo.setCurrentIndex(
            self._material_preset_combo.findData(name))
        self._material_preset_combo.blockSignals(False)

    def _on_material_preset_changed(self, index):
        # Picking a preset fills the three physics numbers (one undo
        # step) and, only with "Also set look" ticked, the roughness
        # and metalness from the visual hint. Picking Custom keeps
        # the values and just renames them — since the name is never
        # stored, the picker re-derives from the numbers at once, so
        # a Custom pick on steel's numbers shows steel again.
        if self._refreshing:
            return
        obj = find_object(self.scene, self.selected_id) if self.selected_id else None
        if obj is None:
            return
        name = self._material_preset_combo.itemData(index)
        if name is None:
            return
        mat = obj.setdefault("material", {})
        if name == "custom":
            self._sync_material_preset(mat)
            return
        preset = MATERIAL_PRESETS[name]
        look = self._material_look_check.isChecked()
        if (mat.get("density") == preset["density"]
                and mat.get("friction") == preset["friction"]
                and mat.get("restitution") == preset["restitution"]
                and (not look
                     or (mat.get("roughness") == preset["roughness"]
                         and mat.get("metalness") == preset["metalness"]))):
            return
        self._clear_material_preview()
        self._push_undo()
        mat["density"] = preset["density"]
        mat["friction"] = preset["friction"]
        mat["restitution"] = preset["restitution"]
        if look:
            mat["roughness"] = preset["roughness"]
            mat["metalness"] = preset["metalness"]
        self._refresh_inspector()
        self._viewport.update()

    def _on_material_physics_changed(self, key, value):
        # A hand-edit (live-edit pattern, like the transform spins):
        # the gesture arms on the first change and commits on
        # editingFinished. The picker re-derives at once, so an edit
        # away from a preset flips it to custom live. No viewport
        # update — the numbers change no pixel until T20 simulates.
        if self._refreshing:
            return
        obj = find_object(self.scene, self.selected_id) if self.selected_id else None
        if obj is None:
            return
        value = round(value, 2)
        mat = obj.setdefault("material", {})
        if mat.get(key) != value:
            self._arm_live_edit()
            mat[key] = value
            self._sync_material_preset(mat)

    def _on_material_param_changed(self, knob, value):
        # A knob edit (PO2, live-edit pattern, like the transform
        # spins): the gesture arms on the first change and commits
        # on editingFinished. The value is converted back to the
        # param's own units — a spacing knob stores the count per
        # tile — and rounded: value knobs to the spin's decimals,
        # counts to 4 places, so the scene file stays tidy. The
        # comparison is against the EFFECTIVE value (defaults
        # merged in), so setting the displayed default on an old
        # scene — whose params dict is empty — is a true no-op:
        # no arm, no re-bake, no undo step. And a value equal to
        # the style's default is not stored at all: params holds
        # overrides only, so a gesture that ends on the default
        # compares equal to its snapshot (T14). The re-render is
        # debounced (see _schedule_param_rebake): a bake costs
        # 100-300 ms, so a drag must not bake on every change.
        if self._refreshing:
            return
        obj = find_object(self.scene, self.selected_id) if self.selected_id else None
        if obj is None:
            return
        if knob["kind"] == "spacing_cm":
            param_value = round(spacing_to_count(value), 4)
        else:
            param_value = round(value, knob["decimals"])
        mat = obj.setdefault("material", {})
        params = mat.setdefault("params", {})
        style = resolve_material_style(mat)
        effective = merged_params(style, params).get(knob["param"])
        if effective != param_value:
            self._arm_live_edit()
            if param_value == style_params(style).get(knob["param"]):
                params.pop(knob["param"], None)
            else:
                params[knob["param"]] = param_value
            self._schedule_param_rebake()

    def _schedule_param_rebake(self):
        # Restart the debounce timer: the texture re-bakes once,
        # 150 ms after the last knob change (see
        # PARAM_REBAKE_MS). Starting an already-running
        # single-shot timer restarts it, so a drag coalesces
        # into one repaint.
        self._param_rebake_timer.start()

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
    def _confirm_unsaved(self):
        """UI5 (U31/U23): ask about unsaved changes. Returns True
        when the caller may proceed (nothing to lose, the user
        saved, or the user chose Discard) and False when it must
        stop (Cancel, or the user cancelled the Save As dialog).
        Cancel is the default and the escape button — the
        destructive choice is never the default."""
        if not self.isWindowModified():
            return True
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Unsaved changes")
        box.setText("This scene has unsaved changes.")
        box.setInformativeText("Save them before continuing?")
        box.setStandardButtons(QtWidgets.QMessageBox.Save
                               | QtWidgets.QMessageBox.Discard
                               | QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        box.setEscapeButton(QtWidgets.QMessageBox.Cancel)
        choice = box.exec()
        if choice == QtWidgets.QMessageBox.Save:
            return self._save_scene()
        return choice == QtWidgets.QMessageBox.Discard

    def _quit(self):
        """UI5 (U31): File > Quit. Asks about unsaved changes
        first; close() itself never prompts (see closeEvent),
        because a programmatic close has no user to answer."""
        if self._confirm_unsaved():
            self.close()

    def _new_scene(self):
        if not self._confirm_unsaved():
            return
        self._push_undo()
        self.scene = new_scene()
        self.scene_path = None
        self.selected_id = None
        self._frame = 0
        self._apply_scene_to_ui()
        # UI5 (U31): a fresh scene has nothing unsaved — clear the
        # flag _push_undo just set.
        self.setWindowModified(False)

    def _open_scene(self):
        if not self._confirm_unsaved():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open scene", "", "Scene files (*.json)")
        if not path:
            return
        # Load before pushing, so a failed load leaves the undo
        # stack and the modified flag exactly as they were.
        try:
            loaded = load_scene(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open failed", str(exc))
            return
        # UI5 (U31): the current scene goes on the undo stack
        # first, so an accidental Open is recoverable with undo.
        self._push_undo()
        self.scene = loaded
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
        # UI5 (U31): what is on screen now is what is on disk.
        self.setWindowModified(False)

    def _save_scene(self):
        """Save to the current path, or Save As when there is none.
        Returns True when the scene was actually written."""
        if self.scene_path:
            save_scene(self.scene, self.scene_path)
            self.statusBar().showMessage("Saved")
            self.setWindowModified(False)
            return True
        return self._save_scene_as()

    def _save_scene_as(self):
        """Save under a new name. Returns True when the scene was
        actually written, False when the user cancelled."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save scene as", "", "Scene files (*.json)")
        if not path:
            return False
        if not path.endswith(".json"):
            path += ".json"
        save_scene(self.scene, path)
        self.scene_path = path
        self.statusBar().showMessage(f"Saved to {os.path.basename(path)}")
        self.setWindowModified(False)
        return True

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
        # UI7 (U16): the busy dot + text for the parse. One event
        # pump lets the dot paint before the blocking parse — the
        # finally clears it on every exit, success or failure.
        self._set_busy("Reading mesh file")
        QtWidgets.QApplication.processEvents()
        try:
            parsed = meshio_mod.parse(path)
        except (OSError, ValueError) as exc:
            # UI6 (U29): the most technical import errors name file
            # internals, so the dialog leads with plain words a
            # person can act on, then the detail.
            QtWidgets.QMessageBox.warning(
                self, "Import failed",
                "Reblend could not read this mesh file.\n\n"
                f"{exc}")
            return
        finally:
            self._clear_busy()
        report = parsed["report"]
        if meshio_mod.is_heavy(report["triangles"]):
            answer = QtWidgets.QMessageBox.question(
                self, "Heavy mesh",
                f"This mesh has {report['triangles']:,} triangles, "
                "so this will be heavy on this machine. "
                "Import it anyway?",
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
                "Nothing to export. The scene is empty")
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
                "Nothing to export. The scene is empty")
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
            # Exports render at full quality (UI11), whatever level
            # the tuner sits at — a render is not interactive load.
            with self._viewport.full_quality():
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
        self._run_frame_export(folder)

    def _run_frame_export(self, folder):
        """The frame-export loop, split from the dialog flow so
        tests can drive it directly (the video export's pattern).
        Renders on the GUI thread with a progress dialog and
        Cancel — frame N of M in the label — pumping events after
        every frame, so the window stays responsive and Cancel
        works. On cancel the frames already written stay, each is
        a complete file, and the status bar says honestly how many
        were kept. Returns the number of frames written."""
        end = self.scene["animation"]["end_frame"]
        total = end + 1
        was_playing = self._playing
        if was_playing:
            self._toggle_play()
        self._set_busy("Exporting frames")
        progress = QtWidgets.QProgressDialog(
            "Exporting frames…", "Cancel", 0, total, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        self._frame_progress = progress
        saved = 0
        # The whole export holds the full-quality override (UI11),
        # so the per-frame grabs nest inside it instead of
        # rebuilding the shadow map once per frame.
        with self._viewport.full_quality():
            try:
                for index, frame in enumerate(range(total)):
                    path = os.path.join(folder, f"frame_{frame:04d}.png")
                    if self.render_frame_to(frame, path):
                        saved += 1
                    progress.setLabelText(
                        f"Exporting frame {index + 1} of {total}")
                    progress.setValue(index + 1)
                    QtWidgets.QApplication.processEvents()
                    if progress.wasCanceled():
                        self.statusBar().showMessage(
                            f"Frame export cancelled. {saved} of {total} "
                            "frames saved.")
                        return saved
            finally:
                self._frame_progress = None
                progress.close()
                self._clear_busy()
        self.statusBar().showMessage(
            f"Exported {saved} of {total} frames to {folder}")
        return saved

    # -- video export (T15) -------------------------------------------------
    def _export_video(self):
        """File > Export video: ffmpeg check, options dialog, save
        dialog, then the export loop. Missing ffmpeg is a plain
        message with the install line — never a crash, never a
        silent failure."""
        ffmpeg = video_mod.find_ffmpeg()
        if ffmpeg is None:
            QtWidgets.QMessageBox.warning(
                self, "ffmpeg is not installed",
                "Video export needs ffmpeg, which is not installed.\n\n"
                "Install it with:\n"
                f"    {video_mod.FFMPEG_INSTALL_LINE}")
            return
        viewport = self._viewport
        dialog = VideoExportDialog(
            self, viewport.width(), viewport.height(),
            self.scene["animation"]["end_frame"],
            self.scene["animation"]["fps"])
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        fmt = dialog.format()
        width, height = dialog.size()
        start, end = dialog.frame_range()
        ext = video_mod.FORMATS[fmt]["extension"]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export video", "",
            f"{fmt.upper()} files (*{ext})")
        if not path:
            return
        if not path.endswith(ext):
            path += ext
        self._run_video_export(ffmpeg, path, width, height, start, end,
                               fmt)

    def _grab_frame_rgb24(self, frame, width, height):
        """Render one frame off the viewport and return packed RGB24
        bytes at (width, height). Same suppression as
        render_frame_to: the gizmo never appears in an export.
        QImage rows can carry padding (bytesPerLine wider than
        width*3), so the rows are packed explicitly."""
        self.set_frame(frame)
        self._viewport.set_gizmo_suppressed(True)
        try:
            # Exports render at full quality (UI11), whatever level
            # the tuner sits at — a render is not interactive load.
            with self._viewport.full_quality():
                self._viewport.update()
                QtWidgets.QApplication.processEvents()
                QtCore.QCoreApplication.sendPostedEvents()
                image = self._viewport.grabFramebuffer()
            if image.width() != width or image.height() != height:
                image = image.scaled(
                    width, height, QtCore.Qt.IgnoreAspectRatio,
                    QtCore.Qt.SmoothTransformation)
            rgb = image.convertToFormat(QtGui.QImage.Format_RGB888)
            stride = rgb.bytesPerLine()
            row_bytes = width * 3
            raw = bytes(rgb.constBits())
            if stride == row_bytes:
                return raw
            return b"".join(raw[y * stride:y * stride + row_bytes]
                            for y in range(height))
        finally:
            self._viewport.set_gizmo_suppressed(False)

    def _run_video_export(self, ffmpeg, path, width, height, start, end,
                          fmt):
        """The export loop, split from the dialog flow so tests can
        drive it directly with a fake ffmpeg. Renders on the GUI
        thread (the OpenGL context lives there) and pumps events
        after every frame, so the window stays responsive and Cancel
        works. Returns True on a finished encode, False on cancel
        or failure. A cancelled or failed export leaves no file.
        UI7: the busy dot + text show for the whole export; a
        finished export ends with the one-shot done moment, a
        cancelled or failed one clears the dot with no moment."""
        fps = self.scene["animation"]["fps"]
        total = end - start + 1
        was_playing = self._playing
        if was_playing:
            self._toggle_play()
        exporter = video_mod.VideoExporter(ffmpeg, path, width, height,
                                           fps, fmt)
        if not exporter.start():
            QtWidgets.QMessageBox.warning(
                self, "Video export failed",
                f"Could not start ffmpeg: {exporter.spawn_error()}")
            return False
        # UI7 (U16): the busy dot + text for the whole export. The
        # outer finally clears them on every exit except success,
        # where the done moment takes over instead.
        self._set_busy("Exporting video")
        moment = False
        try:
            progress = QtWidgets.QProgressDialog(
                "Exporting video…", "Cancel", 0, total, self)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            self._video_progress = progress
            failed = None
            # The whole export holds the full-quality override
            # (UI11), so the per-frame grabs nest inside it instead
            # of rebuilding the shadow map once per frame.
            with self._viewport.full_quality():
                try:
                    for index, frame in enumerate(range(start, end + 1)):
                        rgb = self._grab_frame_rgb24(frame, width, height)
                        if not exporter.write_frame(rgb):
                            failed = "ffmpeg stopped accepting frames"
                            break
                        progress.setValue(index + 1)
                        QtWidgets.QApplication.processEvents()
                        if progress.wasCanceled():
                            exporter.cancel()
                            self.statusBar().showMessage(
                                "Video export cancelled")
                            return False
                finally:
                    self._video_progress = None
                    progress.close()
            if failed is not None:
                _ok, detail = exporter.finish()
                exporter.cancel()
                message = failed
                if detail:
                    message += f"\n\n{detail}"
                QtWidgets.QMessageBox.warning(self, "Video export failed",
                                              message)
                return False
            ok, detail = exporter.finish()
            if not ok:
                exporter.cancel()
                message = "ffmpeg failed to write the video."
                if detail:
                    message += f"\n\n{detail}"
                QtWidgets.QMessageBox.warning(self, "Video export failed",
                                              message)
                return False
            # UI7 (entry 008): the one-shot completion moment. The
            # flag flips after the call, so a failure inside it
            # still falls through to the finally's clear.
            self._show_video_done_moment()
            moment = True
            self.statusBar().showMessage(
                f"Exported video to {os.path.basename(path)} "
                f"({total} frames)")
            return True
        finally:
            if not moment:
                self._clear_busy()

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
        # UI5 (U31): the one edit path that does not go through the
        # undo choke points (end-frame changes are not undoable),
        # so the flag is set here directly. The End spin is never
        # set programmatically, so this fires only on a real edit.
        self.setWindowModified(True)
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
