"""Reblend's visual identity (UI8, U15 + Q4): one stylesheet, one
accent, one press response, one reduced-motion rule.

The palette is measured from the provisional cube icon
(assets/reblend.png), not invented: warm off-white, blue-grey, and
two slates. The look is cream on slate — dark surfaces (3D editors
are dark by convention) with a warm cream accent, so the app feels
like itself (Q4).

This module is Qt-free at import time (the PySide6 import is
guarded): the palette, the stylesheet string, and reduced-motion
detection all work on a bare system without PySide6. Only
install() needs Qt, and it is called by main() — never by a test
that constructs a window directly, so a test that asserts styled
appearance must call install() itself first, the same way main()
does.

One exception to "stylesheet only": the spin/combo arrows. With a
stylesheet active the native arrows are not drawn at all (probed),
so install() generates two tiny chevron PNGs into the user's
cache dir and the stylesheet references them by path. No image
files ship with the app.
"""

from __future__ import annotations

import os
import subprocess
import weakref
from string import Template

try:
    from PySide6 import QtCore as _QtCore
    from PySide6 import QtWidgets as _QtWidgets
    import shiboken6 as _shiboken
except ImportError:  # bare system: no Qt, no filter class
    _QtCore = None
    _QtWidgets = None
    _shiboken = None

# Palette — measured from assets/reblend.png (the provisional cube):
#   #e8e4da warm off-white (the cube's lit face)
#   #8a9aa8 blue-grey (the cube's shaded face)
#   #44525e dark slate (the cube's dark face)
#   #2b333b darker slate (the cube's edge shading)
ACCENT = "#e8e4da"   # the one accent: focus ring, selection, handles
TEXT = "#e8e4da"     # primary text (the accent color — cream on slate)
MUTED = "#8a9aa8"    # secondary text, help lines, hover borders
SURFACE = "#2b333b"  # window + panel background
RAISED = "#44525e"   # buttons, raised controls, selections

# The activity dot (UI7) keeps its amber-busy / green-done semantics;
# the green is brightened from #4e9a51 so it reads on the dark
# surface. The amber was already warm enough to sit with the cream.
BUSY_DOT_COLOR = "#e8a33d"
DONE_DOT_COLOR = "#5da85f"

# Entry 001: a button press dips to 0.97 over 150 ms.
PRESS_DIP_MS = 150
PRESS_DIP_SCALE = 0.97

# Reduced motion: REBLEND_REDUCED_MOTION=1 forces it on (the test
# hook); unset, the desktop's animation setting is read instead.
REDUCED_MOTION_ENV_VAR = "REBLEND_REDUCED_MOTION"

_STYLESHEET_TEMPLATE = Template("""
QWidget {
    background-color: $SURFACE;
    color: $TEXT;
}
QWidget:disabled {
    color: $MUTED;
}
QMainWindow, QDialog {
    background-color: $SURFACE;
}
QMenuBar {
    background-color: $SURFACE;
    color: $TEXT;
}
QMenuBar::item:selected {
    background-color: $RAISED;
}
QMenu {
    background-color: $SURFACE;
    color: $TEXT;
    border: 1px solid $RAISED;
}
QMenu::item:selected {
    background-color: $RAISED;
}
QPushButton {
    background-color: $RAISED;
    color: $TEXT;
    border: 1px solid $SURFACE;
    border-bottom: 2px solid $SURFACE;
    border-radius: 3px;
    padding: 4px 10px;
}
QPushButton:hover {
    border: 1px solid $MUTED;
    border-bottom: 2px solid $MUTED;
}
QPushButton:pressed {
    background-color: $SURFACE;
    border-bottom: 1px solid $SURFACE;
}
QPushButton:checked {
    background-color: $SURFACE;
    border: 1px solid $ACCENT;
}
QPushButton:focus {
    border: 2px solid $ACCENT;
}
QPushButton:disabled {
    background-color: $SURFACE;
}
QPushButton:flat {
    background-color: transparent;
    border: none;
}
QPushButton:flat:hover {
    background-color: $RAISED;
}
QPushButton:flat:focus {
    border: 2px solid $ACCENT;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: $SURFACE;
    color: $TEXT;
    border: 1px solid $RAISED;
    border-radius: 3px;
    padding: 2px 4px;
    selection-background-color: $RAISED;
    selection-color: $TEXT;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 2px solid $ACCENT;
}
/* The spin up/down buttons, the combo drop-down, and their arrows
   are NOT in this template: they live in _ARROW_RULES_TEMPLATE
   and join the sheet in build_stylesheet(), because the arrow
   image paths are only known once install() has generated the
   chevron files. */
QComboBox QAbstractItemView {
    background-color: $SURFACE;
    color: $TEXT;
    selection-background-color: $RAISED;
    selection-color: $TEXT;
    border: 1px solid $RAISED;
}
QCheckBox::indicator {
    width: 13px;
    height: 13px;
    border: 1px solid $MUTED;
    background-color: $SURFACE;
    border-radius: 2px;
}
QCheckBox::indicator:checked {
    background-color: $ACCENT;
}
QCheckBox::indicator:focus {
    border: 2px solid $ACCENT;
}
QSlider::groove:horizontal {
    height: 4px;
    background-color: $RAISED;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    background-color: $ACCENT;
    border-radius: 7px;
    margin: -5px 0;
}
QSlider::groove:vertical {
    width: 4px;
    background-color: $RAISED;
    border-radius: 2px;
}
QSlider::handle:vertical {
    width: 14px;
    height: 14px;
    background-color: $ACCENT;
    border-radius: 7px;
    margin: 0 -5px;
}
QSlider:focus {
    border: 2px solid $ACCENT;
}
QListWidget {
    background-color: $SURFACE;
    border: 1px solid $RAISED;
}
QListWidget::item:selected {
    background-color: $RAISED;
    color: $TEXT;
}
QListWidget:focus {
    border: 2px solid $ACCENT;
}
QScrollArea {
    border: none;
}
QScrollBar:vertical {
    background-color: $SURFACE;
    width: 12px;
    border: none;
}
QScrollBar::handle:vertical {
    background-color: $RAISED;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
QScrollBar:horizontal {
    background-color: $SURFACE;
    height: 12px;
    border: none;
}
QScrollBar::handle:horizontal {
    background-color: $RAISED;
    border-radius: 5px;
    min-width: 20px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}
QSplitter::handle {
    background-color: $RAISED;
}
QSplitter::handle:horizontal {
    width: 4px;
}
QSplitter::handle:vertical {
    height: 4px;
}
QStatusBar {
    background-color: $SURFACE;
    color: $MUTED;
}
QStatusBar::item {
    border: none;
}
QToolTip {
    background-color: $RAISED;
    color: $TEXT;
    border: 1px solid $SURFACE;
}
QProgressBar {
    background-color: $SURFACE;
    border: 1px solid $RAISED;
    border-radius: 3px;
    text-align: center;
    color: $TEXT;
}
QProgressBar::chunk {
    background-color: $MUTED;
    border-radius: 2px;
}
QOpenGLWidget {
    background: transparent;
    border: none;
}
""")

STYLESHEET = _STYLESHEET_TEMPLATE.substitute(
    ACCENT=ACCENT, TEXT=TEXT, MUTED=MUTED,
    SURFACE=SURFACE, RAISED=RAISED)

# -- spin/combo arrows --------------------------------------------------
# With a stylesheet active, the native spin/combo arrows are not
# drawn at all (probed: a logging proxy style saw no Spin/Arrow
# primitive, and the buttons render as flat inset rectangles).
# The standard fix — the one every QSS theme uses — is image files
# on the arrow sub-controls. The chevrons are generated here at
# runtime (stdlib only, no shipped image files) into the user's
# cache dir, and build_stylesheet() references them by absolute
# path. If the files cannot be made, the arrow rules are omitted
# and the buttons simply show no glyphs (the degraded state is a
# missing affordance, never a crash).

_ARROW_W = 12
_ARROW_H = 8
_ARROW_RGB = (0xE8, 0xE4, 0xDA)   # ACCENT: cream chevrons
ARROW_DIR_ENV_VAR = "REBLEND_ARROW_DIR"


def _png_bytes(width, height, rows):
    """Encode RGBA rows as a PNG. Stdlib only (struct + zlib)."""
    import struct
    import zlib
    raw = b"".join(
        b"\x00" + b"".join(bytes(px) for px in row) for row in rows)

    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _chevron_png(up):
    """A 12x8 cream chevron on transparency: a filled triangle,
    apex up (up=True) or down."""
    rows = []
    for y in range(_ARROW_H):
        row_y = y if up else _ARROW_H - 1 - y
        half = 6.0 * (row_y + 1) / _ARROW_H
        row = []
        for x in range(_ARROW_W):
            if abs(x - 5.5) < half:
                row.append(_ARROW_RGB + (255,))
            else:
                row.append((0, 0, 0, 0))
        rows.append(row)
    return _png_bytes(_ARROW_W, _ARROW_H, rows)


def _arrow_paths():
    """Ensure the chevron files exist; return (up_path, down_path)
    as plain strings, or (None, None) when they cannot be made.
    The files live in the user's cache dir (Reblend/arrows/),
    or in $REBLEND_ARROW_DIR when set (the test hook, so tests
    never touch the real cache)."""
    if _QtCore is None:
        return None, None
    override = os.environ.get(ARROW_DIR_ENV_VAR)
    if override:
        folder = override
    else:
        cache = _QtCore.QStandardPaths.writableLocation(
            _QtCore.QStandardPaths.CacheLocation)
        if not cache:
            return None, None
        folder = os.path.join(cache, "Reblend", "arrows")
    try:
        os.makedirs(folder, exist_ok=True)
        up = os.path.join(folder, "up.png")
        down = os.path.join(folder, "down.png")
        for path, is_up in ((up, True), (down, False)):
            data = _chevron_png(is_up)
            current = None
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    current = fh.read()
            if current != data:
                with open(path, "wb") as fh:
                    fh.write(data)
        return up, down
    except OSError:
        return None, None


_ARROW_RULES_TEMPLATE = Template("""
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: $RAISED;
    border: none;
    width: 16px;
}
QSpinBox::up-button:disabled, QSpinBox::down-button:disabled,
QDoubleSpinBox::up-button:disabled,
QDoubleSpinBox::down-button:disabled {
    background-color: $SURFACE;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: url("$ARROW_UP");
    width: 12px;
    height: 8px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: url("$ARROW_DOWN");
    width: 12px;
    height: 8px;
}
QSpinBox::up-arrow:disabled, QSpinBox::down-arrow:disabled,
QDoubleSpinBox::up-arrow:disabled,
QDoubleSpinBox::down-arrow:disabled {
    image: none;
}
QComboBox::drop-down {
    border: none;
    background-color: $RAISED;
    width: 18px;
}
QComboBox::drop-down:disabled {
    background-color: $SURFACE;
}
QComboBox::down-arrow {
    image: url("$ARROW_DOWN");
    width: 12px;
    height: 8px;
}
QComboBox::down-arrow:disabled {
    image: none;
}
""")


def build_stylesheet(arrow_up=None, arrow_down=None):
    """The full app stylesheet: the base sheet plus the spin/combo
    arrow rules when the chevron paths are given. Without paths
    the arrow rules are omitted entirely — a stylesheet with a
    dangling url() would be worse than none."""
    if not arrow_up or not arrow_down:
        return STYLESHEET
    return STYLESHEET + _ARROW_RULES_TEMPLATE.substitute(
        RAISED=RAISED, SURFACE=SURFACE,
        ARROW_UP=arrow_up, ARROW_DOWN=arrow_down)

# -- reduced motion ---------------------------------------------------

_override = None        # None = no override; True/False = forced
_desktop_cache = None   # None = the desktop setting not probed yet


def set_reduced_motion_override(value):
    """Force reduced motion on (True) or off (False) for tests and
    verify scripts. Pass None to clear the override."""
    global _override
    _override = None if value is None else bool(value)


def clear_reduced_motion_override():
    """Drop the forced value; detection reads the environment and
    the desktop again."""
    global _override
    _override = None


def _desktop_reduced_motion():
    """The desktop's animation setting: GNOME and Cinnamon both
    expose an enable-animations key, and false on either means the
    user asked for reduced motion. No gsettings, no schema, or an
    error all mean motion stays on — the setting must be readable
    to turn anything off. Probed once and cached (a subprocess
    spawn per press would be silly)."""
    global _desktop_cache
    if _desktop_cache is not None:
        return _desktop_cache
    result = False
    for schema in ("org.gnome.desktop.interface",
                   "org.cinnamon.desktop.interface"):
        try:
            out = subprocess.run(
                ["gsettings", "get", schema, "enable-animations"],
                capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0 and out.stdout.strip().lower() == "false":
            result = True
            break
    _desktop_cache = result
    return result


def reduced_motion():
    """True when animations must not run (state changes still
    happen). The override wins, then the environment variable,
    then the desktop setting."""
    if _override is not None:
        return _override
    env = os.environ.get(REDUCED_MOTION_ENV_VAR)
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return _desktop_reduced_motion()


# -- press dip (entry 001) --------------------------------------------

def _scaled_rect(rect, scale):
    """The rect scaled about its own centre (a 0.97 dip keeps the
    button's centre where it was)."""
    cx, cy = rect.center().x(), rect.center().y()
    width = max(1, round(rect.width() * scale))
    height = max(1, round(rect.height() * scale))
    return _QtCore.QRect(cx - width // 2, cy - height // 2,
                         width, height)


if _QtCore is not None:

    class _PressDipFilter(_QtCore.QObject):
        """Entry 001, app-wide: every push button dips to 0.97 over
        150 ms while pressed and returns over 150 ms on release.
        The dip animates the button's geometry directly — a layout
        only re-asserts geometry when it activates, and a press
        never activates one, so the animation runs unopposed and
        the layout never shifts (cheap motion only: transform, no
        layout animation). A press mid-flight retargets from the
        current geometry, so fast clicking never strands a button
        small. With reduced motion on, no animation runs at all —
        the :pressed stylesheet state still shows, so the state
        change stays. The filter never consumes the event: the
        button still gets its click.

        animations maps id(button) to (weakref, animation,
        original rect) — the weakref guards against a button
        deleted mid-dip (its id could be reused by a new button).
        The entry lives while the button is pressed: the dip's
        finish does NOT forget it (a press held past 150 ms would
        otherwise find nothing to return on release), and only
        the return animation's finish forgets it. The original
        rect is the button's layout geometry at the first press,
        kept across re-presses, so a dip that starts mid-return
        still ends at 0.97 of the layout size.
        """

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("reblend-press-dip")
            self.animations = {}

        def eventFilter(self, obj, event):
            if isinstance(obj, _QtWidgets.QPushButton):
                if event.type() == _QtCore.QEvent.MouseButtonPress:
                    self._dip(obj)
                elif event.type() == _QtCore.QEvent.MouseButtonRelease:
                    self._undip(obj)
            return False

        def _animation_for(self, button):
            """The live animation for this button, or None. A dead
            weakref (the button was deleted mid-dip) drops the
            stale entry instead of trusting a reused id."""
            entry = self.animations.get(id(button))
            if entry is None:
                return None
            ref, anim, _original = entry
            if ref() is not button or not _shiboken.isValid(anim):
                del self.animations[id(button)]
                return None
            return anim

        def _forget(self, button):
            self.animations.pop(id(button), None)

        def _dip(self, button):
            if reduced_motion():
                return
            if not button.isVisible() or button.geometry().isEmpty():
                return
            old = self._animation_for(button)
            if old is not None:
                old.stop()
            # The original is the layout geometry at the first
            # press — a re-press mid-return keeps it, so the dip
            # always ends at 0.97 of the layout size, never of a
            # half-returned rect.
            entry = self.animations.get(id(button))
            if entry is not None and entry[0]() is button:
                original = entry[2]
            else:
                original = _QtCore.QRect(button.geometry())
            anim = _QtCore.QPropertyAnimation(button, b"geometry",
                                              button)
            anim.setDuration(PRESS_DIP_MS)
            anim.setEasingCurve(_QtCore.QEasingCurve.OutCubic)
            anim.setStartValue(button.geometry())
            anim.setEndValue(_scaled_rect(original, PRESS_DIP_SCALE))
            # No finished handler: the entry must outlive the dip
            # (a held press finishes the dip before the release),
            # and only the return animation forgets it.
            self.animations[id(button)] = (weakref.ref(button), anim,
                                           original)
            anim.start()

        def _undip(self, button):
            entry = self.animations.get(id(button))
            if entry is None:
                return
            ref, anim, original = entry
            if ref() is not button:
                del self.animations[id(button)]
                return
            if _shiboken.isValid(anim):
                anim.stop()
            back = _QtCore.QPropertyAnimation(button, b"geometry",
                                              button)
            back.setDuration(PRESS_DIP_MS)
            back.setEasingCurve(_QtCore.QEasingCurve.OutCubic)
            back.setStartValue(button.geometry())
            back.setEndValue(original)
            back.finished.connect(lambda: self._forget(button))
            self.animations[id(button)] = (ref, back, original)
            back.start()


def _dark_palette():
    """The QPalette companion to the stylesheet, from the same
    constants. Style-drawn glyphs (spin arrows, the combo arrow,
    check marks) are painted from the palette, not the
    stylesheet — without it they render in the system's
    dark-on-light colors and vanish on the dark surface (probed:
    the spin arrows were invisible until the palette landed).
    The stylesheet stays the one source for the look; the
    palette only recolors what the stylesheet cannot reach.
    (The spin/combo arrows are NOT palette-driven: with a
    stylesheet active the native arrows are not drawn at all —
    probed with a logging proxy style, which saw no
    Spin/Arrow primitive — so the arrows come from generated
    image files, see _arrow_paths and _ARROW_RULES_TEMPLATE.)"""
    from PySide6 import QtGui
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(SURFACE))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor(SURFACE))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(RAISED))
    pal.setColor(QtGui.QPalette.Text, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor(RAISED))
    pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.BrightText, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(RAISED))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(RAISED))
    pal.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(MUTED))
    return pal


def install(app):
    """Apply the look to the app: the one stylesheet, its palette
    companion, plus the press-dip filter. Idempotent per app (a
    second call is a no-op), so tests that build their own
    QApplication each get exactly one filter."""
    if _QtCore is None:
        raise ImportError("style.install needs PySide6")
    if app.property("reblend_style_installed"):
        return
    app.setProperty("reblend_style_installed", True)
    app.setStyleSheet(build_stylesheet(*_arrow_paths()))
    app.setPalette(_dark_palette())
    app.installEventFilter(_PressDipFilter(app))
