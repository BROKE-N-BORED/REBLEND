"""One reusable dial widget (R3): the knob the wheel center, the
material editor's knobs, and later the light/camera numeric params
all share. A painted circular dial with a numeric readout that is
always visible — no dial without a typed path (ui-ux-pro-max's
gesture-alternative rule: every drag needs a visible control and a
keyboard alternative).

Interactions, all pinned by QtDialTests (group C28):
- Drag vertically: up raises the value, down lowers it. The full
  min..max range spans DRAG_RANGE_PX of travel, and the value is
  computed from the press point plus the total travel, so a drag
  is exact and deterministic.
- Scroll: one wheel notch is one step, up raises, down lowers.
- Readout: the current value is painted under the dial at all
  times, in the widget's own text — never hidden, never a tooltip.
- Click (press and release without dragging): an editor opens over
  the readout, pre-filled with the current value and selected, so
  typing replaces it. Enter or focus-out commits (clamped to the
  range); a non-number is rejected and the editor stays open with
  the text selected, so the mistake can be fixed in place.
- Arrow keys: Up/Right raise by the step, Down/Left lower by it.
- Esc during a drag restores the value from before the drag and
  ends the drag. Esc in the editor closes it without committing.
  Esc at rest does nothing, so a parent (the wheel, a dialog) can
  use it for its own dismiss.

The step is the keyboard/wheel increment only: drags and typed
values are continuous and never snap to it (typing 10.3 into a
0.5-step dial must give 10.3, not 10.5). The readout's decimals
follow the step (a 0.25 step reads to two decimals) unless the
caller passes decimals. The widget paints from its own rect, so
it scales to whatever size its parent gives it; the size hint is
a comfortable default. No animation of any kind: the arc moves
with the pointer, which is the dial's job.

Signals: valueChanged(float) on every change (drag moves emit
live, so a host can re-render as the hand moves), and
editingFinished() when a gesture ends — drag release, Esc-cancel,
a wheel notch, an arrow step, or the type-in editor closing
(R2: the wheel commits one undo step per gesture on it). The
plain getter is value().
"""

from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

from . import style as style_mod

# Full min..max over this many pixels of vertical drag. 200 px is
# a comfortable full sweep on a 1366x768 screen without being so
# coarse that small moves jump.
DRAG_RANGE_PX = 200.0
# Movement under this many pixels is a click (opens the editor),
# not a drag.
CLICK_SLOP_PX = 3


class Dial(QtWidgets.QWidget):
    """A painted knob with an always-visible numeric readout.

    minimum/maximum bound the value, step is the arrow-key and
    wheel increment, value is the starting value (minimum when
    omitted), unit is the readout suffix ("" for none, "%" joins
    without a space), decimals fixes the readout's decimal places
    (None derives them from the step).
    """

    valueChanged = QtCore.Signal(float)
    editingFinished = QtCore.Signal()

    def __init__(self, minimum=0.0, maximum=100.0, step=1.0, value=None,
                 unit="", decimals=None, parent=None):
        super().__init__(parent)
        if maximum < minimum:
            raise ValueError(
                f"dial maximum ({maximum}) is below its minimum "
                f"({minimum})")
        if step <= 0:
            raise ValueError(f"dial step must be positive, got {step}")
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._step = float(step)
        self._unit = unit
        self._decimals = decimals
        self._value = self._minimum if value is None else self._clamp(value)
        self._press_pos = None       # press point, or None when idle
        self._press_value = None     # value at press (Esc restores this)
        self._dragging = False
        self._hover = False
        self._editor = None          # the QLineEdit overlay, lazy
        self._edit_cancelled = False
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)

    # -- value ----------------------------------------------------------

    def value(self):
        """The current value (a float, always within range)."""
        return self._value

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum

    def step(self):
        return self._step

    def setValue(self, new_value):
        """Set the value, clamped to the range. Emits valueChanged
        only when the value actually changes. A non-finite value
        (NaN, inf) is ignored — clamping it would be a lie."""
        if not math.isfinite(new_value):
            return
        clamped = self._clamp(new_value)
        if clamped != self._value:
            self._value = clamped
            self.valueChanged.emit(clamped)
            self.update()

    def setRange(self, minimum, maximum):
        if maximum < minimum:
            raise ValueError(
                f"dial maximum ({maximum}) is below its minimum "
                f"({minimum})")
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self.setValue(self._value)   # re-clamps + emits if it moved

    def setStep(self, step):
        if step <= 0:
            raise ValueError(f"dial step must be positive, got {step}")
        self._step = float(step)
        self.update()

    def setUnit(self, unit):
        self._unit = unit
        self.update()

    def setDecimals(self, decimals):
        self._decimals = decimals
        self.update()

    def _clamp(self, new_value):
        return max(self._minimum, min(self._maximum, float(new_value)))

    # -- readout --------------------------------------------------------

    def _decimals_for(self):
        """Decimal places for the readout: the caller's override, or
        as many as the step needs (0.25 -> 2, 0.5 -> 1, 1 -> 0),
        capped at 3."""
        if self._decimals is not None:
            return max(0, int(self._decimals))
        step = self._step
        places = 0
        while places < 3:
            if abs(step - round(step)) < 1e-9:
                break
            step *= 10.0
            places += 1
        return places

    def _number_text(self):
        """The value as a plain number, no unit — what the editor
        pre-fills and what a typed value is parsed against."""
        return f"{self._value:.{self._decimals_for()}f}"

    def readout_text(self):
        """The readout string exactly as painted: the number plus
        the unit ("42 cm", "75%"). The test hook for 'the readout
        always shows the current value'."""
        text = self._number_text()
        if not self._unit:
            return text
        if self._unit == "%":
            return text + "%"
        return text + " " + self._unit

    def is_editing(self):
        """True while the type-in editor is open."""
        return self._editor is not None and self._editor.isVisible()

    # -- geometry -------------------------------------------------------

    def sizeHint(self):
        return QtCore.QSize(72, 92)

    def minimumSizeHint(self):
        return QtCore.QSize(48, 64)

    def _dial_rect(self):
        """The square the dial circle is drawn in: the top of the
        widget, leaving the readout strip at the bottom."""
        readout_h = 20
        side = min(self.width(), self.height() - readout_h) - 8
        side = max(side, 16)
        x = (self.width() - side) // 2
        return QtCore.QRect(x, 4, side, side)

    def _readout_rect(self):
        """The strip the readout text (and the editor) live in."""
        return QtCore.QRect(0, self.height() - 22, self.width(), 20)

    def _fraction(self):
        span = self._maximum - self._minimum
        if span <= 0:
            return 0.0
        return (self._value - self._minimum) / span

    # -- painting -------------------------------------------------------

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self._dial_rect()
        centre = rect.center()
        radius = rect.width() / 2.0

        # Dial face: the raised surface, a touch brighter on hover
        # so the pointer's target reads as live.
        face = QtGui.QColor(style_mod.RAISED)
        if self._hover:
            face = face.lighter(115)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(face)
        painter.drawEllipse(centre, int(radius), int(radius))

        # Track arc: 270 degrees, from 7:30 round clockwise to
        # 4:30, leaving the gap at the bottom. Qt angles are
        # 1/16ths of a degree, 0 at 3 o'clock, positive
        # counter-clockwise — so the start is 135 and the sweep
        # is -270.
        track = QtGui.QColor(style_mod.MUTED)
        pen = QtGui.QPen(track, 3)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        arc_rect = rect.adjusted(5, 5, -5, -5)
        painter.drawArc(arc_rect, 135 * 16, -270 * 16)

        # Value arc: the same sweep, filled up to the value, in
        # the accent cream.
        span = -int(round(270 * 16 * self._fraction()))
        if span:
            pen.setColor(QtGui.QColor(style_mod.ACCENT))
            pen.setWidth(4)
            painter.setPen(pen)
            painter.drawArc(arc_rect, 135 * 16, span)

        # Indicator dot at the current angle, on the track.
        angle = math.radians(135.0 - 270.0 * self._fraction())
        dot_r = radius - 7
        dot = QtCore.QPointF(
            centre.x() + dot_r * math.cos(angle),
            centre.y() - dot_r * math.sin(angle))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(style_mod.ACCENT))
        painter.drawEllipse(dot, 4.0, 4.0)

        # Focus ring: the app's 2 px accent border, as a ring.
        if self.hasFocus():
            pen = QtGui.QPen(QtGui.QColor(style_mod.ACCENT), 2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(centre, int(radius) + 1, int(radius) + 1)

        # Readout: always visible — unless the editor is open over
        # it, in which case the editor is the readout.
        if not self.is_editing():
            painter.setPen(QtGui.QColor(style_mod.TEXT))
            painter.drawText(self._readout_rect(),
                             QtCore.Qt.AlignCenter, self.readout_text())
        painter.end()
        super().paintEvent(event)

    # -- mouse: drag or click-to-type -----------------------------------

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.setFocus(QtCore.Qt.MouseFocusReason)
            self._press_pos = event.position().toPoint()
            self._press_value = self._value
            self._dragging = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press_pos is not None:
            dy = self._press_pos.y() - event.position().y()
            if not self._dragging and abs(dy) > CLICK_SLOP_PX:
                self._dragging = True
            if self._dragging:
                span = self._maximum - self._minimum
                self.setValue(self._press_value
                              + dy / DRAG_RANGE_PX * span)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton \
                and self._press_pos is not None:
            was_drag = self._dragging
            self._press_pos = None
            self._press_value = None
            self._dragging = False
            event.accept()
            if was_drag:
                # The gesture ended: the host commits its undo step.
                self.editingFinished.emit()
            else:
                # A plain click: type the value directly.
                self._open_editor()
            return
        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    # -- wheel: one notch is one step -----------------------------------

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.setValue(self._value + steps * self._step)
            # One notch is one gesture: the host commits.
            self.editingFinished.emit()
            event.accept()
            return
        super().wheelEvent(event)

    # -- keys: arrows adjust, Esc cancels a drag ------------------------

    def keyPressEvent(self, event):
        key = event.key()
        if key in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Right):
            self.setValue(self._value + self._step)
            self.editingFinished.emit()
            event.accept()
        elif key in (QtCore.Qt.Key_Down, QtCore.Qt.Key_Left):
            self.setValue(self._value - self._step)
            self.editingFinished.emit()
            event.accept()
        elif key == QtCore.Qt.Key_Escape and self._dragging:
            # Cancel the drag: the value goes back to the press
            # point, and the drag is over (the release that
            # follows is a no-op, not a click).
            self.setValue(self._press_value)
            self._dragging = False
            self._press_pos = None
            self._press_value = None
            self.editingFinished.emit()
            event.accept()
        else:
            # Esc at rest is not ours — a parent (the wheel, a
            # dialog) may use it to dismiss.
            super().keyPressEvent(event)

    # -- the type-in editor ---------------------------------------------

    def _open_editor(self):
        if self._editor is None:
            editor = QtWidgets.QLineEdit(self)
            editor.returnPressed.connect(
                lambda: self._close_editor(commit=True))
            editor.installEventFilter(self)
            self._editor = editor
        self._editor.setGeometry(self._readout_rect())
        self._editor.setText(self._number_text())
        self._edit_cancelled = False
        self._editor.show()
        self._editor.setFocus(QtCore.Qt.MouseFocusReason)
        self._editor.selectAll()
        self.update()

    def _commit_editor(self, keep_open_on_invalid):
        """Parse the editor text and set the value (clamped).
        Returns True when the editor ended up closed, False when
        it stays open on an invalid value so the mistake can be
        fixed in place. A closed or missing editor is already
        closed (True)."""
        editor = self._editor
        if editor is None or not editor.isVisible():
            return True
        text = editor.text().strip()
        # Tolerate a pasted unit ("42 cm", "75%").
        if self._unit and text.endswith(self._unit):
            text = text[:-len(self._unit)].strip()
        try:
            typed = float(text)
        except ValueError:
            if keep_open_on_invalid:
                editor.selectAll()
                return False
            editor.hide()
            return True
        editor.hide()
        self.setValue(typed)
        return True

    def _close_editor(self, commit):
        """End the edit. On commit, parse and set the value; a
        non-number keeps the editor open with the text selected.
        On cancel (Esc), the editor closes and the value is
        untouched."""
        editor = self._editor
        if editor is None or not editor.isVisible():
            return
        if commit and not self._edit_cancelled:
            if not self._commit_editor(keep_open_on_invalid=True):
                return
        else:
            editor.hide()
        self._edit_cancelled = False
        self.setFocus(QtCore.Qt.OtherFocusReason)
        self.update()
        # The edit ended (committed or cancelled): the gesture is
        # over, so the host commits its undo step.
        self.editingFinished.emit()

    def eventFilter(self, watched, event):
        if watched is self._editor:
            if event.type() == QtCore.QEvent.KeyPress \
                    and event.key() == QtCore.Qt.Key_Escape:
                # Esc in the editor: cancel the edit, no commit.
                # The flag guards the focus-out that follows the
                # hide, which would otherwise commit.
                self._edit_cancelled = True
                self._close_editor(commit=False)
                return True
            if event.type() == QtCore.QEvent.FocusOut:
                # Clicking away commits, like Return does — but
                # an invalid value is abandoned, not kept open:
                # the user has already moved on. Only a visible
                # editor is a real edit: the focus-out that
                # follows _close_editor's hide must not emit a
                # second editingFinished.
                if self._editor is not None and self._editor.isVisible():
                    if self._commit_editor(keep_open_on_invalid=False):
                        self._edit_cancelled = False
                        self.update()
                        self.editingFinished.emit()
        return super().eventFilter(watched, event)
