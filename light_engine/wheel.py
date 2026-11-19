"""The wheel (R2) — the material editor.

Replaces UI9's PieMenu and absorbs R1: the wheel IS the material
editor. It is a large panel docked bottom-centre over the viewport,
not a cursor popup and not a flick menu.

Layout, top to bottom:
  * a header row: breadcrumb (PO7) + minimise button
  * a search box: typing filters the Style page by name
  * a help line: plain-words text for the hovered slice (or the
    parameter loaded in the centre dial)
  * two painted rings: the inner ring holds the 10 category pages
    (all visible at once, no paging); the outer ring holds the
    active page's slices (picks, knobs, toggles)
  * the centre dial: the R3 Dial component, showing the last-touched
    parameter; it persists across page switches
  * a bottom row: Save (PO10's home, disabled until the saved
    library lands) + Close, with a slot a pack can add a button to
    (PO8 Surprise plugs in here when installed — §9)

There is NO slice cap anywhere (the caps rule): the Style page
holds all 20 styles plus None on one ring. A click picks; arrows
navigate the ring; Enter activates; Esc or a click outside closes.
The wheel is dumb about materials, like the pie was: it owns
geometry, paint and input and emits signals; the window owns the
model and the apply logic.

U33: no em-dashes in visible UI text.
"""

from PySide6 import QtCore, QtGui, QtWidgets

from .dial import Dial
from .style import ACCENT, MUTED, RAISED, SURFACE, TEXT

# The 10 category pages, in ring order. Each entry: key, label,
# one-line description shown in the help line on hover.
WHEEL_PAGES = (
    ("material", "Material",
     "Material: roughness, metalness, smooth shading, all settings."),
    ("style", "Style",
     "Style: pick a look. Hover to preview it, click to apply it."),
    ("layout", "Layout",
     "Layout: how the pattern is laid out, and its sizes."),
    ("detail", "Detail",
     "Detail: the style's own knobs, like ring spacing."),
    ("form", "Form",
     "Form: what the stuff is made of, in plain words."),
    ("surface", "Surface",
     "Surface: density, friction and bounce, as exact numbers."),
    ("life", "Life",
     "Life: wear and age, like dirt, grime and rust."),
    ("colour", "Colour",
     "Colour: the base colour, plain chips or a picker."),
    ("depth", "Depth",
     "Depth: how sharp the picture is, against how fast it runs."),
    ("saved", "Saved",
     "Saved: materials you saved, ready to pick again."),
)

# Default size of the open wheel, and the smallest it can shrink to.
WHEEL_DEFAULT_SIZE = (640, 520)
WHEEL_MIN_SIZE = (420, 360)


class WheelSlice:
    """One slice on the outer ring.

    kind is "pick" (click applies), "knob" (click loads the centre
    dial), "toggle" (click flips a bool) or "action" (click runs
    something, e.g. opening the colour picker). payload is the
    window's dispatch data and is opaque to the wheel. value is the
    short readout shown under the label (a knob's value, or On/Off).
    checked draws the accent dot (the current style, layout, preset
    or quality mode). enabled False paints the slice muted and
    ignores clicks — used when nothing is selected.
    """

    def __init__(self, key, label, kind="pick", swatch=None, value=None,
                 checked=False, enabled=True, help_key=None, payload=None):
        self.key = key
        self.label = label
        self.kind = kind
        self.swatch = swatch
        self.value = value
        self.checked = checked
        self.enabled = enabled
        self.help_key = help_key
        self.payload = payload or {}


class WheelCanvas(QtWidgets.QWidget):
    """The two painted rings and the centre dial.

    Owns the ring geometry (derived from its own rect, so the wheel
    resizes cleanly), painting, hit-testing and keyboard navigation.
    Emits slice signals with (ring, index); the Wheel translates
    them to (page, key).
    """

    slice_activated = QtCore.Signal(str, int)  # ring, index
    slice_hovered = QtCore.Signal(str, int)    # ring, index
    slice_hover_cleared = QtCore.Signal()
    type_to_search = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)
        self._pages = list(WHEEL_PAGES)
        self._page = self._pages[0][0]
        self._slices = []
        self._empty_note = None
        self._hover = None  # (ring, index) or None
        self._pressed = None  # (ring, index) or None
        self._centre_widget = None
        self._centre_label = None

    # -- model ---------------------------------------------------

    def set_page(self, key):
        self._page = key
        self._hover = None
        self.update()

    def set_slices(self, slices, empty_note=None):
        # The window rebuilds the model on every dial tick; the
        # hover survives a rebuild when the same slice is still
        # at the same index (the usual case — a value text
        # changed), so a dial drag does not clear the highlight.
        old_hover = self._hover
        old_slices = self._slices
        self._slices = list(slices)
        self._empty_note = empty_note
        if old_hover is not None and old_hover[0] == "outer":
            ring, index = old_hover
            if (index < len(self._slices) and index < len(old_slices)
                    and self._slices[index].key
                    == old_slices[index].key):
                self._hover = old_hover
            else:
                # The hovered slice went away (a search filter,
                # a style pick): the preview ends with it.
                self._hover = None
                self.slice_hover_cleared.emit()
        self.update()

    @property
    def page(self):
        return self._page

    @property
    def slices(self):
        return list(self._slices)

    def set_centre_widget(self, widget, label):
        """The dial and its name label, both children of the canvas,
        positioned over the centre circle on resize."""
        self._centre_widget = widget
        self._centre_label = label
        self._place_centre()

    # -- geometry ------------------------------------------------

    def ring_geometry(self):
        """(centre_x, centre_y, outer_r, inner_r, centre_r) — the
        test hook, and the single source of truth for painting and
        hit-testing."""
        rect = self.rect()
        cx = rect.width() / 2.0
        cy = rect.height() / 2.0
        outer = min(rect.width(), rect.height()) / 2.0 - 4.0
        inner = outer * 0.62
        centre = outer * 0.38
        return cx, cy, outer, inner, centre

    def slice_point(self, ring, index):
        """Centre point of a slice, in canvas coordinates. Test hook
        (the pie's slice_point, kept for the same reason)."""
        cx, cy, outer, inner, centre = self.ring_geometry()
        if ring == "inner":
            count = len(self._pages)
            r_mid = (centre + inner) / 2.0
        else:
            count = max(1, len(self._slices))
            r_mid = (inner + outer) / 2.0
        step = 360.0 / count
        angle = index * step
        rad = angle * 3.141592653589793 / 180.0
        import math
        return QtCore.QPoint(
            int(cx + r_mid * math.sin(rad)),
            int(cy - r_mid * math.cos(rad)))

    def _hit(self, pos):
        """(ring, index) for a canvas point, or None for the centre
        circle and the corners outside the disc."""
        import math
        cx, cy, outer, inner, centre = self.ring_geometry()
        dx = pos.x() - cx
        dy = pos.y() - cy
        r = math.hypot(dx, dy)
        if r > outer or r <= centre:
            return None
        angle = math.degrees(math.atan2(dx, -dy)) % 360.0
        if r <= inner:
            count = len(self._pages)
            step = 360.0 / count
            return ("inner", int((angle + step / 2.0) / step) % count)
        count = len(self._slices)
        if count == 0:
            return None
        step = 360.0 / count
        return ("outer", int((angle + step / 2.0) / step) % count)

    def _place_centre(self):
        if self._centre_widget is None:
            return
        cx, cy, _o, _i, centre = self.ring_geometry()
        label_h = 18
        dial_h = max(60, int(2 * centre - label_h - 10))
        dial_w = int(dial_h * 0.78)
        self._centre_label.setGeometry(
            int(cx - centre), int(cy - centre + 4),
            int(2 * centre), label_h)
        self._centre_widget.setGeometry(
            int(cx - dial_w / 2.0),
            int(cy - centre + label_h + 6),
            dial_w, dial_h)

    def resizeEvent(self, event):
        self._place_centre()
        super().resizeEvent(event)

    # -- paint ---------------------------------------------------

    def _wedge(self, cx, cy, r_inner, r_outer, a0, a1):
        """Annular wedge path. Angles in degrees, clockwise from
        north, a1 > a0."""
        path = QtGui.QPainterPath()
        outer_rect = QtCore.QRectF(cx - r_outer, cy - r_outer,
                                   2 * r_outer, 2 * r_outer)
        inner_rect = QtCore.QRectF(cx - r_inner, cy - r_inner,
                                   2 * r_inner, 2 * r_inner)
        path.arcMoveTo(outer_rect, 90.0 - a0)
        path.arcTo(outer_rect, 90.0 - a0, -(a1 - a0))
        path.arcTo(inner_rect, 90.0 - a1, a1 - a0)
        path.closeSubpath()
        return path

    def paintEvent(self, _event):
        import math
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        cx, cy, outer, inner, centre = self.ring_geometry()

        # Back disc.
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(SURFACE))
        painter.drawEllipse(QtCore.QPointF(cx, cy), outer, outer)

        # Outer ring slices.
        slices = self._slices
        if slices:
            step = 360.0 / len(slices)
            for i, s in enumerate(slices):
                a0 = i * step - step / 2.0 + 1.0
                a1 = i * step + step / 2.0 - 1.0
                path = self._wedge(cx, cy, inner, outer, a0, a1)
                hovered = self._hover == ("outer", i)
                fill = QtGui.QColor(RAISED) if hovered else QtGui.QColor(SURFACE)
                if not s.enabled:
                    fill = QtGui.QColor(SURFACE)
                painter.setBrush(fill)
                painter.setPen(QtGui.QColor(MUTED) if hovered
                               else QtGui.QColor(RAISED))
                painter.drawPath(path)
                # Label + swatch/value, centred on the slice.
                mid = i * step
                rad = math.radians(mid)
                r_mid = (inner + outer) / 2.0
                text_color = QtGui.QColor(TEXT)
                if not s.enabled:
                    text_color.setAlpha(90)
                painter.setPen(text_color)
                font = painter.font()
                font.setPointSize(8)
                painter.setFont(font)
                metrics = QtGui.QFontMetrics(font)
                chord = 2 * r_mid * math.sin(math.radians(step / 2.0))
                if s.swatch is not None:
                    dot_x = cx + r_mid * math.sin(rad)
                    dot_y = cy - r_mid * math.cos(rad) - 8
                    painter.setBrush(QtGui.QColor(s.swatch))
                    painter.setPen(QtCore.Qt.NoPen)
                    painter.drawEllipse(QtCore.QPointF(dot_x, dot_y), 8, 8)
                    label_r = r_mid + 12
                else:
                    label_r = r_mid - 6
                label = metrics.elidedText(s.label, QtCore.Qt.ElideRight,
                                           int(max(20, chord - 6)))
                lx = cx + label_r * math.sin(rad)
                ly = cy - label_r * math.cos(rad)
                text_rect = QtCore.QRectF(lx - chord / 2.0, ly - 8,
                                          chord, 16)
                painter.setPen(text_color)
                painter.drawText(text_rect, QtCore.Qt.AlignCenter, label)
                if s.value is not None:
                    value = metrics.elidedText(
                        s.value, QtCore.Qt.ElideRight, int(max(20, chord - 6)))
                    v_rect = QtCore.QRectF(lx - chord / 2.0, ly + 6,
                                           chord, 16)
                    painter.setPen(QtGui.QColor(MUTED))
                    painter.drawText(v_rect, QtCore.Qt.AlignCenter, value)
                if s.checked and s.enabled:
                    dot_r = outer - 8
                    dx = cx + dot_r * math.sin(rad)
                    dy = cy - dot_r * math.cos(rad)
                    painter.setBrush(QtGui.QColor(ACCENT))
                    painter.setPen(QtCore.Qt.NoPen)
                    painter.drawEllipse(QtCore.QPointF(dx, dy), 3.5, 3.5)
        elif self._empty_note:
            note_rect = QtCore.QRectF(cx - outer * 0.7, cy - 40,
                                      outer * 1.4, 80)
            painter.setPen(QtGui.QColor(MUTED))
            painter.drawText(note_rect,
                             QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap,
                             self._empty_note)

        # Inner ring: the 10 pages.
        step = 360.0 / len(self._pages)
        for i, (key, label, _desc) in enumerate(self._pages):
            a0 = i * step - step / 2.0 + 1.0
            a1 = i * step + step / 2.0 - 1.0
            path = self._wedge(cx, cy, centre, inner, a0, a1)
            current = key == self._page
            hovered = self._hover == ("inner", i)
            if current:
                fill = QtGui.QColor(RAISED)
            elif hovered:
                fill = QtGui.QColor(RAISED)
                fill.setAlpha(140)
            else:
                fill = QtGui.QColor(SURFACE)
            painter.setBrush(fill)
            painter.setPen(QtGui.QColor(ACCENT) if current
                           else QtGui.QColor(RAISED))
            painter.drawPath(path)
            mid = i * step
            rad = math.radians(mid)
            r_mid = (centre + inner) / 2.0
            chord = 2 * r_mid * math.sin(math.radians(step / 2.0))
            font = painter.font()
            font.setPointSize(9)
            font.setBold(current)
            painter.setFont(font)
            metrics = QtGui.QFontMetrics(font)
            text = metrics.elidedText(label, QtCore.Qt.ElideRight,
                                      int(max(24, chord - 6)))
            lx = cx + r_mid * math.sin(rad)
            ly = cy - r_mid * math.cos(rad)
            text_rect = QtCore.QRectF(lx - chord / 2.0, ly - 9, chord, 18)
            painter.setPen(QtGui.QColor(ACCENT if current else TEXT))
            painter.drawText(text_rect, QtCore.Qt.AlignCenter, text)

        # Centre disc behind the dial.
        painter.setPen(QtGui.QColor(RAISED))
        painter.setBrush(QtGui.QColor(RAISED))
        painter.drawEllipse(QtCore.QPointF(cx, cy), centre, centre)

    # -- mouse ---------------------------------------------------

    def _set_hover(self, hit):
        if hit == self._hover:
            return
        self._hover = hit
        if hit is None:
            self.slice_hover_cleared.emit()
        else:
            self.slice_hovered.emit(hit[0], hit[1])
        self.update()

    def mousePressEvent(self, event):
        # Accept the press: it is a wheel interaction, and an
        # unaccepted press propagates to the centre pane — where
        # the wheel's own click-outside filter reads it as a
        # click outside and closes the wheel after every pick.
        if event.button() == QtCore.Qt.LeftButton:
            self._pressed = self._hit(event.position().toPoint())
        event.accept()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._pressed:
            hit = self._hit(event.position().toPoint())
            if hit == self._pressed:
                ring, index = hit
                if ring == "outer" and not self._slices[index].enabled:
                    pass
                else:
                    self.slice_activated.emit(ring, index)
            self._pressed = None
        event.accept()
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        self._set_hover(self._hit(event.position().toPoint()))
        event.accept()
        super().mouseMoveEvent(event)

    def leaveEvent(self, _event):
        self._set_hover(None)

    # -- keyboard ------------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()
        if key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
            step = -1 if key == QtCore.Qt.Key_Left else 1
            if self._hover is None:
                self._set_hover(("outer", 0) if self._slices
                                else ("inner", 0))
            else:
                ring, index = self._hover
                count = (len(self._pages) if ring == "inner"
                         else len(self._slices))
                if count:
                    self._set_hover((ring, (index + step) % count))
            event.accept()
            return
        if key in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
            if self._hover is None:
                self._set_hover(("inner", 0))
            elif self._hover[0] == "outer":
                self._set_hover(("inner", 0))
            elif self._slices:
                self._set_hover(("outer", 0))
            event.accept()
            return
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter,
                   QtCore.Qt.Key_Space):
            if self._hover is not None:
                ring, index = self._hover
                if ring == "outer" and not self._slices[index].enabled:
                    pass
                else:
                    self.slice_activated.emit(ring, index)
            event.accept()
            return
        text = event.text()
        if text and text.isprintable() and not (
                event.modifiers() & QtCore.Qt.ControlModifier):
            self.type_to_search.emit(text)
            event.accept()
            return
        super().keyPressEvent(event)


class Wheel(QtWidgets.QWidget):
    """The docked wheel panel: header, search, help line, rings,
    centre dial, bottom row. Parented to the centre pane and
    positioned bottom-centre over the viewport.

    Signals carry (page, key) for outer-ring slices; page switches
    are handled here and announced with page_changed.
    """

    page_changed = QtCore.Signal(str)
    slice_activated = QtCore.Signal(str, str)   # page, key
    slice_hovered = QtCore.Signal(str, str)     # page, key
    slice_hover_cleared = QtCore.Signal()
    search_changed = QtCore.Signal(str)
    save_requested = QtCore.Signal()
    close_requested = QtCore.Signal()
    # True while a text field inside the wheel has focus — the window
    # disables the pane shortcuts (W/E/R/Q) so typing "wood" in the
    # search box does not switch the gizmo.
    text_focus_changed = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.Widget)
        self.setMinimumSize(*WHEEL_MIN_SIZE)
        self.resize(*WHEEL_DEFAULT_SIZE)
        self._moved = False
        self._minimised = False
        self._drag_offset = None
        self._suspend_outside_close = False
        self._filter_installed = False
        self._text_focus = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(4)

        # Header: breadcrumb + minimise.
        self._header = QtWidgets.QWidget(self)
        header_layout = QtWidgets.QHBoxLayout(self._header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        self._breadcrumb = QtWidgets.QLabel(self._header)
        self._breadcrumb.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse)
        header_layout.addWidget(self._breadcrumb, 1)
        self._minimise_btn = QtWidgets.QPushButton("-", self._header)
        self._minimise_btn.setToolTip("Minimise the wheel to a bar")
        self._minimise_btn.setFixedWidth(28)
        self._minimise_btn.clicked.connect(self.toggle_minimised)
        header_layout.addWidget(self._minimise_btn)
        layout.addWidget(self._header)
        self._header.installEventFilter(self)
        self._breadcrumb.installEventFilter(self)

        # Search.
        self._search = QtWidgets.QLineEdit(self)
        self._search.setPlaceholderText("Type to find a style...")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self.search_changed.emit)
        layout.addWidget(self._search)

        # Help line.
        self._help = QtWidgets.QLabel(self)
        self._help.setWordWrap(True)
        self._help.setStyleSheet("color: %s;" % MUTED)
        self._help.setMinimumHeight(28)
        layout.addWidget(self._help)

        # Rings + centre dial.
        self._canvas = WheelCanvas(self)
        layout.addWidget(self._canvas, 1)
        self.dial = Dial(0.0, 1.0, 0.05, 0.0, parent=self._canvas)
        self._dial_label = QtWidgets.QLabel(self._canvas)
        self._dial_label.setAlignment(QtCore.Qt.AlignCenter)
        self._canvas.set_centre_widget(self.dial, self._dial_label)
        self._canvas.slice_activated.connect(self._on_canvas_activated)
        self._canvas.slice_hovered.connect(self._on_canvas_hovered)
        self._canvas.slice_hover_cleared.connect(
            self.slice_hover_cleared.emit)
        self._canvas.type_to_search.connect(self.focus_search)

        # Bottom row: Save, the pack slot, Close.
        bottom = QtWidgets.QWidget(self)
        bottom_layout = QtWidgets.QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        self._save_btn = QtWidgets.QPushButton("Save", bottom)
        self._save_btn.setEnabled(False)
        self._save_btn.setToolTip(
            "Saved materials arrive with the saved library (PO10)")
        self._save_btn.clicked.connect(self.save_requested.emit)
        bottom_layout.addWidget(self._save_btn)
        # The pack slot: empty now. When the Surprise add-on (PO8)
        # is installed, the pack loader adds its button here via
        # add_bottom_button; the base never shows one on its own.
        self._surprise_slot = QtWidgets.QWidget(bottom)
        self._surprise_slot.setLayout(QtWidgets.QHBoxLayout())
        self._surprise_slot.layout().setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(self._surprise_slot, 1)
        self._close_btn = QtWidgets.QPushButton("Close", bottom)
        self._close_btn.clicked.connect(self.close_requested.emit)
        bottom_layout.addWidget(self._close_btn)
        layout.addWidget(bottom)
        self._bottom = bottom

        # Resize grip, bottom-right, positioned on resize.
        self._grip = QtWidgets.QSizeGrip(self)

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._on_focus_changed)

    # -- public API ----------------------------------------------

    @property
    def canvas(self):
        return self._canvas

    @property
    def page(self):
        return self._canvas.page

    @property
    def minimised(self):
        return self._minimised

    @property
    def search_text(self):
        return self._search.text()

    @property
    def save_button(self):
        return self._save_btn

    @property
    def close_button(self):
        return self._close_btn

    @property
    def surprise_slot(self):
        return self._surprise_slot

    def set_breadcrumb(self, text):
        self._breadcrumb.setText(text)

    def breadcrumb_text(self):
        return self._breadcrumb.text()

    def set_help(self, text):
        self._help.setText(text)

    def help_text(self):
        return self._help.text()

    def set_dial_label(self, text):
        self._dial_label.setText(text)

    def dial_label(self):
        return self._dial_label.text()

    def set_page(self, key):
        self._canvas.set_page(key)

    def set_slices(self, slices, empty_note=None):
        self._canvas.set_slices(slices, empty_note)

    def set_save_enabled(self, enabled):
        """PO10 enables Save when the saved library lands."""
        self._save_btn.setEnabled(enabled)

    def add_bottom_button(self, button):
        """The pack contribution hook (§9): a pack adds a button to
        the bottom row (the Surprise add-on plugs in here)."""
        self._surprise_slot.layout().addWidget(button)

    def focus_search(self, text=""):
        self._search.setFocus()
        if text:
            self._search.insert(text)

    def clear_search(self):
        self._search.clear()

    def toggle_minimised(self):
        self._minimised = not self._minimised
        for widget in (self._search, self._help, self._canvas,
                       self._bottom, self._grip):
            widget.setVisible(not self._minimised)
        self._minimise_btn.setText("+" if self._minimised else "-")
        self._minimise_btn.setToolTip(
            "Restore the wheel" if self._minimised
            else "Minimise the wheel to a bar")
        if self._minimised:
            self._full_size = self.size()
            self.setFixedHeight(self._header.sizeHint().height() + 12)
        else:
            self.setMinimumSize(*WHEEL_MIN_SIZE)
            self.setMaximumHeight(16777215)
            if getattr(self, "_full_size", None) is not None:
                self.resize(self._full_size)
        self.update()

    # -- open / close --------------------------------------------

    def open(self):
        """Show docked bottom-centre over the viewport (or at the
        last moved position), and start watching for clicks
        outside."""
        parent = self.parentWidget()
        if parent is not None:
            pw, ph = parent.width(), parent.height()
            w = min(self.width(), pw - 16)
            h = min(self.height(), ph - 16)
            self.resize(w, h)
            if not self._moved:
                self.move((pw - w) // 2, ph - h - 8)
            else:
                x = min(max(0, self.x()), pw - w)
                y = min(max(0, self.y()), ph - h)
                self.move(x, y)
        self.show()
        self.raise_()
        if not self._filter_installed:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.installEventFilter(self)
                self._filter_installed = True
        self._canvas.setFocus()

    def close_wheel(self):
        if self._filter_installed:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
            self._filter_installed = False
        self.hide()

    def suspend_outside_close(self, suspended):
        """While a modal dialog is up (the colour picker), clicks on
        it must not close the wheel."""
        self._suspend_outside_close = suspended

    # -- events --------------------------------------------------

    def eventFilter(self, watched, event):
        # Drag the header to move the wheel. The press and the
        # move both go through global coordinates: the event's
        # local position is relative to whichever child got it
        # (the header or the breadcrumb), and adding the child's
        # offset by hand is wrong for the breadcrumb (its pos is
        # header-local, not wheel-local) and was missing entirely
        # on the move — the wheel drifted by the header's offset
        # on every drag.
        if watched is self._header or watched is self._breadcrumb:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                if event.button() == QtCore.Qt.LeftButton:
                    self._drag_offset = self.mapFromGlobal(
                        event.globalPosition().toPoint())
                    return True
            if event.type() == QtCore.QEvent.MouseMove:
                if self._drag_offset is not None:
                    parent = self.parentWidget()
                    point = self.mapFromGlobal(
                        event.globalPosition().toPoint())
                    new = self.mapToParent(point) - self._drag_offset
                    if parent is not None:
                        new.setX(min(max(0, new.x()),
                                     parent.width() - self.width()))
                        new.setY(min(max(0, new.y()),
                                     parent.height() - self.height()))
                    self.move(new)
                    self._moved = True
                    return True
            if event.type() == QtCore.QEvent.MouseButtonRelease:
                self._drag_offset = None
                return True
            return False
        # Click outside closes the wheel (the pie's pattern).
        # The test is the press POSITION, not the receiving
        # widget: a press is delivered up the parent chain, so
        # a widget check reads a wheel press as outside once it
        # reaches the centre pane — and with two wheels open a
        # press inside one reads as outside to the other and
        # gets swallowed before it lands. Inside the wheel's
        # rect is inside, whichever widget the event is passing
        # through; outside is outside, and the dismissing click
        # is swallowed so no orbit starts and no selection
        # changes.
        if event.type() == QtCore.QEvent.MouseButtonPress:
            if self.isVisible() and not self._suspend_outside_close:
                app = QtWidgets.QApplication.instance()
                if app is not None and app.activeModalWidget() is not None:
                    return False
                top_left = self.mapToGlobal(QtCore.QPoint(0, 0))
                rect = QtCore.QRect(top_left, self.size())
                if not rect.contains(
                        event.globalPosition().toPoint()):
                    self.close_requested.emit()
                    return True
        return False

    def paintEvent(self, _event):
        # Opaque: the wheel is a workspace over the viewport, not
        # a glass pane — the scene must not show through around
        # the rings.
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QColor(RAISED))
        painter.setBrush(QtGui.QColor(SURFACE))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        self._grip.move(self.width() - self._grip.width(),
                        self.height() - self._grip.height())
        super().resizeEvent(event)

    def _on_canvas_activated(self, ring, index):
        if ring == "inner":
            key = WHEEL_PAGES[index][0]
            self._canvas.set_page(key)
            self.page_changed.emit(key)
            return
        slices = self._canvas.slices
        if 0 <= index < len(slices):
            self.slice_activated.emit(self._canvas.page,
                                      slices[index].key)

    def _on_canvas_hovered(self, ring, index):
        if ring == "inner":
            self.set_help(WHEEL_PAGES[index][2])
            return
        slices = self._canvas.slices
        if 0 <= index < len(slices):
            self.slice_hovered.emit(self._canvas.page,
                                    slices[index].key)

    def _on_focus_changed(self, _old, new):
        focused = False
        if new is not None and self.isAncestorOf(new):
            focused = isinstance(new, QtWidgets.QLineEdit)
        if focused != self._text_focus:
            self._text_focus = focused
            self.text_focus_changed.emit(focused)
