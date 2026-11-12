"""Procedural textures. No image files anywhere.

The pattern functions are pure Python (no Qt) so they can be unit-tested
headless; build_qimage turns a pattern into a small QImage the viewport
binds as an OpenGL texture. Patterns modulate the material's base color.
"""
from __future__ import annotations

import math

TEXTURE_NAMES = ("none", "wood_grain", "brushed_metal", "worn_leather")

# Default wood-grain strength (T6). 1.0 is the v0.1 look: the rings
# were too strong and wavy on big flat planes. 0.5 still read as wavy
# stripes on the floor in the screenshot, so the default is 0.3 — the
# grain reads as wood without dominating the surface. The strength
# scales only the deviation from the base brightness, so the mean
# color is unchanged. Per-material tuning is a later item — there is
# no UI for this yet.
WOOD_GRAIN_STRENGTH = 0.3


def _hash(x, y, seed):
    # Deterministic integer hash in [0, 1). No Random objects: build_qimage
    # calls this per pixel, and Random() construction per call was slow.
    h = (x * 374761393 + y * 668265263 + seed * 974634211) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    return h / 4294967295.0


def _smooth_noise(u, v, cells, seed):
    """Cheap value noise in [0, 1]."""
    x, y = u * cells, v * cells
    xi, yi = int(x), int(y)
    xf, yf = x - xi, y - yi
    sx = xf * xf * (3 - 2 * xf)
    sy = yf * yf * (3 - 2 * yf)
    a = _hash(xi, yi, seed)
    b = _hash(xi + 1, yi, seed)
    c = _hash(xi, yi + 1, seed)
    d = _hash(xi + 1, yi + 1, seed)
    return (a * (1 - sx) + b * sx) * (1 - sy) + (c * (1 - sx) + d * sx) * sy


def pattern_factor(name, u, v, wood_strength=WOOD_GRAIN_STRENGTH):
    """Brightness factor in roughly [0.7, 1.15] for a pattern at (u, v).

    wood_strength scales the wood-grain deviation only (1.0 = the old
    v0.1 look); the other patterns ignore it.
    """
    if name == "wood_grain":
        rings = math.sin((u * 14.0 + _smooth_noise(u, v, 3, 7) * 3.0)
                         * math.pi * 2.0)
        grain = _smooth_noise(u * 2.0, v * 24.0, 4, 11)
        return 0.88 + wood_strength * (0.10 * rings + 0.08 * (grain - 0.5))
    if name == "brushed_metal":
        streak = _smooth_noise(u * 40.0, v * 2.0, 6, 3)
        fine = _smooth_noise(u, v, 24, 5)
        return 0.90 + 0.12 * (streak - 0.5) + 0.06 * (fine - 0.5)
    if name == "worn_leather":
        blotch = _smooth_noise(u, v, 5, 9)
        crease = _smooth_noise(u * 3.0, v * 3.0, 9, 13)
        return 0.86 + 0.16 * (blotch - 0.5) + 0.10 * (crease - 0.5)
    return 1.0


def pattern_rgb(name, u, v, base_rgb, wood_strength=WOOD_GRAIN_STRENGTH):
    f = pattern_factor(name, u, v, wood_strength)
    return tuple(max(0.0, min(1.0, c * f)) for c in base_rgb)


def build_qimage(name, base_rgb, size=256,
                 wood_strength=WOOD_GRAIN_STRENGTH):
    """Render a pattern to a QImage (Qt import kept local so the pure
    pattern functions stay testable without Qt)."""
    from PySide6.QtGui import QImage, qRgb
    img = QImage(size, size, QImage.Format_RGB32)
    for y in range(size):
        v = y / max(1, size - 1)
        for x in range(size):
            u = x / max(1, size - 1)
            r, g, b = pattern_rgb(name, u, v, base_rgb, wood_strength)
            img.setPixel(x, y, qRgb(int(r * 255), int(g * 255), int(b * 255)))
    return img
