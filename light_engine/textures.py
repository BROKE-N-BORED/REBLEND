"""Procedural textures. No image files anywhere.

The pattern functions are pure Python (no Qt) so they can be unit-tested
headless; build_qimage turns a pattern into a small QImage the viewport
binds as an OpenGL texture. Patterns modulate the material's base color.
"""
from __future__ import annotations

import math

TEXTURE_NAMES = ("none", "wood_grain", "brushed_metal", "worn_leather")


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


def pattern_factor(name, u, v):
    """Brightness factor in roughly [0.7, 1.15] for a pattern at (u, v)."""
    if name == "wood_grain":
        rings = math.sin((u * 14.0 + _smooth_noise(u, v, 3, 7) * 3.0)
                         * math.pi * 2.0)
        grain = _smooth_noise(u * 2.0, v * 24.0, 4, 11)
        return 0.88 + 0.10 * rings + 0.08 * (grain - 0.5)
    if name == "brushed_metal":
        streak = _smooth_noise(u * 40.0, v * 2.0, 6, 3)
        fine = _smooth_noise(u, v, 24, 5)
        return 0.90 + 0.12 * (streak - 0.5) + 0.06 * (fine - 0.5)
    if name == "worn_leather":
        blotch = _smooth_noise(u, v, 5, 9)
        crease = _smooth_noise(u * 3.0, v * 3.0, 9, 13)
        return 0.86 + 0.16 * (blotch - 0.5) + 0.10 * (crease - 0.5)
    return 1.0


def pattern_rgb(name, u, v, base_rgb):
    f = pattern_factor(name, u, v)
    return tuple(max(0.0, min(1.0, c * f)) for c in base_rgb)


def build_qimage(name, base_rgb, size=256):
    """Render a pattern to a QImage (Qt import kept local so the pure
    pattern functions stay testable without Qt)."""
    from PySide6.QtGui import QImage, qRgb
    img = QImage(size, size, QImage.Format_RGB32)
    for y in range(size):
        v = y / max(1, size - 1)
        for x in range(size):
            u = x / max(1, size - 1)
            r, g, b = pattern_rgb(name, u, v, base_rgb)
            img.setPixel(x, y, qRgb(int(r * 255), int(g * 255), int(b * 255)))
    return img
