"""Procedural textures. No image files anywhere.

The pattern functions are pure Python (no Qt) so they can be unit-tested
headless; build_qimage turns a pattern into a small QImage the viewport
binds as an OpenGL texture. Patterns modulate the material's base color:
each pattern returns a brightness factor, and pattern_rgb multiplies the
base color by it. The pattern is the structure (grain, veins, courses);
the color is the material's, so the same style reads differently on a
brown box than on a grey one.

PO1: the library is a registry of styles in families (STYLES), each
style carrying its family, its plain-words display name, its parameter
defaults, and which layouts (PO4) and wear (PO5) apply to it. Adding a
style is data + one pattern function: add an entry to STYLES and a
function to _PATTERNS — nothing else needs to know about it. The
material dict stores the style key (see scene.py); the three pre-PO1
texture names still resolve through LEGACY_TEXTURE_MAP, so old scenes
load with the matching new style and render exactly as before.
"""
from __future__ import annotations

import math

# Families, in combo order: the internal key and the plain-words
# header shown in the texture dropdown (no em-dash, no bare "&").
FAMILIES = (
    ("wood", "Wood"),
    ("stone", "Stone and marble"),
    ("brick", "Brick and concrete"),
    ("metal", "Metal"),
    ("fabric", "Fabric and leather"),
    ("ground", "Ground"),
)

# Layout vocabulary (PO4 implements these; PO1 only declares which
# styles take which). "solid" is the default: the pattern as-is.
LAYOUTS = ("solid", "boards", "parquet", "chevron", "herringbone",
           "strip", "basket_weave", "stretcher", "flemish", "english",
           "slab")

# Wear vocabulary (PO5 implements these; PO1 only declares which
# styles take which).
WEAR_TYPES = ("dirt", "grime", "edge_wear", "rust", "moss",
              "wetness", "cracks")

# The style registry. Each style: family, display name, parameter
# defaults (PO2 exposes these as knobs; pattern_factor merges a
# caller's params over these), and the layouts + wear that apply.
# Parameter meanings are per style — the pattern function documents
# its own. Keep every pattern's factor in roughly [0.6, 1.15]: the
# suite pins the range, and a pattern that clips to black or white
# reads as broken, not as a material.
STYLES = {
    "wood_cathedral": {
        "family": "wood", "display": "Cathedral oak",
        "params": {"strength": 0.3, "rings": 14.0, "wobble": 3.0},
        "layouts": ("solid", "boards", "parquet", "chevron",
                    "herringbone", "strip"),
        "wear": ("dirt", "grime", "edge_wear", "moss", "wetness"),
    },
    "wood_knotty_pine": {
        "family": "wood", "display": "Knotty pine",
        "params": {"strength": 0.35, "rings": 8.0},
        "layouts": ("solid", "boards", "parquet", "chevron",
                    "herringbone", "strip"),
        "wear": ("dirt", "grime", "edge_wear", "moss", "wetness"),
    },
    "wood_burl": {
        "family": "wood", "display": "Burl",
        "params": {"strength": 0.4, "rings": 22.0, "wobble": 8.0},
        "layouts": ("solid", "boards", "strip"),
        "wear": ("dirt", "grime", "edge_wear", "moss", "wetness"),
    },
    "wood_weathered": {
        "family": "wood", "display": "Weathered wood",
        "params": {"strength": 0.35},
        "layouts": ("solid", "boards", "parquet", "chevron",
                    "herringbone", "strip"),
        "wear": ("dirt", "grime", "edge_wear", "moss", "wetness"),
    },
    "stone_granite": {
        "family": "stone", "display": "Granite",
        "params": {"speckle": 0.18, "blotch": 0.08},
        "layouts": ("solid", "slab"),
        "wear": ("dirt", "grime", "edge_wear", "moss", "cracks",
                 "wetness"),
    },
    "stone_limestone": {
        "family": "stone", "display": "Limestone",
        "params": {"strength": 1.0},
        "layouts": ("solid", "slab"),
        "wear": ("dirt", "grime", "edge_wear", "moss", "cracks",
                 "wetness"),
    },
    "stone_marble_carrara": {
        "family": "stone", "display": "Carrara marble",
        "params": {"veins": 0.3, "bands": 3.0, "warp": 5.0},
        "layouts": ("solid", "slab"),
        "wear": ("dirt", "grime", "edge_wear", "moss", "cracks",
                 "wetness"),
    },
    "stone_slate": {
        "family": "stone", "display": "Slate",
        "params": {"layers": 12.0, "strength": 0.5},
        "layouts": ("solid", "slab", "strip"),
        "wear": ("dirt", "grime", "edge_wear", "moss", "cracks",
                 "wetness"),
    },
    "brick_stretcher": {
        "family": "brick", "display": "Stretcher brick",
        "params": {"rows": 8.0, "cols": 4.0, "mortar": 0.06,
                   "variation": 0.15},
        "layouts": ("stretcher", "flemish", "english",
                    "basket_weave", "herringbone"),
        "wear": ("dirt", "grime", "edge_wear", "moss", "cracks",
                 "wetness"),
    },
    "concrete_raw": {
        "family": "brick", "display": "Raw concrete",
        "params": {"strength": 1.0, "pits": 0.06},
        "layouts": ("solid", "slab"),
        "wear": ("dirt", "grime", "edge_wear", "moss", "cracks",
                 "wetness"),
    },
    "concrete_aged": {
        "family": "brick", "display": "Aged concrete",
        "params": {"stains": 0.2, "strength": 0.06},
        "layouts": ("solid", "slab"),
        "wear": ("dirt", "grime", "edge_wear", "moss", "cracks",
                 "wetness"),
    },
    "metal_brushed": {
        "family": "metal", "display": "Brushed metal",
        "params": {"streak": 0.12, "fine": 0.06},
        "layouts": ("solid",),
        "wear": ("dirt", "grime", "edge_wear", "rust", "wetness"),
    },
    "metal_rusted": {
        "family": "metal", "display": "Rusted metal",
        "params": {"rust": 0.35, "patches": 4.0},
        "layouts": ("solid",),
        "wear": ("dirt", "grime", "edge_wear", "rust", "wetness"),
    },
    "metal_galvanised": {
        "family": "metal", "display": "Galvanised metal",
        "params": {"spangle": 6.0, "strength": 1.0},
        "layouts": ("solid",),
        "wear": ("dirt", "grime", "edge_wear", "rust", "wetness"),
    },
    "fabric_canvas": {
        "family": "fabric", "display": "Canvas",
        "params": {"threads": 24.0, "strength": 1.0},
        "layouts": ("solid",),
        "wear": ("dirt", "grime", "wetness"),
    },
    "fabric_wool_loop": {
        "family": "fabric", "display": "Wool loop",
        "params": {"loops": 16.0, "strength": 1.0},
        "layouts": ("solid",),
        "wear": ("dirt", "grime", "wetness"),
    },
    "fabric_leather": {
        "family": "fabric", "display": "Worn leather",
        "params": {"blotch": 0.16, "crease": 0.10},
        "layouts": ("solid",),
        "wear": ("dirt", "grime", "wetness"),
    },
    "ground_lawn": {
        "family": "ground", "display": "Lawn",
        "params": {"blades": 0.15, "stripes": 4.0},
        "layouts": ("solid",),
        "wear": ("dirt", "grime", "moss", "wetness"),
    },
    "ground_soil": {
        "family": "ground", "display": "Soil",
        "params": {"strength": 1.0, "clumps": 6.0},
        "layouts": ("solid",),
        "wear": ("dirt", "grime", "moss", "wetness"),
    },
    "ground_sand": {
        "family": "ground", "display": "Sand",
        "params": {"ripples": 30.0, "strength": 1.0},
        "layouts": ("solid",),
        "wear": ("dirt", "grime", "moss", "wetness"),
    },
}

# Every name the pattern engine accepts: "none" plus every style key.
# The three pre-PO1 names are not listed — they are aliases, not
# styles, and resolve through LEGACY_TEXTURE_MAP.
TEXTURE_NAMES = ("none",) + tuple(STYLES)

# Pre-PO1 texture names and the style each one became. The old
# pattern code moved into the matching style's function unchanged,
# so an old scene renders exactly as before.
LEGACY_TEXTURE_MAP = {
    "wood_grain": "wood_cathedral",
    "brushed_metal": "metal_brushed",
    "worn_leather": "fabric_leather",
}

# Default wood-grain strength (T6). 1.0 is the v0.1 look: the rings
# were too strong and wavy on big flat planes. 0.5 still read as wavy
# stripes on the floor in the screenshot, so the default is 0.3 — the
# grain reads as wood without dominating the surface. The strength
# scales only the deviation from the base brightness, so the mean
# color is unchanged. It is also the wood_cathedral "strength"
# default — the two must stay equal.
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


def resolve_style(name):
    """The style key for any texture name: a style key passes
    through, a pre-PO1 name maps to its style, anything else
    (including None) is "none". Never raises."""
    if name in STYLES:
        return name
    return LEGACY_TEXTURE_MAP.get(name, "none")


def styles_in_family(family):
    """Style keys in one family, in registry order."""
    return [key for key, entry in STYLES.items()
            if entry["family"] == family]


def style_params(style):
    """A copy of a style's parameter defaults (a copy, so a
    caller tuning the values cannot corrupt the registry).
    Unknown styles get an empty dict."""
    entry = STYLES.get(resolve_style(style))
    return dict(entry["params"]) if entry else {}


def merged_params(style, params=None):
    """A style's parameter values with a caller's overrides
    merged over the defaults: unknown keys and non-numbers are
    ignored, so a malformed params dict cannot crash a render.
    The single source of truth for the merge rule — pattern_factor
    and the viewport's texture cache both use it, so the cache key
    and the baked image can never disagree about which values were
    used. Unknown styles get an empty dict."""
    entry = STYLES.get(resolve_style(style))
    if entry is None:
        return {}
    merged = dict(entry["params"])
    if isinstance(params, dict):
        for key, value in params.items():
            if (key in merged and isinstance(value, (int, float))
                    and not isinstance(value, bool)):
                merged[key] = float(value)
    return merged


# The texture tile is one metre of material (PO2): the pattern
# works in (u, v) space, and the tile spans the object's face, so
# a count parameter (rings, rows, threads) is a count per metre
# of material. A knob that speaks centimeters converts through
# these two: spacing_cm = 100 / count, count = 100 / spacing_cm.
# On a face exactly one metre across the spacing is exact; on
# other faces the pattern stretches with the face (PO4's
# real-world projection will make it exact everywhere). The
# clamps keep a zero or negative input from dividing by zero —
# the UI bounds its spins, so they are a backstop, not a path.
TILE_CM = 100.0


def spacing_to_count(spacing_cm):
    """A feature spacing in cm as a count per tile."""
    return TILE_CM / max(float(spacing_cm), 0.1)


def count_to_spacing(count):
    """A count per tile as a feature spacing in cm."""
    return TILE_CM / max(float(count), 0.1)


def _wood_cathedral(u, v, p):
    # The pre-PO1 wood_grain algorithm, unchanged: rings warped by
    # low-frequency noise, plus fine grain along v. The arithmetic
    # must stay identical with the default params — the tavern
    # renders from this pattern, and its shots are pinned.
    rings = math.sin((u * p["rings"]
                      + _smooth_noise(u, v, 3, 7) * p["wobble"])
                     * math.pi * 2.0)
    grain = _smooth_noise(u * 2.0, v * 24.0, 4, 11)
    return 0.88 + p["strength"] * (0.10 * rings + 0.08 * (grain - 0.5))


def _wood_knotty_pine(u, v, p):
    # Wider, softer rings than cathedral oak, plus knots: a dark
    # ring and core around the nearest of four hash-placed centers
    # on a 2x2 grid.
    rings = math.sin((u * p["rings"]
                      + _smooth_noise(u, v, 3, 17) * 2.0)
                     * math.pi * 2.0)
    grain = _smooth_noise(u * 2.0, v * 18.0, 4, 19)
    base = 0.90 + p["strength"] * (0.10 * rings + 0.06 * (grain - 0.5))
    nearest = 2.0
    for i in range(2):
        for j in range(2):
            cx = (i + 0.25 + 0.5 * _hash(i, j, 29)) / 2.0
            cy = (j + 0.25 + 0.5 * _hash(i, j, 31)) / 2.0
            nearest = min(nearest, math.hypot(u - cx, v - cy))
    ring = math.exp(-((nearest - 0.05) / 0.025) ** 2)
    core = math.exp(-(nearest / 0.03) ** 2)
    return base - 0.15 * ring - 0.12 * core


def _wood_burl(u, v, p):
    # Tight, swirling rings: a high ring count with a heavy wobble,
    # plus a swirling blotch underneath.
    wobble = _smooth_noise(u, v, 4, 37) * p["wobble"]
    rings = math.sin((u * p["rings"] + v * p["rings"] * 0.5 + wobble)
                     * math.pi * 2.0)
    swirl = _smooth_noise(u * 3.0, v * 3.0, 6, 41)
    return 0.88 + p["strength"] * (0.12 * rings + 0.10 * (swirl - 0.5))


def _wood_weathered(u, v, p):
    # Grey and streaky: long vertical streaks plus a faint
    # horizontal banding from old boards.
    streak = _smooth_noise(u * 24.0, v * 2.0, 5, 43)
    band = math.sin(v * math.pi * 6.0) * 0.5 + 0.5
    return 0.90 + p["strength"] * (0.14 * (streak - 0.5)
                                   + 0.08 * (band - 0.5))


def _stone_granite(u, v, p):
    # Dense speckle (per-pixel hash) over a soft blotch.
    speck = _hash(int(u * 128), int(v * 128), 47)
    blotch = _smooth_noise(u, v, 4, 53)
    return 0.88 + p["speckle"] * (speck - 0.5) \
        + p["blotch"] * (blotch - 0.5)


def _stone_limestone(u, v, p):
    # Very soft: low-contrast blotches, barely-there fine grain.
    soft = _smooth_noise(u, v, 5, 59)
    fine = _smooth_noise(u, v, 18, 61)
    return 0.93 + p["strength"] * (0.10 * (soft - 0.5)
                                   + 0.06 * (fine - 0.5))


def _stone_marble_carrara(u, v, p):
    # Veins: sine bands warped by noise, darkened into thin lines
    # where the sine crosses zero (the power sharpens them).
    warp = _smooth_noise(u, v, 3, 67)
    s = math.sin((u * p["bands"] + v * p["bands"] * 0.7
                  + warp * p["warp"]) * math.pi)
    vein = (1.0 - abs(s)) ** 8
    soft = _smooth_noise(u, v, 5, 71)
    return 0.96 - p["veins"] * vein + 0.04 * (soft - 0.5)


def _stone_slate(u, v, p):
    # Horizontal layers, warped slightly, with a fine speckle.
    layer = math.sin((v * p["layers"]
                      + _smooth_noise(u, v, 3, 73) * 2.0)
                     * math.pi * 2.0)
    speck = _hash(int(u * 96), int(v * 96), 79)
    return 0.88 + p["strength"] * (0.10 * layer + 0.08 * (speck - 0.5))


def _brick_stretcher(u, v, p):
    # Courses of bricks, every other row offset by half a brick
    # (the stretcher bond); light mortar joints, per-brick tone
    # from a hash of the brick's row and column.
    rows = int(p["rows"])
    cols = int(p["cols"])
    row = int(v * rows)
    offset = 0.5 if row % 2 else 0.0
    col = int(u * cols + offset)
    fx = (u * cols + offset) - col
    fy = v * rows - row
    mortar = p["mortar"]
    if fx < mortar or fx > 1 - mortar or fy < mortar or fy > 1 - mortar:
        return 1.02
    tone = _hash(col, row, 83)
    return 0.86 + p["variation"] * (tone - 0.5)


def _concrete_raw(u, v, p):
    # Fine noise with a soft undertone, plus tiny dark pits
    # (a per-pixel hash past a threshold).
    fine = _smooth_noise(u, v, 24, 89)
    soft = _smooth_noise(u, v, 6, 91)
    pit = _hash(int(u * 96), int(v * 96), 97)
    pits = -p["pits"] if pit > 0.92 else 0.0
    return 0.92 + p["strength"] * (0.08 * (fine - 0.5)
                                   + 0.06 * (soft - 0.5)) + pits


def _concrete_aged(u, v, p):
    # Dark stains: low-frequency noise squared, so most of the
    # surface stays light and the stains gather in patches.
    stain = _smooth_noise(u, v, 3, 101)
    stain = stain * stain
    fine = _smooth_noise(u, v, 20, 103)
    return 0.90 - p["stains"] * stain + p["strength"] * (fine - 0.5)


def _metal_brushed(u, v, p):
    # The pre-PO1 brushed_metal algorithm, unchanged: long streaks
    # along u plus a fine grain.
    streak = _smooth_noise(u * 40.0, v * 2.0, 6, 3)
    fine = _smooth_noise(u, v, 24, 5)
    return 0.90 + p["streak"] * (streak - 0.5) + p["fine"] * (fine - 0.5)


def _metal_rusted(u, v, p):
    # Rust blooms: low-frequency noise cubed, so the rust gathers
    # in rough-edged patches over the brushed streaks underneath.
    patch = _smooth_noise(u, v, int(p["patches"]), 107)
    patch = patch * patch * patch
    streak = _smooth_noise(u * 40.0, v * 2.0, 6, 3)
    return 0.88 - p["rust"] * patch + 0.06 * (streak - 0.5)


def _metal_galvanised(u, v, p):
    # Crystalline spangle: large soft facets from a coarse cell
    # hash, plus a fine grain.
    cells = int(p["spangle"])
    facet = _hash(int(u * cells), int(v * cells), 109)
    fine = _smooth_noise(u, v, 20, 113)
    return 0.90 + p["strength"] * (0.12 * (facet - 0.5)
                                   + 0.06 * (fine - 0.5))


def _fabric_canvas(u, v, p):
    # Woven cross-hatch: a fine sine grid in both directions over
    # a soft undertone.
    weave = (math.sin(u * math.pi * 2.0 * p["threads"])
             * math.sin(v * math.pi * 2.0 * p["threads"]))
    soft = _smooth_noise(u, v, 5, 127)
    return 0.90 + p["strength"] * (0.10 * weave + 0.06 * (soft - 0.5))


def _fabric_wool_loop(u, v, p):
    # Soft loops: medium-frequency noise folded into ridges, so
    # the tops read light and the gaps read dark.
    loops = _smooth_noise(u, v, int(p["loops"]), 131)
    loops = abs(loops * 2.0 - 1.0)
    fine = _smooth_noise(u, v, 24, 137)
    return 0.88 + p["strength"] * (0.12 * (loops - 0.5)
                                   + 0.06 * (fine - 0.5))


def _fabric_leather(u, v, p):
    # The pre-PO1 worn_leather algorithm, unchanged: soft blotches
    # plus a finer crease.
    blotch = _smooth_noise(u, v, 5, 9)
    crease = _smooth_noise(u * 3.0, v * 3.0, 9, 13)
    return 0.86 + p["blotch"] * (blotch - 0.5) \
        + p["crease"] * (crease - 0.5)


def _ground_lawn(u, v, p):
    # Dense blade speckle, soft mow stripes (alternating bands
    # in v), and a gentle undertone.
    blades = _hash(int(u * 128), int(v * 128), 139)
    stripe = 1.0 if int(v * p["stripes"]) % 2 == 0 else -1.0
    soft = _smooth_noise(u, v, 6, 149)
    return (0.88 + p["blades"] * (blades - 0.5) + 0.04 * stripe
            + 0.05 * (soft - 0.5))


def _ground_soil(u, v, p):
    # Dark clumps with a coarse speckle.
    clump = _smooth_noise(u, v, int(p["clumps"]), 151)
    speck = _hash(int(u * 96), int(v * 96), 157)
    return 0.82 + p["strength"] * (0.12 * (clump - 0.5)
                                   + 0.10 * (speck - 0.5))


def _ground_sand(u, v, p):
    # Wind ripples: sine bands warped by noise, plus a fine grain.
    ripple = math.sin((u * p["ripples"]
                       + _smooth_noise(u, v, 3, 163) * 3.0)
                      * math.pi * 2.0)
    fine = _smooth_noise(u, v, 28, 167)
    return 0.92 + p["strength"] * (0.08 * ripple + 0.06 * (fine - 0.5))


# Style key -> pattern function. A new style adds one entry here
# and one entry in STYLES — that is the whole of it.
_PATTERNS = {
    "wood_cathedral": _wood_cathedral,
    "wood_knotty_pine": _wood_knotty_pine,
    "wood_burl": _wood_burl,
    "wood_weathered": _wood_weathered,
    "stone_granite": _stone_granite,
    "stone_limestone": _stone_limestone,
    "stone_marble_carrara": _stone_marble_carrara,
    "stone_slate": _stone_slate,
    "brick_stretcher": _brick_stretcher,
    "concrete_raw": _concrete_raw,
    "concrete_aged": _concrete_aged,
    "metal_brushed": _metal_brushed,
    "metal_rusted": _metal_rusted,
    "metal_galvanised": _metal_galvanised,
    "fabric_canvas": _fabric_canvas,
    "fabric_wool_loop": _fabric_wool_loop,
    "fabric_leather": _fabric_leather,
    "ground_lawn": _ground_lawn,
    "ground_soil": _ground_soil,
    "ground_sand": _ground_sand,
}


def pattern_factor(name, u, v, wood_strength=None, params=None):
    """Brightness factor in roughly [0.6, 1.15] for a pattern at
    (u, v). The name resolves through resolve_style, so the three
    pre-PO1 names still work. params merges over the style's
    defaults (unknown keys and non-numbers are ignored, so a
    malformed params dict cannot crash a render). wood_strength
    is the v0.1-era override: it sets the "strength" parameter
    of wood_cathedral only, and only when given — None means the
    style's own default."""
    style = resolve_style(name)
    if style == "none":
        return 1.0
    merged = merged_params(style, params)
    if wood_strength is not None and style == "wood_cathedral":
        merged["strength"] = float(wood_strength)
    return _PATTERNS[style](u, v, merged)


def pattern_rgb(name, u, v, base_rgb, wood_strength=None, params=None):
    f = pattern_factor(name, u, v, wood_strength, params)
    return tuple(max(0.0, min(1.0, c * f)) for c in base_rgb)


def build_qimage(name, base_rgb, size=256, wood_strength=None,
                 params=None):
    """Render a pattern to a QImage (Qt import kept local so the pure
    pattern functions stay testable without Qt)."""
    from PySide6.QtGui import QImage, qRgb
    img = QImage(size, size, QImage.Format_RGB32)
    for y in range(size):
        v = y / max(1, size - 1)
        for x in range(size):
            u = x / max(1, size - 1)
            r, g, b = pattern_rgb(name, u, v, base_rgb, wood_strength,
                                  params)
            img.setPixel(x, y, qRgb(int(r * 255), int(g * 255), int(b * 255)))
    return img
