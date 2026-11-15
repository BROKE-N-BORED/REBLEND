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

# Layout vocabulary (PO4 implements these: the shader transforms in
# shaders.py and the brick bonds in _brick_stretcher below).
# "solid" is the default: the pattern as-is.
LAYOUTS = ("solid", "boards", "parquet", "chevron", "herringbone",
           "strip", "basket_weave", "stretcher", "flemish", "english",
           "slab")

# Plain-words names for the layout picker (no em-dash, no bare "&").
LAYOUT_DISPLAY = {
    "solid": "Solid",
    "boards": "Boards",
    "parquet": "Parquet",
    "chevron": "Chevron",
    "herringbone": "Herringbone",
    "strip": "Strip",
    "basket_weave": "Basket weave",
    "stretcher": "Stretcher bond",
    "flemish": "Flemish bond",
    "english": "English bond",
    "slab": "Slabs",
}

# Default piece sizes in real centimetres, per layout: (width,
# length). Both keys are stored in the material's layout_size dict
# as "board_width" and "board_length" whatever the layout calls
# them on screen (the picker relabels them: parquet's length is
# its block size, a slab's width and length are the slab's sides).
# Boards: a 12 cm wide, 90 cm long plank. Strip: narrow 6 cm
# strips. Parquet: 7.5 cm strips in 30 cm blocks. Chevron and
# herringbone: 9 cm by 54 cm pieces (6:1, the classic ratio).
# Basket weave: 10 cm boards in 30 cm squares. Slabs: 60 cm square.
# Solid and the brick bonds have no entry here: solid has no
# pieces, and brick sizes itself from the brick knobs (brick
# length and course height).
LAYOUT_DEFAULTS = {
    "boards": (12.0, 90.0),
    "strip": (6.0, 60.0),
    "parquet": (7.5, 30.0),
    "chevron": (9.0, 54.0),
    "herringbone": (9.0, 54.0),
    "basket_weave": (10.0, 30.0),
    "slab": (60.0, 60.0),
}

# The shader-side layout transform (shaders.py) works in modes,
# and several layouts share one mode with different sizes:
# 0 solid (no transform — the vUV path, exactly as before),
# 1 boards (rows of boards with staggered joints; strip is narrow
# boards), 2 blocks (squares of parallel pieces; parquet and
# basket weave differ only in their sizes), 3 chevron (the
# staircase lattice in 45-degree-rotated coordinates, so the
# pieces meet at mitred joints), 4 herringbone (the staircase
# lattice straight), 5 slabs (a plain grid), 6 tile (no cell
# transform — the brick bonds are baked into the tile by
# _brick_stretcher, and the shader only tiles it at real-world
# size).
LAYOUT_SHADER_MODES = {
    "solid": 0,
    "boards": 1,
    "strip": 1,
    "parquet": 2,
    "basket_weave": 2,
    "chevron": 3,
    "herringbone": 4,
    "slab": 5,
    "stretcher": 6,
    "flemish": 6,
    "english": 6,
}

# Gap between laid pieces, in centimetres. The shader darkens the
# pattern near each piece edge by this much; 0.5 cm reads as a
# joint at arm's length without eating the piece.
LAYOUT_GAP_CM = 0.5

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


def layouts_for_style(style):
    """The layouts a style takes, in registry order. Unknown
    styles (and "none") take only "solid"."""
    entry = STYLES.get(resolve_style(style))
    return list(entry["layouts"]) if entry else ["solid"]


def default_layout(style):
    """A style's default layout: "solid" when the style takes it,
    else the first layout in its list (brick takes no solid, so
    brick defaults to "stretcher")."""
    layouts = layouts_for_style(style)
    return "solid" if "solid" in layouts else layouts[0]


def resolve_layout(style, layout):
    """The layout to use for a style: the requested one when the
    style takes it, else the style's default. Never raises, never
    returns a layout the style does not take — so a stored layout
    from another style (or a hover preview over a different style)
    can never reach the shader unresolved."""
    layouts = layouts_for_style(style)
    return layout if layout in layouts else default_layout(style)


def merged_layout_size(layout, sizes=None):
    """A layout's piece sizes in cm: (board_width, board_length),
    the layout's defaults with a caller's overrides merged over
    them. Unknown keys and non-numbers are ignored, and a size at
    or below zero falls back to its default — the shader guards
    its own sizes too, but a render must never divide by a stored
    zero. Layouts without sizes (solid, and the brick bonds, which
    size themselves from the brick knobs) return (0.0, 0.0)."""
    defaults = LAYOUT_DEFAULTS.get(layout)
    if defaults is None:
        return (0.0, 0.0)
    width, length = defaults
    if isinstance(sizes, dict):
        value = sizes.get("board_width")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and float(value) > 0.0):
            width = float(value)
        value = sizes.get("board_length")
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and float(value) > 0.0):
            length = float(value)
    return (width, length)


def layout_shader_mode(style, layout):
    """The shader mode for a style's layout (see
    LAYOUT_SHADER_MODES). The layout resolves first, so an invalid
    layout for the style yields the default's mode."""
    return LAYOUT_SHADER_MODES[resolve_layout(style, layout)]


def layout_affects_bake(style):
    """True when the layout changes the baked tile rather than the
    shader transform: only brick, whose bonds rearrange the bricks
    in _brick_stretcher. Every other style's layout is shader-side
    and the bake is layout-independent."""
    return resolve_style(style) == "brick_stretcher"


# The wear set for a style-less material ("none" or unknown): the
# wood set, the most generic — dirt, grime, edge wear, moss, and
# wetness. Rust needs metal and cracks need stone or concrete, so
# neither is in the generic set.
GENERIC_WEAR = ("dirt", "grime", "edge_wear", "moss", "wetness")


def wear_for_style(style):
    """The wear types a style takes, in registry order. Unknown
    styles (and "none") take the generic set — wear overlays any
    material, even a plain colour with no pattern."""
    entry = STYLES.get(resolve_style(style))
    if entry is None:
        return list(GENERIC_WEAR)
    return list(entry["wear"])


def merged_wear(style, wear=None):
    """A style's wear values with a caller's overrides merged over
    the defaults: every wear type the style takes, default 0.0
    (clean, dry, no cracks). Percentages clamp to 0-100; cracks
    (a fissure spacing in cm, 0 meaning no cracks) clamps to
    0-100. Unknown keys, non-numbers, and wear the style does not
    take are dropped — a stored rust value must not follow the
    material onto a style that cannot rust. Never raises."""
    merged = {key: 0.0 for key in wear_for_style(style)}
    if isinstance(wear, dict):
        for key, value in wear.items():
            if (key in merged and isinstance(value, (int, float))
                    and not isinstance(value, bool)):
                merged[key] = max(0.0, min(100.0, float(value)))
    return merged


def wear_uniforms(style, wear=None):
    """The wear values as the shader takes them (see shaders.py):
    (uWearA, uWearB), where uWearA is (dirt, grime, edge wear,
    rust) as 0-1 fractions and uWearB is (moss, wetness, crack
    spacing in cm). The packing is the contract with the fragment
    shader — the viewport calls this, never packs by hand, so the
    two can never disagree about which component is which."""
    merged = merged_wear(style, wear)
    wear_a = tuple(merged.get(key, 0.0) / 100.0
                   for key in ("dirt", "grime", "edge_wear", "rust"))
    wear_b = (merged.get("moss", 0.0) / 100.0,
              merged.get("wetness", 0.0) / 100.0,
              merged.get("cracks", 0.0))
    return wear_a, wear_b


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


def _staircase_piece(x, y, L, w):
    """Which piece of a herringbone staircase lattice (x, y) is in.
    Pieces are L long and w wide, at 90 degrees to each other in a
    zigzag chain; the motif is an H piece [0,L]x[0,w] and a V piece
    [L,L+w]x[0,L], tiled on the lattice b1=(L,L), b2=(w,-w). The
    cell area is 2*L*w, exactly one H+V pair, so the lattice tiles
    the plane with no gaps and no overlaps (proven by a dense-grid
    coverage check in the PO4 prototype: 146,882 points, every
    point in exactly one piece, including points on piece edges).
    Returns (kind, lx, ly, i, j): kind "H" or "V", (lx, ly) the
    position inside the piece (H: lx along the length, ly across
    the width; V: lx across the width, ly along the length), and
    (i, j) the lattice indices, which identify the piece. The
    indices are derived, not looped: for H the row is pinned by
    the y-range, for V the column by the x-range, and the i index
    comes from s=(x+y)/2L (only floor(s)-1 and floor(s) can match,
    so four candidate tests cover the plane). Half-open intervals:
    a point on a shared edge belongs to the piece whose lower edge
    it is on. Returns None only if no candidate matches, which the
    coverage proof says cannot happen — callers still handle it.
    The GLSL version in shaders.py is the same math."""
    s = (x + y) / (2.0 * L)
    base = math.floor(s)
    for i in (base - 1, base):
        cy = y - ((y - i * L) % w)
        j = round((i * L - cy) / w)
        cy = i * L - j * w
        cx = i * L + j * w
        if cx <= x < cx + L and cy <= y < cy + w:
            return ("H", x - cx, y - cy, i, j)
        cx = x - ((x - (i * L + L)) % w)
        j = round((cx - i * L - L) / w)
        cx = i * L + j * w + L
        cy = i * L - j * w
        if cx <= x < cx + w and cy <= y < cy + L:
            return ("V", x - cx, y - cy, i, j)
    return None


def _brick_stretcher(u, v, p, bond="stretcher"):
    # Courses of bricks, light mortar joints, per-brick tone from a
    # hash of the brick's row and column. The bond rearranges the
    # bricks; every bond funnels into the same (col, row, fx, fy)
    # tail so the mortar and tone stay identical across bonds.
    # "stretcher" is the PO1 path, unchanged: every other row offset
    # by half a brick. "flemish": stretcher and header alternate
    # along each course on a 3-half-unit period (stretcher 2 units,
    # header 1), odd rows shifted by 1.5 units so each header
    # centres on the stretcher below it — a 1-unit shift would
    # align the joints every 3 units and defeat the bond. "english":
    # whole courses alternate stretcher and header, the header
    # course offset by half a header (the queen-closer shift) so
    # no header joint aligns with a stretcher joint. "basket_weave":
    # squares of side one brick length, two bricks per square,
    # alternating orientation; the square is rows+1 courses tall
    # when rows is odd so the weave tiles seamlessly. "herringbone":
    # the staircase lattice from _staircase_piece, in tile space.
    # An unknown bond falls back to stretcher. Seam notes: the
    # bonds tile seamlessly on the default 8-row / 4-column grid;
    # flemish needs the column count a multiple of 3, english and
    # herringbone need even rows (and herringbone even columns) —
    # on other grids the metre seam shows a bond irregularity (two
    # adjacent stretchers, or a doubled course), never a gap.
    # In basket weave and herringbone the brick width follows the
    # course height, so on a grid where rows is not twice cols the
    # pieces are rectangles rather than true squares; the default
    # grid (8 rows, 4 columns) is exact.
    rows = int(p["rows"])
    cols = int(p["cols"])
    mortar = p["mortar"]
    if bond == "flemish":
        row = int(v * rows)
        shift = 1.5 if row % 2 else 0.0
        xh = u * cols * 2.0 + shift
        xu = int(xh)
        local = (xu % 3) + (xh - xu)
        if local < 2.0:
            col = (xu // 3) * 2
            fx = local / 2.0
        else:
            col = (xu // 3) * 2 + 1
            fx = local - 2.0
        fy = v * rows - row
    elif bond == "english":
        row = int(v * rows)
        fy = v * rows - row
        if row % 2 == 0:
            col = int(u * cols)
            fx = u * cols - col
        else:
            xh = u * cols * 2.0 + 0.5
            col = int(xh)
            fx = xh - col
    elif bond == "basket_weave":
        sq_rows = rows if rows % 2 == 0 else rows + 1
        sq_col = int(u * cols)
        sq_row = int(v * sq_rows / 2.0)
        su = u * cols - sq_col
        sv = v * sq_rows / 2.0 - sq_row
        if (sq_col + sq_row) % 2 == 0:
            half = 0 if su < 0.5 else 1
            fx = su * 2.0 - half
            fy = sv
            col = sq_col * 2 + half
            row = sq_row
        else:
            half = 0 if sv < 0.5 else 1
            fx = su
            fy = sv * 2.0 - half
            col = sq_col
            row = sq_row * 2 + half
    elif bond == "herringbone":
        piece = _staircase_piece(u * cols, v * rows, 1.0, 0.5)
        if piece is None:
            col, row, fx, fy = 0, 0, 0.5, 0.5
        else:
            kind, lx, ly, i, j = piece
            if kind == "H":
                fx, fy = lx, ly / 0.5
            else:
                fx, fy = lx / 0.5, ly
            col = i
            row = j * 2 + (0 if kind == "H" else 1)
    else:  # stretcher (and any unknown bond)
        row = int(v * rows)
        offset = 0.5 if row % 2 else 0.0
        col = int(u * cols + offset)
        fx = (u * cols + offset) - col
        fy = v * rows - row
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


def pattern_factor(name, u, v, wood_strength=None, params=None,
                   layout=None):
    """Brightness factor in roughly [0.6, 1.15] for a pattern at
    (u, v). The name resolves through resolve_style, so the three
    pre-PO1 names still work. params merges over the style's
    defaults (unknown keys and non-numbers are ignored, so a
    malformed params dict cannot crash a render). wood_strength
    is the v0.1-era override: it sets the "strength" parameter
    of wood_cathedral only, and only when given — None means the
    style's own default. layout is the layout key: it only
    affects the bake for brick, whose bond rearranges the bricks
    (other styles lay out in the shader, so their bake is
    layout-independent); None means the style's default layout."""
    style = resolve_style(name)
    if style == "none":
        return 1.0
    merged = merged_params(style, params)
    if wood_strength is not None and style == "wood_cathedral":
        merged["strength"] = float(wood_strength)
    if style == "brick_stretcher":
        bond = resolve_layout(style, layout) if layout is not None \
            else "stretcher"
        return _brick_stretcher(u, v, merged, bond)
    return _PATTERNS[style](u, v, merged)


def pattern_rgb(name, u, v, base_rgb, wood_strength=None, params=None,
                layout=None):
    f = pattern_factor(name, u, v, wood_strength, params, layout)
    return tuple(max(0.0, min(1.0, c * f)) for c in base_rgb)


def build_qimage(name, base_rgb, size=256, wood_strength=None,
                 params=None, layout=None):
    """Render a pattern to a QImage (Qt import kept local so the pure
    pattern functions stay testable without Qt)."""
    from PySide6.QtGui import QImage, qRgb
    img = QImage(size, size, QImage.Format_RGB32)
    for y in range(size):
        v = y / max(1, size - 1)
        for x in range(size):
            u = x / max(1, size - 1)
            r, g, b = pattern_rgb(name, u, v, base_rgb, wood_strength,
                                  params, layout)
            img.setPixel(x, y, qRgb(int(r * 255), int(g * 255), int(b * 255)))
    return img
