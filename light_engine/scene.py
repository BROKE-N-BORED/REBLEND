"""Scene data: JSON format, defaults, save/load. No Qt here.

Scene JSON (format v2, T6):
  {
    "name": "...", "version": 2,
    "objects": [ { "id", "name", "shape", "params",
                   "kind": "object" | "emitter" (default "object"),
                   "emitter": { ... } (emitter objects only, T16 — see
                           default_emitter_params: shape, rate, lifetime,
                           speed, size, colors, opacity, gravity factor,
                           drag, wind response, turbulence, attract,
                           collide, seed, preset),
                   "mesh_file": "assets/....mesh.json" (mesh objects, T7;
                           geometry comes from that asset file — see
                           meshfile.py for the .mesh.json schema),
                   "position": [x, y, z], "rotation": [rx, ry, rz] (degrees),
                   "scale": [sx, sy, sz],
                   "smooth_shading": true|false (default by shape, T3),
                   "cast_shadow": true|false (default true, T5),
                   "receive_shadow": true|false (default true, T5),
                   "material": { "base_color": "#rrggbb", "roughness": 0..1,
                                 "metalness": 0..1,
                                 "style": "none" | a style key from
                                        textures.STYLES (PO1 — the
                                        procedural material library),
                                 "texture": mirrors "style" (the
                                        viewport reads this key; old
                                        scenes store only "texture"
                                        with a pre-PO1 name, mapped
                                        at load — see
                                        LEGACY_TEXTURE_MAP),
                                 "params": { ... } (PO2 — the style's
                                        knob values; overrides only,
                                        so a missing key means the
                                        style's default and an old
                                        scene's {} renders exactly
                                        as before),
                                 "density": 1.0, "friction": 0.5,
                                 "restitution": 0.3 (T19 — physics, see
                                 Design 8; missing in old scenes,
                                 defaults apply) } } ],
    "cameras": [ { "name", "position": [x, y, z], "look_at": [x, y, z] } ],
    "lights":  [ { "name", "kind": "key"|"fill"|"rim"|"ambient",
                   "position": [x, y, z], "color": "#rrggbb",
                   "intensity": 0..1 } ],
    "world": { "gravity": [0, 0, -981], "wind": [0, 0, 0],
               "wind_gust": { "amplitude": 0.0, "frequency": 0.0 } }
               (T16 — scene gravity + wind for particles/physics;
               missing entirely in old scenes, defaults apply),
    "physics": { object_id: { ... } } (T18 — preserved, not used yet),
    "animation": { "fps": 24, "end_frame": 48,
                   "keys": { object_id: { "position": [[frame, [x,y,z]], ...],
                                          "rotation": [...], "scale": [...] } } }
  }
Units are cm, z-up. Loading is tolerant: missing fields get defaults,
and v1 scenes load with defaults for every new field (smooth normals
on for curved primitives, shadows on, kind "object"). Save always
writes v2; no v1 writer is kept.
"""
from __future__ import annotations

import copy
import json
import os
import uuid

from .geometry import default_params
from .textures import LEGACY_TEXTURE_MAP, STYLES, resolve_layout

# Every texture value a material may carry: "none" plus every style
# key in the PO1 library.
TEXTURES = ("none",) + tuple(STYLES)
MATERIAL_DEFAULTS = {"base_color": "#b0b0b0", "roughness": 0.6,
                     "metalness": 0.0, "style": "none",
                     "texture": "none", "params": {},
                     "layout": "solid", "layout_size": {},
                     "wear": {},
                     "density": 1.0, "friction": 0.5,
                     "restitution": 0.3}

# Material physics presets (T19, Design 8): real-world-ish handbook
# midpoints — friction and restitution are pair-dependent in reality,
# so these are standard midpoints, not laboratory measurements. The
# visual hint (roughness / metalness) is what "also set look" applies;
# it is never written silently. The preset name is never stored — only
# the three numbers are, so a scene file stays self-describing; the
# picker derives the name from the numbers (material_preset_for).
MATERIAL_PRESETS = {
    "balsa_wood": {"display": "Balsa wood", "density": 0.16,
                   "friction": 0.4, "restitution": 0.3,
                   "roughness": 0.8, "metalness": 0.0},
    "pine_wood": {"display": "Pine wood", "density": 0.5,
                  "friction": 0.4, "restitution": 0.3,
                  "roughness": 0.7, "metalness": 0.0},
    "oak_wood": {"display": "Oak wood", "density": 0.75,
                 "friction": 0.45, "restitution": 0.35,
                 "roughness": 0.6, "metalness": 0.0},
    "cork": {"display": "Cork", "density": 0.24, "friction": 0.5,
             "restitution": 0.2, "roughness": 0.9, "metalness": 0.0},
    "steel": {"display": "Steel", "density": 7.8, "friction": 0.3,
              "restitution": 0.6, "roughness": 0.3, "metalness": 1.0},
    "aluminium": {"display": "Aluminium", "density": 2.7,
                  "friction": 0.35, "restitution": 0.4,
                  "roughness": 0.4, "metalness": 1.0},
    "copper": {"display": "Copper", "density": 9.0, "friction": 0.3,
               "restitution": 0.5, "roughness": 0.25, "metalness": 1.0},
    "gold": {"display": "Gold", "density": 19.3, "friction": 0.3,
             "restitution": 0.4, "roughness": 0.15, "metalness": 1.0},
    "stone": {"display": "Stone / concrete", "density": 2.4,
              "friction": 0.7, "restitution": 0.1,
              "roughness": 0.9, "metalness": 0.0},
    "glass": {"display": "Glass", "density": 2.5, "friction": 0.2,
              "restitution": 0.5, "roughness": 0.05, "metalness": 0.0},
    "rubber": {"display": "Rubber", "density": 1.1, "friction": 0.9,
               "restitution": 0.8, "roughness": 0.9, "metalness": 0.0},
    "plastic": {"display": "Plastic", "density": 1.2, "friction": 0.4,
                "restitution": 0.4, "roughness": 0.5, "metalness": 0.0},
    "ice": {"display": "Ice", "density": 0.9, "friction": 0.05,
            "restitution": 0.2, "roughness": 0.1, "metalness": 0.0},
    "cloth": {"display": "Cloth", "density": 0.3, "friction": 0.6,
              "restitution": 0.05, "roughness": 1.0, "metalness": 0.0},
    "water": {"display": "Water (as a body)", "density": 1.0,
              "friction": 0.1, "restitution": 0.0,
              "roughness": 0.05, "metalness": 0.0},
}

# Emitter parameter set (T16, Design 5 revised). The emitter dict lives
# on the object under "emitter" — the object's "params" stay the
# primitive geometry params, so the two never collide.
EMITTER_SHAPES = ("point", "box", "sphere", "cone")
EMITTER_COLLIDE = ("none", "kill", "bounce")
EMITTER_PRESETS = ("custom", "fire", "smoke", "sparks", "rain",
                   "snow", "dust", "water")
DEFAULT_GRAVITY = [0.0, 0.0, -981.0]  # cm/s^2, z-up


def _clamp(value, low, high, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def _hex_color(value, default):
    text = str(value or "").strip()
    if len(text) == 7 and text.startswith("#"):
        try:
            int(text[1:], 16)
            return text.lower()
        except ValueError:
            pass
    return default


def default_world(overrides=None):
    """The top-level "world" section: scene gravity + wind (T16).
    Tolerant: a missing or mistyped section gets the defaults."""
    world = {"gravity": list(DEFAULT_GRAVITY), "wind": [0.0, 0.0, 0.0],
             "wind_gust": {"amplitude": 0.0, "frequency": 0.0}}
    if not isinstance(overrides, dict):
        return world
    for key in ("gravity", "wind"):
        try:
            vals = [float(v) for v in overrides.get(key, world[key])]
            if len(vals) == 3:
                world[key] = vals
        except (TypeError, ValueError):
            pass
    gust = overrides.get("wind_gust")
    if isinstance(gust, dict):
        world["wind_gust"]["amplitude"] = _clamp(
            gust.get("amplitude", 0.0), 0.0, 1e9, 0.0)
        world["wind_gust"]["frequency"] = _clamp(
            gust.get("frequency", 0.0), 0.0, 1e9, 0.0)
    return world


def default_emitter_params(overrides=None):
    """The full emitter parameter set (T16, Design 5). Units are cm and
    seconds. Every value is validated: an unknown shape / collide mode /
    preset falls back to its default, numbers clamp into range, and a
    mistyped value keeps the default — loading never fails on a bad
    emitter dict. Nothing here refuses: rate and lifetime only clamp
    at zero (a zero rate simply emits nothing)."""
    params = {
        # Emission shape: point (one spot), box (volume), sphere
        # (volume), cone (apex + direction cone). box_size and
        # sphere_radius size the volume shapes.
        "shape": "point",
        "box_size": [20.0, 20.0, 20.0],
        "sphere_radius": 10.0,
        # Emission: particles per second, seconds of life.
        "rate": 60.0,
        "lifetime": 1.0,
        "lifetime_spread": 0.3,
        # Initial velocity: cm/s along the emitter's forward (+Z,
        # rotated by the object rotation), inside the spread cone.
        "speed": 50.0,
        "speed_spread": 0.3,
        "spread": 15.0,
        # Appearance over life: start size in cm, growth factor at
        # death (smoke grows ~3x, sparks shrink to 0), colors
        # interpolated start -> end, opacity fades as fractions of
        # life (fade in over the first 10%, out over the last 30%).
        "size": 5.0,
        "size_over_life": 1.0,
        "color_start": "#ffffff",
        "color_end": "#ffffff",
        "opacity": {"fade_in": 0.1, "fade_out": 0.3},
        # Forces: gravity_factor multiplies the scene gravity (1 =
        # normal, 0 = floats, negative = rises); drag is the linear
        # drag coefficient (0 = none); wind_response 0-1 is how much
        # the global wind pushes this emitter's particles.
        "gravity_factor": 1.0,
        "drag": 0.0,
        "wind_response": 0.5,
        # Turbulence is the analytic sine wobble strength in cm
        # (T16). Position-dependent turbulence is stepped mode (T18).
        "turbulence": 0.0,
        # Attraction is stepped-only (T18): {"target": object_id,
        # "strength": float} or None. Stored now, ignored by the
        # analytic sim. Collide: "kill" removes a particle at floor
        # impact (analytic, T16); "bounce" needs stepped mode (T18).
        "attract": None,
        "collide": "none",
        # Seed: same seed + same params = same particles, always.
        "seed": 0,
        # Preset: the name the parameters came from ("custom" once
        # any parameter is edited by hand). The parameters are the
        # stored state; the name is a convenience for the picker.
        "preset": "custom",
    }
    if not isinstance(overrides, dict):
        return params
    if overrides.get("shape") in EMITTER_SHAPES:
        params["shape"] = overrides["shape"]
    try:
        vals = [float(v) for v in overrides.get("box_size",
                                                params["box_size"])]
        if len(vals) == 3:
            params["box_size"] = [max(0.0, v) for v in vals]
    except (TypeError, ValueError):
        pass
    params["sphere_radius"] = _clamp(
        overrides.get("sphere_radius", params["sphere_radius"]),
        0.0, 1e9, params["sphere_radius"])
    params["rate"] = _clamp(overrides.get("rate", params["rate"]),
                            0.0, 1e9, params["rate"])
    # Lifetime 0 would mean no particle is ever alive; the floor is a
    # normalization, not a refusal — a 10 ms life still emits.
    params["lifetime"] = _clamp(overrides.get("lifetime",
                                              params["lifetime"]),
                                0.01, 1e9, params["lifetime"])
    params["lifetime_spread"] = _clamp(
        overrides.get("lifetime_spread", params["lifetime_spread"]),
        0.0, 1.0, params["lifetime_spread"])
    params["speed"] = _clamp(overrides.get("speed", params["speed"]),
                             0.0, 1e9, params["speed"])
    params["speed_spread"] = _clamp(
        overrides.get("speed_spread", params["speed_spread"]),
        0.0, 1.0, params["speed_spread"])
    params["spread"] = _clamp(overrides.get("spread", params["spread"]),
                              0.0, 180.0, params["spread"])
    params["size"] = _clamp(overrides.get("size", params["size"]),
                            0.01, 1e9, params["size"])
    params["size_over_life"] = _clamp(
        overrides.get("size_over_life", params["size_over_life"]),
        0.0, 1e9, params["size_over_life"])
    params["color_start"] = _hex_color(
        overrides.get("color_start"), params["color_start"])
    params["color_end"] = _hex_color(overrides.get("color_end"),
                                     params["color_end"])
    opacity = overrides.get("opacity")
    if isinstance(opacity, dict):
        params["opacity"] = {
            "fade_in": _clamp(opacity.get("fade_in", 0.1),
                              0.0, 1.0, 0.1),
            "fade_out": _clamp(opacity.get("fade_out", 0.3),
                               0.0, 1.0, 0.3),
        }
    params["gravity_factor"] = _clamp(
        overrides.get("gravity_factor", params["gravity_factor"]),
        -1e9, 1e9, params["gravity_factor"])
    params["drag"] = _clamp(overrides.get("drag", params["drag"]),
                            0.0, 1e9, params["drag"])
    params["wind_response"] = _clamp(
        overrides.get("wind_response", params["wind_response"]),
        0.0, 1.0, params["wind_response"])
    params["turbulence"] = _clamp(
        overrides.get("turbulence", params["turbulence"]),
        0.0, 1e9, params["turbulence"])
    attract = overrides.get("attract")
    if isinstance(attract, dict) and attract.get("target"):
        params["attract"] = {
            "target": str(attract["target"]),
            "strength": _clamp(attract.get("strength", 0.0),
                               -1e9, 1e9, 0.0),
        }
    if overrides.get("collide") in EMITTER_COLLIDE:
        params["collide"] = overrides["collide"]
    try:
        params["seed"] = int(overrides.get("seed", 0))
    except (TypeError, ValueError):
        params["seed"] = 0
    if overrides.get("preset") in EMITTER_PRESETS:
        params["preset"] = overrides["preset"]
    return params
# Shapes that default to smooth shading (T3): curved primitives.
# Box and plane default flat, so their edges stay crisp.
SMOOTH_SHAPES = ("sphere", "cylinder", "cone", "torus")


def default_smooth_shading(shape):
    """Smooth shading default for a shape."""
    return shape in SMOOTH_SHAPES


def _material_number(value, default, low=None, high=None):
    """One material physics number, tolerant (T19): a missing or
    malformed value (a string, None, a bool) falls back to the
    default, never crashes. A real number is clamped to the given
    range — density and friction have no upper bound (gold is 19.3,
    friction runs 0 to 1 and past), restitution is 0 to 1."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if low is not None:
        number = max(low, number)
    if high is not None:
        number = min(high, number)
    return number


def material_preset_for(mat):
    """The preset whose three physics numbers match a material's
    exactly, or "custom". The preset name is never stored (Design
    8) — the picker derives it from the numbers every time, so a
    hand-edit to exactly steel's numbers shows steel, and any other
    edit shows custom."""
    density = _material_number(mat.get("density"), 1.0, 0.0)
    friction = _material_number(mat.get("friction"), 0.5, 0.0)
    restitution = _material_number(mat.get("restitution"), 0.3, 0.0, 1.0)
    for name, preset in MATERIAL_PRESETS.items():
        if (preset["density"] == density
                and preset["friction"] == friction
                and preset["restitution"] == restitution):
            return name
    return "custom"


def resolve_material_style(overrides):
    """The style a material stores (PO1): the "style" key when it
    names a real style (or "none"), else the legacy "texture" name
    mapped through LEGACY_TEXTURE_MAP (old scenes store only
    "texture"), else "none". Never raises, never returns an
    unknown style."""
    if overrides:
        style = overrides.get("style")
        if style in STYLES or style == "none":
            return style
        return LEGACY_TEXTURE_MAP.get(overrides.get("texture"), "none")
    return "none"


def default_material(overrides=None):
    mat = dict(MATERIAL_DEFAULTS)
    if overrides:
        for key in MATERIAL_DEFAULTS:
            if key in overrides:
                mat[key] = overrides[key]
    # "style" is canonical (PO1); "texture" mirrors it — the
    # viewport reads "texture", so the two stay equal and no
    # viewport change is needed. An old scene's "texture" name
    # maps to its style here (see resolve_material_style).
    mat["style"] = resolve_material_style(overrides)
    mat["texture"] = mat["style"]
    # "layout" (PO4): how the pattern is laid, in real centimetres.
    # Resolved against the style, so a stored layout from another
    # style (or a mistyped one) becomes the style's default —
    # brick defaults to "stretcher", everything else to "solid".
    # "layout_size" holds the piece sizes as overrides only, like
    # "params": {} means the layout's defaults. Normalized to a
    # fresh dict on every call, for the same shared-{} reason.
    mat["layout"] = resolve_layout(mat["style"], mat.get("layout"))
    raw_sizes = mat.get("layout_size")
    if isinstance(raw_sizes, dict):
        mat["layout_size"] = {
            key: float(value)
            for key, value in raw_sizes.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)}
    else:
        mat["layout_size"] = {}
    # "params" (PO2): the style's knob values, overrides only.
    # Normalized to a fresh dict on every call — MATERIAL_DEFAULTS
    # holds one shared {} and a shallow copy must never leak it
    # into two materials. A missing or mistyped value is {}, and
    # a non-number value is dropped (the merge in textures would
    # ignore it anyway); unknown keys are kept, so a scene written
    # by a newer Reblend keeps its values.
    raw_params = mat.get("params")
    if isinstance(raw_params, dict):
        mat["params"] = {
            key: float(value)
            for key, value in raw_params.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)}
    else:
        mat["params"] = {}
    # "wear" (PO5): the weather/age values, overrides only like
    # "params" — {} means clean and dry, no cracks. Normalized
    # the same way: numbers kept, everything else dropped, a
    # fresh dict on every call. The style filter and the clamps
    # live in textures.merged_wear, which the viewport calls at
    # draw time — the scene keeps the stored values as they are.
    raw_wear = mat.get("wear")
    if isinstance(raw_wear, dict):
        mat["wear"] = {
            key: float(value)
            for key, value in raw_wear.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)}
    else:
        mat["wear"] = {}
    mat["roughness"] = max(0.0, min(1.0, float(mat["roughness"])))
    mat["metalness"] = max(0.0, min(1.0, float(mat["metalness"])))
    mat["density"] = _material_number(mat["density"], 1.0, 0.0)
    mat["friction"] = _material_number(mat["friction"], 0.5, 0.0)
    mat["restitution"] = _material_number(mat["restitution"], 0.3,
                                          0.0, 1.0)
    return mat


def new_object(shape, name=None, position=None, params=None, material=None):
    obj = {
        "id": uuid.uuid4().hex[:12],
        "name": name or shape.capitalize(),
        "shape": shape,
        "params": params if params is not None else default_params(shape),
        "position": list(position or [0.0, 0.0, 0.0]),
        "rotation": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "kind": "object",
        "smooth_shading": default_smooth_shading(shape),
        "cast_shadow": True,
        "receive_shadow": True,
        "material": default_material(material),
    }
    return obj


def new_emitter(name=None, position=None, params=None):
    """An emitter object (T16): kind "emitter" with the full parameter
    set. The shape is a small 10 cm box — the emitter's marker in the
    viewport until T17 draws the particles themselves. The emission
    shape (point/box/sphere/cone) is a parameter, not the object
    shape."""
    obj = new_object("box", name=name or "Emitter", position=position,
                     params={"size": [10.0, 10.0, 10.0]})
    obj["kind"] = "emitter"
    obj["emitter"] = default_emitter_params(params)
    return obj


def default_lights():
    return [
        {"name": "Key", "kind": "key", "position": [-300.0, -300.0, 400.0],
         "color": "#ffffff", "intensity": 0.9},
        {"name": "Fill", "kind": "fill", "position": [300.0, -200.0, 150.0],
         "color": "#cfe0ff", "intensity": 0.45},
        {"name": "Rim", "kind": "rim", "position": [0.0, 400.0, 250.0],
         "color": "#ffe8c8", "intensity": 0.5},
        {"name": "Ambient", "kind": "ambient", "position": [0.0, 0.0, 0.0],
         "color": "#ffffff", "intensity": 0.35},
    ]


def new_scene(name="Untitled"):
    return {
        "name": name,
        "version": 2,
        "objects": [],
        "cameras": [],
        "lights": default_lights(),
        "world": default_world(),
        "animation": {"fps": 24, "end_frame": 48, "keys": {}},
    }


def normalize_object(raw):
    """Fill defaults for one object dict loaded from disk."""
    shape = raw.get("shape", "box")
    obj = new_object(shape, name=raw.get("name"),
                     position=raw.get("position"),
                     params=raw.get("params"),
                     material=raw.get("material"))
    obj["id"] = str(raw.get("id") or obj["id"])
    # kind is a v2 field (T6): "object" or "emitter". Anything else
    # — including a v1 file with no kind at all — means "object".
    if raw.get("kind") in ("object", "emitter"):
        obj["kind"] = raw["kind"]
    # The emitter parameter set (T16): normalized with defaults, so
    # an emitter with a missing or partial "emitter" dict still loads
    # whole. Non-emitters never carry the dict.
    if obj["kind"] == "emitter":
        obj["emitter"] = default_emitter_params(raw.get("emitter"))
    # mesh_file is a v2 field (T7): preserved if it is a string, so a
    # scene round-trips intact. The viewport resolves it against the
    # scene's folder and renders the asset; the window drops a mesh
    # object whose file is missing (warns, never fakes).
    if isinstance(raw.get("mesh_file"), str) and raw["mesh_file"]:
        obj["mesh_file"] = raw["mesh_file"]
    # smooth_shading is additive (T3): a real bool in the file wins,
    # a missing or mistyped value keeps the shape default from
    # new_object.
    if isinstance(raw.get("smooth_shading"), bool):
        obj["smooth_shading"] = raw["smooth_shading"]
    # cast_shadow / receive_shadow are additive the same way (T5):
    # a real bool wins, anything else keeps the True default.
    for flag in ("cast_shadow", "receive_shadow"):
        if isinstance(raw.get(flag), bool):
            obj[flag] = raw[flag]
    for key in ("rotation", "scale"):
        try:
            vals = [float(v) for v in raw.get(key, obj[key])]
            if len(vals) == 3:
                obj[key] = vals
        except (TypeError, ValueError):
            pass
    try:
        vals = [float(v) for v in raw.get("position", obj["position"])]
        if len(vals) == 3:
            obj["position"] = vals
    except (TypeError, ValueError):
        pass
    return obj


def normalize_scene(raw):
    scene = new_scene(raw.get("name", "Untitled"))
    # Format v2 (T6): every scene loads as v2, including v1 files —
    # the new fields get their defaults in normalize_object.
    scene["version"] = 2
    scene["objects"] = [normalize_object(o) for o in raw.get("objects", [])
                        if isinstance(o, dict)]
    # world is a v2 section (T16): scene gravity + wind. Old scenes
    # without it load with the defaults.
    scene["world"] = default_world(raw.get("world"))
    # physics is a v2 section (T18): preserved as-is so a scene
    # round-trips intact; nothing reads it yet.
    if isinstance(raw.get("physics"), dict):
        scene["physics"] = copy.deepcopy(raw["physics"])
    cams = []
    for cam in raw.get("cameras", []) or []:
        if not isinstance(cam, dict) or not cam.get("name"):
            continue
        try:
            pos = [float(v) for v in cam.get("position", [0, 0, 0])]
            look = [float(v) for v in cam.get("look_at", [0, 0, 0])]
        except (TypeError, ValueError):
            continue
        if len(pos) == 3 and len(look) == 3:
            cams.append({"name": str(cam["name"]),
                         "position": pos, "look_at": look})
    scene["cameras"] = cams
    lights = raw.get("lights")
    if isinstance(lights, list) and lights:
        scene["lights"] = lights
    anim = raw.get("animation")
    if isinstance(anim, dict):
        try:
            scene["animation"]["fps"] = max(1, int(anim.get("fps", 24)))
            scene["animation"]["end_frame"] = max(
                1, int(anim.get("end_frame", 48)))
        except (TypeError, ValueError):
            pass
        if isinstance(anim.get("keys"), dict):
            scene["animation"]["keys"] = anim["keys"]
    return scene


def load_scene(path):
    with open(path, "r", encoding="utf-8") as fh:
        return normalize_scene(json.load(fh))


def save_scene(scene, path):
    # Save always writes v2 (T6); no v1 writer is kept. The version is
    # forced on a shallow copy so a hand-built scene dict can never
    # write a v1 file, and the caller's dict is not mutated.
    data = dict(scene)
    data["version"] = 2
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def scene_copy(scene):
    return copy.deepcopy(scene)


def find_object(scene, object_id):
    for obj in scene["objects"]:
        if obj["id"] == object_id:
            return obj
    return None


def hex_to_rgb(value):
    value = str(value).lstrip("#")
    try:
        return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return (0.7, 0.7, 0.7)


def total_triangles(scene, base_path=None, mesh_counts=None):
    """Triangle count for the whole scene (primitives tessellate in
    memory, cheap at these sizes; the viewport caches the result in
    GPU buffers). Mesh objects count from the asset's RECORDED
    triangle_count, without building any vertex data (v2 spec):
    mesh_counts maps object id -> count when the caller already has
    the assets loaded (the viewport cache); otherwise, with
    base_path, the count is read from the asset file. Without either
    source a mesh counts 0 — never a tessellated guess."""
    from .geometry import tessellate
    from .meshfile import read_counts
    total = 0
    for obj in scene["objects"]:
        if obj.get("shape") == "mesh":
            if mesh_counts is not None and obj["id"] in mesh_counts:
                total += mesh_counts[obj["id"]]
            elif base_path and obj.get("mesh_file"):
                path = obj["mesh_file"]
                if not os.path.isabs(path):
                    path = os.path.join(base_path, path)
                try:
                    _verts, tris = read_counts(path)
                    total += tris
                except (OSError, ValueError):
                    pass
            continue
        total += tessellate(obj["shape"], obj.get("params", {})).get(
            "tri_count", 0)
    return total
