"""Scene data: JSON format, defaults, save/load. No Qt here.

Scene JSON:
  {
    "name": "...", "version": 1,
    "objects": [ { "id", "name", "shape", "params",
                   "position": [x, y, z], "rotation": [rx, ry, rz] (degrees),
                   "scale": [sx, sy, sz],
                   "material": { "base_color": "#rrggbb", "roughness": 0..1,
                                 "metalness": 0..1,
                                 "texture": "none" | "wood_grain" |
                                            "brushed_metal" | "worn_leather" } } ],
    "cameras": [ { "name", "position": [x, y, z], "look_at": [x, y, z] } ],
    "lights":  [ { "name", "kind": "key"|"fill"|"rim"|"ambient",
                   "position": [x, y, z], "color": "#rrggbb",
                   "intensity": 0..1 } ],
    "animation": { "fps": 24, "end_frame": 48,
                   "keys": { object_id: { "position": [[frame, [x,y,z]], ...],
                                          "rotation": [...], "scale": [...] } } }
  }
Units are cm, z-up. Loading is tolerant: missing fields get defaults.
"""
from __future__ import annotations

import copy
import json
import uuid

from .geometry import default_params

TEXTURES = ("none", "wood_grain", "brushed_metal", "worn_leather")
MATERIAL_DEFAULTS = {"base_color": "#b0b0b0", "roughness": 0.6,
                     "metalness": 0.0, "texture": "none"}


def default_material(overrides=None):
    mat = dict(MATERIAL_DEFAULTS)
    if overrides:
        for key in MATERIAL_DEFAULTS:
            if key in overrides:
                mat[key] = overrides[key]
    if mat["texture"] not in TEXTURES:
        mat["texture"] = "none"
    mat["roughness"] = max(0.0, min(1.0, float(mat["roughness"])))
    mat["metalness"] = max(0.0, min(1.0, float(mat["metalness"])))
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
        "material": default_material(material),
    }
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
        "version": 1,
        "objects": [],
        "cameras": [],
        "lights": default_lights(),
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
    scene["version"] = 1
    scene["objects"] = [normalize_object(o) for o in raw.get("objects", [])
                        if isinstance(o, dict)]
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
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scene, fh, indent=2)


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


def total_triangles(scene):
    """Triangle count for the whole scene (tessellates in memory, cheap
    at these sizes; the viewport caches the result in GPU buffers)."""
    from .geometry import tessellate
    total = 0
    for obj in scene["objects"]:
        total += tessellate(obj["shape"], obj.get("params", {})).get(
            "tri_count", 0)
    return total
