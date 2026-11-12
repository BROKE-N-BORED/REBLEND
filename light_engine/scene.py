"""Scene data: JSON format, defaults, save/load. No Qt here.

Scene JSON (format v2, T6):
  {
    "name": "...", "version": 2,
    "objects": [ { "id", "name", "shape", "params",
                   "kind": "object" | "emitter" (default "object"; emitters
                           are T16 — the v2 loader only preserves the field),
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
                                 "texture": "none" | "wood_grain" |
                                            "brushed_metal" | "worn_leather" } } ],
    "cameras": [ { "name", "position": [x, y, z], "look_at": [x, y, z] } ],
    "lights":  [ { "name", "kind": "key"|"fill"|"rim"|"ambient",
                   "position": [x, y, z], "color": "#rrggbb",
                   "intensity": 0..1 } ],
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

TEXTURES = ("none", "wood_grain", "brushed_metal", "worn_leather")
MATERIAL_DEFAULTS = {"base_color": "#b0b0b0", "roughness": 0.6,
                     "metalness": 0.0, "texture": "none"}
# Shapes that default to smooth shading (T3): curved primitives.
# Box and plane default flat, so their edges stay crisp.
SMOOTH_SHAPES = ("sphere", "cylinder", "cone", "torus")


def default_smooth_shading(shape):
    """Smooth shading default for a shape."""
    return shape in SMOOTH_SHAPES


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
        "kind": "object",
        "smooth_shading": default_smooth_shading(shape),
        "cast_shadow": True,
        "receive_shadow": True,
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
        "version": 2,
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
    # kind is a v2 field (T6): "object" or "emitter" (emitters are T16;
    # the loader only accepts and preserves the value). Anything else
    # — including a v1 file with no kind at all — means "object".
    if raw.get("kind") in ("object", "emitter"):
        obj["kind"] = raw["kind"]
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
