"""Mesh import entry point (T8/T9). No Qt here.

Both importers run through import_mesh(path): it dispatches on the
file extension — .obj (meshio_obj) or .glb (meshio_glb) — and returns
(objects, report), where objects is a list of {name, mesh, material}
dicts (.glb items also carry position/rotation/scale from their node
transform) and report is {objects, triangles, warnings} in plain
language.

The window uses the two-step form so the import dialog can show the
report before the user picks a scale: parse(path) reads the file
(scale is NOT applied yet), build(parsed, scale) turns the groups
into mesh dicts with positions scaled. import_mesh is parse +
build in one call.
"""
from __future__ import annotations

import os

from . import meshio_glb
from . import meshio_obj

# .obj files are usually in meters or unitless; Reblend works in cm.
DEFAULT_SCALE = 100.0
# A mesh past this many triangles gets a plain warning before import
# ("this will be heavy on this machine") with a proceed/cancel choice
# — a warning with a choice is not a refusal (Design 2).
HEAVY_TRIANGLES = 1_000_000


def is_heavy(triangle_count):
    """True when a mesh is past the heavy warning threshold."""
    return triangle_count > HEAVY_TRIANGLES


def parse(path):
    """Read a mesh file into a parsed dict (unscaled). Raises
    ValueError with a plain message for an unsupported file or a
    malformed one, FileNotFoundError for a missing one."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        return meshio_obj.parse_obj(path)
    if ext == ".glb":
        return meshio_glb.parse_glb(path)
    raise ValueError(
        f"'{os.path.basename(path)}' is not a supported mesh file — "
        "try a .obj or .glb file")


def build(parsed, scale=DEFAULT_SCALE):
    """[{name, mesh, material}] for a parsed file, positions scaled
    by `scale`."""
    if parsed["format"] == "obj":
        return meshio_obj.build_meshes(parsed, scale)
    if parsed["format"] == "glb":
        return meshio_glb.build_meshes(parsed, scale)
    raise ValueError(
        f"cannot build meshes from a '{parsed['format']}' file")


def import_mesh(path, scale=DEFAULT_SCALE):
    """(objects, report) for a mesh file: parse it, build its groups
    into mesh dicts with positions scaled by `scale`."""
    parsed = parse(path)
    return build(parsed, scale), parsed["report"]


def _sanitize(name):
    """A filename-safe stem: letters, digits, dot, dash, underscore
    survive; anything else (including path separators) becomes a
    dash. Runs of dashes collapse, and leading dots/dashes go, so a
    group name can never escape the assets folder or hide a file."""
    out = "".join(ch if (ch.isalnum() or ch in "._-") else "-"
                  for ch in name)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip(".-") or "mesh"


def unique_asset_name(stem, group, assets_dir, used=None):
    """A collision-safe .mesh.json filename for one imported group:
    the source stem when the group is unnamed or matches the stem,
    otherwise stem-group, sanitized; -2, -3, ... when the name is
    taken. `used` tracks names already claimed in this import run so
    two groups cannot land on the same file."""
    if used is None:
        used = set()
    if not group or group.lower() == stem.lower():
        base = _sanitize(stem)
    else:
        base = _sanitize(f"{stem}-{group}")
    candidate = f"{base}.mesh.json"
    n = 2
    while candidate in used or os.path.exists(
            os.path.join(assets_dir, candidate)):
        candidate = f"{base}-{n}.mesh.json"
        n += 1
    used.add(candidate)
    return candidate
