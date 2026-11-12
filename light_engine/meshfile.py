"""Imported mesh assets: the .mesh.json format (T7). No Qt here.

A mesh object in a scene stores {"shape": "mesh", "mesh_file":
"assets/<name>.mesh.json"}; the vertex data lives in that file, next
to the scene JSON, so scene files stay small and readable. The
importers (T8 .obj, T9 .glb) write this format through save_mesh;
the viewport reads it through load_mesh.

.mesh.json schema (version 1):
  {
    "format": "reblend-mesh",
    "version": 1,
    "source": "chair.obj",            # file the mesh came from
    "imported": "2026-11-12T13:30:00",# ISO date the asset was written
    "vertex_count": 24,               # recorded counts — total_triangles
    "triangle_count": 8,              #   reads these WITHOUT building
                                      #   any vertex data (v2 spec)
    "positions": [x, y, z, ...],      # flat, 3 per vertex, cm, z-up
    "normals":   [x, y, z, ...],      # flat, 3 per vertex, as stored —
                                      #   the asset's normals are used
                                      #   as-is, never re-averaged
    "uvs":       [u, v, ...],         # flat, 2 per vertex
    "indices":   [i, j, k, ...]       # flat, 3 per triangle
  }
load_mesh validates the arrays and raises ValueError with a plain
message for a corrupt file; a missing file raises FileNotFoundError,
which the caller turns into the missing-file warning.
"""
from __future__ import annotations

import json
import math

FORMAT = "reblend-mesh"
VERSION = 1


def mesh_bounds(positions):
    """(min, max) of a flat positions list. Empty mesh -> a 1 cm box
    at the origin, so bounds code never sees infinities."""
    if not positions:
        return (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)
    xs = positions[0::3]
    ys = positions[1::3]
    zs = positions[2::3]
    return ((min(xs), min(ys), min(zs)),
            (max(xs), max(ys), max(zs)))


def _as_floats(values, field):
    """A flat list of finite numbers, or a plain ValueError."""
    if not isinstance(values, list):
        raise ValueError(f"'{field}' must be a list of numbers")
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError):
        raise ValueError(f"'{field}' must be a list of numbers")
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"'{field}' has a non-finite number")
    return out


def _as_indices(values, vertex_count):
    if not isinstance(values, list):
        raise ValueError("'indices' must be a list of integers")
    try:
        out = [int(v) for v in values]
    except (TypeError, ValueError):
        raise ValueError("'indices' must be a list of integers")
    if len(out) % 3 != 0:
        raise ValueError(
            f"'indices' has {len(out)} entries, not a multiple of 3")
    for i in out:
        if i < 0 or i >= vertex_count:
            raise ValueError(
                f"index {i} is outside the {vertex_count} vertices")
    return out


def load_mesh(path):
    """Load and validate a .mesh.json file. Returns a dict with the
    arrays, the recorded counts, the source/imported metadata, and
    "bounds" (min, max) computed from the vertices. The recorded
    counts must agree with the arrays — a mismatch is corruption,
    not something to guess around."""
    with open(path, "r", encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON ({exc})")
    if not isinstance(raw, dict):
        raise ValueError("the file does not hold a mesh object")
    if raw.get("format") not in (None, FORMAT):
        raise ValueError(
            f"format is {raw.get('format')!r}, not {FORMAT!r}")
    positions = _as_floats(raw.get("positions"), "positions")
    normals = _as_floats(raw.get("normals"), "normals")
    uvs = _as_floats(raw.get("uvs"), "uvs")
    if len(positions) % 3 != 0:
        raise ValueError(
            f"'positions' has {len(positions)} numbers, not a multiple "
            "of 3")
    vertex_count = len(positions) // 3
    if len(normals) != len(positions):
        raise ValueError(
            f"'normals' has {len(normals)} numbers for "
            f"{vertex_count} vertices (expected {len(positions)})")
    if len(uvs) != vertex_count * 2:
        raise ValueError(
            f"'uvs' has {len(uvs)} numbers for {vertex_count} vertices "
            f"(expected {vertex_count * 2})")
    indices = _as_indices(raw.get("indices"), vertex_count)
    triangle_count = len(indices) // 3
    # The recorded counts are authoritative for counting (v2 spec), so
    # they must agree with the arrays when present.
    for field, actual in (("vertex_count", vertex_count),
                          ("triangle_count", triangle_count)):
        recorded = raw.get(field)
        if recorded is not None and recorded != actual:
            raise ValueError(
                f"'{field}' says {recorded} but the arrays hold {actual}")
    return {
        "positions": positions,
        "normals": normals,
        "uvs": uvs,
        "indices": indices,
        "vertex_count": vertex_count,
        "triangle_count": triangle_count,
        "bounds": mesh_bounds(positions),
        "source": raw.get("source", ""),
        "imported": raw.get("imported", ""),
    }


def save_mesh(path, mesh, source="", imported=""):
    """Write a mesh dict in .mesh.json format. The importers (T8/T9)
    build the mesh dict and call this; counts are recorded from the
    arrays, so they can never disagree with them."""
    positions = mesh["positions"]
    indices = mesh["indices"]
    data = {
        "format": FORMAT,
        "version": VERSION,
        "source": source,
        "imported": imported,
        "vertex_count": len(positions) // 3,
        "triangle_count": len(indices) // 3,
        "positions": list(positions),
        "normals": list(mesh["normals"]),
        "uvs": list(mesh["uvs"]),
        "indices": list(indices),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def read_counts(path):
    """(vertex_count, triangle_count) from a .mesh.json file, without
    building any vertex data. Reads the recorded counts when present;
    falls back to the array lengths when a file omits them."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError("the file does not hold a mesh object")
    verts = raw.get("vertex_count")
    tris = raw.get("triangle_count")
    if verts is None:
        verts = len(raw.get("positions", [])) // 3
    if tris is None:
        tris = len(raw.get("indices", [])) // 3
    return int(verts), int(tris)
