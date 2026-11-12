"""Wavefront .obj import (T8) and export (T10). Stdlib only, no Qt.

Import parses v/vt/vn/f (triangles and quads, fan-triangulated on
read), o/g groups (each group becomes one object), and mtllib +
usemtl for basic material colors (Kd diffuse -> base_color). Unknown
lines are skipped, never fatal. Negative (relative) face indices are
resolved against the vertices defined so far. A malformed file
raises ValueError with a plain message naming the problem — never a
traceback to the user, never a crash.

Two steps, so the import dialog can show a report before the user
picks a scale: parse_obj(path) reads the file into a parsed dict
(positions unscaled, faces as resolved triangle tuples, materials,
warnings, and a report of objects/triangles/warnings);
build_meshes(parsed, scale) turns the groups into mesh dicts in the
T7 .mesh.json shape (indexed positions/normals/uvs + triangle
indices), with positions scaled by `scale` (default 100: meters to
cm). The meshio.py dispatcher calls these; nothing else should.

Normals: when every corner of an object has a vn, the authored
normals are kept exactly (vertices are keyed by the full v/vt/vn
tuple, so hard edges survive). When any corner lacks a vn, the
object's normals are computed instead — the soup is welded with the
T2 angle rule (geometry.weld_soup), so smooth surfaces come out
smooth and crisp edges stay crisp. Missing vt means uvs of 0.

Export writes v/vt/vn/f plus an .mtl with Kd base colors, one `o`
group + `usemtl` per object. The soup is de-indexed on write
(Design 2): each triangle corner gets its own v/vt/vn entries —
simple, correct, slightly larger files; nothing is re-welded.
Normals come from the soup the viewport would draw (smooth objects
through the indexed path with T3 averaged normals, flat objects as
the raw soup, mesh objects from their asset), written as-is. .obj
has no per-object transform, so each object's world transform
(position, rotation, scale) is applied to its vertices before
writing, and its normals go through the inverse-transpose (a pure
rotation comes back as itself; a non-uniform scale gets the correct
matrix, so normals stay perpendicular to the transformed surface).
Export scale defaults to 0.01 (cm -> m), the mirror of the import
default, so export-then-import round-trips the geometry.
"""
from __future__ import annotations

import math
import os

from . import geometry as geometry_mod
from . import meshfile as meshfile_mod

DEFAULT_COLOR = "#b0b0b0"


class ObjError(ValueError):
    """A plain-language .obj problem (bad numbers, no faces, ...)."""


def _rgb_to_hex(r, g, b):
    def channel(v):
        return max(0, min(255, round(float(v) * 255.0)))
    return "#%02x%02x%02x" % (channel(r), channel(g), channel(b))


def _parse_mtllib(path, warnings):
    """{material name: base_color hex} from an .mtl file. Only Kd is
    read; anything else is skipped. A missing or unreadable file
    warns and returns what it has (usually nothing)."""
    materials = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        warnings.append(
            f"Material file not found: {os.path.basename(path)} — "
            "objects use the default color.")
        return materials
    current = None
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "newmtl" and len(parts) >= 2:
            current = parts[1]
            materials.setdefault(current, DEFAULT_COLOR)
        elif parts[0] == "Kd" and current is not None and len(parts) >= 4:
            try:
                materials[current] = _rgb_to_hex(parts[1], parts[2], parts[3])
            except ValueError:
                warnings.append(
                    f"Material '{current}' has a bad diffuse color — "
                    "the default color is used.")
                materials[current] = DEFAULT_COLOR
    return materials


def _resolve(index, count, what, line_no):
    """An .obj index (1-based, negative = relative to the end) to a
    0-based index, or a plain ObjError naming the line."""
    if index > 0:
        resolved = index - 1
    else:
        resolved = count + index
    if resolved < 0 or resolved >= count:
        raise ObjError(
            f"line {line_no}: a face refers to {what} {index}, but the "
            f"file has only {count}")
    return resolved


def _parse_face_ref(token, n_verts, n_uvs, n_normals, line_no):
    """One face corner (v, v/vt, v//vn or v/vt/vn) to a
    (vertex, uv, normal) tuple of 0-based indices, None where the
    file gives nothing."""
    fields = token.split("/")
    if len(fields) > 3 or not fields[0]:
        raise ObjError(f"line {line_no}: bad face corner '{token}'")
    try:
        vi = int(fields[0])
        vti = int(fields[1]) if len(fields) > 1 and fields[1] else None
        vni = int(fields[2]) if len(fields) > 2 and fields[2] else None
    except ValueError:
        raise ObjError(f"line {line_no}: bad face corner '{token}'")
    vi = _resolve(vi, n_verts, "vertex", line_no)
    vti = (_resolve(vti, n_uvs, "texture coordinate", line_no)
           if vti is not None else None)
    vni = (_resolve(vni, n_normals, "normal", line_no)
           if vni is not None else None)
    return (vi, vti, vni)


def parse_obj(path):
    """Read an .obj file into a parsed dict (see the module docstring).
    Raises ObjError with a plain message for a malformed file, and
    FileNotFoundError for a missing one."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    if not lines:
        raise ObjError("the file is empty")
    source = os.path.basename(path)
    folder = os.path.dirname(os.path.abspath(path))
    stem = os.path.splitext(source)[0]
    vertices, texcoords, normals = [], [], []
    materials = {}
    warnings = []
    # Groups: each o/g line starts a new object, except that a group
    # with no faces yet is just renamed — Blender writes `o Name`
    # followed by `g Name`, and that must not make an empty object.
    groups = [{"name": stem, "faces": [], "material": None,
               "materials_seen": []}]
    current = groups[0]

    def start_group(name):
        nonlocal current
        if current["faces"] and name != current["name"]:
            current = {"name": name, "faces": [], "material": None,
                       "materials_seen": []}
            groups.append(current)
        else:
            current["name"] = name

    for line_no, raw in enumerate(lines, 1):
        parts = raw.split()
        if not parts or parts[0].startswith("#"):
            continue
        kind = parts[0]
        if kind == "v":
            try:
                vertices.append((float(parts[1]), float(parts[2]),
                                 float(parts[3])))
            except (IndexError, ValueError):
                raise ObjError(
                    f"line {line_no}: a vertex has a bad number")
        elif kind == "vt":
            try:
                texcoords.append((float(parts[1]), float(parts[2])))
            except (IndexError, ValueError):
                raise ObjError(
                    f"line {line_no}: a texture coordinate has a bad "
                    "number")
        elif kind == "vn":
            try:
                normals.append((float(parts[1]), float(parts[2]),
                                float(parts[3])))
            except (IndexError, ValueError):
                raise ObjError(
                    f"line {line_no}: a normal has a bad number")
        elif kind in ("o", "g"):
            if len(parts) >= 2:
                start_group(parts[1])
        elif kind == "usemtl":
            if len(parts) >= 2:
                name = parts[1]
                if name not in current["materials_seen"]:
                    current["materials_seen"].append(name)
                if current["material"] is None:
                    current["material"] = name
                elif current["material"] != name:
                    warnings.append(
                        f"Object '{current['name']}' uses more than "
                        "one material; only the first was kept.")
        elif kind == "mtllib":
            for name in parts[1:]:
                materials.update(_parse_mtllib(
                    os.path.join(folder, name), warnings))
        elif kind == "f":
            if len(parts) < 4:
                warnings.append(
                    f"line {line_no}: a face has fewer than 3 corners "
                    "and was skipped.")
                continue
            corners = [_parse_face_ref(tok, len(vertices), len(texcoords),
                                       len(normals), line_no)
                       for tok in parts[1:]]
            # Fan triangulation: a quad becomes 2 triangles, an
            # n-gon becomes n-2.
            for i in range(1, len(corners) - 1):
                current["faces"].append(
                    (corners[0], corners[i], corners[i + 1]))
        # Every other line (s, l, p, vp, cstype, ...) is skipped —
        # never fatal.
    objects = [g for g in groups if g["faces"]]
    if not objects:
        raise ObjError("no faces found — there is nothing to import")
    triangles = sum(len(g["faces"]) for g in objects)
    # De-duplicate warnings, keeping first-seen order: a file with
    # fifty multi-material groups warns once per group name, and the
    # same missing mtllib is not repeated per group.
    seen = set()
    unique_warnings = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            unique_warnings.append(w)
    report = {"objects": len(objects), "triangles": triangles,
              "warnings": unique_warnings}
    return {"format": "obj", "source": source, "path": path,
            "objects": objects, "vertices": vertices,
            "texcoords": texcoords, "normals": normals,
            "materials": materials, "warnings": unique_warnings,
            "report": report}


def _build_indexed(group, parsed, scale):
    """Indexed mesh for one group, using its authored normals (every
    corner has a vn). Vertices are keyed by the full v/vt/vn tuple,
    so a hard edge that shares a position and uv still gets its own
    vertex with its own normal."""
    vertices = parsed["vertices"]
    texcoords = parsed["texcoords"]
    normals = parsed["normals"]
    positions, norms, uvs, indices = [], [], [], []
    seen = {}
    for tri in group["faces"]:
        for vi, vti, vni in tri:
            key = (vi, vti, vni)
            idx = seen.get(key)
            if idx is None:
                idx = len(positions) // 3
                seen[key] = idx
                x, y, z = vertices[vi]
                positions.extend((x * scale, y * scale, z * scale))
                norms.extend(normals[vni])
                u, v = texcoords[vti] if vti is not None else (0.0, 0.0)
                uvs.extend((u, v))
            indices.append(idx)
    return {"positions": positions, "normals": norms, "uvs": uvs,
            "indices": indices}


def _build_computed(group, parsed, scale):
    """Indexed mesh for one group whose corners lack vn: build a soup
    with per-face normals, then weld it with the T2 angle rule, so
    smooth surfaces come out smooth and crisp edges stay crisp."""
    vertices = parsed["vertices"]
    texcoords = parsed["texcoords"]
    soup_positions, soup_normals, soup_uvs = [], [], []
    for tri in group["faces"]:
        pts = []
        for vi, vti, _vni in tri:
            x, y, z = vertices[vi]
            pts.append((x * scale, y * scale, z * scale))
        nx, ny, nz = geometry_mod.face_normal(*pts)
        for (vi, vti, _vni), pt in zip(tri, pts):
            soup_positions.extend(pt)
            soup_normals.extend((nx, ny, nz))
            u, v = texcoords[vti] if vti is not None else (0.0, 0.0)
            soup_uvs.extend((u, v))
    soup = {"positions": soup_positions, "normals": soup_normals,
            "uvs": soup_uvs, "tri_count": len(group["faces"])}
    welded = geometry_mod.weld_soup(soup)
    return {"positions": welded["positions"], "normals": welded["normals"],
            "uvs": welded["uvs"], "indices": welded["indices"]}


def build_meshes(parsed, scale=100.0):
    """[{name, mesh, material}] for a parsed .obj: one mesh object per
    group, positions scaled by `scale`. The material is the group's
    first usemtl mapped through the mtllib Kd colors, or the default
    gray when the file names no material."""
    out = []
    for group in parsed["objects"]:
        has_normals = all(vni is not None
                          for tri in group["faces"] for _vi, _vti, vni in tri)
        if has_normals:
            mesh = _build_indexed(group, parsed, scale)
        else:
            mesh = _build_computed(group, parsed, scale)
        mat_name = group["material"]
        color = parsed["materials"].get(mat_name, DEFAULT_COLOR) \
            if mat_name else DEFAULT_COLOR
        out.append({"name": group["name"], "mesh": mesh,
                    "material": {"base_color": color}})
    return out


# --- export (T10) ------------------------------------------------------

# Reblend works in cm; .obj files are usually in meters. Export scales
# cm -> m, the mirror of the import default (100, meters -> cm): an
# export at 0.01 followed by an import at 100 round-trips the
# geometry (within float tolerance).
DEFAULT_EXPORT_SCALE = 0.01

# Shapes that default to smooth shading (T3) — mirrors
# scene.default_smooth_shading. A mesh object always uses its asset
# normals, so the rule only matters for primitives.
_SMOOTH_SHAPES = ("sphere", "cylinder", "cone", "torus")


def _sanitize_name(name):
    """A name safe for an .obj `o` or `usemtl` line. The importer
    reads only the first token of those lines, so spaces and path
    separators become dashes — the same rule as meshio._sanitize, so
    an exported name re-imports unchanged."""
    out = "".join(ch if (ch.isalnum() or ch in "._-") else "-"
                  for ch in str(name))
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip(".-") or "object"


def _hex_to_rgb_fractions(value):
    """'#rrggbb' -> (r, g, b) in 0..1, or the default gray for a bad
    value (the same fallback as scene.hex_to_rgb)."""
    text = str(value).lstrip("#")
    try:
        return tuple(int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return (0.7, 0.7, 0.7)


def _deindex(mesh):
    """An indexed mesh (positions/normals/uvs + indices) to a soup:
    one corner per triangle vertex, in triangle order. The writer
    de-indexes on write (Design 2) — simple, correct, slightly
    larger files; nothing is re-welded."""
    positions, normals, uvs, indices = (mesh["positions"], mesh["normals"],
                                        mesh["uvs"], mesh["indices"])
    soup_pos, soup_nor, soup_uv = [], [], []
    for idx in indices:
        soup_pos.extend(positions[idx * 3:idx * 3 + 3])
        soup_nor.extend(normals[idx * 3:idx * 3 + 3])
        soup_uv.extend(uvs[idx * 2:idx * 2 + 2])
    return {"positions": soup_pos, "normals": soup_nor, "uvs": soup_uv,
            "tri_count": len(indices) // 3}


def _object_soup(obj, base_path):
    """(soup, warning) for one scene object in LOCAL space: the soup
    the viewport would draw — smooth objects through the indexed
    path (T3 averaged normals), flat objects as the raw soup, mesh
    objects de-indexed from their asset. A mesh whose asset cannot
    be loaded returns (None, warning naming the file); the caller
    skips it, never fakes it."""
    name = obj.get("name") or obj.get("id", "object")
    if obj.get("shape") == "mesh":
        mesh_file = obj.get("mesh_file")
        if not mesh_file:
            return None, (f"Mesh object '{name}' has no mesh file, "
                          "so it was skipped.")
        path = mesh_file if os.path.isabs(mesh_file) else (
            os.path.join(base_path, mesh_file) if base_path else None)
        if path is None:
            return None, (f"Mesh file {mesh_file} cannot be found "
                          f"without a saved scene — object '{name}' "
                          "was skipped.")
        try:
            mesh = meshfile_mod.load_mesh(path)
        except FileNotFoundError:
            return None, (f"Mesh file not found: {mesh_file} — object "
                          f"'{name}' was skipped.")
        except (OSError, ValueError) as exc:
            return None, (f"Mesh file could not be read: {mesh_file} "
                          f"({exc}) — object '{name}' was skipped.")
        return _deindex(mesh), None
    smooth = obj.get("smooth_shading",
                     obj.get("shape") in _SMOOTH_SHAPES)
    if smooth:
        indexed = geometry_mod.tessellate_indexed(
            obj.get("shape", "box"), obj.get("params", {}))
        return _deindex(indexed), None
    return geometry_mod.tessellate(obj.get("shape", "box"),
                                   obj.get("params", {})), None


def plan_export(objects, base_path=None):
    """([(obj, soup, material)], report) for an export: resolves each
    object's local-space soup, skips non-geometry objects and
    unloadable meshes with a warning naming them. No files are
    written — the export dialog shows the report, and export_obj
    writes from the same plan."""
    items = []
    warnings = []
    triangles = 0
    for obj in objects:
        name = obj.get("name") or obj.get("id", "object")
        if obj.get("kind", "object") != "object":
            warnings.append(
                f"Object '{name}' is not geometry, so it was skipped.")
            continue
        soup, warning = _object_soup(obj, base_path)
        if warning is not None:
            warnings.append(warning)
            continue
        triangles += soup["tri_count"]
        items.append((obj, soup, obj.get("material") or {}))
    report = {"objects": len(items), "triangles": triangles,
              "warnings": warnings}
    return items, report


def _normal_matrix(model):
    """The 3x3 inverse-transpose of the model's linear part, for
    transforming normals. A pure rotation comes back as itself; a
    non-uniform scale gets the correct (non-rotation) matrix, so the
    exported normals stay perpendicular to the transformed surface.
    The 4x4 inverse's upper 3x3 is the linear inverse, and the
    normal matrix is its transpose."""
    inv = geometry_mod.mat_inverse(model)
    return [[inv[j][i] for j in range(3)] for i in range(3)]


def _fmt(value):
    """A float formatted for the file: repr round-trips exactly, so
    an export at 0.01 followed by an import at 100 lands back on the
    original coordinates (within float tolerance)."""
    return repr(float(value))


def write_obj(items, path, scale=DEFAULT_EXPORT_SCALE):
    """Write planned export items to path (.obj) + the .mtl next to
    it (same stem). Vertices go through the object's world transform
    (position, rotation, scale) and the export scale; normals go
    through the inverse-transpose and are re-normalized. One material
    per distinct base color, named from the color, so objects that
    share a color share one .mtl entry. Raises OSError when the
    files cannot be written."""
    stem = os.path.splitext(path)[0]
    mtl_path = stem + ".mtl"
    mtl_name = os.path.basename(mtl_path)
    materials = {}  # base color hex -> material name
    taken = set()
    for _obj, _soup, material in items:
        color = str(material.get("base_color", DEFAULT_COLOR)).lower()
        if color not in materials:
            base = _sanitize_name("mat_" + color.lstrip("#"))
            name = base
            n = 2
            while name in taken:
                name = f"{base}-{n}"
                n += 1
            taken.add(name)
            materials[color] = name
    obj_lines = ["# Reblend export", f"mtllib {mtl_name}"]
    corner = 0  # running 1-based vertex index across the whole file
    for obj, soup, material in items:
        name = _sanitize_name(obj.get("name") or obj.get("id", "object"))
        color = str(material.get("base_color", DEFAULT_COLOR)).lower()
        obj_lines.append(f"o {name}")
        obj_lines.append(f"usemtl {materials[color]}")
        model = geometry_mod.compose_matrix(
            obj.get("position", [0.0, 0.0, 0.0]),
            obj.get("rotation", [0.0, 0.0, 0.0]),
            obj.get("scale", [1.0, 1.0, 1.0]))
        nmat = _normal_matrix(model)
        pos, nor, uv = soup["positions"], soup["normals"], soup["uvs"]
        corners = len(pos) // 3
        for i in range(corners):
            x, y, z = geometry_mod.apply_matrix(
                model, (pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]))
            obj_lines.append(f"v {_fmt(x * scale)} {_fmt(y * scale)} "
                             f"{_fmt(z * scale)}")
        for i in range(corners):
            obj_lines.append(f"vt {_fmt(uv[i * 2])} {_fmt(uv[i * 2 + 1])}")
        for i in range(corners):
            nx = (nmat[0][0] * nor[i * 3] + nmat[0][1] * nor[i * 3 + 1]
                  + nmat[0][2] * nor[i * 3 + 2])
            ny = (nmat[1][0] * nor[i * 3] + nmat[1][1] * nor[i * 3 + 1]
                  + nmat[1][2] * nor[i * 3 + 2])
            nz = (nmat[2][0] * nor[i * 3] + nmat[2][1] * nor[i * 3 + 1]
                  + nmat[2][2] * nor[i * 3 + 2])
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            if length < 1e-9:
                nx, ny, nz = 0.0, 0.0, 1.0
            else:
                nx, ny, nz = nx / length, ny / length, nz / length
            obj_lines.append(f"vn {_fmt(nx)} {_fmt(ny)} {_fmt(nz)}")
        for t in range(soup["tri_count"]):
            a = corner + t * 3 + 1
            obj_lines.append(f"f {a}/{a}/{a} {a + 1}/{a + 1}/{a + 1} "
                             f"{a + 2}/{a + 2}/{a + 2}")
        corner += corners
    mtl_lines = ["# Reblend export"]
    for color, mat_name in materials.items():
        r, g, b = _hex_to_rgb_fractions(color)
        mtl_lines.append(f"newmtl {mat_name}")
        mtl_lines.append(f"Kd {r:.4f} {g:.4f} {b:.4f}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(obj_lines) + "\n")
    with open(mtl_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(mtl_lines) + "\n")


def export_obj(objects, path, scale=DEFAULT_EXPORT_SCALE, base_path=None):
    """The one .obj export entry point: plan the export (resolve
    geometry, skip what cannot be written, with warnings), write the
    .obj + .mtl pair, and return the report {objects, triangles,
    warnings}. Raises OSError when the files cannot be written."""
    items, report = plan_export(objects, base_path)
    write_obj(items, path, scale)
    return report
