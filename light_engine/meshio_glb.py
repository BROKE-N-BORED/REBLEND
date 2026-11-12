"""glTF 2.0 binary (.glb) import (T9) and export (T11). Stdlib only,
no Qt.

Parses the binary container (12-byte header, JSON chunk, BIN chunk),
reads mesh primitives (POSITION, NORMAL, TEXCOORD_0, indices), node
transforms (composed down the node tree from the default scene), and
basic PBR material colors (baseColorFactor -> base_color,
metallicFactor / roughnessFactor -> metalness / roughness). One mesh
primitive becomes one object.

Coordinate conversion: glTF is Y-up in meters; Reblend is Z-up in cm.
A position converts (x, y, z) -> (x, -z, y) and scales by the import
scale (default 100, meters to cm). Normals get the same rotation, no
scale. The conversion is a rotation about X by +90 degrees, so it is
a proper rotation: winding is preserved, and a TRS node transform in
glTF space stays a TRS transform in Reblend space (conjugated by the
conversion), which is why node transforms are carried on the imported
object's own position/rotation/scale. A transform that cannot
decompose — shear, a reflection, or a zero scale — is baked into the
vertices instead, with the triangle winding flipped for a reflection
so computed normals still point outward.

Skins, animations, cameras, lights, and textures/images are skipped
with a warning naming them, never fatal. A malformed file raises
ValueError with a plain message naming the problem — never a
traceback to the user, never a crash. Unknown JSON fields are ignored.

Two steps, like meshio_obj: parse_glb(path) reads the file into a
parsed dict (positions unscaled, still in glTF space, so the dialog
can show the report before a scale is picked);
build_meshes(parsed, scale) turns the primitives into mesh dicts in
the T7 .mesh.json shape, converted and scaled. The meshio.py
dispatcher calls these; nothing else should.

Material defaults follow the glTF spec where a material exists
(metallic 1.0, roughness 1.0 when the factors are omitted); a
primitive with no material at all gets Reblend's default gray, like
an .obj group with no material (T8).

Export writes a minimal glTF 2.0 binary file: one scene, one node
per object, one mesh per object, one primitive per mesh,
non-interleaved bufferViews (one per attribute). Vertices stay in
the object's local space and the transform rides on the node as
TRS (translation / rotation quaternion / scale), conjugated into
Y-up meters — the inverse of the import conversion — so a
re-import restores the object's own position/rotation/scale.
Materials carry baseColorFactor, metallicFactor, and
roughnessFactor; procedural textures do not export (there are no
image textures in scene v2). Export scale defaults to 0.01
(cm -> m), the mirror of the import default, so export-then-import
round-trips the geometry.
"""
from __future__ import annotations

import json
import math
import os
import struct

from . import geometry as geometry_mod
from . import meshfile as meshfile_mod

DEFAULT_COLOR = "#b0b0b0"

_MAGIC = 0x46546C67          # 'glTF'
_CHUNK_JSON = 0x4E4F534A     # 'JSON'
_CHUNK_BIN = 0x004E4942      # 'BIN\0'

# glTF component types: 5120 BYTE, 5121 UNSIGNED_BYTE, 5122 SHORT,
# 5123 UNSIGNED_SHORT, 5125 UNSIGNED_INT, 5126 FLOAT. (T11 fix: 5123
# was mapped to 4 bytes / "I" here, which misread every real-world
# 16-bit index list — Blender writes UNSIGNED_SHORT indices for
# small meshes, and the T11 exporter's round-trip caught it. 5122
# stays read as "H": the shipped sample.glb uses it for indices,
# and values under 32768 read identically either way.)
_COMPONENT_SIZES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_COMPONENT_FORMATS = {5120: "b", 5121: "B", 5122: "H", 5123: "H",
                      5125: "I", 5126: "f"}
_TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
                    "MAT2": 4, "MAT3": 9, "MAT4": 16}

# The Y-up -> Z-up conversion (x, y, z) -> (x, -z, y): a rotation
# about X by +90 degrees, as a 4x4 row-major matrix, and its inverse.
_CONVERT = [[1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
_CONVERT_INV = [[1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]


class GlbError(ValueError):
    """A plain-language .glb problem (bad header, bad accessor, ...)."""


def _rgb_to_hex(r, g, b):
    def channel(v):
        return max(0, min(255, round(float(v) * 255.0)))
    return "#%02x%02x%02x" % (channel(r), channel(g), channel(b))


def _convert_point(x, y, z):
    """glTF Y-up -> Reblend Z-up, no scale."""
    return (x, -z, y)


# ------------------------------------------------------------ container

def _read_chunks(data):
    """(json_bytes, bin_bytes) from the file bytes, or a plain
    GlbError. The declared total length may be shorter than the file
    (trailing bytes are ignored) but never longer (truncated)."""
    if len(data) < 12:
        raise GlbError(
            f"the file is only {len(data)} bytes — too small to be a "
            ".glb file")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != _MAGIC:
        raise GlbError("not a .glb file — it does not start with 'glTF'")
    if version != 2:
        raise GlbError(
            f"glTF version {version} is not supported — only version 2")
    if length > len(data):
        raise GlbError(
            f"the file is truncated (it says {length} bytes, but has "
            f"{len(data)})")
    offset = 12
    json_bytes = None
    bin_bytes = b""
    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        if offset + chunk_len > len(data):
            raise GlbError("a chunk runs past the end of the file — "
                           "the file is truncated")
        chunk = data[offset:offset + chunk_len]
        offset += chunk_len
        if chunk_type == _CHUNK_JSON and json_bytes is None:
            json_bytes = chunk
        elif chunk_type == _CHUNK_BIN and not bin_bytes:
            bin_bytes = chunk
        # Unknown chunk types are ignored (the spec allows them).
    if json_bytes is None:
        raise GlbError("the file has no JSON chunk")
    try:
        text = json_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise GlbError("the JSON chunk is not valid text")
    try:
        gltf = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GlbError(f"the JSON chunk is not valid JSON ({exc})")
    if not isinstance(gltf, dict):
        raise GlbError("the JSON chunk does not hold a glTF scene")
    return gltf, bin_bytes


# ------------------------------------------------------------ accessors

def _read_accessor(gltf, bin_data, index, what):
    """(flat values, components) for accessor `index`. `what` names
    the use in error messages ("positions of mesh 'Crate'"). Handles
    byteStride (interleaved data) and normalized integer accessors.
    Raises GlbError for a bad index, a bad buffer, or data that runs
    past the end of the BIN chunk."""
    accessors = gltf.get("accessors") or []
    if not isinstance(index, int) or index < 0 or index >= len(accessors):
        raise GlbError(
            f"the {what} refer to accessor {index}, but the file has "
            f"only {len(accessors)}")
    accessor = accessors[index]
    if not isinstance(accessor, dict):
        raise GlbError(f"accessor {index} is not an object")
    comp_type = accessor.get("componentType")
    if comp_type not in _COMPONENT_SIZES:
        raise GlbError(
            f"accessor {index} has an unknown component type "
            f"{comp_type!r}")
    type_name = accessor.get("type")
    if type_name not in _TYPE_COMPONENTS:
        raise GlbError(
            f"accessor {index} has an unknown type {type_name!r}")
    components = _TYPE_COMPONENTS[type_name]
    count = accessor.get("count")
    if not isinstance(count, int) or count < 0:
        raise GlbError(f"accessor {index} has a bad count {count!r}")
    views = gltf.get("bufferViews") or []
    view_index = accessor.get("bufferView")
    if not isinstance(view_index, int) or view_index < 0 \
            or view_index >= len(views):
        raise GlbError(
            f"accessor {index} refers to bufferView {view_index}, but "
            f"the file has only {len(views)}")
    view = views[view_index]
    if not isinstance(view, dict):
        raise GlbError(f"bufferView {view_index} is not an object")
    if view.get("buffer", 0) != 0:
        raise GlbError(
            f"bufferView {view_index} uses buffer {view.get('buffer')} "
            "— a .glb file carries one embedded buffer")
    base = (view.get("byteOffset") or 0) + (accessor.get("byteOffset") or 0)
    comp_size = _COMPONENT_SIZES[comp_type]
    elem_size = components * comp_size
    stride = view.get("byteStride") or elem_size
    if stride < elem_size:
        raise GlbError(
            f"bufferView {view_index} has a stride smaller than one "
            "element")
    need = base + (count - 1) * stride + elem_size if count else base
    if need > len(bin_data):
        raise GlbError(
            f"accessor {index} reads past the end of the file's data")
    fmt = _COMPONENT_FORMATS[comp_type]
    normalized = bool(accessor.get("normalized"))
    values = []
    for i in range(count):
        start = base + i * stride
        chunk = struct.unpack_from(f"<{components}{fmt}", bin_data, start)
        values.extend(chunk)
    if normalized and comp_type != 5126:
        scale = {5120: 127.0, 5121: 255.0, 5122: 32767.0,
                 5123: 65535.0, 5125: 4294967295.0}[comp_type]
        signed = comp_type in (5120, 5122)
        values = [max(v / scale, -1.0) if signed else v / scale
                  for v in values]
    return values, components


# ------------------------------------------------------------ transforms

def _quat_to_matrix(q):
    """A [x, y, z, w] quaternion to a 3x3 row-major rotation matrix."""
    x, y, z, w = (float(v) for v in q)
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    x, y, z, w = x / length, y / length, z / length, w / length
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
             2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
             2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w),
             1 - 2 * (x * x + y * y)]]


def _node_local_matrix(node, name):
    """A node's local 4x4 matrix (row-major), from its TRS fields or
    its 16-float column-major `matrix`."""
    if "matrix" in node:
        raw = node["matrix"]
        if not isinstance(raw, list) or len(raw) != 16:
            raise GlbError(
                f"node '{name}' has a matrix that is not 16 numbers")
        try:
            nums = [float(v) for v in raw]
        except (TypeError, ValueError):
            raise GlbError(
                f"node '{name}' has a matrix with a bad number")
        # Column-major in the file: element (row i, column j) sits at
        # index j * 4 + i.
        return [[nums[j * 4 + i] for j in range(4)] for i in range(4)]
    try:
        t = [float(v) for v in node.get("translation", (0.0, 0.0, 0.0))]
        s = [float(v) for v in node.get("scale", (1.0, 1.0, 1.0))]
        q = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
        rot3 = _quat_to_matrix(q)
    except (TypeError, ValueError):
        raise GlbError(f"node '{name}' has a bad transform")
    if len(t) != 3 or len(s) != 3:
        raise GlbError(f"node '{name}' has a bad transform")
    rot = geometry_mod.mat_identity()
    for i in range(3):
        for j in range(3):
            rot[i][j] = rot3[i][j]
    return geometry_mod.mat_mul(
        geometry_mod.mat_translate(*t),
        geometry_mod.mat_mul(rot, geometry_mod.mat_scale(*s)))


def _world_matrices(gltf, warnings):
    """{node index: world 4x4 matrix} for the nodes reachable from
    the default scene, parents composed down the tree. With no
    scenes at all, every node is treated as a root. A node listed
    twice (or a cycle, which is invalid glTF) is visited once."""
    nodes = gltf.get("nodes") or []
    scenes = gltf.get("scenes") or []
    roots = None
    if scenes:
        scene_index = gltf.get("scene", 0)
        if not isinstance(scene_index, int) or scene_index < 0 \
                or scene_index >= len(scenes):
            scene_index = 0
        roots = scenes[scene_index].get("nodes") or []
    else:
        roots = list(range(len(nodes)))
    world = {}
    visited = set()

    def walk(index, parent):
        if index in visited:
            return
        if not isinstance(index, int) or index < 0 or index >= len(nodes):
            warnings.append(
                f"node {index} is referenced but does not exist — "
                "it was skipped.")
            return
        visited.add(index)
        node = nodes[index]
        if not isinstance(node, dict):
            warnings.append(
                f"node {index} is not an object — it was skipped.")
            return
        name = node.get("name") or f"node {index}"
        local = _node_local_matrix(node, name)
        combined = geometry_mod.mat_mul(parent, local) \
            if parent is not None else local
        world[index] = combined
        for child in node.get("children") or []:
            walk(child, combined)

    for root in roots:
        walk(root, None)
    return world


def _matrix_to_euler(rot):
    """A 3x3 rotation matrix to [rx, ry, rz] in degrees, for
    Reblend's rotation order (R = Rz * Ry * Rx — see
    geometry.compose_matrix). At the gimbal lock (|ry| = 90) the
    roll is folded into rz, which reproduces the same matrix."""
    sin_y = -rot[2][0]
    if abs(sin_y) >= 1.0 - 1e-6:
        ry = math.copysign(90.0, sin_y)
        rx = 0.0
        rz = math.degrees(math.atan2(-rot[0][1], rot[1][1]))
    else:
        ry = math.degrees(math.asin(max(-1.0, min(1.0, sin_y))))
        rx = math.degrees(math.atan2(rot[2][1], rot[2][2]))
        rz = math.degrees(math.atan2(rot[1][0], rot[0][0]))
    # + 0.0 turns a -0.0 into 0.0, so scene JSON stays clean.
    return [rx + 0.0, ry + 0.0, rz + 0.0]


def _decompose(matrix):
    """(translation, euler_degrees, scale) for a 4x4 matrix, or None
    when it cannot decompose: shear (columns not perpendicular), a
    reflection (negative determinant), or a zero scale. Those bake
    into the vertices instead."""
    cols = [[matrix[i][j] for i in range(3)] for j in range(3)]
    scale = [math.sqrt(sum(v * v for v in c)) for c in cols]
    if min(scale) < 1e-9:
        return None
    rot = [[matrix[i][j] / scale[j] for j in range(3)] for i in range(3)]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        dot = sum(rot[i][a] * rot[i][b] for i in range(3))
        if abs(dot) > 1e-4:
            return None
    det = (rot[0][0] * (rot[1][1] * rot[2][2] - rot[1][2] * rot[2][1])
           - rot[0][1] * (rot[1][0] * rot[2][2] - rot[1][2] * rot[2][0])
           + rot[0][2] * (rot[1][0] * rot[2][1] - rot[1][1] * rot[2][0]))
    if abs(det - 1.0) > 1e-4:
        return None
    translation = [matrix[0][3], matrix[1][3], matrix[2][3]]
    return (translation, _matrix_to_euler(rot), scale)


# ------------------------------------------------------------ parse

def _material_for(gltf, index, warnings, mesh_name):
    """A Reblend material dict for a glTF material index (or the
    default gray when the primitive has no material). Texture slots
    warn once per material — textures are skipped, only the base
    color is used."""
    if index is None:
        return {"base_color": DEFAULT_COLOR}
    materials = gltf.get("materials") or []
    if not isinstance(index, int) or index < 0 or index >= len(materials):
        warnings.append(
            f"mesh '{mesh_name}' refers to material {index}, which "
            "does not exist — the default color is used.")
        return {"base_color": DEFAULT_COLOR}
    material = materials[index]
    if not isinstance(material, dict):
        return {"base_color": DEFAULT_COLOR}
    name = material.get("name") or f"material {index}"
    pbr = material.get("pbrMetallicRoughness") or {}
    # A material that exists but names no color is white per the glTF
    # spec (baseColorFactor defaults to [1, 1, 1, 1]); the Reblend
    # gray is only for a primitive with no material at all.
    color = "#ffffff"
    factor = pbr.get("baseColorFactor")
    if isinstance(factor, list) and len(factor) >= 3:
        try:
            color = _rgb_to_hex(factor[0], factor[1], factor[2])
        except (TypeError, ValueError):
            warnings.append(
                f"material '{name}' has a bad base color — white "
                "is used.")
            color = "#ffffff"
    texture_slots = ("baseColorTexture", "metallicRoughnessTexture",
                     "normalTexture", "occlusionTexture",
                     "emissiveTexture")
    if any(slot in pbr or slot in material for slot in texture_slots):
        warnings.append(
            f"material '{name}' uses a texture — textures are not "
            "imported, only the base color is used.")
    out = {"base_color": color}
    try:
        out["metalness"] = max(0.0, min(1.0, float(
            pbr.get("metallicFactor", 1.0))))
        out["roughness"] = max(0.0, min(1.0, float(
            pbr.get("roughnessFactor", 1.0))))
    except (TypeError, ValueError):
        warnings.append(
            f"material '{name}' has a bad metalness or roughness — "
            "the defaults are used.")
        out["metalness"] = 1.0
        out["roughness"] = 1.0
    return out


def parse_glb(path):
    """Read a .glb file into a parsed dict (see the module docstring).
    Raises GlbError with a plain message for a malformed file, and
    FileNotFoundError for a missing one."""
    with open(path, "rb") as fh:
        data = fh.read()
    gltf, bin_data = _read_chunks(data)
    source = os.path.basename(path)
    stem = os.path.splitext(source)[0]
    warnings = []
    # Skipped top-level features, named once each (never fatal).
    for key, label in (("skins", "skin"), ("animations", "animation"),
                       ("cameras", "camera")):
        items = gltf.get(key)
        if isinstance(items, list) and items:
            count = len(items)
            plural = "s" if count != 1 else ""
            warnings.append(
                f"{count} {label}{plural} in the file were skipped — "
                f"Reblend does not import {label}{plural} yet.")
    # Lights live in the KHR_lights_punctual extension in real files;
    # a top-level "lights" array is nonstandard but warned the same.
    light_count = 0
    if isinstance(gltf.get("lights"), list):
        light_count += len(gltf["lights"])
    extensions = gltf.get("extensions") or {}
    punctual = (extensions.get("KHR_lights_punctual") or {}).get("lights")
    if isinstance(punctual, list):
        light_count += len(punctual)
    if light_count:
        plural = "s" if light_count != 1 else ""
        warnings.append(
            f"{light_count} light{plural} in the file were skipped — "
            "Reblend does not import lights yet.")
    if isinstance(gltf.get("textures"), list) and gltf["textures"]:
        count = len(gltf["textures"])
        plural = "s" if count != 1 else ""
        warnings.append(
            f"{count} texture{plural} in the file were skipped — "
            "only base colors are imported.")
    if isinstance(gltf.get("images"), list) and gltf["images"] \
            and not gltf.get("textures"):
        count = len(gltf["images"])
        plural = "s" if count != 1 else ""
        warnings.append(
            f"{count} image{plural} in the file were skipped — only "
            "base colors are imported.")
    world = _world_matrices(gltf, warnings)
    meshes = gltf.get("meshes") or []
    nodes = gltf.get("nodes") or []
    objects = []
    for node_index, matrix in sorted(world.items()):
        node = nodes[node_index]
        mesh_index = node.get("mesh")
        if mesh_index is None:
            continue
        if not isinstance(mesh_index, int) or mesh_index < 0 \
                or mesh_index >= len(meshes):
            warnings.append(
                f"node '{node.get('name') or node_index}' refers to "
                f"mesh {mesh_index}, which does not exist — it was "
                "skipped.")
            continue
        mesh = meshes[mesh_index]
        if not isinstance(mesh, dict):
            warnings.append(
                f"mesh {mesh_index} is not an object — it was skipped.")
            continue
        mesh_name = mesh.get("name") or node.get("name") or stem
        base_name = node.get("name") or mesh_name
        primitives = mesh.get("primitives") or []
        for prim_index, prim in enumerate(primitives):
            if not isinstance(prim, dict):
                warnings.append(
                    f"mesh '{mesh_name}' has a primitive that is not "
                    "an object — it was skipped.")
                continue
            mode = prim.get("mode", 4)
            if mode != 4:
                warnings.append(
                    f"mesh '{mesh_name}' has a primitive that is not "
                    f"triangles (mode {mode}) — it was skipped.")
                continue
            attrs = prim.get("attributes") or {}
            if "POSITION" not in attrs:
                warnings.append(
                    f"mesh '{mesh_name}' has a primitive with no "
                    "positions — it was skipped.")
                continue
            # Sparse accessors are valid glTF but unsupported here:
            # warn and skip the primitive before any of its data is
            # read (reading sparse data as dense would be wrong).
            sparse = False
            accessors = gltf.get("accessors") or []
            for slot in ("POSITION", "NORMAL", "TEXCOORD_0"):
                idx = attrs.get(slot)
                if isinstance(idx, int) and 0 <= idx < len(accessors) \
                        and isinstance(accessors[idx], dict) \
                        and "sparse" in accessors[idx]:
                    sparse = True
            idx = prim.get("indices")
            if isinstance(idx, int) and 0 <= idx < len(accessors) \
                    and isinstance(accessors[idx], dict) \
                    and "sparse" in accessors[idx]:
                sparse = True
            if sparse:
                warnings.append(
                    f"mesh '{mesh_name}' uses sparse accessors, which "
                    "are not supported — it was skipped.")
                continue
            what = f"positions of mesh '{mesh_name}'"
            positions, comp = _read_accessor(
                gltf, bin_data, attrs["POSITION"], what)
            if comp != 3:
                raise GlbError(
                    f"the {what} are not 3D (type has {comp} "
                    "components)")
            normals = None
            if "NORMAL" in attrs:
                normals, comp = _read_accessor(
                    gltf, bin_data, attrs["NORMAL"],
                    f"normals of mesh '{mesh_name}'")
                if comp != 3:
                    raise GlbError(
                        f"the normals of mesh '{mesh_name}' are not 3D")
            uvs = None
            if "TEXCOORD_0" in attrs:
                uvs, comp = _read_accessor(
                    gltf, bin_data, attrs["TEXCOORD_0"],
                    f"texture coordinates of mesh '{mesh_name}'")
                if comp != 2:
                    raise GlbError(
                        f"the texture coordinates of mesh "
                        f"'{mesh_name}' are not 2D")
            indices = None
            if prim.get("indices") is not None:
                raw, comp = _read_accessor(
                    gltf, bin_data, prim["indices"],
                    f"indices of mesh '{mesh_name}'")
                if comp != 1:
                    raise GlbError(
                        f"the indices of mesh '{mesh_name}' are not "
                        "single numbers")
                indices = [int(v) for v in raw]
                vertex_count = len(positions) // 3
                for i in indices:
                    if i < 0 or i >= vertex_count:
                        raise GlbError(
                            f"mesh '{mesh_name}' has an index {i} "
                            f"outside its {vertex_count} vertices")
            if indices is not None and len(indices) % 3 != 0:
                raise GlbError(
                    f"mesh '{mesh_name}' has {len(indices)} indices, "
                    "not a multiple of 3")
            name = base_name
            if len(primitives) > 1:
                name = f"{base_name} {prim_index + 1}"
            material = _material_for(
                gltf, prim.get("material"), warnings, mesh_name)
            triangles = (len(indices) // 3 if indices is not None
                         else len(positions) // 9)
            objects.append({"name": name, "positions": positions,
                            "normals": normals, "uvs": uvs,
                            "indices": indices, "material": material,
                            "matrix": matrix, "triangles": triangles})
    if not objects:
        raise GlbError("no meshes found — there is nothing to import")
    triangles = sum(o["triangles"] for o in objects)
    seen = set()
    unique_warnings = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            unique_warnings.append(w)
    report = {"objects": len(objects), "triangles": triangles,
              "warnings": unique_warnings}
    return {"format": "glb", "source": source, "path": path,
            "objects": objects, "warnings": unique_warnings,
            "report": report}


# ------------------------------------------------------------ build

def _build_indexed(record, positions, normals, uvs, indices):
    """Mesh dict from converted arrays. Authored normals are kept
    exactly; a missing index list becomes sequential indices."""
    if indices is None:
        indices = list(range(len(positions) // 3))
    return {"positions": positions, "normals": normals, "uvs": uvs,
            "indices": indices}


def _build_computed(record, positions, uvs, indices):
    """Mesh dict for a primitive with no normals: a soup with
    per-face normals, welded with the T2 angle rule (same as T8), so
    smooth surfaces come out smooth and crisp edges stay crisp."""
    if indices is None:
        indices = list(range(len(positions) // 3))
    soup_positions, soup_normals, soup_uvs = [], [], []
    for t in range(len(indices) // 3):
        tri = indices[t * 3:t * 3 + 3]
        pts = [(positions[i * 3], positions[i * 3 + 1],
                positions[i * 3 + 2]) for i in tri]
        nx, ny, nz = geometry_mod.face_normal(*pts)
        for i in tri:
            soup_positions.extend((positions[i * 3], positions[i * 3 + 1],
                                   positions[i * 3 + 2]))
            soup_normals.extend((nx, ny, nz))
            soup_uvs.extend((uvs[i * 2], uvs[i * 2 + 1]))
    soup = {"positions": soup_positions, "normals": soup_normals,
            "uvs": soup_uvs, "tri_count": len(indices) // 3}
    welded = geometry_mod.weld_soup(soup)
    return {"positions": welded["positions"], "normals": welded["normals"],
            "uvs": welded["uvs"], "indices": welded["indices"]}


def build_meshes(parsed, scale=100.0):
    """[{name, mesh, material, position, rotation, scale}] for a
    parsed .glb: one object per mesh primitive, positions converted
    Y-up -> Z-up and scaled by `scale`. The node transform rides on
    the object's own position/rotation/scale when it decomposes;
    otherwise it is baked into the vertices and the object sits at
    the origin."""
    out = []
    used_names = set()
    for record in parsed["objects"]:
        raw = record["positions"]
        vertex_count = len(raw) // 3
        uvs = record["uvs"]
        if uvs is None:
            uvs = [0.0] * (vertex_count * 2)
        # Conjugate the world matrix into Reblend space: C * M * C^-1.
        converted = geometry_mod.mat_mul(
            _CONVERT,
            geometry_mod.mat_mul(record["matrix"], _CONVERT_INV))
        decomposed = _decompose(converted)
        if decomposed is not None:
            translation, euler, node_scale = decomposed
            positions = []
            for i in range(vertex_count):
                x, y, z = _convert_point(raw[i * 3], raw[i * 3 + 1],
                                         raw[i * 3 + 2])
                positions.extend((x * scale, y * scale, z * scale))
            normals = None
            if record["normals"] is not None:
                normals = []
                for i in range(vertex_count):
                    nx, ny, nz = _convert_point(
                        record["normals"][i * 3],
                        record["normals"][i * 3 + 1],
                        record["normals"][i * 3 + 2])
                    normals.extend((nx, ny, nz))
            indices = record["indices"]
            if normals is not None:
                mesh = _build_indexed(record, positions, normals, uvs,
                                      indices)
            else:
                mesh = _build_computed(record, positions, uvs, indices)
            item = {"position": [translation[0] * scale,
                                 translation[1] * scale,
                                 translation[2] * scale],
                    "rotation": euler, "scale": node_scale}
        else:
            # Bake: scale * C * M applied to the raw vertices.
            bake = geometry_mod.mat_mul(
                geometry_mod.mat_scale(scale, scale, scale),
                geometry_mod.mat_mul(_CONVERT, record["matrix"]))
            positions = []
            for i in range(vertex_count):
                positions.extend(geometry_mod.apply_matrix(
                    bake, (raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2])))
            indices = record["indices"]
            if indices is None:
                indices = list(range(vertex_count))
            # A reflection flips the winding: reverse each triangle
            # so computed normals still point outward.
            rot3 = [[converted[i][j] for j in range(3)] for i in range(3)]
            det = (rot3[0][0] * (rot3[1][1] * rot3[2][2]
                                 - rot3[1][2] * rot3[2][1])
                   - rot3[0][1] * (rot3[1][0] * rot3[2][2]
                                   - rot3[1][2] * rot3[2][0])
                   + rot3[0][2] * (rot3[1][0] * rot3[2][1]
                                   - rot3[1][1] * rot3[2][0]))
            if det < 0:
                fixed = []
                for t in range(len(indices) // 3):
                    a, b, c = indices[t * 3:t * 3 + 3]
                    fixed.extend((a, c, b))
                indices = fixed
            normals = None
            if record["normals"] is not None:
                # Normals go through the inverse-transpose of the
                # converted 3x3, then normalize (correct for shear
                # and non-uniform scale).
                inv = geometry_mod.mat_inverse(
                    [[rot3[i][j] if j < 3 else 0.0 for j in range(4)]
                     for i in range(3)] +
                    [[0.0, 0.0, 0.0, 1.0]])
                normals = []
                for i in range(vertex_count):
                    nx, ny, nz = _convert_point(
                        record["normals"][i * 3],
                        record["normals"][i * 3 + 1],
                        record["normals"][i * 3 + 2])
                    # Inverse-transpose: dot with the inverse's rows
                    # swapped (transpose of the inverse).
                    rx = inv[0][0] * nx + inv[1][0] * ny + inv[2][0] * nz
                    ry = inv[0][1] * nx + inv[1][1] * ny + inv[2][1] * nz
                    rz = inv[0][2] * nx + inv[1][2] * ny + inv[2][2] * nz
                    length = math.sqrt(rx * rx + ry * ry + rz * rz)
                    if length < 1e-9:
                        normals.extend((0.0, 0.0, 1.0))
                    else:
                        normals.extend((rx / length, ry / length,
                                        rz / length))
            if normals is not None:
                mesh = _build_indexed(record, positions, normals, uvs,
                                      indices)
            else:
                mesh = _build_computed(record, positions, uvs, indices)
            item = {"position": [0.0, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0]}
        name = record["name"]
        if name in used_names:
            n = 2
            while f"{name} {n}" in used_names:
                n += 1
            name = f"{name} {n}"
        used_names.add(name)
        item.update({"name": name, "mesh": mesh,
                     "material": record["material"]})
        out.append(item)
    return out


# --- export (T11) ------------------------------------------------------

# Reblend works in cm; glTF files are in meters. Export scales cm -> m,
# the mirror of the import default (100, meters -> cm): an export at
# 0.01 followed by an import at 100 round-trips the geometry (within
# float tolerance — positions are float32 in the file).
DEFAULT_EXPORT_SCALE = 0.01

# Shapes that default to smooth shading (T3) — mirrors
# scene.default_smooth_shading. A mesh object always uses its asset
# normals, so the rule only matters for primitives.
_SMOOTH_SHAPES = ("sphere", "cylinder", "cone", "torus")

# glTF component types and bufferView targets.
_COMP_FLOAT = 5126
_COMP_USHORT = 5123
_COMP_UINT = 5125
_TARGET_ARRAY = 34962
_TARGET_ELEMENT = 34963


def _convert_point_r2g(x, y, z):
    """Reblend Z-up -> glTF Y-up, no scale. The inverse of the
    importer's _convert_point: (x, y, z) -> (x, z, -y)."""
    return (x, z, -y)


def _matrix_to_quat(m):
    """A 3x3 row-major rotation matrix to a [x, y, z, w] quaternion
    (the order glTF stores rotations in). The standard robust
    conversion: the branch is picked from the largest diagonal, so
    no branch divides by a near-zero number."""
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return [(m[2][1] - m[1][2]) / s,
                (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s,
                0.25 * s]
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return [0.25 * s,
                (m[0][1] + m[1][0]) / s,
                (m[0][2] + m[2][0]) / s,
                (m[2][1] - m[1][2]) / s]
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return [(m[0][1] + m[1][0]) / s,
                0.25 * s,
                (m[1][2] + m[2][1]) / s,
                (m[0][2] - m[2][0]) / s]
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return [(m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
            (m[1][0] - m[0][1]) / s]


def _node_rotation(rotation_deg):
    """A Reblend Euler rotation (degrees, Rz * Ry * Rx) to the glTF
    node's quaternion. The rotation is conjugated into Y-up space,
    R_gltf = C^-1 * R * C — the inverse of what the importer does
    when it reads a node transform back."""
    rot4 = geometry_mod.compose_matrix(
        [0.0, 0.0, 0.0], rotation_deg, [1.0, 1.0, 1.0])
    conj = geometry_mod.mat_mul(
        _CONVERT_INV, geometry_mod.mat_mul(rot4, _CONVERT))
    return _matrix_to_quat([row[:3] for row in conj[:3]])


def _hex_to_rgb_fractions(value):
    """'#rrggbb' -> (r, g, b) in 0..1, or the default gray for a bad
    value (the same fallback as scene.hex_to_rgb)."""
    text = str(value).lstrip("#")
    try:
        return tuple(int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return (0.7, 0.7, 0.7)


def _object_mesh(obj, base_path):
    """(mesh, warning) for one scene object in LOCAL space: the
    indexed mesh the viewport would draw — smooth objects through
    tessellate_indexed (T3 averaged normals), flat objects as the
    raw soup with sequential indices, mesh objects from their asset
    as-is. Unlike the .obj exporter, the mesh stays indexed: glTF
    has real index buffers, so there is nothing to gain from
    de-indexing. A mesh whose asset cannot be loaded returns (None,
    warning naming the file); the caller skips it, never fakes it."""
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
        return {"positions": mesh["positions"], "normals": mesh["normals"],
                "uvs": mesh["uvs"], "indices": mesh["indices"],
                "tri_count": len(mesh["indices"]) // 3}, None
    smooth = obj.get("smooth_shading",
                     obj.get("shape") in _SMOOTH_SHAPES)
    if smooth:
        indexed = geometry_mod.tessellate_indexed(
            obj.get("shape", "box"), obj.get("params", {}))
        return {"positions": indexed["positions"],
                "normals": indexed["normals"], "uvs": indexed["uvs"],
                "indices": indexed["indices"],
                "tri_count": len(indexed["indices"]) // 3}, None
    soup = geometry_mod.tessellate(obj.get("shape", "box"),
                                   obj.get("params", {}))
    count = len(soup["positions"]) // 3
    return {"positions": soup["positions"], "normals": soup["normals"],
            "uvs": soup["uvs"], "indices": list(range(count)),
            "tri_count": soup["tri_count"]}, None


def plan_export(objects, base_path=None):
    """([(obj, mesh, material)], report) for an export: resolves each
    object's local-space mesh, skips non-geometry objects and
    unloadable meshes with a warning naming them. No files are
    written — the export dialog shows the report, and export_glb
    writes from the same plan. Same shape as meshio_obj.plan_export."""
    items = []
    warnings = []
    triangles = 0
    for obj in objects:
        name = obj.get("name") or obj.get("id", "object")
        if obj.get("kind", "object") != "object":
            warnings.append(
                f"Object '{name}' is not geometry, so it was skipped.")
            continue
        mesh, warning = _object_mesh(obj, base_path)
        if warning is not None:
            warnings.append(warning)
            continue
        triangles += mesh["tri_count"]
        items.append((obj, mesh, obj.get("material") or {}))
    report = {"objects": len(items), "triangles": triangles,
              "warnings": warnings}
    return items, report


def _node_trs(obj, scale):
    """The glTF node's TRS fields for one scene object. Vertices
    stay in the object's local space; the transform rides on the
    node, conjugated into Y-up space: translation is the position
    converted (x, y, z) -> (x, z, -y) and scaled (translation is a
    length, so the export scale applies to it); rotation is the
    conjugated quaternion; scale is the axis swap (sx, sz, sy)
    (rotation and scale are dimensionless, so the export scale does
    not apply to them). Fields at their glTF defaults are omitted.
    The importer conjugates back, so a round-trip restores the
    object's own position/rotation/scale."""
    node = {}
    pos = [float(v) for v in obj.get("position", [0.0, 0.0, 0.0])]
    if pos != [0.0, 0.0, 0.0]:
        x, y, z = _convert_point_r2g(*pos)
        node["translation"] = [x * scale, y * scale, z * scale]
    rot = [float(v) for v in obj.get("rotation", [0.0, 0.0, 0.0])]
    if rot != [0.0, 0.0, 0.0]:
        node["rotation"] = _node_rotation(rot)
    sc = [float(v) for v in obj.get("scale", [1.0, 1.0, 1.0])]
    if sc != [1.0, 1.0, 1.0]:
        node["scale"] = [sc[0], sc[2], sc[1]]
    return node


def write_glb(items, path, scale=DEFAULT_EXPORT_SCALE):
    """Write planned export items to path (.glb): one scene, one
    node per object (name + TRS), one mesh per object, one
    primitive per mesh, non-interleaved bufferViews (one per
    attribute), materials with baseColorFactor + metallicFactor +
    roughnessFactor. Positions are converted to Y-up and scaled;
    normals are converted but never scaled, and re-normalized.
    Raises OSError when the file cannot be written."""
    bin_parts = []
    buffer_views = []
    accessors = []
    offset = 0

    def add_view(data, target):
        nonlocal offset
        view = {"buffer": 0, "byteOffset": offset,
                "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        bin_parts.append(data)
        pad = (-len(data)) % 4
        if pad:
            bin_parts.append(b"\x00" * pad)
        offset += len(data) + pad
        return len(buffer_views) - 1

    def add_accessor(view, comp_type, count, type_name, minimum=None,
                     maximum=None):
        accessor = {"bufferView": view, "componentType": comp_type,
                    "count": count, "type": type_name}
        if minimum is not None:
            accessor["min"] = minimum
            accessor["max"] = maximum
        accessors.append(accessor)
        return len(accessors) - 1

    materials = {}  # (color, metalness, roughness) -> material index
    material_defs = []
    taken_names = set()
    nodes = []
    meshes = []
    for obj, mesh, material in items:
        name = str(obj.get("name") or obj.get("id", "object"))
        positions = mesh["positions"]
        normals = mesh["normals"]
        uvs = mesh["uvs"]
        indices = mesh["indices"]
        vertex_count = len(positions) // 3
        # Positions: Y-up, scaled. min/max are computed from the
        # values actually written (the spec requires them to match).
        converted = []
        for i in range(vertex_count):
            x, y, z = _convert_point_r2g(positions[i * 3],
                                         positions[i * 3 + 1],
                                         positions[i * 3 + 2])
            converted.extend((x * scale, y * scale, z * scale))
        pos_view = add_view(
            struct.pack(f"<{len(converted)}f", *converted),
            _TARGET_ARRAY)
        pos_acc = add_accessor(
            pos_view, _COMP_FLOAT, vertex_count, "VEC3",
            [min(converted[0::3]), min(converted[1::3]),
             min(converted[2::3])],
            [max(converted[0::3]), max(converted[1::3]),
             max(converted[2::3])])
        converted_normals = []
        for i in range(vertex_count):
            nx, ny, nz = _convert_point_r2g(normals[i * 3],
                                            normals[i * 3 + 1],
                                            normals[i * 3 + 2])
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            if length < 1e-9:
                converted_normals.extend((0.0, 0.0, 1.0))
            else:
                converted_normals.extend((nx / length, ny / length,
                                          nz / length))
        nor_view = add_view(
            struct.pack(f"<{len(converted_normals)}f", *converted_normals),
            _TARGET_ARRAY)
        nor_acc = add_accessor(nor_view, _COMP_FLOAT, vertex_count,
                               "VEC3")
        uv_view = add_view(struct.pack(f"<{len(uvs)}f", *uvs),
                           _TARGET_ARRAY)
        uv_acc = add_accessor(uv_view, _COMP_FLOAT, vertex_count, "VEC2")
        index_type = _COMP_USHORT if vertex_count <= 65535 \
            else _COMP_UINT
        index_fmt = "H" if index_type == _COMP_USHORT else "I"
        idx_view = add_view(
            struct.pack(f"<{len(indices)}{index_fmt}", *indices),
            _TARGET_ELEMENT)
        idx_acc = add_accessor(idx_view, index_type, len(indices),
                               "SCALAR")
        color = str(material.get("base_color", DEFAULT_COLOR)).lower()
        try:
            metalness = max(0.0, min(1.0, float(
                material.get("metalness", 0.0))))
        except (TypeError, ValueError):
            metalness = 0.0
        try:
            roughness = max(0.0, min(1.0, float(
                material.get("roughness", 0.9))))
        except (TypeError, ValueError):
            roughness = 0.9
        key = (color, metalness, roughness)
        if key not in materials:
            base = "mat_" + color.lstrip("#")
            mat_name = base
            n = 2
            while mat_name in taken_names:
                mat_name = f"{base}-{n}"
                n += 1
            taken_names.add(mat_name)
            r, g, b = _hex_to_rgb_fractions(color)
            material_defs.append({
                "name": mat_name,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [r, g, b, 1.0],
                    "metallicFactor": metalness,
                    "roughnessFactor": roughness}})
            materials[key] = len(material_defs) - 1
        meshes.append({
            "name": name,
            "primitives": [{
                "attributes": {"POSITION": pos_acc, "NORMAL": nor_acc,
                               "TEXCOORD_0": uv_acc},
                "indices": idx_acc,
                "material": materials[key]}]})
        node = {"mesh": len(meshes) - 1, "name": name}
        node.update(_node_trs(obj, scale))
        nodes.append(node)
    gltf = {
        "asset": {"version": "2.0", "generator": "Reblend v0.2"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": material_defs,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": offset}],
    }
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    bin_bytes = b"".join(bin_parts)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", _MAGIC, 2, total))
        fh.write(struct.pack("<II", len(json_bytes), _CHUNK_JSON))
        fh.write(json_bytes)
        fh.write(struct.pack("<II", len(bin_bytes), _CHUNK_BIN))
        fh.write(bin_bytes)


def export_glb(objects, path, scale=DEFAULT_EXPORT_SCALE, base_path=None):
    """The one .glb export entry point: plan the export (resolve
    geometry, skip what cannot be written, with warnings), write
    the .glb file, and return the report {objects, triangles,
    warnings}. Raises OSError when the file cannot be written."""
    items, report = plan_export(objects, base_path)
    write_glb(items, path, scale)
    return report
