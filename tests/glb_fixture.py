"""Build small valid glTF 2.0 .glb files for the T9 tests and the
shipped sample (struct + json, stdlib only).

A .glb is a 12-byte header (magic 'glTF', version 2, total length),
a JSON chunk, and a BIN chunk — each chunk length-prefixed and
4-byte aligned (JSON padded with spaces, BIN with zero bytes).
GlbBuilder collects buffer data and JSON pieces and writes a file a
real glTF parser accepts: accessors carry min/max (POSITION requires
them), bufferViews are 4-byte aligned, and indices use UNSIGNED_SHORT
unless they need UNSIGNED_INT.
"""
import json
import struct

F32 = 5126
U16 = 5122
U32 = 5125

_TYPE_NAMES = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}
# 5123 is UNSIGNED_SHORT, the spec type for 16-bit indices (the
# T11 regression test builds a file with it). U16 stays 5122: the
# T9 tests build their files with it and the importer reads it.
_FORMATS = {F32: ("f", 4), U16: ("H", 2), U32: ("I", 4),
            5123: ("H", 2)}


def box_verts(half=0.5):
    """A box centered on the origin (glTF Y-up), per-face vertices
    with authored normals and uvs: 24 verts, 36 indices, 12 tris.
    Each face's corners are ordered so (B-A)x(C-A) is the outward
    normal — the faces are CCW from outside, as glTF requires."""
    h = half
    # (normal, u, v) per face, with u x v == normal.
    faces = [
        ((1, 0, 0), (0, 0, -1), (0, 1, 0)),    # +X
        ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),    # -X
        ((0, 1, 0), (1, 0, 0), (0, 0, -1)),    # +Y top
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),    # -Y bottom
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),     # +Z
        ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),   # -Z
    ]
    positions, normals, uvs, indices = [], [], [], []
    face_uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for n, u, v in faces:
        base = len(positions)
        center = (n[0] * h, n[1] * h, n[2] * h)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            positions.append((center[0] + su * u[0] * h + sv * v[0] * h,
                              center[1] + su * u[1] * h + sv * v[1] * h,
                              center[2] + su * u[2] * h + sv * v[2] * h))
            normals.append(n)
        uvs.extend(face_uvs)
        indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))
    return positions, normals, uvs, indices


def pyramid_verts(half=0.5, height=1.0):
    """A square pyramid, base at y=0, apex at y=height: 5 verts,
    18 indices, 6 tris (2 base + 4 sides). No normals or uvs — the
    sample uses this to exercise computed normals and zero uvs."""
    s, apex = half, (0.0, height, 0.0)
    positions = [(-s, 0.0, -s), (s, 0.0, -s), (s, 0.0, s), (-s, 0.0, s),
                 apex]
    indices = [0, 1, 2, 0, 2, 3,   # base (normal -Y)
               3, 2, 4, 2, 1, 4, 1, 0, 4, 0, 3, 4]  # sides
    return positions, indices


def quat_y(degrees):
    """Quaternion [x, y, z, w] for a rotation about the Y axis."""
    import math
    half = math.radians(degrees) / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def quat_x(degrees):
    """Quaternion [x, y, z, w] for a rotation about the X axis."""
    import math
    half = math.radians(degrees) / 2.0
    return [math.sin(half), 0.0, 0.0, math.cos(half)]


def quat_z(degrees):
    """Quaternion [x, y, z, w] for a rotation about the Z axis."""
    import math
    half = math.radians(degrees) / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


class GlbBuilder:
    """Collects one .glb file. Accessors are written as they are
    added; meshes, nodes, materials, and the scene are JSON dicts
    assembled at build time."""

    def __init__(self):
        self._bin = bytearray()
        self.buffer_views = []
        self.accessors = []
        self.meshes = []
        self.nodes = []
        self.materials = []
        self.scenes = []
        self.scene = 0
        self.extras = {}

    def add_accessor(self, values, components, comp_type=F32):
        """Append flat `values` to the BIN chunk and return the new
        accessor index. `components` is 1/2/3/4 (SCALAR..VEC4)."""
        fmt, size = _FORMATS[comp_type]
        count = len(values) // components
        while len(self._bin) % 4:
            self._bin.append(0)
        offset = len(self._bin)
        self._bin.extend(struct.pack(f"<{len(values)}{fmt}", *values))
        view_index = len(self.buffer_views)
        self.buffer_views.append({"buffer": 0, "byteOffset": offset,
                                  "byteLength": len(values) * size})
        accessor = {"bufferView": view_index, "componentType": comp_type,
                    "count": count, "type": _TYPE_NAMES[components]}
        if count:
            accessor["min"] = [min(values[i::components])
                               for i in range(components)]
            accessor["max"] = [max(values[i::components])
                               for i in range(components)]
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def add_vec3(self, triples):
        flat = [c for t in triples for c in t]
        return self.add_accessor(flat, 3, F32)

    def add_vec2(self, pairs):
        flat = [c for t in pairs for c in t]
        return self.add_accessor(flat, 2, F32)

    def add_indices(self, indices):
        comp_type = U32 if max(indices) >= 65536 else U16
        return self.add_accessor(list(indices), 1, comp_type)

    def add_interleaved(self, arrays):
        """One bufferView holding several accessors interleaved
        (the layout real exporters use for POSITION + NORMAL):
        arrays is [(values, components, comp_type), ...], one entry
        per accessor, all with the same element count. Returns the
        list of new accessor indices, in order."""
        count = len(arrays[0][0]) // arrays[0][1]
        stride = sum(_FORMATS[ct][1] * comp for _v, comp, ct in arrays)
        while len(self._bin) % 4:
            self._bin.append(0)
        offset = len(self._bin)
        for i in range(count):
            for values, comp, ct in arrays:
                fmt = _FORMATS[ct][0]
                chunk = values[i * comp:(i + 1) * comp]
                self._bin.extend(struct.pack(f"<{comp}{fmt}", *chunk))
        view_index = len(self.buffer_views)
        self.buffer_views.append({"buffer": 0, "byteOffset": offset,
                                  "byteLength": count * stride,
                                  "byteStride": stride})
        indices = []
        running = 0
        for values, comp, ct in arrays:
            accessor = {"bufferView": view_index,
                        "byteOffset": running,
                        "componentType": ct, "count": count,
                        "type": _TYPE_NAMES[comp]}
            accessor["min"] = [min(values[j::comp]) for j in range(comp)]
            accessor["max"] = [max(values[j::comp]) for j in range(comp)]
            self.accessors.append(accessor)
            indices.append(len(self.accessors) - 1)
            running += _FORMATS[ct][1] * comp
        return indices

    def add_material(self, name=None, base_color=None, metallic=None,
                     roughness=None, with_texture=False):
        """A PBR material. base_color is [r, g, b, a]; metallic and
        roughness are omitted when None (the glTF defaults, 1.0,
        then apply). with_texture adds a baseColorTexture slot plus
        a textures/images stub, for the skip warning."""
        pbr = {}
        if base_color is not None:
            pbr["baseColorFactor"] = list(base_color)
        if metallic is not None:
            pbr["metallicFactor"] = metallic
        if roughness is not None:
            pbr["roughnessFactor"] = roughness
        if with_texture:
            pbr["baseColorTexture"] = {"index": 0}
            self.extras["textures"] = [{"source": 0}]
            self.extras["images"] = [{"uri": "texture.png"}]
        material = {"pbrMetallicRoughness": pbr}
        if name is not None:
            material["name"] = name
        self.materials.append(material)
        return len(self.materials) - 1

    def add_mesh(self, name, primitives):
        """primitives: [{attributes: {POSITION: idx, ...},
        indices: idx or None, material: idx or None, mode: int}]."""
        mesh = {"primitives": []}
        if name is not None:
            mesh["name"] = name
        for prim in primitives:
            out = {"attributes": dict(prim["attributes"])}
            if prim.get("indices") is not None:
                out["indices"] = prim["indices"]
            if prim.get("material") is not None:
                out["material"] = prim["material"]
            if prim.get("mode") is not None:
                out["mode"] = prim["mode"]
            mesh["primitives"].append(out)
        self.meshes.append(mesh)
        return len(self.meshes) - 1

    def add_node(self, name=None, mesh=None, translation=None,
                 rotation=None, scale=None, matrix=None, children=None):
        node = {}
        if name is not None:
            node["name"] = name
        if mesh is not None:
            node["mesh"] = mesh
        if matrix is not None:
            node["matrix"] = [float(v) for v in matrix]
        else:
            if translation is not None:
                node["translation"] = [float(v) for v in translation]
            if rotation is not None:
                node["rotation"] = [float(v) for v in rotation]
            if scale is not None:
                node["scale"] = [float(v) for v in scale]
        if children:
            node["children"] = list(children)
        self.nodes.append(node)
        return len(self.nodes) - 1

    def set_scene(self, node_indices, name=None):
        scene = {"nodes": list(node_indices)}
        if name is not None:
            scene["name"] = name
        self.scenes = [scene]
        self.scene = 0

    def build(self):
        """The .glb file as bytes."""
        gltf = {"asset": {"version": "2.0", "generator": "reblend-test"}}
        if self.scenes:
            gltf["scene"] = self.scene
            gltf["scenes"] = self.scenes
        if self.nodes:
            gltf["nodes"] = self.nodes
        if self.meshes:
            gltf["meshes"] = self.meshes
        if self.materials:
            gltf["materials"] = self.materials
        if self.accessors:
            gltf["accessors"] = self.accessors
        if self.buffer_views:
            gltf["bufferViews"] = self.buffer_views
        if self._bin:
            gltf["buffers"] = [{"byteLength": len(self._bin)}]
        gltf.update(self.extras)
        json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
        bin_bytes = bytes(self._bin)
        bin_bytes += b"\x00" * ((4 - len(bin_bytes) % 4) % 4)
        total = 12 + 8 + len(json_bytes)
        if bin_bytes:
            total += 8 + len(bin_bytes)
        out = bytearray()
        out.extend(struct.pack("<III", 0x46546C67, 2, total))
        out.extend(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        out.extend(json_bytes)
        if bin_bytes:
            out.extend(struct.pack("<II", len(bin_bytes), 0x004E4942))
            out.extend(bin_bytes)
        return bytes(out)

    def write(self, path):
        with open(path, "wb") as fh:
            fh.write(self.build())


def build_sample():
    """The shipped sample: a red crate (box, authored normals +
    uvs, 12 tris) under a group node — translated and rotated 30°
    about Y — and a teal pyramid (no normals, no uvs, 6 tris).
    18 triangles total, counted from the indices."""
    builder = GlbBuilder()
    red = builder.add_material("CrateMat", base_color=[0.8, 0.2, 0.15, 1.0],
                               metallic=0.0, roughness=0.8)
    teal = builder.add_material("PyramidMat",
                                base_color=[0.1, 0.6, 0.55, 1.0],
                                metallic=0.1, roughness=0.4)
    pos, nor, uv, idx = box_verts(0.5)
    crate = builder.add_mesh("Crate", [{
        "attributes": {"POSITION": builder.add_vec3(pos),
                       "NORMAL": builder.add_vec3(nor),
                       "TEXCOORD_0": builder.add_vec2(uv)},
        "indices": builder.add_indices(idx),
        "material": red,
    }])
    ppos, pidx = pyramid_verts(0.5, 1.0)
    pyramid = builder.add_mesh("Pyramid", [{
        "attributes": {"POSITION": builder.add_vec3(ppos)},
        "indices": builder.add_indices(pidx),
        "material": teal,
    }])
    crate_node = builder.add_node(
        "Crate", mesh=crate, translation=[-0.8, 0.5, 0.0],
        rotation=quat_y(30.0))
    group_node = builder.add_node("Pair", translation=[0.0, 0.0, 0.5],
                                  children=[crate_node])
    pyramid_node = builder.add_node("Pyramid", mesh=pyramid,
                                    translation=[0.9, 0.0, 0.0])
    builder.set_scene([group_node, pyramid_node], name="Sample")
    return builder
