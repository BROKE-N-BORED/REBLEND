"""Primitive tessellation: pure functions, no Qt.

Pattern note: the tessellation approach (flat-shaded triangle soup in cm,
z-up, one builder per shape) follows the pattern in the Design Shell v1.0
Bench viewport (design-shell/designshell/qtbench.py). This is new code
written for this engine; nothing is imported from Design Shell.

Every builder returns a dict:
  positions: flat list of floats, 3 per vertex (triangle soup)
  normals:   flat list of floats, 3 per vertex (flat, per-face)
  uvs:       flat list of floats, 2 per vertex
  tri_count: number of triangles
Geometry is centred on the origin; the object transform places it.

tessellate_indexed returns the same triangles in indexed form: shared
vertices plus a triangle index list (indices: flat list of ints, 3 per
triangle). Vertices that share a position and uv are welded into one,
with their face normals averaged and re-normalized — but only across
faces whose normals are within SMOOTH_ANGLE_DEGREES of each other, so
curved primitives come out smooth-shaded while box corners and
cylinder cap edges keep their crisp edges. The indexed form is what
smooth-shaded objects (T3), mesh import, and sculpting use; flat
objects keep the soup by default.
"""
from __future__ import annotations

import math

SHAPES = ("box", "sphere", "cylinder", "cone", "torus", "plane")

# Segment defaults kept low on purpose: this engine targets a Ryzen 3 with
# integrated graphics. Higher detail is the user's choice, never forced.
DEFAULT_SEGMENTS = 20


def _normal(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < 1e-9:
        return (0.0, 0.0, 1.0)
    return (nx / length, ny / length, nz / length)


def face_normal(a, b, c):
    """Unit normal of the triangle (a, b, c). The importers use this
    for files that carry no normals (T8)."""
    return _normal(a, b, c)


class _Soup:
    """Collects triangles into flat position/normal/uv arrays."""

    def __init__(self):
        self.pos: list[float] = []
        self.nor: list[float] = []
        self.uv: list[float] = []
        self.tris = 0

    def tri(self, a, b, c, uv_a=(0.0, 0.0), uv_b=(1.0, 0.0), uv_c=(0.0, 1.0)):
        nx, ny, nz = _normal(a, b, c)
        for v in (a, b, c):
            self.pos.extend(v)
            self.nor.extend((nx, ny, nz))
        for u in (uv_a, uv_b, uv_c):
            self.uv.extend(u)
        self.tris += 1

    def quad(self, a, b, c, d, uv_a=(0.0, 0.0), uv_b=(1.0, 0.0),
             uv_c=(1.0, 1.0), uv_d=(0.0, 1.0)):
        self.tri(a, b, c, uv_a, uv_b, uv_c)
        self.tri(a, c, d, uv_a, uv_c, uv_d)

    def result(self):
        return {"positions": self.pos, "normals": self.nor,
                "uvs": self.uv, "tri_count": self.tris}


def _box(params) -> dict:
    sx, sy, sz = params.get("size", [100.0, 100.0, 100.0])
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    v = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
         (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    s = _Soup()
    s.quad(v[0], v[3], v[2], v[1])
    s.quad(v[4], v[5], v[6], v[7])
    s.quad(v[0], v[1], v[5], v[4])
    s.quad(v[2], v[3], v[7], v[6])
    s.quad(v[1], v[2], v[6], v[5])
    s.quad(v[3], v[0], v[4], v[7])
    return s.result()


def _plane(params) -> dict:
    sx, sy = params.get("size", [100.0, 100.0])[:2]
    hx, hy = sx / 2.0, sy / 2.0
    s = _Soup()
    s.quad((-hx, -hy, 0.0), (hx, -hy, 0.0), (hx, hy, 0.0), (-hx, hy, 0.0))
    return s.result()


def _cylinder(params, top_radius=None) -> dict:
    r_bot = float(params.get("radius", 50.0))
    r_top = r_bot if top_radius is None else float(top_radius)
    h = float(params.get("height", 100.0))
    seg = max(6, int(params.get("segments", DEFAULT_SEGMENTS)))
    z0, z1 = -h / 2.0, h / 2.0
    ring0, ring1 = [], []
    for i in range(seg):
        t = 2 * math.pi * i / seg
        ring0.append((r_bot * math.cos(t), r_bot * math.sin(t), z0))
        ring1.append((r_top * math.cos(t), r_top * math.sin(t), z1))
    s = _Soup()
    for i in range(seg):
        j = (i + 1) % seg
        u0, u1 = i / seg, (i + 1) / seg
        s.quad(ring0[i], ring0[j], ring1[j], ring1[i],
               (u0, 0.0), (u1, 0.0), (u1, 1.0), (u0, 1.0))
    top_c, bot_c = (0.0, 0.0, z1), (0.0, 0.0, z0)
    for i in range(seg):
        j = (i + 1) % seg
        if r_top > 0:
            s.tri(top_c, ring1[i], ring1[j],
                  (0.5, 0.5), (0.5 + 0.5 * math.cos(2 * math.pi * i / seg),
                               0.5 + 0.5 * math.sin(2 * math.pi * i / seg)),
                  (0.5 + 0.5 * math.cos(2 * math.pi * j / seg),
                   0.5 + 0.5 * math.sin(2 * math.pi * j / seg)))
        s.tri(bot_c, ring0[j], ring0[i])
    return s.result()


def _sphere(params) -> dict:
    r = float(params.get("radius", 50.0))
    seg = max(8, int(params.get("segments", DEFAULT_SEGMENTS)))
    rings = max(4, int(params.get("rings", 12)))
    pts = []
    for ri in range(rings + 1):
        phi = math.pi * ri / rings
        for si in range(seg):
            theta = 2 * math.pi * si / seg
            pts.append((r * math.sin(phi) * math.cos(theta),
                        r * math.sin(phi) * math.sin(theta),
                        -r * math.cos(phi)))
    s = _Soup()
    for ri in range(rings):
        for si in range(seg):
            j = (si + 1) % seg
            a, b = ri * seg + si, ri * seg + j
            c, d = (ri + 1) * seg + j, (ri + 1) * seg + si
            u0, u1 = si / seg, (si + 1) / seg
            v0, v1 = ri / rings, (ri + 1) / rings
            s.quad(pts[a], pts[b], pts[c], pts[d],
                   (u0, v0), (u1, v0), (u1, v1), (u0, v1))
    return s.result()


def _torus(params) -> dict:
    major = float(params.get("major_radius", 50.0))
    minor = float(params.get("minor_radius", 20.0))
    seg = max(8, int(params.get("segments", DEFAULT_SEGMENTS)))
    sides = max(6, int(params.get("sides", 10)))
    pts = []
    for i in range(seg):
        u = 2 * math.pi * i / seg
        for j in range(sides):
            v = 2 * math.pi * j / sides
            x = (major + minor * math.cos(v)) * math.cos(u)
            y = (major + minor * math.cos(v)) * math.sin(u)
            z = minor * math.sin(v)
            pts.append((x, y, z))
    s = _Soup()
    for i in range(seg):
        ni = (i + 1) % seg
        for j in range(sides):
            nj = (j + 1) % sides
            a, b = i * sides + j, ni * sides + j
            c, d = ni * sides + nj, i * sides + nj
            u0, u1 = i / seg, (i + 1) / seg
            v0, v1 = j / sides, (j + 1) / sides
            s.quad(pts[a], pts[b], pts[c], pts[d],
                   (u0, v0), (u1, v0), (u1, v1), (u0, v1))
    return s.result()


_BUILDERS = {
    "box": _box,
    "plane": _plane,
    "cylinder": lambda p: _cylinder(p),
    "cone": lambda p: _cylinder(p, top_radius=0.0),
    "sphere": _sphere,
    "torus": _torus,
}


def tessellate(shape: str, params: dict | None = None) -> dict:
    """Build the triangle soup for one primitive. Unknown shapes -> box."""
    builder = _BUILDERS.get(shape, _box)
    return builder(dict(params or {}))


# Smoothing angle for the weld: face normals closer than this share
# one vertex and average; wider angles get separate vertices, so
# crisp edges survive the indexed path. 45 degrees: the default
# torus turns 36 degrees per side step, so it still welds smooth —
# a 30-degree threshold would leave it faceted. Box edges are 90
# degrees, so they never weld (T2's weld had no threshold and
# softened 4 of the 24 box corners where two faces share a uv).
SMOOTH_ANGLE_DEGREES = 45.0
_SMOOTH_COS = math.cos(math.radians(SMOOTH_ANGLE_DEGREES))


def _weld(soup: dict) -> dict:
    """Weld soup vertices that share a position and uv into shared
    vertices, averaging their face normals — but only across faces
    whose normals are within SMOOTH_ANGLE_DEGREES of the group's
    first normal. Wider edges get their own vertex, so a box keeps
    all 24 crisp corners. Index order follows soup order, so
    triangle i of the indexed mesh is triangle i of the soup.
    Vertex keys use exact float equality: shared corners come from
    the same computed points, so their floats are identical."""
    pos, nor, uv = soup["positions"], soup["normals"], soup["uvs"]
    verts = len(pos) // 3
    # key -> list of [vertex index, seed normal (nx, ny, nz)]
    groups_of = {}
    positions: list[float] = []
    uvs: list[float] = []
    normal_sums: list[list[float]] = []
    indices: list[int] = []
    for i in range(verts):
        key = (pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2],
               uv[i * 2], uv[i * 2 + 1])
        nx, ny, nz = nor[i * 3], nor[i * 3 + 1], nor[i * 3 + 2]
        groups = groups_of.get(key)
        idx = None
        if groups is not None:
            for g_idx, (sx, sy, sz) in groups:
                if nx * sx + ny * sy + nz * sz >= _SMOOTH_COS:
                    idx = g_idx
                    break
        if idx is None:
            idx = len(positions) // 3
            if groups is None:
                groups_of[key] = [(idx, (nx, ny, nz))]
            else:
                groups.append((idx, (nx, ny, nz)))
            positions.extend((pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]))
            uvs.extend((uv[i * 2], uv[i * 2 + 1]))
            normal_sums.append([nx, ny, nz])
        else:
            s = normal_sums[idx]
            s[0] += nx
            s[1] += ny
            s[2] += nz
        indices.append(idx)
    normals: list[float] = []
    for s in normal_sums:
        length = math.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2])
        if length < 1e-9:
            normals.extend((0.0, 0.0, 1.0))
        else:
            normals.extend((s[0] / length, s[1] / length, s[2] / length))
    return {"positions": positions, "normals": normals, "uvs": uvs,
            "indices": indices, "tri_count": soup["tri_count"]}


def tessellate_indexed(shape: str, params: dict | None = None) -> dict:
    """Indexed mesh for one primitive: shared vertices + triangle index
    list. Same triangles as tessellate(), fewer vertices. Unknown
    shapes -> box, as with tessellate."""
    return _weld(tessellate(shape, params))


def weld_soup(soup: dict) -> dict:
    """Weld a triangle soup into an indexed mesh: shared vertices +
    triangle index list, with face normals averaged across faces
    within SMOOTH_ANGLE_DEGREES (see _weld). The importers use this
    for files that carry no normals (T8): the soup's per-face normals
    become smooth normals, and crisp edges stay crisp."""
    return _weld(soup)


def default_params(shape: str) -> dict:
    if shape == "box":
        return {"size": [100.0, 100.0, 100.0]}
    if shape == "plane":
        return {"size": [400.0, 400.0]}
    if shape == "sphere":
        return {"radius": 50.0, "segments": 20, "rings": 12}
    if shape == "cylinder":
        return {"radius": 50.0, "height": 100.0, "segments": 20}
    if shape == "cone":
        return {"radius": 50.0, "height": 100.0, "segments": 20}
    if shape == "torus":
        return {"major_radius": 50.0, "minor_radius": 20.0,
                "segments": 24, "sides": 10}
    if shape == "mesh":
        # Mesh geometry comes from the asset file (T7); params are
        # unused, so a mesh object carries an empty dict.
        return {}
    return {"size": [100.0, 100.0, 100.0]}


def local_bounds(shape: str, params: dict | None = None) -> tuple:
    """(min, max) of the primitive in its own space, without tessellating."""
    p = dict(params or {})
    if shape == "box":
        sx, sy, sz = p.get("size", [100.0, 100.0, 100.0])
        return (-sx / 2, -sy / 2, -sz / 2), (sx / 2, sy / 2, sz / 2)
    if shape == "plane":
        sx, sy = p.get("size", [400.0, 400.0])[:2]
        return (-sx / 2, -sy / 2, 0.0), (sx / 2, sy / 2, 0.0)
    if shape == "sphere":
        r = float(p.get("radius", 50.0))
        return (-r, -r, -r), (r, r, r)
    if shape in ("cylinder", "cone"):
        r = float(p.get("radius", 50.0))
        h = float(p.get("height", 100.0))
        return (-r, -r, -h / 2), (r, r, h / 2)
    if shape == "torus":
        m = float(p.get("major_radius", 50.0)) + float(p.get("minor_radius", 20.0))
        mn = float(p.get("minor_radius", 20.0))
        return (-m, -m, -mn), (m, m, mn)
    return (-50.0, -50.0, -50.0), (50.0, 50.0, 50.0)


# ------------------------------------------------------------ transforms
# Plain 4x4 row-major matrices, stdlib only. Rotation order: Z, then Y,
# then X (applied to the point as Rz * Ry * Rx * point).

def mat_identity():
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4))
             for j in range(4)] for i in range(4)]


def mat_translate(x, y, z):
    m = mat_identity()
    m[0][3], m[1][3], m[2][3] = x, y, z
    return m


def mat_scale(x, y, z):
    m = mat_identity()
    m[0][0], m[1][1], m[2][2] = x, y, z
    return m


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def compose_matrix(position, rotation_deg, scale):
    """Model matrix from position, Euler rotation (degrees), scale."""
    rx, ry, rz = (math.radians(v) for v in rotation_deg)
    m = mat_mul(_rot_z(rz), mat_mul(_rot_y(ry), _rot_x(rx)))
    m = mat_mul(m, mat_scale(*scale))
    m = mat_mul(mat_translate(*position), m)
    return m


def apply_matrix(m, point):
    x, y, z = point
    return (m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
            m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
            m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3])


def mat_inverse(m):
    """Inverse of a 4x4 matrix (Gauss-Jordan). Falls back to identity."""
    n = 4
    aug = [list(m[i]) + [1.0 if i == j else 0.0 for j in range(n)]
           for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return mat_identity()
        aug[col], aug[piv] = aug[piv], aug[col]
        div = aug[col][col]
        aug[col] = [v / div for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def ray_hits_aabb(origin, direction, mn, mx):
    """Slab test. Returns the distance along the ray, or None."""
    tmin, tmax = 0.0, float("inf")
    for i in range(3):
        o, d = origin[i], direction[i]
        lo, hi = mn[i], mx[i]
        if abs(d) < 1e-12:
            if o < lo or o > hi:
                return None
        else:
            t1, t2 = (lo - o) / d, (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    return tmin
