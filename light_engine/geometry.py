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


# ------------------------------------------------------------ gizmo picking
# Ray tests for the gizmo handles (T13). All take an origin and a
# direction and return the ray parameter of the nearest hit in front
# of the origin, or None. The direction need not be unit — t is then
# in direction units, not cm — but the viewport passes a unit ray,
# so t is a distance in cm there.

def ray_vs_plane(origin, direction, point, normal):
    """Distance along the ray to the plane through `point` with unit
    `normal`, or None when the ray runs parallel to the plane or the
    plane is behind the ray origin. The drag math uses this for both
    the move/scale plane (perpendicular to the camera) and the rotate
    plane (the ring's own plane)."""
    denom = sum(direction[i] * normal[i] for i in range(3))
    if abs(denom) < 1e-12:
        return None
    t = sum((point[i] - origin[i]) * normal[i] for i in range(3)) / denom
    if t < 0.0:
        return None
    return t


def ray_vs_cylinder_segment(origin, direction, a, b, radius):
    """Distance along the ray to a finite cylinder (side + end caps)
    running from `a` to `b` with `radius`, or None. The gizmo's move
    handles read as one cylinder per axis, from the gizmo centre to
    the arrow tip: the shaft (0 to 0.75) and the cone (0.75 to 1.0,
    base radius 0.09) both fit inside a radius-0.12 cylinder, so one
    test covers the whole arrow."""
    axis = [b[i] - a[i] for i in range(3)]
    length = math.sqrt(sum(v * v for v in axis))
    if length < 1e-12 or radius <= 0.0:
        return None
    axis = [v / length for v in axis]
    delta = [origin[i] - a[i] for i in range(3)]
    d_dot = sum(direction[i] * axis[i] for i in range(3))
    delta_dot = sum(delta[i] * axis[i] for i in range(3))
    d_perp = [direction[i] - d_dot * axis[i] for i in range(3)]
    delta_perp = [delta[i] - delta_dot * axis[i] for i in range(3)]
    best = None
    # Side: |delta_perp + t * d_perp|^2 = radius^2.
    aa = sum(v * v for v in d_perp)
    if aa > 1e-18:
        bb = 2.0 * sum(delta_perp[i] * d_perp[i] for i in range(3))
        cc = sum(v * v for v in delta_perp) - radius * radius
        disc = bb * bb - 4.0 * aa * cc
        if disc >= 0.0:
            root = math.sqrt(disc)
            for t in ((-bb - root) / (2.0 * aa), (-bb + root) / (2.0 * aa)):
                if t < 0.0:
                    continue
                s = delta_dot + t * d_dot
                if -1e-9 <= s <= length + 1e-9:
                    if best is None or t < best:
                        best = t
    # End caps: the two planes at s = 0 and s = length, with the hit
    # inside the cap disk.
    if abs(d_dot) > 1e-12:
        for s_cap in (0.0, length):
            t = (s_cap - delta_dot) / d_dot
            if t < 0.0 or (best is not None and t >= best):
                continue
            radial = math.sqrt(max(0.0, sum(
                (delta_perp[i] + t * d_perp[i]) ** 2 for i in range(3))))
            if radial <= radius:
                best = t
    return best


def ray_vs_ring(origin, direction, center, axis, radius, thickness):
    """Distance along the ray to a thin ring (the gizmo's rotate
    handles, T13), or None. The ray meets the ring's plane — the
    plane through `center` with unit normal `axis` — and the hit
    counts when it lands within `thickness` of the circle of
    `radius` around the centre. A torus test would need the tube
    radius too; the drawn ring is a zero-width line, so the flat
    annulus is the honest test for what is on screen."""
    t = ray_vs_plane(origin, direction, center, axis)
    if t is None:
        return None
    rel = [origin[i] + direction[i] * t - center[i] for i in range(3)]
    along = sum(rel[i] * axis[i] for i in range(3))
    radial = math.sqrt(max(0.0, sum(v * v for v in rel) - along * along))
    if abs(radial - radius) <= thickness:
        return t
    return None


# Gizmo geometry (T12/T13), moved here from viewport.py in F1:
# every function below is pure (no GL, no Qt), so the headless
# tests import it without PySide6. viewport.py imports these
# names back — geometry.py is the single source.
# Gizmo (T12): drawn with the lines program only, in unit space centred
# on the selected object's position; the draw scales it by
# gizmo_size(distance) so it stays the same size on screen. Axis colors
# match the grid axes. Cones and cubes are wireframe outlines through
# the lines program — T12 adds no solid-geometry pass.
GIZMO_MODES = ("move", "rotate", "scale")
GIZMO_AXIS_COLORS = {"x": (0.85, 0.25, 0.25),
                     "y": (0.25, 0.75, 0.30),
                     "z": (0.30, 0.45, 0.9)}
# The gizmo's world-space size is distance * this factor. At the
# 45-degree field of view the visible height at the target plane is
# about 0.83 * distance, so 0.12 puts the gizmo at roughly 15% of the
# viewport height whatever the zoom.
GIZMO_SIZE_FACTOR = 0.12
GIZMO_RING_SEGMENTS = 48
GIZMO_CONE_SEGMENTS = 8
# Handle extents in unit-gizmo space (T12 values, named in T13 so the
# drawing and the picking share one source of truth — the builders
# below and pick_gizmo both read these, so the pick can never drift
# from what is drawn). Move: shaft to 0.75, cone base radius 0.09 at
# 0.75, apex at 1.0. Rotate: ring radius 1.0. Scale: line to 0.7,
# cube centred at 0.81 with half-side 0.11.
GIZMO_MOVE_SHAFT_END = 0.75
GIZMO_MOVE_TIP = 1.0
GIZMO_CONE_BASE_RADIUS = 0.09
GIZMO_RING_RADIUS = 1.0
GIZMO_SCALE_LINE_END = 0.7
GIZMO_SCALE_CUBE_CENTER = 0.81
GIZMO_SCALE_CUBE_HALF = 0.11
# Pick tolerance in unit-gizmo space (T13): the drawn lines have no
# width, so the pick tests fatten each handle by this much — about
# 11 screen pixels at the tavern view, a grabbable handle without
# swallowing the scene around it.
GIZMO_PICK_TOLERANCE = 0.12
# Shift-snap steps (T13): move 10 cm, rotate 15 degrees, scale 0.1.
GIZMO_SNAP_MOVE = 10.0
GIZMO_SNAP_ROTATE = 15.0
GIZMO_SNAP_SCALE = 0.1
# A scale drag never takes an axis below this: 0 would collapse the
# object and a negative would flip it inside out. The inspector still
# accepts any value typed in — this clamp is drag-only.
GIZMO_MIN_SCALE = 0.01


def gizmo_size(distance):
    """World-space gizmo size for a camera distance (T12): the gizmo
    grows with distance so it keeps the same size on screen."""
    return float(distance) * GIZMO_SIZE_FACTOR


def _gizmo_axis_frame(axis):
    """(unit, perp1, perp2) for an axis: the axis direction and two
    perpendicular unit vectors, right-handed."""
    frames = {"x": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
              "y": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
              "z": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))}
    return frames[axis]


def _gizmo_add(verts, a, b, color):
    verts += [a[0], a[1], a[2], *color, b[0], b[1], b[2], *color]


def _gizmo_bright(color):
    """The hover highlight for an axis color (T13): each channel moved
    55% toward white, so the hovered handle reads brighter without
    changing hue."""
    return tuple(c + (1.0 - c) * 0.55 for c in color)


def _gizmo_move_axis(verts, axis, color):
    """One move arrow: a shaft from the origin to 0.75, then a cone
    outline — an 8-segment base circle of radius 0.09 at 0.75 and
    four side lines to the apex at 1.0."""
    u, p, q = _gizmo_axis_frame(axis)
    tip = tuple(GIZMO_MOVE_SHAFT_END * v for v in u)
    _gizmo_add(verts, (0.0, 0.0, 0.0), tip, color)
    base = []
    for i in range(GIZMO_CONE_SEGMENTS):
        angle = 2.0 * math.pi * i / GIZMO_CONE_SEGMENTS
        base.append(tuple(tip[j] + GIZMO_CONE_BASE_RADIUS
                          * (math.cos(angle) * p[j]
                             + math.sin(angle) * q[j])
                          for j in range(3)))
    for i in range(GIZMO_CONE_SEGMENTS):
        _gizmo_add(verts, base[i], base[(i + 1) % GIZMO_CONE_SEGMENTS],
                   color)
    apex = tuple(GIZMO_MOVE_TIP * v for v in u)
    for i in range(0, GIZMO_CONE_SEGMENTS, 2):
        _gizmo_add(verts, base[i], apex, color)


def _gizmo_rotate_axis(verts, axis, color):
    """One rotate ring: a 48-segment circle of radius 1.0 in the
    plane perpendicular to the axis."""
    _u, p, q = _gizmo_axis_frame(axis)
    ring = []
    for i in range(GIZMO_RING_SEGMENTS):
        angle = 2.0 * math.pi * i / GIZMO_RING_SEGMENTS
        ring.append(tuple(GIZMO_RING_RADIUS
                          * (math.cos(angle) * p[j]
                             + math.sin(angle) * q[j])
                          for j in range(3)))
    for i in range(GIZMO_RING_SEGMENTS):
        _gizmo_add(verts, ring[i], ring[(i + 1) % GIZMO_RING_SEGMENTS],
                   color)


def _gizmo_scale_axis(verts, axis, color):
    """One scale handle: a short line from the origin to 0.7, ending
    in a small cube outline (side 0.22) centred at 0.81 along the
    axis."""
    u, p, q = _gizmo_axis_frame(axis)
    end = tuple(GIZMO_SCALE_LINE_END * v for v in u)
    _gizmo_add(verts, (0.0, 0.0, 0.0), end, color)
    centre = tuple(GIZMO_SCALE_CUBE_CENTER * v for v in u)
    half = GIZMO_SCALE_CUBE_HALF
    corners = []
    for su in (-1.0, 1.0):
        for sp in (-1.0, 1.0):
            for sq in (-1.0, 1.0):
                corners.append(tuple(
                    centre[j] + half * (su * u[j] + sp * p[j] + sq * q[j])
                    for j in range(3)))
    # Corner order is (su, sp, sq) with each sign varying slowest to
    # fastest; the edges join corners differing in exactly one sign.
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6),
             (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        _gizmo_add(verts, corners[a], corners[b], color)


def gizmo_lines(mode, highlight=None):
    """Interleaved pos(3)+color(3) floats for one gizmo mode, in unit
    space centred on the origin (T12). The draw scales the unit gizmo
    by gizmo_size(distance) and translates it to the selected
    object's position. Raises ValueError on an unknown mode.
    highlight (T13) is None or one axis ("x", "y", "z"): that axis's
    lines come out in the bright hover color, the others unchanged."""
    builders = {"move": _gizmo_move_axis,
                "rotate": _gizmo_rotate_axis,
                "scale": _gizmo_scale_axis}
    build = builders.get(mode)
    if build is None:
        raise ValueError(f"unknown gizmo mode: {mode!r} "
                         f"(expected one of {', '.join(GIZMO_MODES)})")
    if highlight is not None and highlight not in ("x", "y", "z"):
        raise ValueError(f"unknown gizmo highlight: {highlight!r} "
                         "(expected None or one of 'x', 'y', 'z')")
    verts = []
    for axis in ("x", "y", "z"):
        color = GIZMO_AXIS_COLORS[axis]
        if axis == highlight:
            color = _gizmo_bright(color)
        build(verts, axis, color)
    return verts
