"""Particle simulation (T16 analytic, T18 stepped; Design 5 revised).
Pure Python, no Qt, stdlib only.

Two modes, picked AUTOMATICALLY from the emitter parameters
(sim_mode) — the user never picks a mode directly:

- ANALYTIC (the default): a particle's position is a CLOSED-FORM
  function of its age and the emitter seed — no stepping, no wall
  clock. Scrubbing to any frame is O(1) per particle and exact: the
  same frame always gives the same particles, forward or backward.
  The per-particle variation (life, speed, direction, spawn,
  phases) is memoized across frames (T17): it is a pure function of
  (params, rotation, index), and the Random seeding inside it is the
  sim's biggest per-particle cost — a particle is evaluated once per
  frame while alive, so the memo turns ~36 re-seeds per particle
  into one. The memo is bounded with FIFO eviction and changes no
  result: the sim is still a pure function of its inputs, with or
  without it.
- STEPPED (T18): a fixed-timestep simulation (one step per frame,
  semi-implicit Euler) re-run deterministically from frame 0, with
  per-frame memoization (SteppedSim). Needed for forces that depend
  on position (curl turbulence, attraction) and for floor bounce,
  which changes the velocity history. Scrubbing forward continues
  from the cached frame; repeat visits are free; scrubbing back to
  an evicted frame restarts from frame 0. Deterministic because the
  timestep is fixed, the variation is seeded, and no wall-clock
  input exists anywhere in the sim.

The `turbulence` parameter means something different per mode
(Design 5's parameter description): in analytic mode it is the
sine-wobble strength in cm (a function of age and seed, not of
position — that is what keeps it analytic); in stepped mode it is
the curl-noise acceleration strength in cm/s^2 (a true
position-dependent curl field, _curl_noise). Turbulence alone does
NOT switch modes — the force table is authoritative, and the T16/T17
proofs pin the presets' analytic wobble exactly. An emitter goes
stepped when it needs a position-dependent force or a
history-changing collision: collide "bounce", or attract with a
target and nonzero strength.

The model, per particle i (born at t_birth = i / rate seconds):
  - Seeded variation: lifetime, speed, direction inside the spread
    cone, spawn offset inside the emission shape, and turbulence
    phases all come from a Random seeded by (emitter seed, index).
    The draw order is FIXED (life, speed, direction, spawn, phases)
    — changing it re-randomizes every existing scene, so don't.
  - Motion: gravity (x gravity_factor) + constant wind (x
    wind_response) + sinusoidal wind gust + linear drag, all in
    closed form via the integrating factor (see _ballistic). The
    gust is a function of ABSOLUTE scene time, so a particle born
    at t_birth sees phase omega * t_birth.
  - Turbulence wobble: a per-particle sine offset added to the
    position (a function of age and seed, not of position — that is
    what keeps it analytic; position-dependent turbulence is stepped
    mode, T18, where the same parameter drives curl noise instead).
  - Floor kill (collide "kill"): the particle dies at the first
    floor crossing of its ballistic path (z = 0), found by bracket
    + bisection on the closed form. "bounce" and "attract" are
    stepped-only (T18): in stepped mode a bounce reflects the
    velocity with FLOOR_RESTITUTION and an attractor pulls with a
    spring force toward its target object.
  - Appearance over life: size interpolates start -> start x
    size_over_life, color start -> end, alpha fades in/out by the
    opacity fractions.

Caps warn, never refuse (Design 5): past MAX_LIVE_PER_EMITTER the
OLDEST particles die first (pool recycling) and the result reports
capped=True with the dropped count, so the UI can warn. The
scene-wide soft cap is reported the same way by
evaluate_scene_emitters. Nothing is ever blocked for size.

Rendering (T17): build_render_data merges every live particle of
every emitter into ONE interleaved float array — position (3) +
color (3) + size (1) + alpha (1) = 8 floats per particle — for a
single VBO upload per frame. Alpha-blend particles come first,
additive ones second, so the viewport draws both blend modes as
two glDrawArrays ranges from the one buffer. The blend mode is a
property of the preset: fire and sparks glow (additive); every
other preset, including custom and unknown names, is plain alpha
— a non-glowing particle must never add light to the scene.

Conventions: units are cm and seconds; z-up. The emitter's forward
is its local +Z rotated by the object rotation (the engine's Euler
convention, geometry.compose_matrix). Spawn offsets are in emitter
local space, rotated the same way. An animated emitter spawns each
particle from the emitter's transform AT THE PARTICLE'S BIRTH
FRAME (via transform_at) — particles never teleport with a moving
emitter.
"""
from __future__ import annotations

import heapq
import math
import random
from array import array

from . import animation as anim_mod
from . import scene as scene_mod
from .geometry import apply_matrix, compose_matrix

# Caps (Design 5): warn, never refuse. Past the per-emitter cap the
# oldest particles die first; the result says so.
MAX_LIVE_PER_EMITTER = 2000
SCENE_SOFT_CAP = 10000

# Turbulence wobble frequencies (Hz), one per axis — distinct so the
# path does not repeat trivially.
_WOBBLE_FREQ = (0.9, 1.3, 0.7)

# Per-particle variation memo (T17). Two levels: the outer key is
# (variation fingerprint, birth rotation) and the inner dict maps a
# particle index to its _variation result tuple. _variation is a
# pure function of those three, so a memo hit returns exactly what a
# fresh call would. Two levels so the static-emitter case — one
# rotation for every particle — looks the outer dict up once per
# evaluation and then hits int keys only. Bounded: past
# _VARIATION_CACHE_MAX entries the oldest outer entries are evicted
# whole (dicts keep insertion order), so a long playback cannot grow
# memory without bound — an evicted entry is simply recomputed. A
# continuously rotating emitter defeats the memo (every birth
# rotation is new) and degrades to T16 speed, never worse.
_VARIATION_CACHE = {}
_VARIATION_CACHE_COUNT = 0
_VARIATION_CACHE_MAX = 20000


def clear_variation_cache():
    """Empty the variation memo (tests only — production code never
    needs it; the memo changes no result)."""
    global _VARIATION_CACHE_COUNT
    _VARIATION_CACHE.clear()
    _VARIATION_CACHE_COUNT = 0


def _memo_inner(var_key):
    """The inner dict for one (fingerprint, rotation) pair,
    created on first use."""
    inner = _VARIATION_CACHE.get(var_key)
    if inner is None:
        inner = {}
        _VARIATION_CACHE[var_key] = inner
    return inner


def _memo_put(var_key, inner, index, varied):
    """Store one memoized variation, making room first. Eviction
    drops the oldest outer entries whole; when only the entry being
    written to remains, its own oldest particle goes instead (the
    earliest-born, most likely dead) — the memo stays bounded in
    every case."""
    global _VARIATION_CACHE_COUNT
    while _VARIATION_CACHE_COUNT >= _VARIATION_CACHE_MAX:
        oldest = next(iter(_VARIATION_CACHE))
        if oldest == var_key:
            inner.pop(next(iter(inner)))
            _VARIATION_CACHE_COUNT -= 1
        else:
            _VARIATION_CACHE_COUNT -= len(_VARIATION_CACHE.pop(oldest))
    inner[index] = varied
    _VARIATION_CACHE_COUNT += 1


# -- stepped mode (T18) ----------------------------------------------------
# Floor bounce: the floor is a fixed plane (z = 0) with a fixed
# restitution — there is no material to read it from (material
# restitution arrives with T19/T20 for objects, not for the floor).
FLOOR_RESTITUTION = 0.4
# Below this bounce height a bounce settles into a rest instead:
# the particle sits on the floor (z = 0, vz = 0) and slides,
# instead of micro-bouncing every step. The test is on the APEX
# the bounce would reach (vz^2 / 2g), not on the impact speed — one
# step of gravity alone is ~41 cm/s, so a speed threshold can never
# catch the settle: the particle would bounce forever on
# sub-centimeter hops. A resting particle is NOT counted as a
# bounce — only a real impact is.
FLOOR_REST_HEIGHT = 1.0

# Curl-noise field (T18): the vector potential lives in noise space,
# one lattice cell per _NOISE_CELL_CM of world space, so the swirls
# are ~60 cm across — visible in a smoke plume, small next to a room.
_NOISE_CELL_CM = 60.0
_NOISE_SCALE = 1.0 / _NOISE_CELL_CM
# The three potential fields are the same noise at three seed
# offsets, so they are independent fields, not copies.
_NOISE_SEED_OFFSETS = (0, 101, 202)
# The lattice hash is the classic Perlin permutation: a fixed
# shuffle (random.Random with a fixed integer seed is deterministic
# on every machine — Mersenne Twister, stable across versions).
# Doubled so (index & 255) never overruns.
_NOISE_PERM = list(range(256))
random.Random(1337).shuffle(_NOISE_PERM)
_NOISE_PERM = _NOISE_PERM + _NOISE_PERM


def sim_mode(params):
    """The simulation mode for an emitter's parameters: "stepped" or
    "analytic". Stepped when the emitter needs a position-dependent
    force or a history-changing collision — collide "bounce", or
    attract with a target and nonzero strength. Everything else is
    analytic, including turbulence (the sine wobble) and floor kill
    (the impact time comes from the ballistic equation). The user
    never picks a mode directly; the parameters decide."""
    params = scene_mod.default_emitter_params(params)
    if params["collide"] == "bounce":
        return "stepped"
    attract = params["attract"]
    if (isinstance(attract, dict) and attract.get("target")
            and attract.get("strength", 0.0) != 0.0):
        return "stepped"
    return "analytic"


def _noise_gradient(x, y, z, seed):
    """Analytic gradient of the value-noise field at (x, y, z) in
    noise space. Returns (dv/dx, dv/dy, dv/dz). The field is
    trilinear interpolation over a permutation-hashed lattice with
    smoothstep easing; the gradient is the exact derivative of that
    interpolant (each partial is a bilinear interpolation of the
    four corner differences along its axis, times the smoothstep
    slope), so no finite differences are needed — one 8-corner
    evaluation gives the value's slope in all three directions.
    Deterministic: pure integer lattice math, no Python hash()."""
    perm = _NOISE_PERM
    ix = math.floor(x)
    iy = math.floor(y)
    iz = math.floor(z)
    fx = x - ix
    fy = y - iy
    fz = z - iz
    sx = fx * fx * (3.0 - 2.0 * fx)
    sy = fy * fy * (3.0 - 2.0 * fy)
    sz = fz * fz * (3.0 - 2.0 * fz)
    dsx = 6.0 * fx * (1.0 - fx)
    dsy = 6.0 * fy * (1.0 - fy)
    dsz = 6.0 * fz * (1.0 - fz)
    seed = seed & 255
    # The z-level hashes are shared across the 8 corners (2 + 4 + 8
    # lookups, not 24): the lattice hash nests z inside y inside x.
    pz0 = perm[(iz + seed) & 255]
    pz1 = perm[(iz + 1 + seed) & 255]
    py00 = perm[(iy + pz0) & 255]
    py10 = perm[(iy + 1 + pz0) & 255]
    py01 = perm[(iy + pz1) & 255]
    py11 = perm[(iy + 1 + pz1) & 255]
    c000 = perm[(ix + py00) & 255] / 255.0
    c100 = perm[(ix + 1 + py00) & 255] / 255.0
    c010 = perm[(ix + py10) & 255] / 255.0
    c110 = perm[(ix + 1 + py10) & 255] / 255.0
    c001 = perm[(ix + py01) & 255] / 255.0
    c101 = perm[(ix + 1 + py01) & 255] / 255.0
    c011 = perm[(ix + py11) & 255] / 255.0
    c111 = perm[(ix + 1 + py11) & 255] / 255.0
    ix0 = 1.0 - sx
    iy0 = 1.0 - sy
    iz0 = 1.0 - sz
    dvdx = dsx * ((c100 - c000) * iy0 * iz0
                  + (c110 - c010) * sy * iz0
                  + (c101 - c001) * iy0 * sz
                  + (c111 - c011) * sy * sz)
    dvdy = dsy * ((c010 - c000) * ix0 * iz0
                  + (c110 - c100) * sx * iz0
                  + (c011 - c001) * ix0 * sz
                  + (c111 - c101) * sx * sz)
    dvdz = dsz * ((c001 - c000) * ix0 * iy0
                  + (c101 - c100) * sx * iy0
                  + (c011 - c010) * ix0 * sy
                  + (c111 - c110) * sx * sy)
    return (dvdx * 2.0, dvdy * 2.0, dvdz * 2.0)


def _curl_noise(position, seed):
    """The curl of the noise potential field at a world position:
    a genuine curl (divergence-free — the field swirls, it never
    sources or sinks), so turbulence bends trajectories without
    piling particles up. Components are O(1) (mean magnitude ~1.5),
    so multiplying by the emitter's `turbulence` gives an
    acceleration in cm/s^2. Deterministic: the same position and
    seed give the same vector, always."""
    qx = position[0] * _NOISE_SCALE
    qy = position[1] * _NOISE_SCALE
    qz = position[2] * _NOISE_SCALE
    gx = _noise_gradient(qx, qy, qz, seed + _NOISE_SEED_OFFSETS[0])
    gy = _noise_gradient(qx, qy, qz, seed + _NOISE_SEED_OFFSETS[1])
    gz = _noise_gradient(qx, qy, qz, seed + _NOISE_SEED_OFFSETS[2])
    return (gz[1] - gy[2], gx[2] - gz[0], gy[0] - gx[1])


def _freeze(value):
    """A hashable canonical form of JSON-ish data (dicts, lists,
    tuples, scalars) for cache fingerprints. Dict keys sort, so
    two dicts with the same content freeze identically."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value

# Presets (Design 5). Each fills every parameter in one tap; the
# numbers are the stored state, the name is a convenience. Water is
# the T16 splash baseline — T26 extends it with pour/drip variants.
PRESETS = {
    "fire": {
        "shape": "cone", "rate": 80.0, "lifetime": 0.8,
        "lifetime_spread": 0.3, "speed": 120.0, "speed_spread": 0.3,
        "spread": 20.0, "size": 8.0, "size_over_life": 0.2,
        "color_start": "#ffdd55", "color_end": "#ff3300",
        "opacity": {"fade_in": 0.05, "fade_out": 0.4},
        "gravity_factor": -0.5, "drag": 0.5, "wind_response": 0.3,
        "turbulence": 3.0, "collide": "none",
    },
    "smoke": {
        "shape": "point", "rate": 30.0, "lifetime": 3.0,
        "lifetime_spread": 0.3, "speed": 40.0, "speed_spread": 0.3,
        "spread": 25.0, "size": 10.0, "size_over_life": 3.0,
        "color_start": "#999999", "color_end": "#555555",
        "opacity": {"fade_in": 0.1, "fade_out": 0.3},
        "gravity_factor": -0.15, "drag": 1.0, "wind_response": 0.8,
        "turbulence": 5.0, "collide": "none",
    },
    "sparks": {
        "shape": "point", "rate": 100.0, "lifetime": 0.5,
        "lifetime_spread": 0.4, "speed": 300.0, "speed_spread": 0.4,
        "spread": 45.0, "size": 2.0, "size_over_life": 0.0,
        "color_start": "#ffffaa", "color_end": "#ff6600",
        "opacity": {"fade_in": 0.0, "fade_out": 0.5},
        "gravity_factor": 1.0, "drag": 0.2, "wind_response": 0.1,
        "turbulence": 0.0, "collide": "none",
    },
    "rain": {
        "shape": "box", "box_size": [400.0, 400.0, 10.0],
        "rate": 200.0, "lifetime": 1.0, "lifetime_spread": 0.1,
        "speed": 0.0, "speed_spread": 0.0, "spread": 0.0,
        "size": 1.5, "size_over_life": 1.0,
        "color_start": "#aaccff", "color_end": "#aaccff",
        "opacity": {"fade_in": 0.0, "fade_out": 0.0},
        "gravity_factor": 1.0, "drag": 0.0, "wind_response": 0.5,
        "turbulence": 0.0, "collide": "kill",
    },
    "snow": {
        "shape": "box", "box_size": [400.0, 400.0, 10.0],
        "rate": 60.0, "lifetime": 4.0, "lifetime_spread": 0.3,
        "speed": 0.0, "speed_spread": 0.0, "spread": 0.0,
        "size": 3.0, "size_over_life": 1.0,
        "color_start": "#ffffff", "color_end": "#ffffff",
        "opacity": {"fade_in": 0.05, "fade_out": 0.1},
        "gravity_factor": 0.08, "drag": 1.5, "wind_response": 1.0,
        "turbulence": 8.0, "collide": "kill",
    },
    "dust": {
        "shape": "sphere", "sphere_radius": 30.0,
        "rate": 20.0, "lifetime": 5.0, "lifetime_spread": 0.5,
        "speed": 5.0, "speed_spread": 0.5, "spread": 90.0,
        "size": 2.0, "size_over_life": 1.5,
        "color_start": "#ccbb99", "color_end": "#ccbb99",
        "opacity": {"fade_in": 0.2, "fade_out": 0.3},
        "gravity_factor": 0.05, "drag": 2.0, "wind_response": 1.0,
        "turbulence": 4.0, "collide": "none",
    },
    "water": {
        "shape": "cone", "rate": 150.0, "lifetime": 0.7,
        "lifetime_spread": 0.3, "speed": 250.0, "speed_spread": 0.4,
        "spread": 30.0, "size": 3.0, "size_over_life": 0.5,
        "color_start": "#cceeff", "color_end": "#66aaff",
        "opacity": {"fade_in": 0.0, "fade_out": 0.3},
        "gravity_factor": 1.0, "drag": 0.1, "wind_response": 0.2,
        "turbulence": 0.0, "collide": "kill",
    },
}


def preset_names():
    """Preset names the picker offers (T16 set)."""
    return ["custom"] + sorted(PRESETS)


def apply_preset(params, name):
    """A NEW emitter param dict with a preset's values filled in.
    "custom" returns the defaults unchanged. Unknown names raise
    ValueError — the caller validates against preset_names first."""
    if name == "custom":
        merged = scene_mod.default_emitter_params(params)
        merged["preset"] = "custom"
        return merged
    if name not in PRESETS:
        raise ValueError(f"unknown particle preset: {name!r}")
    # Picking a preset fills EVERY parameter — the caller's old
    # values do not survive.
    merged = scene_mod.default_emitter_params(PRESETS[name])
    merged["preset"] = name
    return merged


def _particle_rng(seed, index):
    """One Random per (seed, particle index). Integer arithmetic, so
    the stream is identical on every run and every machine — no
    string hashing (PYTHONHASHSEED would vary it). Distinct (seed,
    index) pairs give distinct streams for index < 1,000,003, far
    past any scene's particle count."""
    return random.Random(int(seed) * 1000003 + int(index))


def _basis(axis):
    """Two unit vectors perpendicular to the unit axis, for the
    spread cone."""
    ax, ay, az = axis
    ref = (1.0, 0.0, 0.0) if abs(ax) < 0.9 else (0.0, 1.0, 0.0)
    ux = ay * ref[2] - az * ref[1]
    uy = az * ref[0] - ax * ref[2]
    uz = ax * ref[1] - ay * ref[0]
    ulen = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / ulen, uy / ulen, uz / ulen
    vx = ay * uz - az * uy
    vy = az * ux - ax * uz
    vz = ax * uy - ay * ux
    return (ux, uy, uz), (vx, vy, vz)


def _variation(params, index, rot_matrix):
    """Seeded per-particle variation. Returns (life, speed, direction,
    spawn_offset, phases). Direction and spawn offset are in WORLD
    space (the emitter rotation is applied here). Draw order is part
    of the deterministic contract — see the module docstring."""
    rng = _particle_rng(params["seed"], index)
    # 1. Lifetime: uniform in lifetime x (1 +/- spread), never below
    # 5% of the base lifetime (a zero life is never alive).
    spread = params["lifetime_spread"]
    life = params["lifetime"] * (1.0 + spread * (2.0 * rng.random() - 1.0))
    life = max(life, params["lifetime"] * 0.05)
    # 2. Speed: same shape, floored at zero.
    speed = params["speed"] * (
        1.0 + params["speed_spread"] * (2.0 * rng.random() - 1.0))
    speed = max(speed, 0.0)
    # 3. Direction: uniform over the spread cone around the emitter's
    # forward (+Z, rotated). cos(theta) uniform gives equal area.
    axis = apply_matrix(rot_matrix, (0.0, 0.0, 1.0))
    spread_rad = math.radians(params["spread"])
    cos_max = math.cos(spread_rad)
    cos_theta = 1.0 - rng.random() * (1.0 - cos_max)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * math.pi * rng.random()
    u, v = _basis(axis)
    direction = tuple(
        axis[i] * cos_theta
        + (u[i] * math.cos(phi) + v[i] * math.sin(phi)) * sin_theta
        for i in range(3))
    # 4. Spawn offset in emitter local space, rotated to world.
    shape = params["shape"]
    if shape == "box":
        sx, sy, sz = params["box_size"]
        local = ((rng.random() - 0.5) * sx,
                 (rng.random() - 0.5) * sy,
                 (rng.random() - 0.5) * sz)
    elif shape == "sphere":
        radius = params["sphere_radius"] * rng.random() ** (1.0 / 3.0)
        theta = math.acos(2.0 * rng.random() - 1.0)
        psi = 2.0 * math.pi * rng.random()
        local = (radius * math.sin(theta) * math.cos(psi),
                 radius * math.sin(theta) * math.sin(psi),
                 radius * math.cos(theta))
    else:  # point and cone both spawn at the apex
        local = (0.0, 0.0, 0.0)
    spawn = apply_matrix(rot_matrix, local)
    # 5. Turbulence phases, one per axis.
    phases = (2.0 * math.pi * rng.random(),
              2.0 * math.pi * rng.random(),
              2.0 * math.pi * rng.random())
    return life, speed, direction, spawn, phases


def _decay_integral(drag, tau):
    """Integral of e^(-k*s) from 0 to tau = (1 - e^(-k*tau)) / k.
    expm1 keeps it accurate for small k*tau; the k = 0 limit is tau."""
    if drag < 1e-9:
        return tau
    return -math.expm1(-drag * tau) / drag


def _accel_integral(drag, tau):
    """Integral of the decay integral from 0 to tau — the position
    term for a constant acceleration under linear drag. The k = 0
    limit is tau^2 / 2."""
    if drag < 1e-9:
        return tau * tau / 2.0
    return (tau - _decay_integral(drag, tau)) / drag


def _ballistic(p0, v0, a_const, gust_vec, omega, phase, drag, tau,
               gust_amp=None):
    """Closed-form position at age tau (no turbulence wobble — that
    is a visual offset added after). Solves v' = a_const +
    gust*sin(omega*t + phase) - drag*v, v(0) = v0, by the
    integrating factor. The gust phase is omega * t_birth: the gust
    is a function of absolute scene time, and the particle is born
    into it part-way through its cycle. gust_amp, when given, is
    the precomputed |gust_vec| — the caller computes it once per
    evaluation instead of once per particle (T18); None computes it
    here, as before."""
    decay = _decay_integral(drag, tau)
    accel = _accel_integral(drag, tau)
    pos = [p0[i] + v0[i] * decay + a_const[i] * accel for i in range(3)]
    if gust_amp is None:
        gust_amp = math.sqrt(sum(g * g for g in gust_vec))
    if gust_amp <= 0.0 or omega <= 0.0:
        return tuple(pos)
    if drag < 1e-9:
        # No drag: double integral of A*sin(omega*t + phase).
        coef = 1.0 / omega
        gust = (tau * math.cos(phase)
                - (math.sin(omega * tau + phase) - math.sin(phase))
                / omega)
        for i in range(3):
            pos[i] += gust_vec[i] * coef * gust
        return tuple(pos)
    # With drag: the integrating-factor closed form. C is the
    # integration constant from the gust's phase at birth.
    denom = drag * drag + omega * omega
    const = drag * math.sin(phase) - omega * math.cos(phase)
    gust = ((drag / omega) * (math.cos(phase)
                              - math.cos(omega * tau + phase))
            - (math.sin(omega * tau + phase) - math.sin(phase))
            + const * math.expm1(-drag * tau) / drag)
    for i in range(3):
        pos[i] += gust_vec[i] * gust / denom
    return tuple(pos)


def _wobble(turbulence, phases, tau):
    """Sine turbulence offset at age tau (analytic: a function of
    age and seed only). Zero when turbulence is 0."""
    if turbulence <= 0.0:
        return (0.0, 0.0, 0.0)
    return tuple(
        turbulence * math.sin(2.0 * math.pi * _WOBBLE_FREQ[i] * tau
                              + phases[i])
        for i in range(3))


def _killed_by_floor(p0, v0, a_const, gust_vec, omega, phase, drag,
                     tau, gust_amp=None):
    """True if the ballistic path (wobble excluded — it is visual)
    crosses z = 0 during [0, tau]. A particle spawned below the
    floor is dead at birth. Fast path: with no gust and no upward
    acceleration, the vertical velocity is monotonic, so the path
    crosses at most once — the endpoints decide. Otherwise a
    bracket search finds the first crossing. gust_amp, when given,
    is the precomputed |gust_vec| (T18); None computes it here."""
    if p0[2] < 0.0:
        return True
    if tau <= 0.0:
        return False
    if gust_amp is None:
        gust_amp = math.sqrt(sum(g * g for g in gust_vec))
    z_end = _ballistic(p0, v0, a_const, gust_vec, omega, phase,
                       drag, tau, gust_amp)[2]
    if gust_amp <= 0.0 and a_const[2] <= 0.0:
        return z_end < 0.0
    if z_end < 0.0:
        return True
    # Gust or upward acceleration: the path can dip and recover, so
    # scan for the first sign change, then bisect it.
    samples = 16
    prev_t = 0.0
    for i in range(1, samples + 1):
        t = tau * i / samples
        z = _ballistic(p0, v0, a_const, gust_vec, omega, phase,
                       drag, t, gust_amp)[2]
        if z < 0.0:
            lo, hi = prev_t, t
            for _ in range(40):
                mid = (lo + hi) / 2.0
                if _ballistic(p0, v0, a_const, gust_vec, omega,
                              phase, drag, mid, gust_amp)[2] < 0.0:
                    hi = mid
                else:
                    lo = mid
            return True
        prev_t = t
    return False


def _appearance(size0, size_over_life, fade_in, fade_out, age,
                life, colors):
    """(size, color, alpha) at an age within a life. Size and color
    interpolate over life; alpha fades in over the first fade_in
    fraction and out over the last fade_out fraction. The emitter
    constants are passed in hoisted — both sim paths read them once
    per evaluation, not once per particle (T17 parsed the colors
    once; T18 hoists the rest the same way). The arithmetic is the
    T16 arithmetic, unchanged: the same inputs give bit-identical
    outputs."""
    frac = age / life if life > 0.0 else 1.0
    frac = max(0.0, min(1.0, frac))
    size = size0 * (1.0 + (size_over_life - 1.0) * frac)
    c0, c1 = colors
    # Explicit components, not a generator expression — the same
    # arithmetic, without the generator machinery per particle.
    color = (c0[0] + (c1[0] - c0[0]) * frac,
             c0[1] + (c1[1] - c0[1]) * frac,
             c0[2] + (c1[2] - c0[2]) * frac)
    alpha = 1.0
    if fade_in > 0.0:
        alpha = min(alpha, frac / fade_in)
    if fade_out > 0.0:
        alpha = min(alpha, (1.0 - frac) / fade_out)
    return size, color, max(0.0, min(1.0, alpha))


# Per-sim frame cache bound: 96 frames of state. At the 2,000 cap
# a frame holds 64 bytes per particle (indices + positions +
# velocities + lives + bounce counts, all compact arrays), so a
# maxed-out sim holds ~12 MB; the module cache holds 4 sims.
STEPPED_MAX_FRAMES = 96
_STEPPED_SIMS = {}
_STEPPED_SIMS_MAX = 4


def clear_stepped_cache():
    """Empty the stepped-sim cache (tests only — production code
    never needs it; the cache changes no result)."""
    _STEPPED_SIMS.clear()


class SteppedSim:
    """Fixed-timestep deterministic particle sim for ONE emitter
    (T18, stepped mode). One step per frame (dt = 1/fps),
    semi-implicit Euler, re-simulated from frame 0 with per-frame
    memoization: scrubbing forward continues from the cached frame,
    repeat visits are free, scrubbing back to an evicted frame
    restarts from frame 0. Deterministic because the timestep is
    fixed, the variation is seeded (the same _variation + memo as
    the analytic path, so a particle's life/speed/direction/spawn
    are identical in both modes), and no wall-clock input exists
    anywhere in the sim.

    State per frame (compact arrays, 64 bytes per particle):
    indices, positions, velocities, lives, bounce counts, plus the
    cap report (capped, dropped). The result dict is built from the
    state on demand, so repeat visits return bit-identical dicts.
    Frames are integer boundaries: a particle born mid-step gets
    one partial step from its birth time to the next boundary, so
    birth times are exact, not quantized to the step grid."""

    def __init__(self, params, world, fps, position, rotation,
                 transform_at, target_at):
        # params/world are already normalized by evaluate_emitter.
        self._params = params
        self._fps = max(1, int(fps))
        self._position = tuple(position)
        self._rotation = tuple(rotation)
        self._transform_at = transform_at
        self._target_at = target_at
        gravity = world["gravity"]
        wind = world["wind"]
        gust_amp = world["wind_gust"]["amplitude"]
        gust_freq = world["wind_gust"]["frequency"]
        self._omega = 2.0 * math.pi * gust_freq
        wind_response = params["wind_response"]
        self._a_const = tuple(
            gravity[i] * params["gravity_factor"]
            + wind[i] * wind_response for i in range(3))
        wind_len = math.sqrt(sum(w * w for w in wind))
        if wind_len > 1e-9:
            gust_dir = tuple(w / wind_len for w in wind)
        else:
            gust_dir = (1.0, 0.0, 0.0)
        self._gust_vec = tuple(gust_amp * wind_response * d
                               for d in gust_dir)
        self._gust_amp = math.sqrt(
            sum(g * g for g in self._gust_vec))
        self._drag = params["drag"]
        self._turbulence = params["turbulence"]
        self._seed = params["seed"]
        self._rate = params["rate"]
        self._collide = params["collide"]
        attract = params["attract"]
        self._attract_strength = (
            attract["strength"] if isinstance(attract, dict) else 0.0)
        self._colors = (scene_mod.hex_to_rgb(params["color_start"]),
                        scene_mod.hex_to_rgb(params["color_end"]))
        self._size0 = params["size"]
        self._size_over_life = params["size_over_life"]
        self._fade_in = params["opacity"]["fade_in"]
        self._fade_out = params["opacity"]["fade_out"]
        self._var_fingerprint = (
            params["seed"], params["lifetime"],
            params["lifetime_spread"], params["speed"],
            params["speed_spread"], params["spread"], params["shape"],
            tuple(params["box_size"]), params["sphere_radius"])
        self._rot_cache = {}
        self._static_rot_matrix = None
        self._static_var_key = None
        self._static_inner = None
        if transform_at is None:
            self._static_rot_matrix = compose_matrix(
                (0.0, 0.0, 0.0), rotation, (1.0, 1.0, 1.0))
            self._static_var_key = (self._var_fingerprint,
                                    tuple(rotation))
            self._static_inner = _memo_inner(self._static_var_key)
        self._frames = {}
        self._max_frame = -1
        # Uncapped live-count accounting (for the capped/dropped
        # report): the state is capped, so the state alone cannot
        # say how many live particles the cap hid. Every surviving
        # birth pushes its death time (t_birth + life) onto a heap
        # and bumps _born_count; each step pops the deaths up to
        # the frame time. uncapped_live = born - dead. Exact for
        # collide "none"/"bounce" (old age is the only death);
        # for "kill" it over-counts the floor-killed (their heap
        # entries pop at life end, not at impact) — the safe
        # direction for a budget warning, and kill is analytic
        # unless attraction is also set.
        self._death_heap = []
        self._born_count = 0
        self._dead_count = 0

    def evaluate(self, frame):
        """The result dict at a frame — the same shape as
        evaluate_emitter's (count, indices, ages, lives, positions,
        sizes, colors, alphas, capped, dropped)."""
        frame = max(0, int(frame))
        state = self._frames.get(frame)
        if state is None:
            if frame < self._max_frame:
                # Evicted frame: restart from frame 0.
                self._frames.clear()
                self._death_heap.clear()
                self._born_count = 0
                self._dead_count = 0
                self._max_frame = -1
            if self._max_frame < 0:
                self._initial_state()
            while self._max_frame < frame:
                self._step(self._max_frame + 1)
            state = self._frames[frame]
        return self._build_result(frame, state)

    def bounce_counts(self, frame):
        """{particle index: bounce count} for the live particles at
        a frame. Tests and proofs only — the result dict does not
        carry bounce counts."""
        frame = max(0, int(frame))
        self.evaluate(frame)
        indices, _, _, _, bounces, _, _ = self._frames[frame]
        return {indices[k]: bounces[k]
                for k in range(len(indices))}

    def _varied(self, index, birth_rot, rot_matrix):
        """The memoized seeded variation for one particle (shared
        with the analytic path — same life, speed, direction,
        spawn in both modes)."""
        if self._transform_at is None:
            inner = self._static_inner
            var_key = self._static_var_key
        else:
            var_key = (self._var_fingerprint, tuple(birth_rot))
            inner = _memo_inner(var_key)
        varied = inner.get(index)
        if varied is None:
            varied = _variation(self._params, index, rot_matrix)
            _memo_put(var_key, inner, index, varied)
        return varied

    def _birth_transform(self, birth_frame):
        """(position, rotation, rotation matrix) at a birth frame."""
        if self._transform_at is None:
            return (self._position, self._rotation,
                    self._static_rot_matrix)
        birth_pos, birth_rot = self._transform_at(birth_frame)
        rot_key = tuple(birth_rot)
        rot_matrix = self._rot_cache.get(rot_key)
        if rot_matrix is None:
            rot_matrix = compose_matrix((0.0, 0.0, 0.0), birth_rot,
                                        (1.0, 1.0, 1.0))
            self._rot_cache[rot_key] = rot_matrix
        return (birth_pos, birth_rot, rot_matrix)

    def _integrate(self, p0, v0, t_start, dt):
        """One semi-implicit Euler step: v1 = v0 + a(p0, t_start)
        * dt, p1 = p0 + v1 * dt. Forces at the step start: constant
        (gravity + base wind), the sinusoidal gust at the absolute
        time, linear drag, curl turbulence, and the spring
        attraction toward the target's position at the step start."""
        ax = self._a_const[0] - self._drag * v0[0]
        ay = self._a_const[1] - self._drag * v0[1]
        az = self._a_const[2] - self._drag * v0[2]
        if self._gust_amp > 0.0 and self._omega > 0.0:
            gust = math.sin(self._omega * t_start)
            ax += self._gust_vec[0] * gust
            ay += self._gust_vec[1] * gust
            az += self._gust_vec[2] * gust
        if self._turbulence > 0.0:
            curl = _curl_noise(p0, self._seed)
            ax += curl[0] * self._turbulence
            ay += curl[1] * self._turbulence
            az += curl[2] * self._turbulence
        if (self._attract_strength != 0.0
                and self._target_at is not None):
            target = self._target_at(t_start * self._fps)
            ax += self._attract_strength * (target[0] - p0[0])
            ay += self._attract_strength * (target[1] - p0[1])
            az += self._attract_strength * (target[2] - p0[2])
        v1 = (v0[0] + ax * dt, v0[1] + ay * dt, v0[2] + az * dt)
        p1 = (p0[0] + v1[0] * dt, p0[1] + v1[1] * dt,
              p0[2] + v1[2] * dt)
        return p1, v1

    def _floor(self, p0, v0, p1, v1):
        """Floor resolution after a step. Returns (p1, v1, bounced,
        dead). "none" lets the particle fall through. "kill" kills
        it if it ends below the floor, or started below it (spawned
        under the floor is dead at birth, as in the analytic sim).
        "bounce": a particle at rest on the floor (z == 0, vz == 0
        at the step start) that gravity would push under stays at
        rest — no bounce counted; a particle moving up but still
        below (spawned under the floor) is put on the floor and
        keeps rising; a real impact reflects with FLOOR_RESTITUTION
        and counts one bounce; and an impact whose bounce would
        rise less than FLOOR_REST_HEIGHT settles into a rest
        instead of micro-bouncing (the apex test uses the sustained
        downward acceleration, so it scales with the gravity)."""
        if self._collide == "none":
            return p1, v1, False, False
        if self._collide == "kill":
            if p1[2] < 0.0 or p0[2] < 0.0:
                return p1, v1, False, True
            return p1, v1, False, False
        # bounce
        if p1[2] >= 0.0:
            return p1, v1, False, False
        if p0[2] == 0.0 and v0[2] == 0.0 and v1[2] < 0.0:
            return (p1[0], p1[1], 0.0), (v1[0], v1[1], 0.0), \
                False, False
        if v1[2] >= 0.0:
            return (p1[0], p1[1], 0.0), v1, False, False
        vz = -v1[2] * FLOOR_RESTITUTION
        g_down = -self._a_const[2]
        if g_down > 0.0 and vz * vz < 2.0 * g_down * FLOOR_REST_HEIGHT:
            return (p1[0], p1[1], 0.0), (v1[0], v1[1], 0.0), \
                False, False
        return (p1[0], p1[1], -p1[2]), (v1[0], v1[1], vz), True, False

    def _birth(self, index, t1):
        """A particle born at index/rate, stepped to time t1 (one
        partial step from its birth time, or none if born exactly
        at t1). Returns (position, velocity, life, bounced), or
        None if it dies before t1 (short life, or floor kill)."""
        t_birth = index / self._rate
        age = t1 - t_birth
        birth_pos, birth_rot, rot_matrix = self._birth_transform(
            t_birth * self._fps)
        life, speed, direction, spawn, _phases = self._varied(
            index, birth_rot, rot_matrix)
        if age >= life:
            return None
        p0 = tuple(birth_pos[i] + spawn[i] for i in range(3))
        v0 = tuple(direction[i] * speed for i in range(3))
        if age <= 0.0:
            p1, v1 = p0, v0
        else:
            p1, v1 = self._integrate(p0, v0, t_birth, age)
        p1, v1, bounced, dead = self._floor(p0, v0, p1, v1)
        if dead:
            return None
        heapq.heappush(self._death_heap, t_birth + life)
        self._born_count += 1
        return (p1, v1, life, bounced)

    def _cap_report(self, t, state_count):
        """(capped, dropped) at time t from the uncapped live
        count: born minus the deaths popped off the heap so far.
        The state is capped, so only the heap knows how many live
        particles the cap hid. Called with monotonically
        increasing t (frame 0, then each step in order)."""
        heap = self._death_heap
        while heap and heap[0] <= t:
            heapq.heappop(heap)
            self._dead_count += 1
        live = self._born_count - self._dead_count
        if live > MAX_LIVE_PER_EMITTER:
            return True, live - state_count
        return False, 0

    def _initial_state(self):
        """The frame-0 state: only particle 0 is born (t_birth =
        0), at age 0."""
        indices = array("i")
        pos = array("d")
        vel = array("d")
        lives = array("d")
        bounces = array("i")
        if self._rate > 0.0:
            born = self._birth(0, 0.0)
            if born is not None:
                p1, v1, life, bounced = born
                indices.append(0)
                pos.extend(p1)
                vel.extend(v1)
                lives.append(life)
                bounces.append(1 if bounced else 0)
        capped, dropped = self._cap_report(0.0, len(indices))
        state = (indices, pos, vel, lives, bounces, capped, dropped)
        self._frames[0] = state
        self._max_frame = 0
        return state

    def _step(self, frame):
        """The state at `frame`, stepped from the state at frame-1:
        survivors integrate one full step (dying of old age or
        floor kill on the way), then the particles born during the
        step join with partial steps. Survivors precede births and
        both are ascending, so indices stay oldest-first — and the
        cap drops the oldest, as in the analytic path."""
        fps = self._fps
        t1 = frame / fps
        t0 = (frame - 1) / fps
        dt = 1.0 / fps
        prev = self._frames[frame - 1]
        indices, pos, vel, lives, bounces, _, _ = prev
        rate = self._rate
        new_indices = array("i")
        new_pos = array("d")
        new_vel = array("d")
        new_lives = array("d")
        new_bounces = array("i")
        for k in range(len(indices)):
            index = indices[k]
            life = lives[k]
            if t1 - index / rate >= life:
                continue
            p0 = (pos[3 * k], pos[3 * k + 1], pos[3 * k + 2])
            v0 = (vel[3 * k], vel[3 * k + 1], vel[3 * k + 2])
            p1, v1 = self._integrate(p0, v0, t0, dt)
            p1, v1, bounced, dead = self._floor(p0, v0, p1, v1)
            if dead:
                continue
            new_indices.append(index)
            new_pos.extend(p1)
            new_vel.extend(v1)
            new_lives.append(life)
            new_bounces.append(bounces[k] + (1 if bounced else 0))
        # Births: t_birth in (t0, t1] — the same boundary rule as
        # the analytic path's i_max = int(t * rate).
        i_lo = int(t0 * rate)
        i_hi = int(t1 * rate)
        for index in range(i_lo + 1, i_hi + 1):
            born = self._birth(index, t1)
            if born is None:
                continue
            p1, v1, life, bounced = born
            new_indices.append(index)
            new_pos.extend(p1)
            new_vel.extend(v1)
            new_lives.append(life)
            new_bounces.append(1 if bounced else 0)
        count = len(new_indices)
        if count > MAX_LIVE_PER_EMITTER:
            drop = count - MAX_LIVE_PER_EMITTER
            del new_indices[:drop]
            del new_pos[:3 * drop]
            del new_vel[:3 * drop]
            del new_lives[:drop]
            del new_bounces[:drop]
        # The reported capped/dropped come from the heap, not from
        # this step's overflow: the state was already capped in
        # earlier frames, so the overflow here would under-count
        # (and could even report capped=False while 10,000 live
        # particles are hidden behind the cap).
        capped, dropped = self._cap_report(t1, len(new_indices))
        state = (new_indices, new_pos, new_vel, new_lives,
                 new_bounces, capped, dropped)
        self._frames[frame] = state
        while len(self._frames) > STEPPED_MAX_FRAMES:
            self._frames.pop(next(iter(self._frames)))
        self._max_frame = frame
        return state

    def _build_result(self, frame, state):
        """The evaluate_emitter-shaped result dict for a cached
        state. Appearance is a pure function of (age, life), so it
        is derived here, not stored."""
        indices, pos, _vel, lives, _bounces, capped, dropped = state
        t = frame / self._fps
        rate = self._rate
        result = {"count": len(indices), "indices": list(indices),
                  "ages": [], "lives": list(lives), "positions": [],
                  "sizes": [], "colors": [], "alphas": [],
                  "capped": capped, "dropped": dropped}
        for k in range(len(indices)):
            age = t - indices[k] / rate
            result["ages"].append(age)
            result["positions"].append(
                (pos[3 * k], pos[3 * k + 1], pos[3 * k + 2]))
            size, color, alpha = _appearance(
                self._size0, self._size_over_life, self._fade_in,
                self._fade_out, age, lives[k], self._colors)
            result["sizes"].append(size)
            result["colors"].append(color)
            result["alphas"].append(alpha)
        return result


def _stepped_sim_for(params, world, fps, position, rotation,
                     transform_at, target_at, transform_key,
                     target_key):
    """The cached SteppedSim for an emitter's inputs, or a fresh
    uncached one when the inputs cannot be fingerprinted (an
    animated emitter or target without a key — a direct
    evaluate_emitter call with a closure). The fingerprint covers
    everything the sim reads: the normalized params and world, the
    fps, the emitter's transform (static position/rotation, or the
    caller's animation key), and the target's animation key when
    attraction is active. A changed input makes a new sim, so a
    cached sim is never stale."""
    if transform_at is not None and transform_key is None:
        return SteppedSim(params, world, fps, position, rotation,
                          transform_at, target_at)
    if target_at is not None and target_key is None:
        return SteppedSim(params, world, fps, position, rotation,
                          transform_at, target_at)
    if transform_at is None:
        transform_sig = (tuple(position), tuple(rotation))
    else:
        transform_sig = transform_key
    key = (_freeze(params), _freeze(world), fps, transform_sig,
           target_key)
    sim = _STEPPED_SIMS.get(key)
    if sim is None:
        sim = SteppedSim(params, world, fps, position, rotation,
                         transform_at, target_at)
        while len(_STEPPED_SIMS) >= _STEPPED_SIMS_MAX:
            _STEPPED_SIMS.pop(next(iter(_STEPPED_SIMS)))
        _STEPPED_SIMS[key] = sim
    return sim


def evaluate_emitter(params, world=None, fps=24, frame=0,
                     position=(0.0, 0.0, 0.0),
                     rotation=(0.0, 0.0, 0.0), transform_at=None,
                     target_at=None, transform_key=None,
                     target_key=None):
    """Live particles of one emitter at a frame. Pure function of
    (params, world, fps, frame, transform): the same inputs give
    bit-identical outputs, so scrubbing is exact in both
    directions. transform_at, when given, is a callable taking a
    (possibly fractional) frame and returning (position, rotation)
    — the emitter's transform at each particle's BIRTH frame, for
    animated emitters; without it the emitter is static at the
    given position/rotation. target_at, when given, is a callable
    taking a frame and returning the attract target's position
    (stepped mode only). transform_key / target_key are hashable
    fingerprints of the animated emitter's / target's keyframes —
    the scene path passes them so the stepped sim can cache across
    frames; without them a stepped evaluation builds a fresh sim
    (correct, just not cached).

    The mode is automatic (sim_mode): analytic emitters take the
    closed-form path below; stepped emitters go through a
    SteppedSim. Returns a dict of flat lists, indices ascending
    (oldest first): count, indices, ages, lives, positions
    [(x, y, z)], sizes, colors [(r, g, b)], alphas, capped (the
    2,000 cap bit, oldest dropped first), dropped (how many the cap
    removed)."""
    params = scene_mod.default_emitter_params(params)
    world = scene_mod.default_world(world)
    fps = max(1, int(fps))
    if sim_mode(params) == "stepped":
        sim = _stepped_sim_for(params, world, fps, position,
                               rotation, transform_at, target_at,
                               transform_key, target_key)
        return sim.evaluate(frame)
    t = max(0.0, frame / fps)
    rate = params["rate"]
    if rate <= 0.0:
        return {"count": 0, "indices": [], "ages": [], "lives": [],
                "positions": [], "sizes": [], "colors": [],
                "alphas": [], "capped": False, "dropped": 0}
    gravity = world["gravity"]
    wind = world["wind"]
    gust_amp = world["wind_gust"]["amplitude"]
    gust_freq = world["wind_gust"]["frequency"]
    omega = 2.0 * math.pi * gust_freq
    wind_response = params["wind_response"]
    # Constant acceleration: gravity (x gravity_factor) + base wind
    # (x wind_response). The gust direction follows the wind when
    # there is one, else +X.
    a_const = tuple(
        gravity[i] * params["gravity_factor"]
        + wind[i] * wind_response for i in range(3))
    ax, ay, az = a_const
    wind_len = math.sqrt(sum(w * w for w in wind))
    if wind_len > 1e-9:
        gust_dir = tuple(w / wind_len for w in wind)
    else:
        gust_dir = (1.0, 0.0, 0.0)
    gust_vec = tuple(gust_amp * wind_response * d for d in gust_dir)
    # |gust_vec|, computed once (T18): _ballistic and
    # _killed_by_floor used to recompute it per particle.
    gust_vec_amp = math.sqrt(sum(g * g for g in gust_vec))
    drag = params["drag"]
    turbulence = params["turbulence"]
    # The drag-free, gust-free fast path (T18): with no drag the
    # decay integral is tau and the acceleration integral is
    # tau^2/2, and with no gust there is no gust term — the
    # position is p0 + v0*tau + a*tau^2/2, computed inline in the
    # same arithmetic _ballistic uses, so the results are
    # bit-identical to the function call.
    simple_ballistic = drag < 1e-9 and gust_vec_amp <= 0.0
    # Per-emitter constants, parsed once (T17): the colors never
    # change mid-evaluation, and the rotation matrix is a pure
    # function of the birth rotation — a static emitter's particles
    # all share one matrix, so composing it per particle was the
    # frame's biggest cost (measured: ~2/3 of the sim time at 2,000
    # live particles). The cache is keyed by the rotation tuple, so
    # an animated emitter still gets the right matrix per birth
    # frame; results are bit-identical to composing every time.
    # T18 hoists the rest of the per-particle params reads the same
    # way (appearance constants, turbulence) — the loop body reads
    # locals only.
    colors = (scene_mod.hex_to_rgb(params["color_start"]),
              scene_mod.hex_to_rgb(params["color_end"]))
    size0 = params["size"]
    size_over_life = params["size_over_life"]
    fade_in = params["opacity"]["fade_in"]
    fade_out = params["opacity"]["fade_out"]
    rot_cache = {}
    # The variation fingerprint: everything _variation reads from
    # the params (seed, life, speed, spread, shape). Together with
    # the birth rotation and the particle index it identifies one
    # variation result exactly — see the _VARIATION_CACHE note.
    var_fingerprint = (
        params["seed"], params["lifetime"],
        params["lifetime_spread"], params["speed"],
        params["speed_spread"], params["spread"], params["shape"],
        tuple(params["box_size"]), params["sphere_radius"])
    collide_kill = params["collide"] == "kill"
    # Static emitters (no transform_at) have one birth rotation for
    # every particle: resolve the rotation matrix and the memo's
    # inner dict once, before the loop, instead of per particle.
    static_rot_matrix = None
    static_var_key = None
    static_inner = None
    if transform_at is None:
        static_rot_key = tuple(rotation)
        static_rot_matrix = compose_matrix(
            (0.0, 0.0, 0.0), rotation, (1.0, 1.0, 1.0))
        static_var_key = (var_fingerprint, static_rot_key)
        static_inner = _memo_inner(static_var_key)

    # Index range alive at t: born at i / rate, alive while
    # t_birth <= t < t_birth + life_i. life_i never exceeds the max
    # spread life, which bounds the low end of the range.
    life_max = params["lifetime"] * (1.0 + params["lifetime_spread"])
    i_max = int(t * rate)
    i_min = max(0, math.ceil((t - life_max) * rate))
    result = {"count": 0, "indices": [], "ages": [], "lives": [],
              "positions": [], "sizes": [], "colors": [],
              "alphas": [], "capped": False, "dropped": 0}
    out_indices = result["indices"]
    out_ages = result["ages"]
    out_lives = result["lives"]
    out_positions = result["positions"]
    out_sizes = result["sizes"]
    out_colors = result["colors"]
    out_alphas = result["alphas"]
    for index in range(i_min, i_max + 1):
        t_birth = index / rate
        age = t - t_birth
        if age < 0.0:
            continue
        if transform_at is not None:
            birth_pos, birth_rot = transform_at(t_birth * fps)
            rot_key = tuple(birth_rot)
            rot_matrix = rot_cache.get(rot_key)
            if rot_matrix is None:
                rot_matrix = compose_matrix((0.0, 0.0, 0.0), birth_rot,
                                            (1.0, 1.0, 1.0))
                rot_cache[rot_key] = rot_matrix
            var_key = (var_fingerprint, rot_key)
            inner = _memo_inner(var_key)
            varied = inner.get(index)
            if varied is None:
                varied = _variation(params, index, rot_matrix)
                _memo_put(var_key, inner, index, varied)
        else:
            birth_pos, birth_rot = position, rotation
            rot_matrix = static_rot_matrix
            varied = static_inner.get(index)
            if varied is None:
                varied = _variation(params, index, rot_matrix)
                _memo_put(static_var_key, static_inner, index, varied)
        life, speed, direction, spawn, phases = varied
        if age >= life:
            continue
        p0x = birth_pos[0] + spawn[0]
        p0y = birth_pos[1] + spawn[1]
        p0z = birth_pos[2] + spawn[2]
        v0x = direction[0] * speed
        v0y = direction[1] * speed
        v0z = direction[2] * speed
        phase = omega * t_birth
        if collide_kill and _killed_by_floor(
                (p0x, p0y, p0z), (v0x, v0y, v0z), a_const, gust_vec,
                omega, phase, drag, age, gust_vec_amp):
            continue
        if simple_ballistic:
            accel = age * age / 2.0
            pos = (p0x + v0x * age + ax * accel,
                   p0y + v0y * age + ay * accel,
                   p0z + v0z * age + az * accel)
        else:
            pos = _ballistic((p0x, p0y, p0z), (v0x, v0y, v0z),
                             a_const, gust_vec, omega, phase, drag,
                             age, gust_vec_amp)
        if turbulence > 0.0:
            wob = _wobble(turbulence, phases, age)
            pos = (pos[0] + wob[0], pos[1] + wob[1],
                   pos[2] + wob[2])
        size, color, alpha = _appearance(
            size0, size_over_life, fade_in, fade_out, age, life,
            colors)
        out_indices.append(index)
        out_ages.append(age)
        out_lives.append(life)
        out_positions.append(pos)
        out_sizes.append(size)
        out_colors.append(color)
        out_alphas.append(alpha)
    result["count"] = len(result["indices"])
    # Cap (warn, never refuse): past 2,000 live, the OLDEST die
    # first — keep the youngest (highest indices). The result
    # reports it so the UI can warn.
    if result["count"] > MAX_LIVE_PER_EMITTER:
        drop = result["count"] - MAX_LIVE_PER_EMITTER
        for key in ("indices", "ages", "lives", "positions", "sizes",
                    "colors", "alphas"):
            result[key] = result[key][drop:]
        result["count"] = MAX_LIVE_PER_EMITTER
        result["capped"] = True
        result["dropped"] = drop
    return result


def evaluate_scene_emitters(scene, frame):
    """Every emitter in a scene at a frame, with the scene's world
    and fps. Animated emitters sample their transform at each
    particle's birth frame; static emitters (no keyframes at all)
    take the static path (T18 — the per-particle transform sampling
    was the scene path's biggest cost at 2,000 live, and for a
    static emitter the sampled transform is the stored one, so the
    results are bit-identical). Stepped emitters get their attract
    target resolved from the scene (a deleted target leaves the
    attraction inactive) and animation keys for the stepped-sim
    cache. Returns {"emitters": {object_id: result}, "total": live
    count, "scene_capped": the 10,000 scene-wide soft cap
    tripped}."""
    world = scene.get("world")
    fps = scene.get("animation", {}).get("fps", 24)
    keys_by_id = scene.get("animation", {}).get("keys", {})
    objects = scene.get("objects", [])
    out = {"emitters": {}, "total": 0, "scene_capped": False}
    for obj in objects:
        if obj.get("kind") != "emitter":
            continue
        params = obj.get("emitter")
        keys = keys_by_id.get(obj["id"], {})
        if keys:
            def transform_at(birth_frame, _obj=obj, _scene=scene):
                sampled = anim_mod.sample_object(_scene, _obj,
                                                 birth_frame)
                return sampled["position"], sampled["rotation"]
            transform_key = (tuple(obj["position"]),
                             tuple(obj["rotation"]), _freeze(keys))
        else:
            transform_at = None
            transform_key = None
        target_at = None
        target_key = None
        attract = (params or {}).get("attract")
        if isinstance(attract, dict) and attract.get("target"):
            target = next(
                (o for o in objects
                 if o.get("id") == attract["target"]), None)
            if target is not None:
                def target_at(target_frame, _target=target,
                              _scene=scene):
                    return anim_mod.sample_object(
                        _scene, _target, target_frame)["position"]
                target_keys = keys_by_id.get(target["id"], {})
                target_key = (tuple(target["position"]),
                              tuple(target["rotation"]),
                              _freeze(target_keys))
        result = evaluate_emitter(
            params, world, fps, frame, position=obj["position"],
            rotation=obj["rotation"], transform_at=transform_at,
            target_at=target_at, transform_key=transform_key,
            target_key=target_key)
        out["emitters"][obj["id"]] = result
        out["total"] += result["count"]
    out["scene_capped"] = out["total"] > SCENE_SOFT_CAP
    return out


# -- render data (T17) ----------------------------------------------------
# Blend modes: "additive" particles add their light to the framebuffer
# (fire, sparks — things that glow); "alpha" particles blend normally
# (smoke, dust, snow, rain, water — things that occlude). The mapping
# is by preset name, and anything unlisted — custom, or a name a
# hand-edited scene invented — is alpha: the safe default, because an
# alpha particle can only darken toward its own color, while an
# additive one would wrongly brighten the scene.
BLEND_MODE_ALPHA = "alpha"
BLEND_MODE_ADDITIVE = "additive"
ADDITIVE_PRESETS = frozenset({"fire", "sparks"})

# Interleaved render layout, floats per particle: position (3) +
# color (3) + size (1) + alpha (1). Matches the points program's
# attribute locations 0-3 and the viewport's 32-byte stride.
PARTICLE_FLOATS = 8
PARTICLE_STRIDE_BYTES = PARTICLE_FLOATS * 4


def blend_mode_for_preset(preset):
    """The blend mode for a preset name: additive for the glowing
    presets (fire, sparks), alpha for everything else."""
    if preset in ADDITIVE_PRESETS:
        return BLEND_MODE_ADDITIVE
    return BLEND_MODE_ALPHA


def merge_render_data(evaluated):
    """Merge live particles into one interleaved float array for a
    single VBO upload. `evaluated` is a list of (preset, result)
    pairs, result being an evaluate_emitter dict. Alpha-blend
    particles come first, additive second, in scene order within
    each group — so the viewport draws the alpha range
    [0, alpha_count) and the additive range
    [alpha_count, alpha_count + additive_count) from the one
    buffer. Returns {"data": array("f"), "alpha_count": int,
    "additive_count": int, "total": int}. Pure function of its
    inputs: the same frame gives bit-identical bytes."""
    # Build plain lists, convert once (T18): eight array appends
    # per particle was the merge's cost; a list extend per particle
    # plus one array('f', ...) conversion gives bit-identical bytes
    # (each float converts to float32 the same way either route).
    alpha_list = []
    additive_list = []
    alpha_count = 0
    additive_count = 0
    for preset, result in evaluated:
        if result["count"] == 0:
            continue
        additive = (blend_mode_for_preset(preset)
                    == BLEND_MODE_ADDITIVE)
        target = additive_list if additive else alpha_list
        positions = result["positions"]
        colors = result["colors"]
        sizes = result["sizes"]
        alphas = result["alphas"]
        for i in range(result["count"]):
            pos = positions[i]
            color = colors[i]
            target.extend((pos[0], pos[1], pos[2], color[0],
                           color[1], color[2], sizes[i], alphas[i]))
        if additive:
            additive_count += result["count"]
        else:
            alpha_count += result["count"]
    return {"data": array("f", alpha_list + additive_list),
            "alpha_count": alpha_count,
            "additive_count": additive_count,
            "total": alpha_count + additive_count}


def build_render_data(scene, frame):
    """The render data for a scene at a frame: evaluate every emitter
    (evaluate_scene_emitters) and merge the results by each emitter's
    preset. Same shape as merge_render_data's return, plus "capped"
    (any emitter past its 2,000 cap) and "scene_capped" (the 10,000
    scene-wide soft cap tripped) so the caller can warn."""
    evaluated = evaluate_scene_emitters(scene, frame)
    pairs = []
    for obj in scene.get("objects", []):
        if obj.get("kind") != "emitter":
            continue
        result = evaluated["emitters"].get(obj["id"])
        if result is None:
            continue
        preset = (obj.get("emitter") or {}).get("preset", "custom")
        pairs.append((preset, result))
    merged = merge_render_data(pairs)
    merged["capped"] = any(
        result["capped"] for result in evaluated["emitters"].values())
    merged["scene_capped"] = evaluated["scene_capped"]
    return merged
