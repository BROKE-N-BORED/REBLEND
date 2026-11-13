"""Analytic particle simulation (T16, Design 5 revised). Pure Python,
no Qt, stdlib only.

A particle's position is a CLOSED-FORM function of its age and the
emitter seed — no stepping, no stored state, no wall clock. Scrubbing
to any frame is O(1) per particle and exact: the same frame always
gives the same particles, forward or backward.

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
    mode, T18).
  - Floor kill (collide "kill"): the particle dies at the first
    floor crossing of its ballistic path (z = 0), found by bracket
    + bisection on the closed form. "bounce" and "attract" are
    stepped-only (T18): stored on the emitter, ignored here.
  - Appearance over life: size interpolates start -> start x
    size_over_life, color start -> end, alpha fades in/out by the
    opacity fractions.

Caps warn, never refuse (Design 5): past MAX_LIVE_PER_EMITTER the
OLDEST particles die first (pool recycling) and the result reports
capped=True with the dropped count, so the UI can warn. The
scene-wide soft cap is reported the same way by
evaluate_scene_emitters. Nothing is ever blocked for size.

Conventions: units are cm and seconds; z-up. The emitter's forward
is its local +Z rotated by the object rotation (the engine's Euler
convention, geometry.compose_matrix). Spawn offsets are in emitter
local space, rotated the same way. An animated emitter spawns each
particle from the emitter's transform AT THE PARTICLE'S BIRTH
FRAME (via transform_at) — particles never teleport with a moving
emitter.
"""
from __future__ import annotations

import math
import random

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


def _ballistic(p0, v0, a_const, gust_vec, omega, phase, drag, tau):
    """Closed-form position at age tau (no turbulence wobble — that
    is a visual offset added after). Solves v' = a_const +
    gust*sin(omega*t + phase) - drag*v, v(0) = v0, by the
    integrating factor. The gust phase is omega * t_birth: the gust
    is a function of absolute scene time, and the particle is born
    into it part-way through its cycle."""
    decay = _decay_integral(drag, tau)
    accel = _accel_integral(drag, tau)
    pos = [p0[i] + v0[i] * decay + a_const[i] * accel for i in range(3)]
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
                     tau):
    """True if the ballistic path (wobble excluded — it is visual)
    crosses z = 0 during [0, tau]. A particle spawned below the
    floor is dead at birth. Fast path: with no gust and no upward
    acceleration, the vertical velocity is monotonic, so the path
    crosses at most once — the endpoints decide. Otherwise a
    bracket search finds the first crossing."""
    if p0[2] < 0.0:
        return True
    if tau <= 0.0:
        return False
    z_end = _ballistic(p0, v0, a_const, gust_vec, omega, phase,
                       drag, tau)[2]
    gust_amp = math.sqrt(sum(g * g for g in gust_vec))
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
                       drag, t)[2]
        if z < 0.0:
            lo, hi = prev_t, t
            for _ in range(40):
                mid = (lo + hi) / 2.0
                if _ballistic(p0, v0, a_const, gust_vec, omega,
                              phase, drag, mid)[2] < 0.0:
                    hi = mid
                else:
                    lo = mid
            return True
        prev_t = t
    return False


def _appearance(params, age, life):
    """(size, color, alpha) at an age within a life. Size and color
    interpolate over life; alpha fades in over the first fade_in
    fraction and out over the last fade_out fraction."""
    frac = age / life if life > 0.0 else 1.0
    frac = max(0.0, min(1.0, frac))
    size = params["size"] * (1.0 + (params["size_over_life"] - 1.0)
                             * frac)
    c0 = scene_mod.hex_to_rgb(params["color_start"])
    c1 = scene_mod.hex_to_rgb(params["color_end"])
    color = tuple(c0[i] + (c1[i] - c0[i]) * frac for i in range(3))
    alpha = 1.0
    fade_in = params["opacity"]["fade_in"]
    fade_out = params["opacity"]["fade_out"]
    if fade_in > 0.0:
        alpha = min(alpha, frac / fade_in)
    if fade_out > 0.0:
        alpha = min(alpha, (1.0 - frac) / fade_out)
    return size, color, max(0.0, min(1.0, alpha))


def evaluate_emitter(params, world=None, fps=24, frame=0,
                     position=(0.0, 0.0, 0.0),
                     rotation=(0.0, 0.0, 0.0), transform_at=None):
    """Live particles of one emitter at a frame. Pure function of
    (params, world, fps, frame, transform): the same inputs give
    bit-identical outputs, so scrubbing is exact in both
    directions. transform_at, when given, is a callable taking a
    (possibly fractional) frame and returning (position, rotation)
    — the emitter's transform at each particle's BIRTH frame, for
    animated emitters; without it the emitter is static at the
    given position/rotation.

    Returns a dict of flat lists, indices ascending (oldest first):
    count, indices, ages, lives, positions [(x, y, z)], sizes,
    colors [(r, g, b)], alphas, capped (the 2,000 cap bit, oldest
    dropped first), dropped (how many the cap removed)."""
    params = scene_mod.default_emitter_params(params)
    world = scene_mod.default_world(world)
    fps = max(1, int(fps))
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
    wind_len = math.sqrt(sum(w * w for w in wind))
    if wind_len > 1e-9:
        gust_dir = tuple(w / wind_len for w in wind)
    else:
        gust_dir = (1.0, 0.0, 0.0)
    gust_vec = tuple(gust_amp * wind_response * d for d in gust_dir)
    drag = params["drag"]

    # Index range alive at t: born at i / rate, alive while
    # t_birth <= t < t_birth + life_i. life_i never exceeds the max
    # spread life, which bounds the low end of the range.
    life_max = params["lifetime"] * (1.0 + params["lifetime_spread"])
    i_max = int(t * rate)
    i_min = max(0, math.ceil((t - life_max) * rate))
    result = {"count": 0, "indices": [], "ages": [], "lives": [],
              "positions": [], "sizes": [], "colors": [],
              "alphas": [], "capped": False, "dropped": 0}
    for index in range(i_min, i_max + 1):
        t_birth = index / rate
        age = t - t_birth
        if age < 0.0:
            continue
        if transform_at is not None:
            birth_pos, birth_rot = transform_at(t_birth * fps)
        else:
            birth_pos, birth_rot = position, rotation
        rot_matrix = compose_matrix((0.0, 0.0, 0.0), birth_rot,
                                    (1.0, 1.0, 1.0))
        life, speed, direction, spawn, phases = _variation(
            params, index, rot_matrix)
        if age >= life:
            continue
        p0 = tuple(birth_pos[i] + spawn[i] for i in range(3))
        v0 = tuple(direction[i] * speed for i in range(3))
        phase = omega * t_birth
        if params["collide"] == "kill" and _killed_by_floor(
                p0, v0, a_const, gust_vec, omega, phase, drag, age):
            continue
        pos = _ballistic(p0, v0, a_const, gust_vec, omega, phase,
                         drag, age)
        wob = _wobble(params["turbulence"], phases, age)
        pos = tuple(pos[i] + wob[i] for i in range(3))
        size, color, alpha = _appearance(params, age, life)
        result["indices"].append(index)
        result["ages"].append(age)
        result["lives"].append(life)
        result["positions"].append(pos)
        result["sizes"].append(size)
        result["colors"].append(color)
        result["alphas"].append(alpha)
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
    particle's birth frame. Returns {"emitters": {object_id:
    result}, "total": live count, "scene_capped": the 10,000
    scene-wide soft cap tripped}."""
    world = scene.get("world")
    fps = scene.get("animation", {}).get("fps", 24)
    out = {"emitters": {}, "total": 0, "scene_capped": False}
    for obj in scene.get("objects", []):
        if obj.get("kind") != "emitter":
            continue
        params = obj.get("emitter")
        def transform_at(birth_frame, _obj=obj, _scene=scene):
            sampled = anim_mod.sample_object(_scene, _obj, birth_frame)
            return sampled["position"], sampled["rotation"]
        result = evaluate_emitter(params, world, fps, frame,
                                  transform_at=transform_at)
        out["emitters"][obj["id"]] = result
        out["total"] += result["count"]
    out["scene_capped"] = out["total"] > SCENE_SOFT_CAP
    return out
