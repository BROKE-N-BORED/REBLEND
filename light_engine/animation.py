"""Keyframes: per-object position/rotation/scale tracks, linear
interpolation. Pure Python, no Qt.

Keys live in scene["animation"]["keys"]:
  { object_id: { "position": [[frame, [x, y, z]], ...],
                 "rotation": [...], "scale": [...] } }
Frames are ints, sorted by frame. Sampling outside the key range clamps
to the nearest key; an object with no keys keeps its scene transform.
"""
from __future__ import annotations

TRACKS = ("position", "rotation", "scale")


def get_keys(scene, object_id, track):
    keys = scene.get("animation", {}).get("keys", {})
    return list(keys.get(object_id, {}).get(track, []))


def set_key(scene, object_id, track, frame, value):
    anim = scene.setdefault("animation", {"fps": 24, "end_frame": 48,
                                          "keys": {}})
    tracks = anim.setdefault("keys", {}).setdefault(object_id, {})
    keys = [k for k in tracks.get(track, []) if k[0] != int(frame)]
    keys.append([int(frame), [float(v) for v in value]])
    keys.sort(key=lambda k: k[0])
    tracks[track] = keys


def delete_key(scene, object_id, track, frame):
    keys = scene.get("animation", {}).get("keys", {})
    tracks = keys.get(object_id)
    if not tracks:
        return
    tracks[track] = [k for k in tracks.get(track, []) if k[0] != int(frame)]
    if not any(tracks.get(t) for t in TRACKS):
        del keys[object_id]


def sample_track(keys, frame, fallback):
    """Linear interpolation of one track at a frame."""
    if not keys:
        return list(fallback)
    keys = sorted(keys, key=lambda k: k[0])
    if frame <= keys[0][0]:
        return list(keys[0][1])
    if frame >= keys[-1][0]:
        return list(keys[-1][1])
    for i in range(len(keys) - 1):
        f0, v0 = keys[i]
        f1, v1 = keys[i + 1]
        if f0 <= frame <= f1:
            t = 0.0 if f1 == f0 else (frame - f0) / (f1 - f0)
            return [v0[j] + (v1[j] - v0[j]) * t for j in range(3)]
    return list(fallback)


def sample_object(scene, obj, frame):
    """Full sampled transform for one object at a frame."""
    keys = scene.get("animation", {}).get("keys", {}).get(obj["id"], {})
    return {
        "position": sample_track(keys.get("position", []), frame,
                                 obj["position"]),
        "rotation": sample_track(keys.get("rotation", []), frame,
                                 obj["rotation"]),
        "scale": sample_track(keys.get("scale", []), frame, obj["scale"]),
    }


def keyframes_near(scene, object_id, frame):
    """True if the object has any key exactly on this frame."""
    keys = scene.get("animation", {}).get("keys", {}).get(object_id, {})
    return any(k[0] == int(frame)
               for track in TRACKS for k in keys.get(track, []))
