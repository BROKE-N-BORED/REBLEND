"""GLSL shader sources for the OpenGL 3.3 Core renderer (T1).

Four programs, all GLSL 330:

- ``pbr_lite``: the object program. Per-fragment Cook-Torrance PBR
  (GGX distribution, Schlick Fresnel, Smith geometry) with the
  scene's 3-point lights + ambient, roughness/metalness response,
  procedural texture sampling, fitted ACES tone mapping (T4), and
  one shadow map from the key light (T5).
- ``depth``: vertex-only program for the shadow pass (T5). Renders
  the scene from the key light's viewpoint into the depth map; the
  FBO has no color attachment, so the fragment shader is empty.
  Its position attribute is at location 0, the same index the
  object VAOs were built with — a different location would draw
  garbage in the shadow pass.
- ``lines``: color-only program for the grid, axes, and selection box
  (and the gizmo, in a later task).
- ``points``: point sprites with size attenuation and a per-particle
  alpha attribute (T17, location 3) for opacity over life. The
  viewport draws live particles with it: one merged VBO, two draw
  calls by blend mode (alpha for smoke/dust/snow, additive for
  fire/sparks).

The sources are plain module constants so they can be inspected and
tested without Qt. Compiling needs a current GL context; the compile
helpers import Qt locally and raise ShaderError with the driver's log
on failure — the viewport shows that message plainly and refuses to
guess.
"""
from __future__ import annotations

# -- pbr_lite -------------------------------------------------------------
# Attributes match the viewport's interleaved VBO layout: position (0),
# normal (1), uv (2). Units are cm, z-up, lights in world space.

PBR_LITE_VERT = """#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat3 uNormalMatrix;
uniform mat4 uLightSpace;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUV;
out vec4 vLightSpacePos;
out vec3 vLocalPos;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = uNormalMatrix * aNormal;
    vUV = aUV;
    // Position in the key light's clip space, for the shadow test.
    vLightSpacePos = uLightSpace * worldPos;
    // Object-local position in cm (PO4): the layout transform
    // samples the pattern from the surface itself, in real size,
    // so it needs the position before the model matrix (and before
    // the object's scale — sizes are exact on unscaled objects).
    vLocalPos = aPos;
    gl_Position = uProj * uView * worldPos;
}
"""

PBR_LITE_FRAG = """#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;
in vec4 vLightSpacePos;
in vec3 vLocalPos;

uniform vec3 uBaseColor;
uniform float uRoughness;
uniform float uMetalness;
uniform sampler2D uTex;
uniform int uUseTexture;
uniform vec3 uAmbient;
uniform vec3 uLightPos[3];
uniform vec3 uLightColor[3];
uniform vec3 uViewPos;
uniform float uExposure;
uniform sampler2D uShadowMap;
uniform int uUseShadow;
uniform int uLayoutMode;
uniform vec2 uLayoutSize;

out vec4 FragColor;

const float PI = 3.14159265359;

// Cook-Torrance BRDF, T4. GGX normal distribution, Schlick Fresnel,
// Smith geometry (Schlick-GGX both ways). Diffuse is energy-
// conserving: what the Fresnel term reflects cannot also diffuse,
// and metals get no diffuse at all. The output runs through the
// fitted ACES curve, so highlights roll off instead of clipping.
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return geometrySchlickGGX(NdotV, roughness)
         * geometrySchlickGGX(NdotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Fitted ACES tone-mapping curve (Narkowicz). Maps HDR light to
// [0, 1] with a filmic shoulder: midtones lift slightly, bright
// highlights compress instead of clipping to white.
vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Shadow test (T5): 1.0 lit, 0.0 fully in shadow. The fragment's
// light-space position is compared against the key light's depth
// map with a 3x3 PCF average, which softens the shadow edge. The
// bias keeps a surface from shadowing itself (shadow acne); the
// shadow pass also uses slope-scale polygon offset for the same
// reason. Fragments outside the light frustum are lit — the ortho
// box covers the scene, so that only happens off the edges.
float shadowFactor(vec4 lightSpacePos) {
    if (uUseShadow == 0) {
        return 1.0;
    }
    vec3 proj = lightSpacePos.xyz / lightSpacePos.w * 0.5 + 0.5;
    if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0
        || proj.z > 1.0) {
        return 1.0;
    }
    float bias = 0.001;
    float shadow = 0.0;
    vec2 texel = 1.0 / vec2(textureSize(uShadowMap, 0));
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float depth = texture(uShadowMap,
                                  proj.xy + vec2(x, y) * texel).r;
            shadow += (proj.z - bias > depth) ? 0.0 : 1.0;
        }
    }
    return shadow / 9.0;
}

// Layouts (PO4): when a layout is active, the pattern is sampled
// from the object's own surface in real centimetres instead of the
// mesh UVs, so a 12 cm board is 12 cm on any object. The face
// plane comes from the derivative face normal (cross of the
// position derivatives), which is robust for smooth-shaded objects
// where the interpolated normal varies across the face. Solid
// (mode 0) never reaches this code — the vUV path runs exactly as
// before, so existing renders are pixel-identical. Modes are the
// LAYOUT_SHADER_MODES values in textures.py. The branch on
// uLayoutMode is uniform (the mode is a uniform), so the
// derivatives inside are well-defined. Sizes are guarded against
// zero — a stored zero must never divide. The gap between pieces
// is 0.5 cm darkened on each piece's edge (LAYOUT_GAP_CM), so a
// joint reads about 1 cm wide; the joint darkens the albedo to
// 45% at the seam.

float layoutHash(vec2 p) {
    // A stable per-piece id hash in [0, 1): adjacent pieces sample
    // the pattern at different offsets, so the grain varies.
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float layoutEdge(float f, float w) {
    // 0 at the piece edge, 1 a distance w inside: the joint line.
    return smoothstep(0.0, w, f) * smoothstep(1.0, 1.0 - w, f);
}

// The herringbone staircase lattice: pieces L long, w wide, at 90
// degrees in a zigzag chain. Motif H [0,L]x[0,w], V [L,L+w]x[0,L],
// lattice b1=(L,L), b2=(w,-w). The indices are derived, not looped:
// for H the row is pinned by the y-range, for V the column by the
// x-range, and i comes from s=(x+y)/2L — only floor(s)-1 and
// floor(s) can match. Returns (sample u, sample v, gap factor,
// piece id hash). The sample runs along the grain, so H pieces
// sample (ly, lx) and V pieces (lx, ly) — the pattern's v axis is
// the grain direction. The Python version in textures.py
// (_staircase_piece) is the same math, proven by a dense-grid
// coverage check.
vec4 layoutStaircase(vec2 p, float L, float w) {
    float s = (p.x + p.y) / (2.0 * L);
    float base = floor(s);
    for (int k = 0; k < 2; k++) {
        float i = base - 1.0 + float(k);
        // H candidate: cy in (y-w, y] congruent to iL mod w.
        float cy = p.y - mod(p.y - i * L, w);
        float j = round((i * L - cy) / w);
        cy = i * L - j * w;
        float cx = i * L + j * w;
        if (cx <= p.x && p.x < cx + L && cy <= p.y && p.y < cy + w) {
            float lx = p.x - cx;
            float ly = p.y - cy;
            float edge = layoutEdge(lx / L, 0.5 / L)
                       * layoutEdge(ly / w, 0.5 / w);
            float id = layoutHash(vec2(i, j));
            return vec4(ly / 100.0 + id, lx / 100.0 + id * 0.37,
                        mix(0.45, 1.0, edge), id);
        }
        // V candidate: cx in (x-w, x] congruent to iL+L mod w.
        cx = p.x - mod(p.x - (i * L + L), w);
        j = round((cx - i * L - L) / w);
        cx = i * L + j * w + L;
        cy = i * L - j * w;
        if (cx <= p.x && p.x < cx + w && cy <= p.y && p.y < cy + L) {
            float lx = p.x - cx;
            float ly = p.y - cy;
            float edge = layoutEdge(lx / w, 0.5 / w)
                       * layoutEdge(ly / L, 0.5 / L);
            float id = layoutHash(vec2(i, j) + vec2(0.5, 0.0));
            return vec4(lx / 100.0 + id, ly / 100.0 + id * 0.37,
                        mix(0.45, 1.0, edge), id);
        }
    }
    // Unreachable: the lattice tiles the plane. Fall back to the
    // plain position so a driver rounding quirk still renders.
    return vec4(p.x / 100.0, p.y / 100.0, 1.0, 0.0);
}

// Layout sample: maps a surface point (in object-local cm) to the
// texture sample coordinate and a gap factor, as (u, v, gap).
vec3 layoutSample(vec3 localPos, int mode, vec2 size) {
    float w = max(size.x, 1.0);
    float L = max(size.y, 1.0);
    vec3 fn = normalize(cross(dFdx(localPos), dFdy(localPos)));
    vec3 a = abs(fn);
    vec2 p;
    if (a.x >= a.y && a.x >= a.z) {
        p = vec2(localPos.y, localPos.z);
    } else if (a.y >= a.z) {
        p = vec2(localPos.x, localPos.z);
    } else {
        p = vec2(localPos.x, localPos.y);
    }
    if (mode == 6) {
        // Tile: no cell transform — the baked tile (brick bonds)
        // repeats at real-world size, one tile per metre.
        return vec3(p / 100.0, 1.0);
    }
    if (mode == 3) {
        // Chevron: the staircase in 45-degree-rotated coordinates,
        // so the rectangles become mitred parallelograms.
        float c = 0.7071067811865476;
        vec2 r = vec2((p.x - p.y) * c, (p.x + p.y) * c);
        vec4 st = layoutStaircase(r, L, w);
        return vec3(st.xy, st.z);
    }
    if (mode == 4) {
        vec4 st = layoutStaircase(p, L, w);
        return vec3(st.xy, st.z);
    }
    if (mode == 5) {
        // Slabs: a plain grid, w by L.
        vec2 cell = vec2(mod(p.x, w), mod(p.y, L));
        vec2 id = vec2(floor(p.x / w), floor(p.y / L));
        float h = layoutHash(id);
        float h2 = layoutHash(id + vec2(13.7, 7.1));
        float edge = layoutEdge(cell.x / w, 0.5 / w)
                   * layoutEdge(cell.y / L, 0.5 / L);
        return vec3(cell.x / 100.0 + h, cell.y / 100.0 + h2,
                    mix(0.45, 1.0, edge));
    }
    if (mode == 2) {
        // Blocks: squares of side L, each holding parallel pieces
        // w wide; the pieces run along x in even blocks and along y
        // in odd blocks (a checkerboard of orientations). Parquet
        // and basket weave share this mode with different sizes.
        vec2 b = vec2(floor(p.x / L), floor(p.y / L));
        vec2 local = vec2(mod(p.x, L), mod(p.y, L));
        float orient = mod(b.x + b.y, 2.0);
        float edge;
        vec2 q;
        if (orient < 0.5) {
            float strip = floor(local.y / w);
            float ly = local.y - strip * w;
            float h = layoutHash(b * 7.0 + vec2(strip, 0.0));
            float h2 = layoutHash(b * 7.0 + vec2(strip, 3.0));
            edge = layoutEdge(local.x / L, 0.5 / L)
                 * layoutEdge(ly / w, 0.5 / w);
            q = vec2(ly / 100.0 + h, local.x / 100.0 + h2);
        } else {
            float strip = floor(local.x / w);
            float lx = local.x - strip * w;
            float h = layoutHash(b * 7.0 + vec2(strip, 1.0));
            float h2 = layoutHash(b * 7.0 + vec2(strip, 4.0));
            edge = layoutEdge(lx / w, 0.5 / w)
                 * layoutEdge(local.y / L, 0.5 / L);
            q = vec2(lx / 100.0 + h, local.y / 100.0 + h2);
        }
        return vec3(q, mix(0.45, 1.0, edge));
    }
    // Mode 1 (boards, and strip as narrow boards): rows w tall,
    // each row offset by half a board length on alternate rows, so
    // the joints stagger. Grain runs along the board: the sample's
    // v axis is the position along the length.
    float row = floor(p.y / w);
    float shift = (mod(row, 2.0) == 0.0) ? 0.0 : 0.5 * L;
    float x = p.x + shift;
    float col = floor(x / L);
    float lx = x - col * L;
    float ly = p.y - row * w;
    vec2 id = vec2(col, row);
    float h = layoutHash(id);
    float h2 = layoutHash(id + vec2(13.7, 7.1));
    float edge = layoutEdge(lx / L, 0.5 / L)
               * layoutEdge(ly / w, 0.5 / w);
    return vec3(ly / 100.0 + h, lx / 100.0 + h2,
                mix(0.45, 1.0, edge));
}

void main() {
    vec3 N = normalize(vNormal);
    // Two-sided lighting, as the fixed-function renderer had: back
    // faces (the inside of an open shape) light with a flipped normal.
    if (!gl_FrontFacing) {
        N = -N;
    }
    vec3 albedo = uBaseColor;
    if (uUseTexture == 1) {
        // The procedural textures already carry the base color.
        // A layout (mode != 0) resamples from the surface in real
        // centimetres and darkens the joints; solid (mode 0) uses
        // the mesh UVs exactly as before.
        if (uLayoutMode == 0) {
            albedo = texture(uTex, vUV).rgb;
        } else {
            vec3 lay = layoutSample(vLocalPos, uLayoutMode, uLayoutSize);
            albedo = texture(uTex, lay.xy).rgb * lay.z;
        }
    }
    vec3 V = normalize(uViewPos - vWorldPos);
    // Reflectance at normal incidence: dielectrics 0.04, metals the
    // albedo color. Same mix the PBR metallic/roughness model uses.
    vec3 F0 = mix(vec3(0.04), albedo, uMetalness);
    // A roughness floor keeps the GGX denominator sane; a perfectly
    // smooth surface would be a mirror with an infinitely tight lobe.
    float roughness = clamp(uRoughness, 0.05, 1.0);
    vec3 color = albedo * uAmbient * (1.0 - uMetalness);
    for (int i = 0; i < 3; i++) {
        vec3 L = normalize(uLightPos[i] - vWorldPos);
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 0.0) {
            continue;
        }
        vec3 H = normalize(L + V);
        float NdotV = max(dot(N, V), 0.0);
        float VdotH = max(dot(V, H), 0.0);
        float D = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(VdotH, F0);
        vec3 specular = D * F * G / max(4.0 * NdotV * NdotL, 0.001);
        vec3 kS = F;
        vec3 kD = (vec3(1.0) - kS) * (1.0 - uMetalness);
        vec3 diffuse = kD * albedo / PI;
        // Only the key light casts a shadow (light slot 0 — the
        // viewport always puts the key light there). Ambient, fill
        // and rim stay unshadowed.
        float shadow = (i == 0) ? shadowFactor(vLightSpacePos) : 1.0;
        color += shadow * uLightColor[i] * (diffuse + specular) * NdotL;
    }
    // Environment reflection: this engine has no environment map, so
    // a bare metal would reflect nothing and render black. A cheap
    // vertical gradient — warm ground below, cool sky above —
    // sampled in the reflection direction gives metals something to
    // reflect. Scaled by the Fresnel term (metals reflect their
    // albedo color, dielectrics only ~4%) and faded with roughness,
    // which stands in for the blur a rough surface would apply.
    vec3 R = reflect(-V, N);
    float skyMix = clamp(R.z * 0.5 + 0.5, 0.0, 1.0);
    vec3 envColor = mix(vec3(0.32, 0.26, 0.19),
                        vec3(0.52, 0.62, 0.76), skyMix);
    vec3 Fenv = fresnelSchlick(max(dot(N, V), 0.0), F0);
    color += envColor * Fenv * mix(1.0, 0.25, roughness);
    FragColor = vec4(aces(color * uExposure), 1.0);
}
"""

# -- depth ----------------------------------------------------------------
# The shadow pass program (T5). Position at location 0 — the same
# index the object VAOs bind position to, so the shadow pass draws
# through the very same VAOs as the main pass. The FBO has no color
# attachment, so the fragment shader is an empty main.

DEPTH_VERT = """#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat4 uModel;
uniform mat4 uLightSpace;

void main() {
    gl_Position = uLightSpace * uModel * vec4(aPos, 1.0);
}
"""

DEPTH_FRAG = """#version 330 core
// Depth only: nothing is written but the depth buffer.
void main() {
}
"""

# -- lines ----------------------------------------------------------------
# Per-vertex colors, one model matrix. The grid passes identity; the
# selection box passes its world transform.

LINES_VERT = """#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vColor;

void main() {
    vColor = aColor;
    gl_Position = uProj * uView * uModel * vec4(aPos, 1.0);
}
"""

LINES_FRAG = """#version 330 core
in vec3 vColor;

out vec4 FragColor;

void main() {
    FragColor = vec4(vColor, 1.0);
}
"""

# -- points ---------------------------------------------------------------
# Point sprites with size attenuation: aSize is the world-space size,
# uPointScale is viewport_height / (2 * tan(fov / 2)) at draw time.
# aAlpha (location 3, T17) is the per-particle opacity from the sim's
# fade in/out over life; the fragment shader multiplies it by the
# soft-circle falloff, so a particle fades both at its edge and over
# its life. The viewport draws the two blend modes (alpha, additive)
# as two draw calls from the one merged VBO.

POINTS_VERT = """#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in float aSize;
layout(location = 3) in float aAlpha;

uniform mat4 uView;
uniform mat4 uProj;
uniform float uPointScale;

out vec3 vColor;
out float vAlpha;

void main() {
    vec4 eyePos = uView * vec4(aPos, 1.0);
    gl_Position = uProj * eyePos;
    float dist = max(1.0, -eyePos.z);
    gl_PointSize = aSize * uPointScale / dist;
    vColor = aColor;
    vAlpha = aAlpha;
}
"""

POINTS_FRAG = """#version 330 core
in vec3 vColor;
in float vAlpha;

out vec4 FragColor;

void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = length(d);
    if (r > 0.5) {
        discard;
    }
    float alpha = smoothstep(0.5, 0.2, r) * vAlpha;
    FragColor = vec4(vColor, alpha);
}
"""

PROGRAMS = {
    "pbr_lite": (PBR_LITE_VERT, PBR_LITE_FRAG),
    "depth": (DEPTH_VERT, DEPTH_FRAG),
    "lines": (LINES_VERT, LINES_FRAG),
    "points": (POINTS_VERT, POINTS_FRAG),
}


class ShaderError(Exception):
    """A shader failed to compile or link. The message carries the
    driver's log, in plain words the user can report."""


def compile_program(name):
    """Compile and link one program by name. Needs a current GL
    context. Raises ShaderError on any failure."""
    from PySide6.QtOpenGL import (QOpenGLShader, QOpenGLShaderProgram)

    try:
        vert_src, frag_src = PROGRAMS[name]
    except KeyError:
        raise ShaderError(f"Unknown shader program: {name}") from None
    prog = QOpenGLShaderProgram()
    if not prog.addShaderFromSourceCode(QOpenGLShader.Vertex, vert_src):
        raise ShaderError(
            f"Vertex shader failed to compile ({name}): {prog.log()}")
    if not prog.addShaderFromSourceCode(QOpenGLShader.Fragment, frag_src):
        raise ShaderError(
            f"Fragment shader failed to compile ({name}): {prog.log()}")
    if not prog.link():
        raise ShaderError(
            f"Shader program failed to link ({name}): {prog.log()}")
    return prog


def compile_all():
    """Compile and link every program. Returns {name: program}."""
    return {name: compile_program(name) for name in PROGRAMS}
