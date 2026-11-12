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
- ``points``: point sprites with size attenuation. Written here so the
  program set is complete; nothing draws points until the particle
  task (T16/T17) uses it.

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

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = uNormalMatrix * aNormal;
    vUV = aUV;
    // Position in the key light's clip space, for the shadow test.
    vLightSpacePos = uLightSpace * worldPos;
    gl_Position = uProj * uView * worldPos;
}
"""

PBR_LITE_FRAG = """#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;
in vec4 vLightSpacePos;

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
        albedo = texture(uTex, vUV).rgb;
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

POINTS_VERT = """#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in float aSize;

uniform mat4 uView;
uniform mat4 uProj;
uniform float uPointScale;

out vec3 vColor;

void main() {
    vec4 eyePos = uView * vec4(aPos, 1.0);
    gl_Position = uProj * eyePos;
    float dist = max(1.0, -eyePos.z);
    gl_PointSize = aSize * uPointScale / dist;
    vColor = aColor;
}
"""

POINTS_FRAG = """#version 330 core
in vec3 vColor;

out vec4 FragColor;

void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = length(d);
    if (r > 0.5) {
        discard;
    }
    float alpha = smoothstep(0.5, 0.2, r);
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
