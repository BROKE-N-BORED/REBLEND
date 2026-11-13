"""Unit tests for Reblend v0.2. Standard library unittest only.

Run from the reblend folder:  bash test.sh
Qt tests run only when LIGHT_ENGINE_QT_TESTS=1 (they need a display;
under Xvfb + xcb, QOpenGLWidget works — it does not under offscreen).
"""
import base64
import json
import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from light_engine import animation as anim_mod  # noqa: E402
from light_engine import geometry  # noqa: E402
from light_engine import meshfile  # noqa: E402
from light_engine import meshio  # noqa: E402
from light_engine import meshio_glb  # noqa: E402
from light_engine import meshio_obj  # noqa: E402
from light_engine import scene as scene_mod  # noqa: E402
from light_engine import shaders  # noqa: E402
from light_engine import textures  # noqa: E402

import glb_fixture  # noqa: E402

EXAMPLE = HERE / "examples" / "tavern-tabletop.json"
MESH_DEMO = HERE / "examples" / "mesh-demo.json"
GEM_ASSET = HERE / "examples" / "assets" / "gem.mesh.json"
HOUSE_OBJ = HERE / "examples" / "house.obj"


_SAMPLE_TMPDIR = None


def _sample_glb_path():
    """The sample.glb path. The repo carries only sample.glb.b64 (the
    snapshot push cannot carry binaries), so a fresh clone has no
    sample.glb — decode the .b64 in that case. The decoded file is
    named sample.glb inside a temp dir (held for the run): the import
    names its assets from the source file's stem, so a random temp
    name would break the asset-name assertions. When neither file
    exists the real path is returned and the sample tests fail as
    before (a missing fixture, not an import error)."""
    global _SAMPLE_TMPDIR
    real = HERE / "examples" / "sample.glb"
    if real.exists():
        return real
    b64 = HERE / "examples" / "sample.glb.b64"
    if b64.exists():
        _SAMPLE_TMPDIR = tempfile.TemporaryDirectory(
            prefix="reblend-sample-")
        path = Path(_SAMPLE_TMPDIR.name) / "sample.glb"
        path.write_bytes(base64.b64decode(b64.read_text()))
        return path
    return real


SAMPLE_GLB = _sample_glb_path()
QT_TESTS = os.environ.get("LIGHT_ENGINE_QT_TESTS") == "1"


class SceneTests(unittest.TestCase):
    def test_new_scene_defaults(self):
        scene = scene_mod.new_scene("Test")
        self.assertEqual(scene["name"], "Test")
        self.assertEqual(scene["objects"], [])
        self.assertEqual(len(scene["lights"]), 4)
        self.assertEqual(scene["animation"]["fps"], 24)

    def test_material_defaults_and_clamps(self):
        mat = scene_mod.default_material({"roughness": 5.0, "metalness": -1.0,
                                          "texture": "bogus"})
        self.assertEqual(mat["roughness"], 1.0)
        self.assertEqual(mat["metalness"], 0.0)
        self.assertEqual(mat["texture"], "none")
        self.assertEqual(mat["base_color"], "#b0b0b0")

    def test_new_object_has_id_and_defaults(self):
        obj = scene_mod.new_object("sphere")
        self.assertTrue(obj["id"])
        self.assertEqual(obj["shape"], "sphere")
        self.assertEqual(obj["position"], [0.0, 0.0, 0.0])
        self.assertEqual(obj["scale"], [1.0, 1.0, 1.0])

    def test_round_trip(self):
        scene = scene_mod.new_scene("Round")
        scene["objects"].append(scene_mod.new_object(
            "box", name="Crate", position=[10.0, 20.0, 30.0]))
        scene["cameras"].append({"name": "Cam1", "position": [1, 2, 3],
                                 "look_at": [0, 0, 0]})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.json")
            scene_mod.save_scene(scene, path)
            loaded = scene_mod.load_scene(path)
        self.assertEqual(loaded["name"], "Round")
        self.assertEqual(len(loaded["objects"]), 1)
        self.assertEqual(loaded["objects"][0]["name"], "Crate")
        self.assertEqual(loaded["objects"][0]["position"], [10.0, 20.0, 30.0])
        self.assertEqual(len(loaded["cameras"]), 1)

    def test_normalize_tolerates_missing_fields(self):
        scene = scene_mod.normalize_scene({"objects": [{"shape": "cone"}]})
        obj = scene["objects"][0]
        self.assertEqual(obj["shape"], "cone")
        self.assertTrue(obj["id"])
        self.assertEqual(obj["material"]["texture"], "none")

    def test_hex_to_rgb(self):
        self.assertEqual(scene_mod.hex_to_rgb("#ff0000"), (1.0, 0.0, 0.0))
        self.assertEqual(scene_mod.hex_to_rgb("bad"), (0.7, 0.7, 0.7))

    def test_find_object(self):
        scene = scene_mod.new_scene()
        obj = scene_mod.new_object("box")
        scene["objects"].append(obj)
        self.assertIs(scene_mod.find_object(scene, obj["id"]), obj)
        self.assertIsNone(scene_mod.find_object(scene, "nope"))

    def test_scene_copy_is_deep(self):
        scene = scene_mod.new_scene()
        scene["objects"].append(scene_mod.new_object("box"))
        copy = scene_mod.scene_copy(scene)
        copy["objects"][0]["name"] = "Changed"
        self.assertNotEqual(scene["objects"][0]["name"], "Changed")


class GeometryTests(unittest.TestCase):
    def test_box_has_12_tris(self):
        mesh = geometry.tessellate("box", {"size": [10, 20, 30]})
        self.assertEqual(mesh["tri_count"], 12)
        self.assertEqual(len(mesh["positions"]), 12 * 9)

    def test_plane_has_2_tris(self):
        self.assertEqual(geometry.tessellate("plane")["tri_count"], 2)

    def test_each_shape_produces_triangles(self):
        for shape in geometry.SHAPES:
            mesh = geometry.tessellate(shape, geometry.default_params(shape))
            self.assertGreater(mesh["tri_count"], 0, shape)
            n = len(mesh["positions"])
            self.assertEqual(len(mesh["normals"]), n, shape)
            self.assertEqual(len(mesh["uvs"]), n // 3 * 2, shape)

    def test_unknown_shape_falls_back_to_box(self):
        self.assertEqual(geometry.tessellate("weird")["tri_count"], 12)

    def test_cone_is_cylinder_with_zero_top(self):
        cone = geometry.tessellate("cone", {"radius": 10, "height": 20,
                                            "segments": 8})
        cyl = geometry.tessellate("cylinder", {"radius": 10, "height": 20,
                                               "segments": 8})
        self.assertLess(cone["tri_count"], cyl["tri_count"])

    def test_local_bounds(self):
        mn, mx = geometry.local_bounds("box", {"size": [10, 20, 30]})
        self.assertEqual(mn, (-5.0, -10.0, -15.0))
        self.assertEqual(mx, (5.0, 10.0, 15.0))

    def test_compose_and_apply_matrix(self):
        m = geometry.compose_matrix([10, 0, 0], [0, 0, 0], [1, 1, 1])
        self.assertEqual(geometry.apply_matrix(m, (1, 2, 3)), (11.0, 2.0, 3.0))
        m = geometry.compose_matrix([0, 0, 0], [0, 0, 90], [1, 1, 1])
        x, y, z = geometry.apply_matrix(m, (1, 0, 0))
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 1.0, places=6)
        self.assertAlmostEqual(z, 0.0, places=6)

    def test_matrix_inverse_round_trip(self):
        m = geometry.compose_matrix([5, -3, 2], [10, 20, 30], [2, 1, 0.5])
        inv = geometry.mat_inverse(m)
        point = (1.0, 2.0, 3.0)
        back = geometry.apply_matrix(inv, geometry.apply_matrix(m, point))
        for a, b in zip(back, point):
            self.assertAlmostEqual(a, b, places=6)

    def test_ray_hits_aabb(self):
        t = geometry.ray_hits_aabb((0, 0, -10), (0, 0, 1),
                                   (-1, -1, -1), (1, 1, 1))
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 9.0)
        self.assertIsNone(geometry.ray_hits_aabb((5, 5, -10), (0, 0, 1),
                                                 (-1, -1, -1), (1, 1, 1)))


class IndexedGeometryTests(unittest.TestCase):
    """tessellate_indexed (T2): same triangles as the soup, shared
    vertices, valid indices."""

    def test_indices_valid_and_in_range(self):
        for shape in geometry.SHAPES:
            mesh = geometry.tessellate_indexed(
                shape, geometry.default_params(shape))
            verts = len(mesh["positions"]) // 3
            self.assertEqual(len(mesh["indices"]),
                             mesh["tri_count"] * 3, shape)
            self.assertEqual(len(mesh["normals"]),
                             len(mesh["positions"]), shape)
            self.assertEqual(len(mesh["uvs"]), verts * 2, shape)
            for i in mesh["indices"]:
                self.assertGreaterEqual(i, 0, shape)
                self.assertLess(i, verts, shape)

    def test_sphere_shares_vertices(self):
        soup = geometry.tessellate("sphere",
                                   geometry.default_params("sphere"))
        indexed = geometry.tessellate_indexed(
            "sphere", geometry.default_params("sphere"))
        self.assertLess(len(indexed["positions"]),
                        len(soup["positions"]))

    def test_same_triangle_count_as_soup(self):
        for shape in geometry.SHAPES:
            params = geometry.default_params(shape)
            soup = geometry.tessellate(shape, params)
            indexed = geometry.tessellate_indexed(shape, params)
            self.assertEqual(indexed["tri_count"],
                             soup["tri_count"], shape)

    def test_indexed_triangles_match_soup(self):
        """Triangle i of the indexed mesh is triangle i of the soup,
        vertex for vertex."""
        for shape in geometry.SHAPES:
            params = geometry.default_params(shape)
            soup = geometry.tessellate(shape, params)
            indexed = geometry.tessellate_indexed(shape, params)
            for v in range(len(soup["positions"]) // 3):
                idx = indexed["indices"][v]
                got = tuple(indexed["positions"][idx * 3 + j]
                            for j in range(3))
                want = tuple(soup["positions"][v * 3 + j]
                             for j in range(3))
                self.assertEqual(got, want, (shape, v))

    def test_indexed_normals_are_unit_length(self):
        for shape in geometry.SHAPES:
            mesh = geometry.tessellate_indexed(
                shape, geometry.default_params(shape))
            for v in range(len(mesh["positions"]) // 3):
                n = mesh["normals"][v * 3:v * 3 + 3]
                length = math.sqrt(sum(c * c for c in n))
                self.assertAlmostEqual(length, 1.0, places=6, msg=shape)

    def test_unknown_shape_falls_back_to_box(self):
        self.assertEqual(geometry.tessellate_indexed("weird")["tri_count"],
                         12)


class SmoothShadingTests(unittest.TestCase):
    """Per-object smooth shading (T3): shape defaults, save/load,
    and the angle-threshold weld fix."""

    def test_smooth_defaults_by_shape(self):
        smooth = ("sphere", "cylinder", "cone", "torus")
        flat = ("box", "plane")
        for shape in smooth:
            self.assertTrue(
                scene_mod.new_object(shape)["smooth_shading"], shape)
        for shape in flat:
            self.assertFalse(
                scene_mod.new_object(shape)["smooth_shading"], shape)

    def test_normalize_preserves_flag(self):
        obj = scene_mod.normalize_object({"shape": "box",
                                          "smooth_shading": True})
        self.assertTrue(obj["smooth_shading"])
        obj = scene_mod.normalize_object({"shape": "sphere",
                                          "smooth_shading": False})
        self.assertFalse(obj["smooth_shading"])
        # Missing or mistyped values fall back to the shape default.
        obj = scene_mod.normalize_object({"shape": "sphere"})
        self.assertTrue(obj["smooth_shading"])
        obj = scene_mod.normalize_object({"shape": "box"})
        self.assertFalse(obj["smooth_shading"])
        obj = scene_mod.normalize_object({"shape": "box",
                                          "smooth_shading": "yes"})
        self.assertFalse(obj["smooth_shading"])

    def test_smooth_flag_round_trips(self):
        scene = scene_mod.new_scene()
        box = scene_mod.new_object("box")
        box["smooth_shading"] = True
        sphere = scene_mod.new_object("sphere")
        sphere["smooth_shading"] = False
        scene["objects"] = [box, sphere]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.json")
            scene_mod.save_scene(scene, path)
            loaded = scene_mod.load_scene(path)
        self.assertTrue(loaded["objects"][0]["smooth_shading"])
        self.assertFalse(loaded["objects"][1]["smooth_shading"])
        self.assertEqual(loaded["version"], 2)

    def test_indexed_box_keeps_24_vertices(self):
        """The angle threshold keeps box faces apart: all 24 corners
        survive (T2's weld softened 4 of them into 20)."""
        mesh = geometry.tessellate_indexed(
            "box", geometry.default_params("box"))
        self.assertEqual(len(mesh["positions"]) // 3, 24)

    def test_indexed_box_normals_are_axis_aligned(self):
        """No averaged normals on a box: every indexed box normal is
        a unit axis direction, never a blend of two faces."""
        mesh = geometry.tessellate_indexed(
            "box", geometry.default_params("box"))
        axes = {(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)}
        for v in range(len(mesh["positions"]) // 3):
            n = tuple(round(c, 6)
                      for c in mesh["normals"][v * 3:v * 3 + 3])
            self.assertIn(n, axes)

    def test_sphere_indexed_normals_are_averaged(self):
        """Welding averages face normals: at least one indexed vertex
        normal differs from the face normal of its soup triangle."""
        params = geometry.default_params("sphere")
        soup = geometry.tessellate("sphere", params)
        indexed = geometry.tessellate_indexed("sphere", params)
        averaged = 0
        for v in range(len(soup["positions"]) // 3):
            idx = indexed["indices"][v]
            face = soup["normals"][v * 3:v * 3 + 3]
            smooth = indexed["normals"][idx * 3:idx * 3 + 3]
            if any(abs(a - b) > 1e-6 for a, b in zip(face, smooth)):
                averaged += 1
        self.assertGreater(averaged, 0)


class AnimationTests(unittest.TestCase):
    def test_linear_midpoint(self):
        keys = [[0, [0.0, 0.0, 0.0]], [10, [10.0, 0.0, 0.0]]]
        self.assertEqual(anim_mod.sample_track(keys, 5, [0, 0, 0]),
                         [5.0, 0.0, 0.0])

    def test_clamps_outside_range(self):
        keys = [[10, [1.0, 2.0, 3.0]], [20, [4.0, 5.0, 6.0]]]
        self.assertEqual(anim_mod.sample_track(keys, 0, [9, 9, 9]),
                         [1.0, 2.0, 3.0])
        self.assertEqual(anim_mod.sample_track(keys, 99, [9, 9, 9]),
                         [4.0, 5.0, 6.0])

    def test_no_keys_returns_fallback(self):
        self.assertEqual(anim_mod.sample_track([], 7, [1, 2, 3]), [1, 2, 3])

    def test_set_and_delete_key(self):
        scene = scene_mod.new_scene()
        anim_mod.set_key(scene, "obj1", "position", 5, [1, 2, 3])
        anim_mod.set_key(scene, "obj1", "position", 5, [4, 5, 6])
        keys = anim_mod.get_keys(scene, "obj1", "position")
        self.assertEqual(len(keys), 1)  # same frame replaces
        self.assertEqual(keys[0][1], [4.0, 5.0, 6.0])
        anim_mod.delete_key(scene, "obj1", "position", 5)
        self.assertEqual(anim_mod.get_keys(scene, "obj1", "position"), [])

    def test_sample_object_uses_fallbacks(self):
        scene = scene_mod.new_scene()
        obj = scene_mod.new_object("box", position=[7.0, 8.0, 9.0])
        sampled = anim_mod.sample_object(scene, obj, 3)
        self.assertEqual(sampled["position"], [7.0, 8.0, 9.0])
        anim_mod.set_key(scene, obj["id"], "position", 0, [0, 0, 0])
        anim_mod.set_key(scene, obj["id"], "position", 10, [10, 0, 0])
        self.assertEqual(anim_mod.sample_object(scene, obj, 5)["position"],
                         [5.0, 0.0, 0.0])


class TextureTests(unittest.TestCase):
    def test_factor_is_deterministic(self):
        for name in textures.TEXTURE_NAMES:
            a = textures.pattern_factor(name, 0.3, 0.7)
            b = textures.pattern_factor(name, 0.3, 0.7)
            self.assertEqual(a, b)

    def test_factor_stays_in_sane_range(self):
        for name in textures.TEXTURE_NAMES:
            for u in (0.0, 0.25, 0.5, 0.75, 1.0):
                for v in (0.0, 0.5, 1.0):
                    f = textures.pattern_factor(name, u, v)
                    self.assertGreater(f, 0.4, (name, u, v))
                    self.assertLess(f, 1.6, (name, u, v))

    def test_none_is_flat(self):
        self.assertEqual(textures.pattern_factor("none", 0.1, 0.9), 1.0)

    def test_patterns_vary(self):
        values = {textures.pattern_factor("wood_grain", u / 10.0, 0.5)
                  for u in range(10)}
        self.assertGreater(len(values), 1)

    def _grain_std(self, strength):
        """Standard deviation of the wood-grain factor over a grid."""
        values = [textures.pattern_factor("wood_grain", u / 20.0, v / 20.0,
                                          strength)
                  for u in range(21) for v in range(21)]
        mean = sum(values) / len(values)
        return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

    def test_wood_grain_default_is_tamer_than_v01(self):
        # 1.0 is the v0.1 look (too strong and wavy on big planes).
        # The lowered default must cut the pattern contrast measurably
        # while still varying — it has to read as wood, not flat paint.
        old = self._grain_std(1.0)
        new = self._grain_std(textures.WOOD_GRAIN_STRENGTH)
        self.assertLess(textures.WOOD_GRAIN_STRENGTH, 1.0)
        self.assertLess(new, 0.6 * old)
        self.assertGreater(new, 0.0)

    def test_wood_grain_strength_zero_is_flat(self):
        for u, v in ((0.1, 0.2), (0.5, 0.5), (0.9, 0.1)):
            self.assertEqual(
                textures.pattern_factor("wood_grain", u, v, 0.0), 0.88)

    def test_pattern_rgb_clamps(self):
        r, g, b = textures.pattern_rgb("wood_grain", 0.2, 0.8, (1.0, 1.0, 1.0))
        self.assertTrue(0.0 <= r <= 1.0 and 0.0 <= g <= 1.0
                        and 0.0 <= b <= 1.0)


class FileIOTests(unittest.TestCase):
    def test_example_scene_loads_and_is_light(self):
        self.assertTrue(EXAMPLE.exists(), "example scene missing")
        scene = scene_mod.load_scene(str(EXAMPLE))
        self.assertGreater(len(scene["objects"]), 5)
        self.assertGreaterEqual(len(scene["cameras"]), 1)
        tris = scene_mod.total_triangles(scene)
        self.assertLess(tris, 10000, f"example scene is {tris} tris")
        shapes = {o["shape"] for o in scene["objects"]}
        self.assertIn("box", shapes)
        self.assertIn("torus", shapes)

    def test_save_produces_valid_json(self):
        scene = scene_mod.new_scene()
        scene["objects"].append(scene_mod.new_object("torus"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.json")
            scene_mod.save_scene(scene, path)
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
        self.assertEqual(raw["objects"][0]["shape"], "torus")


class ShaderSourceTests(unittest.TestCase):
    """Headless checks on the GLSL sources and the draw path (T1)."""

    def test_all_four_programs_exist(self):
        self.assertEqual(set(shaders.PROGRAMS),
                         {"pbr_lite", "depth", "lines", "points"})

    def test_all_sources_are_glsl_330_core(self):
        for name, (vert, frag) in shaders.PROGRAMS.items():
            self.assertTrue(vert.startswith("#version 330 core"), name)
            self.assertTrue(frag.startswith("#version 330 core"), name)

    def test_pbr_lite_has_the_uniforms_the_viewport_sets(self):
        vert, frag = shaders.PROGRAMS["pbr_lite"]
        for uniform in ("uModel", "uView", "uProj", "uNormalMatrix",
                        "uLightSpace"):
            self.assertIn(uniform, vert, uniform)
        for uniform in ("uBaseColor", "uRoughness", "uMetalness", "uTex",
                        "uUseTexture", "uAmbient", "uLightPos[3]",
                        "uLightColor[3]", "uViewPos", "uExposure",
                        "uShadowMap", "uUseShadow"):
            self.assertIn(uniform, frag, uniform)

    def test_lines_and_points_have_their_uniforms(self):
        vert, _frag = shaders.PROGRAMS["lines"]
        for uniform in ("uModel", "uView", "uProj"):
            self.assertIn(uniform, vert, uniform)
        vert, _frag = shaders.PROGRAMS["points"]
        for uniform in ("uView", "uProj", "uPointScale"):
            self.assertIn(uniform, vert, uniform)

    def test_viewport_has_no_fixed_function_calls(self):
        """T1 acceptance: the draw path is shaders only. Any of these
        tokens back in viewport.py means the fixed-function renderer
        crept back in."""
        # GL_TEXTURE_2D is deliberately NOT in this list: it is a
        # core-profile texture target, and T5 uses it to attach the
        # depth texture to the shadow FBO. What stays forbidden is
        # the fixed-function use, glEnable(GL_TEXTURE_2D), pinned
        # below as a full call.
        forbidden = (
            "glBegin", "glEnd", "glLight", "glMaterial", "glColor3f",
            "glMatrixMode", "glLoadMatrix", "glPushMatrix", "glPopMatrix",
            "glMultMatrix", "glEnableClientState", "glVertexPointer",
            "glNormalPointer", "glTexCoordPointer",
            "glEnable(GL_TEXTURE_2D)",
            "GL_LIGHTING", "GL_COLOR_MATERIAL", "GL_LIGHT0", "GL_LIGHT1",
            "GL_LIGHT2", "GL_PROJECTION", "GL_MODELVIEW",
            "GL_FRONT", "GL_AMBIENT", "GL_DIFFUSE", "GL_SPECULAR",
            "GL_POSITION", "GL_SHININESS", "GL_LIGHT_MODEL_AMBIENT",
            "GL_LIGHT_MODEL_TWO_SIDE", "GL_VERTEX_ARRAY", "GL_NORMAL_ARRAY",
            "GL_TEXTURE_COORD_ARRAY",
            "QOpenGLFunctions_2_1", "CompatibilityProfile",
        )
        source = (HERE / "light_engine" / "viewport.py").read_text(
            encoding="utf-8")
        for token in forbidden:
            self.assertNotIn(token, source, token)

    def test_viewport_requests_core_profile(self):
        source = (HERE / "light_engine" / "viewport.py").read_text(
            encoding="utf-8")
        self.assertIn("QOpenGLFunctions_3_3_Core", source)
        self.assertIn("CoreProfile", source)
        self.assertIn("setVersion(3, 3)", source)


class PbrAcesTests(unittest.TestCase):
    """Headless checks on the T4 BRDF + tone mapping: the fragment
    shader must be a real Cook-Torrance PBR-lite with fitted ACES
    output, not the old Blinn-Phong specular."""

    def test_fragment_has_cook_torrance_terms(self):
        _vert, frag = shaders.PROGRAMS["pbr_lite"]
        for term in ("distributionGGX", "geometrySmith",
                     "geometrySchlickGGX", "fresnelSchlick"):
            self.assertIn(term, frag, term)

    def test_fragment_has_aces_and_exposure(self):
        _vert, frag = shaders.PROGRAMS["pbr_lite"]
        self.assertIn("aces(", frag)
        self.assertIn("uniform float uExposure;", frag)
        self.assertIn("aces(color * uExposure)", frag)

    def test_fragment_is_energy_conserving(self):
        """Diffuse is scaled by one minus Fresnel and by one minus
        metalness; metals get no diffuse at all."""
        _vert, frag = shaders.PROGRAMS["pbr_lite"]
        self.assertIn("kS = F", frag)
        self.assertIn("kD = (vec3(1.0) - kS) * (1.0 - uMetalness)", frag)
        self.assertIn("diffuse = kD * albedo / PI", frag)
        self.assertIn("albedo * uAmbient * (1.0 - uMetalness)", frag)

    def test_no_blinn_phong_specular_left(self):
        _vert, frag = shaders.PROGRAMS["pbr_lite"]
        self.assertNotIn("shininess", frag)
        self.assertNotIn("spec = pow", frag)

    def test_metals_get_environment_reflection(self):
        """Without an environment term a bare metal reflects nothing
        and renders black — the gradient env reflection is what makes
        metal read as metal."""
        _vert, frag = shaders.PROGRAMS["pbr_lite"]
        self.assertIn("reflect(-V, N)", frag)
        self.assertIn("envColor", frag)
        self.assertIn("Fenv", frag)

    def test_viewport_sets_exposure(self):
        source = (HERE / "light_engine" / "viewport.py").read_text(
            encoding="utf-8")
        self.assertIn('"uExposure"', source)
        self.assertIn("self._exposure", source)


class ShadowTests(unittest.TestCase):
    """One shadow map from the key light (T5): the depth program,
    the shadow test in the pbr_lite shader, and the per-object
    cast/receive flags."""

    def test_depth_program_is_position_only_at_location_zero(self):
        """The depth program draws through the object VAOs, whose
        position attribute is at index 0 — a different location
        would draw garbage in the shadow pass."""
        vert, frag = shaders.PROGRAMS["depth"]
        self.assertIn("layout(location = 0) in vec3 aPos", vert)
        self.assertIn("uniform mat4 uModel", vert)
        self.assertIn("uniform mat4 uLightSpace", vert)
        self.assertIn("uLightSpace * uModel", vert)
        # Depth only: no color output, no texture sampling.
        self.assertNotIn("FragColor", frag)
        self.assertNotIn("texture(", frag)

    def test_pbr_lite_vertex_outputs_light_space_position(self):
        vert, _frag = shaders.PROGRAMS["pbr_lite"]
        self.assertIn("uniform mat4 uLightSpace", vert)
        self.assertIn("out vec4 vLightSpacePos", vert)
        self.assertIn("vLightSpacePos = uLightSpace * worldPos", vert)

    def test_pbr_lite_fragment_samples_shadow_map(self):
        _vert, frag = shaders.PROGRAMS["pbr_lite"]
        self.assertIn("uniform sampler2D uShadowMap", frag)
        self.assertIn("uniform int uUseShadow", frag)
        self.assertIn("shadowFactor", frag)
        # 3x3 PCF: nine taps averaged.
        self.assertIn("textureSize(uShadowMap, 0)", frag)
        self.assertIn("shadow / 9.0", frag)
        # Only the key light (slot 0) is shadowed.
        self.assertIn("(i == 0) ? shadowFactor(vLightSpacePos) : 1.0",
                      frag)

    def test_shadow_flags_default_true(self):
        for shape in ("box", "sphere", "cylinder", "cone", "torus",
                      "plane"):
            obj = scene_mod.new_object(shape)
            self.assertTrue(obj["cast_shadow"], shape)
            self.assertTrue(obj["receive_shadow"], shape)

    def test_normalize_preserves_shadow_flags(self):
        obj = scene_mod.normalize_object({"shape": "box",
                                          "cast_shadow": False})
        self.assertFalse(obj["cast_shadow"])
        self.assertTrue(obj["receive_shadow"])
        obj = scene_mod.normalize_object({"shape": "sphere",
                                          "receive_shadow": False})
        self.assertFalse(obj["receive_shadow"])
        self.assertTrue(obj["cast_shadow"])
        # Missing or mistyped values fall back to True.
        obj = scene_mod.normalize_object({"shape": "box"})
        self.assertTrue(obj["cast_shadow"])
        self.assertTrue(obj["receive_shadow"])
        obj = scene_mod.normalize_object({"shape": "box",
                                          "cast_shadow": "yes",
                                          "receive_shadow": 1})
        self.assertTrue(obj["cast_shadow"])
        self.assertTrue(obj["receive_shadow"])

    def test_shadow_flags_round_trip(self):
        scene = scene_mod.new_scene()
        box = scene_mod.new_object("box")
        box["cast_shadow"] = False
        sphere = scene_mod.new_object("sphere")
        sphere["receive_shadow"] = False
        scene["objects"] = [box, sphere]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.json")
            scene_mod.save_scene(scene, path)
            loaded = scene_mod.load_scene(path)
        self.assertFalse(loaded["objects"][0]["cast_shadow"])
        self.assertTrue(loaded["objects"][0]["receive_shadow"])
        self.assertTrue(loaded["objects"][1]["cast_shadow"])
        self.assertFalse(loaded["objects"][1]["receive_shadow"])
        self.assertEqual(loaded["version"], 2)

    def test_viewport_renders_a_shadow_pass(self):
        source = (HERE / "light_engine" / "viewport.py").read_text(
            encoding="utf-8")
        self.assertIn("SHADOW_MAP_SIZE = 1024", source)
        self.assertIn("_render_shadow_map", source)
        self.assertIn("_light_space_matrix", source)
        self.assertIn("_build_shadow_map", source)
        # Slope-scale polygon offset against shadow acne.
        self.assertIn("glPolygonOffset", source)
        self.assertIn("GL_POLYGON_OFFSET_FILL", source)


class SceneFormatV2Tests(unittest.TestCase):
    """Scene format v2 (T6): tolerant v1 load, v2-only writer, future
    fields (mesh_file / kind / physics) accepted and preserved."""

    def _v1_scene(self):
        """A v1 fixture: no v2 fields anywhere, one object with the
        T3/T5 additive flags set explicitly (v1 files could carry
        those — the version stayed 1 through T5)."""
        return {
            "name": "Old",
            "version": 1,
            "objects": [
                {"id": "box1", "name": "Box", "shape": "box",
                 "position": [0.0, 0.0, 0.0]},
                {"id": "ball1", "name": "Ball", "shape": "sphere",
                 "position": [10.0, 0.0, 0.0]},
                {"id": "cyl1", "name": "Peg", "shape": "cylinder",
                 "position": [20.0, 0.0, 0.0],
                 "smooth_shading": False, "cast_shadow": False},
            ],
            "cameras": [],
            "lights": [],
            "animation": {"fps": 24, "end_frame": 48, "keys": {}},
        }

    def test_v1_loads_with_v2_defaults(self):
        scene = scene_mod.normalize_scene(self._v1_scene())
        self.assertEqual(scene["version"], 2)
        by_id = {o["id"]: o for o in scene["objects"]}
        # Smooth normals on for curved primitives, off for the box.
        self.assertFalse(by_id["box1"]["smooth_shading"])
        self.assertTrue(by_id["ball1"]["smooth_shading"])
        # Shadows on unless the file says otherwise.
        self.assertTrue(by_id["box1"]["cast_shadow"])
        self.assertTrue(by_id["box1"]["receive_shadow"])
        self.assertFalse(by_id["cyl1"]["smooth_shading"])
        self.assertFalse(by_id["cyl1"]["cast_shadow"])
        self.assertTrue(by_id["cyl1"]["receive_shadow"])
        # Every object is an "object" unless the file says emitter.
        for obj in scene["objects"]:
            self.assertEqual(obj["kind"], "object")

    def test_v1_mistyped_fields_fall_back_to_defaults(self):
        raw = self._v1_scene()
        raw["objects"][0]["smooth_shading"] = "yes"
        raw["objects"][0]["cast_shadow"] = 1
        raw["objects"][0]["kind"] = "thing"
        scene = scene_mod.normalize_scene(raw)
        box = scene["objects"][0]
        self.assertFalse(box["smooth_shading"])  # box default
        self.assertTrue(box["cast_shadow"])
        self.assertEqual(box["kind"], "object")

    def test_save_writes_v2_and_reloads_identically(self):
        scene = scene_mod.new_scene("Round trip")
        box = scene_mod.new_object("box")
        box["smooth_shading"] = True
        box["cast_shadow"] = False
        sphere = scene_mod.new_object("sphere")
        sphere["receive_shadow"] = False
        scene["objects"] = [box, sphere]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.json")
            scene_mod.save_scene(scene, path)
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            loaded = scene_mod.load_scene(path)
        self.assertEqual(raw["version"], 2)
        self.assertEqual(loaded["version"], 2)
        self.assertEqual(loaded["name"], "Round trip")
        by_id = {o["id"]: o for o in loaded["objects"]}
        self.assertTrue(by_id[box["id"]]["smooth_shading"])
        self.assertFalse(by_id[box["id"]]["cast_shadow"])
        self.assertFalse(by_id[sphere["id"]]["receive_shadow"])
        self.assertEqual(by_id[box["id"]]["kind"], "object")

    def test_no_v1_writer_remains(self):
        # A v1 scene loaded and saved comes back as v2 on disk, and
        # the save does not mutate the caller's dict.
        raw_v1 = self._v1_scene()
        scene = scene_mod.normalize_scene(raw_v1)
        scene["version"] = 1  # hand-built / stale version must not leak
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.json")
            scene_mod.save_scene(scene, path)
            with open(path, encoding="utf-8") as fh:
                written = json.load(fh)
        self.assertEqual(written["version"], 2)
        self.assertEqual(scene["version"], 1)

    def test_future_fields_are_preserved(self):
        # mesh_file (T7), kind "emitter" (T16) and a physics section
        # (T18) load without error and survive a save round-trip.
        # Nothing in T6 builds those features — tolerance only.
        scene = scene_mod.new_scene("Future")
        mesh_obj = scene_mod.new_object("box")
        mesh_obj["mesh_file"] = "assets/chair.mesh.json"
        emitter = scene_mod.new_object("sphere")
        emitter["kind"] = "emitter"
        scene["objects"] = [mesh_obj, emitter]
        scene["physics"] = {mesh_obj["id"]: {"mass": 2.5,
                                             "static": False}}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.json")
            scene_mod.save_scene(scene, path)
            loaded = scene_mod.load_scene(path)
        by_id = {o["id"]: o for o in loaded["objects"]}
        self.assertEqual(by_id[mesh_obj["id"]]["mesh_file"],
                         "assets/chair.mesh.json")
        self.assertEqual(by_id[emitter["id"]]["kind"], "emitter")
        self.assertEqual(loaded["physics"],
                         {mesh_obj["id"]: {"mass": 2.5, "static": False}})

    def test_example_tavern_is_v2(self):
        scene = scene_mod.load_scene(str(EXAMPLE))
        self.assertEqual(scene["version"], 2)
        self.assertEqual(len(scene["objects"]), 19)
        for obj in scene["objects"]:
            self.assertIn("smooth_shading", obj)
            self.assertIn("cast_shadow", obj)
            self.assertIn("receive_shadow", obj)
            self.assertEqual(obj["kind"], "object")


class MeshFileTests(unittest.TestCase):
    """The .mesh.json format (T7): load, validate, round-trip."""

    def test_gem_asset_loads(self):
        self.assertTrue(GEM_ASSET.exists(), "gem asset missing")
        mesh = meshfile.load_mesh(str(GEM_ASSET))
        self.assertEqual(mesh["vertex_count"], 24)
        self.assertEqual(mesh["triangle_count"], 8)
        self.assertEqual(len(mesh["positions"]), 24 * 3)
        self.assertEqual(len(mesh["normals"]), 24 * 3)
        self.assertEqual(len(mesh["uvs"]), 24 * 2)
        self.assertEqual(len(mesh["indices"]), 8 * 3)
        self.assertEqual(mesh["bounds"],
                         ((-45.0, -45.0, -60.0), (45.0, 45.0, 60.0)))
        self.assertEqual(mesh["source"],
                         "hand-written octahedron (T7 example)")

    def test_read_counts_matches_arrays(self):
        verts, tris = meshfile.read_counts(str(GEM_ASSET))
        self.assertEqual((verts, tris), (24, 8))

    def test_save_and_load_round_trip(self):
        mesh = meshfile.load_mesh(str(GEM_ASSET))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "copy.mesh.json")
            meshfile.save_mesh(path, mesh, source="round trip",
                               imported="2026-11-12")
            again = meshfile.load_mesh(path)
        self.assertEqual(again["positions"], mesh["positions"])
        self.assertEqual(again["indices"], mesh["indices"])
        self.assertEqual(again["triangle_count"], 8)
        self.assertEqual(again["source"], "round trip")

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            meshfile.load_mesh("/nonexistent/nope.mesh.json")

    def _write_bad_mesh(self, tmp, data):
        path = os.path.join(tmp, "bad.mesh.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        return path

    def test_inconsistent_counts_raise(self):
        mesh = meshfile.load_mesh(str(GEM_ASSET))
        mesh["triangle_count"] = 9  # arrays still hold 8
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_bad_mesh(tmp, mesh)
            with self.assertRaises(ValueError):
                meshfile.load_mesh(path)

    def test_index_out_of_range_raises(self):
        mesh = meshfile.load_mesh(str(GEM_ASSET))
        mesh["indices"][0] = 24  # only 24 vertices: 0..23
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_bad_mesh(tmp, mesh)
            with self.assertRaises(ValueError):
                meshfile.load_mesh(path)

    def test_mismatched_normals_raise(self):
        mesh = meshfile.load_mesh(str(GEM_ASSET))
        mesh["normals"] = mesh["normals"][:-3]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_bad_mesh(tmp, mesh)
            with self.assertRaises(ValueError):
                meshfile.load_mesh(path)

    def test_not_json_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.mesh.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            with self.assertRaises(ValueError):
                meshfile.load_mesh(path)


class MeshObjectTests(unittest.TestCase):
    """Mesh objects in the scene (T7): shape, counts, round-trip."""

    def test_new_mesh_object_has_empty_params(self):
        obj = scene_mod.new_object("mesh")
        self.assertEqual(obj["shape"], "mesh")
        self.assertEqual(obj["params"], {})
        self.assertNotIn("mesh_file", obj)

    def test_total_triangles_uses_recorded_count(self):
        # The recorded count (8) is what counts — tessellate("mesh")
        # would fall back to a 12-tri box, so this test distinguishes
        # the two paths.
        scene = scene_mod.load_scene(str(MESH_DEMO))
        tris = scene_mod.total_triangles(
            scene, base_path=str(MESH_DEMO.parent))
        # Floor 2 + gem 8 + crate 12.
        self.assertEqual(tris, 22)

    def test_total_triangles_with_mesh_counts_map(self):
        scene = scene_mod.load_scene(str(MESH_DEMO))
        tris = scene_mod.total_triangles(
            scene, mesh_counts={"gem": 8})
        self.assertEqual(tris, 22)

    def test_total_triangles_without_a_source_counts_zero(self):
        # No cache and no base path: the mesh counts 0, never a
        # tessellated guess, and nothing crashes.
        scene = scene_mod.load_scene(str(MESH_DEMO))
        tris = scene_mod.total_triangles(scene)
        # Floor 2 + crate 12; the gem counts 0.
        self.assertEqual(tris, 14)

    def test_mesh_object_round_trip(self):
        scene = scene_mod.new_scene("Mesh round trip")
        obj = scene_mod.new_object("mesh", name="Gem")
        obj["mesh_file"] = "assets/gem.mesh.json"
        obj["position"] = [1.0, 2.0, 3.0]
        scene["objects"].append(obj)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.json")
            scene_mod.save_scene(scene, path)
            loaded = scene_mod.load_scene(path)
        self.assertEqual(loaded["objects"][0]["shape"], "mesh")
        self.assertEqual(loaded["objects"][0]["mesh_file"],
                         "assets/gem.mesh.json")
        self.assertEqual(loaded["objects"][0]["position"], [1.0, 2.0, 3.0])

    def test_demo_scene_is_v2_with_a_mesh(self):
        scene = scene_mod.load_scene(str(MESH_DEMO))
        self.assertEqual(scene["version"], 2)
        gem = scene_mod.find_object(scene, "gem")
        self.assertIsNotNone(gem)
        self.assertEqual(gem["shape"], "mesh")
        self.assertEqual(gem["mesh_file"], "assets/gem.mesh.json")
        self.assertTrue(gem["cast_shadow"])
        self.assertTrue(gem["receive_shadow"])


# A Blender-style .obj fixture: o + g groups, quads, vt/vn, mtllib +
# usemtl, and unknown lines (s, l, vp) that must be skipped, never
# fatal. Seat is a 6-quad box (12 tris), Leg is one triangle.
OBJ_FIXTURE = """# test file
mtllib test.mtl
o Seat
v -0.5 -0.5 0.0
v 0.5 -0.5 0.0
v 0.5 0.5 0.0
v -0.5 0.5 0.0
v -0.5 -0.5 0.4
v 0.5 -0.5 0.4
v 0.5 0.5 0.4
v -0.5 0.5 0.4
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
vn 0.0 0.0 1.0
vn 0.0 0.0 -1.0
vn 0.0 -1.0 0.0
vn 0.0 1.0 0.0
vn 1.0 0.0 0.0
vn -1.0 0.0 0.0
usemtl Wood
s off
f 1/1/2 4/4/2 3/3/2 2/2/2
f 5/1/1 6/2/1 7/3/1 8/4/1
f 1/1/3 2/2/3 6/3/3 5/4/3
f 4/1/4 8/4/4 7/3/4 3/2/4
f 2/1/5 3/2/5 7/3/5 6/4/5
f 1/1/6 5/4/6 8/3/6 4/2/6
g Leg
usemtl Metal
v -0.1 -0.1 0.0
v 0.1 -0.1 0.0
v 0.0 0.1 0.0
f 9/1/1 10/2/1 11/3/1
l 1 2
vp 0.0 0.0 0.0
"""

MTL_FIXTURE = """newmtl Wood
Kd 0.8 0.4 0.2
newmtl Metal
Kd 0.5 0.5 0.55
"""

# No vn and no vt: normals must be computed, uvs default to 0.
OBJ_NO_NORMALS = """o Cube
v -1.0 -1.0 0.0
v 1.0 -1.0 0.0
v 1.0 1.0 0.0
v -1.0 1.0 0.0
v -1.0 -1.0 2.0
v 1.0 -1.0 2.0
v 1.0 1.0 2.0
v -1.0 1.0 2.0
f 1 4 3 2
f 5 6 7 8
f 1 2 6 5
f 4 8 7 3
f 2 3 7 6
f 1 5 8 4
"""


class ObjImportTests(unittest.TestCase):
    """Wavefront .obj import (T8): parsing, materials, scale, the
    import_mesh entry point, and asset naming."""

    def _write(self, tmp, obj_text, mtl_text=None, name="test.obj"):
        obj_path = os.path.join(tmp, name)
        with open(obj_path, "w", encoding="utf-8") as fh:
            fh.write(obj_text)
        if mtl_text is not None:
            with open(os.path.join(tmp, "test.mtl"), "w",
                      encoding="utf-8") as fh:
                fh.write(mtl_text)
        return obj_path

    def test_quads_triangulated_and_groups_split(self):
        """6 quads become 12 triangles; o and g groups become
        separate objects; unknown lines are skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, OBJ_FIXTURE, MTL_FIXTURE)
            parsed = meshio_obj.parse_obj(path)
        self.assertEqual(parsed["report"]["objects"], 2)
        self.assertEqual(parsed["report"]["triangles"], 13)
        names = [g["name"] for g in parsed["objects"]]
        self.assertEqual(names, ["Seat", "Leg"])
        self.assertEqual(len(parsed["objects"][0]["faces"]), 12)
        self.assertEqual(len(parsed["objects"][1]["faces"]), 1)

    def test_negative_indices_resolve(self):
        """-1 is the last vertex defined, -3 the third from last."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "o Neg\n"
                               "v 0.0 0.0 0.0\n"
                               "v 1.0 0.0 0.0\n"
                               "v 0.0 1.0 0.0\n"
                               "f -3 -2 -1\n")
            objects, _report = meshio.import_mesh(path, scale=1.0)
        mesh = objects[0]["mesh"]
        self.assertEqual(mesh["positions"],
                         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        # The computed normal of that triangle is +z, exactly.
        self.assertEqual(mesh["normals"],
                         [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    def test_mtllib_kd_maps_to_base_color(self):
        """Kd 0.8 0.4 0.2 is #cc6633; Kd 0.5 0.5 0.55 is #80808c."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, OBJ_FIXTURE, MTL_FIXTURE)
            objects, _report = meshio.import_mesh(path)
        by_name = {o["name"]: o for o in objects}
        self.assertEqual(by_name["Seat"]["material"]["base_color"],
                         "#cc6633")
        self.assertEqual(by_name["Leg"]["material"]["base_color"],
                         "#80808c")

    def test_missing_mtllib_warns_and_uses_default(self):
        """A mtllib that is not there warns once and the objects get
        the default gray — never a crash, never a refusal."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, OBJ_FIXTURE.replace(
                "mtllib test.mtl", "mtllib nope.mtl"))
            objects, report = meshio.import_mesh(path)
        self.assertTrue(any("nope.mtl" in w for w in report["warnings"]),
                        report["warnings"])
        for obj in objects:
            self.assertEqual(obj["material"]["base_color"], "#b0b0b0")

    def test_missing_normals_are_computed(self):
        """No vn in the file: the cube's normals are computed with
        the T2 angle rule — every normal is unit length and
        axis-aligned, so the box keeps its crisp corners."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, OBJ_NO_NORMALS)
            objects, _report = meshio.import_mesh(path)
        mesh = objects[0]["mesh"]
        self.assertEqual(len(mesh["indices"]) // 3, 12)
        normals = mesh["normals"]
        for i in range(0, len(normals), 3):
            nx, ny, nz = normals[i:i + 3]
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            self.assertAlmostEqual(length, 1.0, places=6)
            aligned = sum(1 for v in (nx, ny, nz) if abs(abs(v) - 1.0)
                          < 1e-6)
            self.assertEqual(aligned, 1,
                             f"normal {(nx, ny, nz)} is not axis-aligned")

    def test_missing_uvs_default_to_zero(self):
        """No vt in the file: every uv is 0."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, OBJ_NO_NORMALS)
            objects, _report = meshio.import_mesh(path)
        self.assertTrue(all(v == 0.0 for v in objects[0]["mesh"]["uvs"]))

    def test_first_material_wins_and_warns(self):
        """One group with two usemtl lines keeps the first color and
        records one warning about the second."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp,
                               "mtllib test.mtl\n"
                               "o Mixed\n"
                               "v 0.0 0.0 0.0\n"
                               "v 1.0 0.0 0.0\n"
                               "v 0.0 1.0 0.0\n"
                               "v 0.0 0.0 1.0\n"
                               "usemtl Wood\n"
                               "f 1 2 3\n"
                               "usemtl Metal\n"
                               "f 1 3 4\n", MTL_FIXTURE)
            objects, report = meshio.import_mesh(path)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["material"]["base_color"], "#cc6633")
        self.assertTrue(any("more than one material" in w
                            for w in report["warnings"]),
                        report["warnings"])

    def test_malformed_files_fail_plainly(self):
        """Bad numbers, an empty file, no faces, and an out-of-range
        index all raise ValueError with a plain message naming the
        problem — never a traceback-shaped error."""
        cases = [
            ("v 0.0 oops 0.0\nf 1 1 1\n", "line 1"),
            ("", "empty"),
            ("v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\n",
             "no faces"),
            ("v 0.0 0.0 0.0\nf 1 2 99\n", "line 2"),
            ("v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\n"
             "f a b c\n", "line 4"),
        ]
        for text, needle in cases:
            with tempfile.TemporaryDirectory() as tmp:
                path = self._write(tmp, text)
                with self.assertRaises(ValueError) as caught:
                    meshio_obj.parse_obj(path)
            self.assertIn(needle, str(caught.exception),
                          f"{text!r} raised {caught.exception!r}")

    def test_scale_applied(self):
        """Default scale is x100 (meters to cm); a custom scale is
        honored exactly."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "o Unit\n"
                               "v 1.0 0.0 0.0\n"
                               "v 0.0 1.0 0.0\n"
                               "v 0.0 0.0 1.0\n"
                               "f 1 2 3\n")
            default, _ = meshio.import_mesh(path)
            custom, _ = meshio.import_mesh(path, scale=2.5)
        self.assertEqual(default[0]["mesh"]["positions"][0], 100.0)
        self.assertEqual(custom[0]["mesh"]["positions"][0], 2.5)

    def test_import_mesh_entry_point(self):
        """import_mesh dispatches .obj and reports counts; an
        unsupported extension and a missing file fail plainly."""
        objects, report = meshio.import_mesh(str(HOUSE_OBJ))
        self.assertEqual(report["objects"], 2)
        self.assertEqual(report["triangles"], 16)
        self.assertEqual(report["warnings"], [])
        by_name = {o["name"]: o for o in objects}
        self.assertEqual(by_name["Walls"]["material"]["base_color"],
                         "#ebdbbd")
        self.assertEqual(by_name["Roof"]["material"]["base_color"],
                         "#8c402e")
        with tempfile.TemporaryDirectory() as tmp:
            txt = os.path.join(tmp, "notes.txt")
            with open(txt, "w", encoding="utf-8") as fh:
                fh.write("hello")
            with self.assertRaises(ValueError) as caught:
                meshio.import_mesh(txt)
            self.assertIn("not a supported mesh file",
                          str(caught.exception))
            with self.assertRaises(FileNotFoundError):
                meshio.import_mesh(os.path.join(tmp, "missing.obj"))

    def test_heavy_threshold(self):
        """Exactly 1,000,000 triangles is fine; one more warns."""
        self.assertEqual(meshio.HEAVY_TRIANGLES, 1_000_000)
        self.assertFalse(meshio.is_heavy(1_000_000))
        self.assertTrue(meshio.is_heavy(1_000_001))
        self.assertFalse(meshio.is_heavy(16))

    def test_asset_naming(self):
        """Names are sanitized (no path separators, no leading dots),
        the stem alone is used when the group matches it, and
        collisions get -2, -3 — both against files on disk and
        against names claimed earlier in the same import."""
        with tempfile.TemporaryDirectory() as tmp:
            name = meshio.unique_asset_name("My Chair!", "Seat & Back",
                                            tmp)
            self.assertEqual(name, "My-Chair-Seat-Back.mesh.json")
            self.assertEqual(
                meshio.unique_asset_name("house", "House", tmp),
                "house.mesh.json")
            self.assertEqual(
                meshio.unique_asset_name("house", "", tmp),
                "house.mesh.json")
            open(os.path.join(tmp, "house.mesh.json"), "w").close()
            self.assertEqual(
                meshio.unique_asset_name("house", "House", tmp),
                "house-2.mesh.json")
            used = set()
            first = meshio.unique_asset_name("stool", "Leg", tmp, used)
            second = meshio.unique_asset_name("stool", "Leg", tmp, used)
            self.assertEqual(first, "stool-Leg.mesh.json")
            self.assertEqual(second, "stool-Leg-2.mesh.json")


@unittest.skipUnless(QT_TESTS, "Qt tests need LIGHT_ENGINE_QT_TESTS=1")
class QtViewportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets
        cls._app = QtWidgets.QApplication.instance()
        if cls._app is None:
            cls._app = QtWidgets.QApplication([])
        from light_engine.viewport import Viewport
        from light_engine.window import MainWindow
        cls.Viewport = Viewport
        cls.MainWindow = MainWindow

    def test_window_title_says_reblend(self):
        # The title is set in __init__; Xvfb may not draw window-manager
        # chrome in screenshots, so pin windowTitle() directly.
        window = self.MainWindow(scene=scene_mod.new_scene())
        self.assertEqual(window.windowTitle(), "Reblend — v0.2")
        window.close()

    def test_viewport_renders_and_saves_png(self):
        from PySide6 import QtCore, QtWidgets
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        image = window._viewport.grabFramebuffer()
        self.assertFalse(image.isNull())
        self.assertGreater(image.width(), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "render.png")
            self.assertTrue(image.save(path))
            self.assertGreater(os.path.getsize(path), 1000)
        window.close()

    def test_pick_selects_an_object(self):
        from PySide6 import QtCore, QtWidgets
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        vp = window._viewport
        picked = vp.pick(vp.width() // 2, vp.height() // 2)
        # The table fills the framed center; some object must be under it.
        self.assertIsNotNone(picked)
        window.close()

    def test_numeric_edit_is_undoable(self):
        from PySide6 import QtCore, QtWidgets
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        for _ in range(10):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        window._select("mug")
        spin = window._spins[("position", "x")]
        self.assertEqual(spin.value(), -60.0)
        spin.setValue(123.0)
        spin.editingFinished.emit()
        obj = scene_mod.find_object(window.scene, "mug")
        self.assertEqual(obj["position"][0], 123.0)
        window.undo()
        obj = scene_mod.find_object(window.scene, "mug")
        self.assertEqual(obj["position"][0], -60.0)
        window.close()

    def test_mid_animation_frame_renders(self):
        from PySide6 import QtCore, QtWidgets
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "frame24.png")
            self.assertTrue(window.render_frame_to(24, path))
            self.assertGreater(os.path.getsize(path), 1000)
        window.close()

    def test_shaders_compile_and_link(self):
        from PySide6 import QtCore, QtWidgets
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        vp = window._viewport
        self.assertIsNone(vp._shader_error)
        self.assertEqual(set(vp._programs),
                         {"pbr_lite", "depth", "lines", "points"})
        for name, prog in vp._programs.items():
            self.assertTrue(prog.isLinked(), name)
        window.close()

    def test_viewport_uses_core_profile(self):
        from PySide6 import QtCore, QtGui, QtWidgets
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        fmt = window._viewport.format()
        self.assertEqual(fmt.profile(), QtGui.QSurfaceFormat.CoreProfile)
        self.assertGreaterEqual((fmt.majorVersion(), fmt.minorVersion()),
                                (3, 3))
        window.close()

    def test_selection_box_renders(self):
        from PySide6 import QtCore, QtWidgets
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        window._select("mug")
        for _ in range(10):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        image = window._viewport.grabFramebuffer()
        self.assertFalse(image.isNull())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "selected.png")
            self.assertTrue(image.save(path))
            self.assertGreater(os.path.getsize(path), 1000)
        window.close()

    def test_indexed_mesh_renders(self):
        """T2: an object routed through the indexed path builds a
        VAO+VBO+IBO and draws via glDrawElements without GL errors."""
        from PySide6 import QtCore, QtWidgets
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        vp = window._viewport
        vp.set_indexed("loaf", True)
        for _ in range(10):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        entry = vp._vbos.get("loaf")
        self.assertIsNotNone(entry)
        vbo, vao, ibo, count, indexed, _key = entry
        self.assertTrue(indexed)
        self.assertIsNotNone(ibo)
        self.assertTrue(ibo.isCreated())
        self.assertTrue(vao.isCreated())
        self.assertTrue(vbo.isCreated())
        # Index count: 3 per triangle of the loaf's own params
        # (the tavern loaf is not the default 20x12 sphere).
        loaf = scene_mod.find_object(scene, "loaf")
        expected = geometry.tessellate_indexed(
            "sphere", loaf.get("params", {}))["tri_count"] * 3
        self.assertEqual(count, expected)
        # Flat objects next to it keep drawing the soup way. (The mug
        # is a cylinder, so it is smooth/indexed by default since T3.)
        floor = vp._vbos.get("floor")
        self.assertIsNotNone(floor)
        self.assertFalse(floor[4])
        self.assertIsNone(floor[2])
        image = vp.grabFramebuffer()
        self.assertFalse(image.isNull())
        vp.makeCurrent()
        try:
            self.assertEqual(vp._gl.glGetError(), 0)
        finally:
            vp.doneCurrent()
        window.close()

    def test_set_indexed_rebuilds_entry(self):
        """T2: the mesh key covers the indexed flag, so flipping it
        rebuilds the object's buffers through the other builder.
        Uses the floor (plane, flat by default) so the hook is the
        only thing routing it — smooth objects are indexed anyway
        since T3."""
        from PySide6 import QtCore, QtWidgets
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        vp = window._viewport
        self.assertFalse(vp._vbos["floor"][4])
        vp.set_indexed("floor", True)
        for _ in range(10):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        self.assertTrue(vp._vbos["floor"][4])
        self.assertIsNotNone(vp._vbos["floor"][2])
        vp.set_indexed("floor", False)
        for _ in range(10):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        self.assertFalse(vp._vbos["floor"][4])
        self.assertIsNone(vp._vbos["floor"][2])
        window.close()

    def test_smooth_object_renders_and_toggle_rebuilds(self):
        """T3: smooth objects (sphere/cylinder) draw through the
        indexed path, flat ones (plane/box) through the soup, and
        toggling the inspector checkbox rebuilds the buffers —
        undoable in one step."""
        from PySide6 import QtCore, QtWidgets
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        vp = window._viewport
        # Defaults: loaf (sphere) and mug (cylinder) smooth/indexed,
        # floor (plane) and tabletop (box) flat/soup.
        self.assertTrue(vp._vbos["loaf"][4])
        self.assertIsNotNone(vp._vbos["loaf"][2])
        self.assertTrue(vp._vbos["mug"][4])
        self.assertFalse(vp._vbos["floor"][4])
        self.assertIsNone(vp._vbos["floor"][2])
        self.assertFalse(vp._vbos["tabletop"][4])
        # Toggle smooth off through the inspector checkbox.
        window._select("loaf")
        self.assertTrue(window._smooth_check.isChecked())
        window._smooth_check.setChecked(False)
        loaf = scene_mod.find_object(window.scene, "loaf")
        self.assertFalse(loaf["smooth_shading"])
        for _ in range(10):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        self.assertFalse(vp._vbos["loaf"][4])
        self.assertIsNone(vp._vbos["loaf"][2])
        # Undo restores smooth shading and the indexed buffers.
        window.undo()
        for _ in range(10):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        loaf = scene_mod.find_object(window.scene, "loaf")
        self.assertTrue(loaf["smooth_shading"])
        self.assertTrue(window._smooth_check.isChecked())
        self.assertTrue(vp._vbos["loaf"][4])
        image = vp.grabFramebuffer()
        self.assertFalse(image.isNull())
        vp.makeCurrent()
        try:
            self.assertEqual(vp._gl.glGetError(), 0)
        finally:
            vp.doneCurrent()
        window.close()

    def test_vao_lifecycle_drop_rebuild_and_prune(self):
        """T2: dropping all buffers rebuilds them on the next paint,
        and deleting an object prunes its buffers instead of leaking
        them until the next scene change."""
        from PySide6 import QtCore, QtWidgets
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        vp = window._viewport
        self.assertEqual(len(vp._vbos), 19)
        vp._drop_vbos()
        self.assertEqual(vp._vbos, {})
        vp.update()
        for _ in range(10):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        self.assertEqual(len(vp._vbos), 19)
        # Delete one object: its buffers must go on the next paint.
        window.scene["objects"] = [o for o in window.scene["objects"]
                                   if o["id"] != "flame"]
        vp.update()
        for _ in range(10):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        self.assertEqual(len(vp._vbos), 18)
        self.assertNotIn("flame", vp._vbos)
        image = vp.grabFramebuffer()
        self.assertFalse(image.isNull())
        window.close()

    def test_pbr_aces_renders_without_clipped_whites(self):
        """T4: the tavern renders through the Cook-Torrance + ACES
        shader with no GL errors, and no pixel clips to full white —
        the ACES shoulder rolls highlights off instead."""
        from PySide6 import QtCore, QtGui, QtWidgets
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        vp = window._viewport
        self.assertIsNone(vp._shader_error)
        image = vp.grabFramebuffer()
        self.assertFalse(image.isNull())
        brightest = 0
        for y in range(0, image.height(), 3):
            for x in range(0, image.width(), 3):
                rgb = image.pixel(x, y)
                brightest = max(brightest, QtGui.qRed(rgb),
                                QtGui.qGreen(rgb), QtGui.qBlue(rgb))
        self.assertLess(brightest, 255,
                        "a pixel clipped to full white")
        vp.makeCurrent()
        try:
            self.assertEqual(vp._gl.glGetError(), 0)
        finally:
            vp.doneCurrent()
        window.close()

    def test_shadow_map_is_complete(self):
        """T5: the depth FBO builds and is valid, the depth texture
        exists, and a paint leaves no GL errors."""
        from PySide6 import QtCore, QtWidgets
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        vp = window._viewport
        self.assertIsNone(vp._shadow_error)
        self.assertIsNotNone(vp._shadow_fbo)
        self.assertTrue(vp._shadow_fbo.isValid())
        self.assertIsNotNone(vp._shadow_tex)
        self.assertTrue(vp._shadow_tex.isCreated())
        self.assertGreater(vp._shadow_tex.textureId(), 0)
        image = vp.grabFramebuffer()
        self.assertFalse(image.isNull())
        vp.makeCurrent()
        try:
            self.assertEqual(vp._gl.glGetError(), 0)
        finally:
            vp.doneCurrent()
        window.close()

    def test_shadows_darken_the_scene(self):
        """T5: rendering with all cast flags off is measurably
        brighter than with shadows on — the shadow test actually
        darkens shadowed pixels. Thresholds come from measured
        numbers at this window size (mean difference 0.257, 435
        pixels differing by >8), with wide margins; llvmpipe
        renders are deterministic (on-vs-on differs in 0 pixels)."""
        from PySide6 import QtCore, QtGui, QtWidgets
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        vp = window._viewport

        def mean_brightness(img):
            total, count = 0.0, 0
            for y in range(0, img.height(), 3):
                for x in range(0, img.width(), 3):
                    rgb = img.pixel(x, y)
                    total += (QtGui.qRed(rgb) + QtGui.qGreen(rgb)
                              + QtGui.qBlue(rgb)) / 3.0
                    count += 1
            return total / max(1, count)

        def differing(img_a, img_b, threshold=8):
            diff = 0
            for y in range(0, img_a.height(), 2):
                for x in range(0, img_a.width(), 2):
                    a, b = img_a.pixel(x, y), img_b.pixel(x, y)
                    if (abs(QtGui.qRed(a) - QtGui.qRed(b)) > threshold
                            or abs(QtGui.qGreen(a) - QtGui.qGreen(b))
                            > threshold
                            or abs(QtGui.qBlue(a) - QtGui.qBlue(b))
                            > threshold):
                        diff += 1
            return diff

        img_on = vp.grabFramebuffer()
        mean_on = mean_brightness(img_on)
        for obj in scene["objects"]:
            obj["cast_shadow"] = False
        vp.update()
        for _ in range(10):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        img_off = vp.grabFramebuffer()
        mean_off = mean_brightness(img_off)
        self.assertLess(mean_on, mean_off - 0.05,
                        "shadows did not darken the scene")
        self.assertGreater(differing(img_on, img_off), 100,
                           "too few pixels changed with shadows off")
        window.close()

    def test_shadow_checkboxes_are_undoable(self):
        """T5: the Cast/Receive shadow checkboxes flip the flags
        and undo restores them, like the T3 smooth-shading box."""
        from PySide6 import QtCore, QtWidgets
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        window._select("mug")
        mug = scene_mod.find_object(window.scene, "mug")
        self.assertTrue(window._cast_check.isChecked())
        self.assertTrue(window._receive_check.isChecked())
        window._cast_check.setChecked(False)
        self.assertFalse(mug["cast_shadow"])
        window._receive_check.setChecked(False)
        self.assertFalse(mug["receive_shadow"])
        # Undo replaces the scene with a snapshot, so re-fetch the
        # object each time — the old dict reference is stale.
        window.undo()
        mug = scene_mod.find_object(window.scene, "mug")
        self.assertTrue(mug["receive_shadow"])
        self.assertTrue(window._receive_check.isChecked())
        self.assertFalse(mug["cast_shadow"])
        window.undo()
        mug = scene_mod.find_object(window.scene, "mug")
        self.assertTrue(mug["cast_shadow"])
        self.assertTrue(window._cast_check.isChecked())
        image = window._viewport.grabFramebuffer()
        self.assertFalse(image.isNull())
        window.close()


@unittest.skipUnless(QT_TESTS, "Qt tests need LIGHT_ENGINE_QT_TESTS=1")
class QtMeshTests(unittest.TestCase):
    """Mesh objects in the live window (T7): render, pick, transform,
    shadow, and the missing-file warning."""

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets
        cls._app = QtWidgets.QApplication.instance()
        if cls._app is None:
            cls._app = QtWidgets.QApplication([])
        from light_engine.window import MainWindow
        cls.MainWindow = MainWindow

    def _show_demo(self):
        from PySide6 import QtCore, QtWidgets
        scene = scene_mod.load_scene(str(MESH_DEMO))
        window = self.MainWindow(scene=scene, scene_path=str(MESH_DEMO))
        window.resize(800, 600)
        window.show()
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)
        return window

    def _pump(self, rounds=10):
        from PySide6 import QtCore, QtWidgets
        for _ in range(rounds):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)

    def test_demo_scene_renders_mesh(self):
        """The gem renders from its asset through the indexed path:
        VAO+VBO+IBO, 24 indices (8 triangles), no GL errors, and the
        triangle readout counts the recorded 8."""
        from PySide6 import QtWidgets
        window = self._show_demo()
        vp = window._viewport
        self.assertIsNone(vp._shader_error)
        entry = vp._vbos.get("gem")
        self.assertIsNotNone(entry, "gem has no GPU buffers")
        vbo, vao, ibo, count, indexed, _key = entry
        self.assertTrue(indexed)
        self.assertIsNotNone(ibo)
        self.assertTrue(ibo.isCreated())
        self.assertTrue(vao.isCreated())
        self.assertTrue(vbo.isCreated())
        self.assertEqual(count, 24)
        self.assertIn("22 triangles", window._tri_label.text())
        image = vp.grabFramebuffer()
        self.assertFalse(image.isNull())
        vp.makeCurrent()
        try:
            self.assertEqual(vp._gl.glGetError(), 0)
        finally:
            vp.doneCurrent()
        window.close()

    def test_mesh_picks_by_click(self):
        """Clicking the gem's projected centre selects it — picking
        uses the asset's bounds, not a primitive fallback."""
        from PySide6 import QtGui, QtWidgets
        window = self._show_demo()
        vp = window._viewport
        proj, view = vp._matrices()
        combined = proj * view
        # The gem sits at (0, 0, 60) in the demo scene.
        ndc = combined.map(QtGui.QVector3D(0.0, 0.0, 60.0))
        sx = int((ndc.x() + 1.0) / 2.0 * vp.width())
        sy = int((1.0 - ndc.y()) / 2.0 * vp.height())
        self.assertEqual(vp.pick(sx, sy), "gem")
        window.close()

    def test_mesh_transform_is_undoable(self):
        """A mesh object transforms through the numeric inspector and
        undoes like a primitive — and its pick bounds move with it."""
        from PySide6 import QtGui, QtWidgets
        window = self._show_demo()
        vp = window._viewport
        window._select("gem")
        spin = window._spins[("position", "x")]
        self.assertEqual(spin.value(), 0.0)
        spin.setValue(120.0)
        spin.editingFinished.emit()
        gem = scene_mod.find_object(window.scene, "gem")
        self.assertEqual(gem["position"][0], 120.0)
        self._pump()
        # The gem moved: its new centre picks, the old one does not.
        proj, view = vp._matrices()
        combined = proj * view
        ndc = combined.map(QtGui.QVector3D(120.0, 0.0, 60.0))
        sx = int((ndc.x() + 1.0) / 2.0 * vp.width())
        sy = int((1.0 - ndc.y()) / 2.0 * vp.height())
        self.assertEqual(vp.pick(sx, sy), "gem")
        window.undo()
        gem = scene_mod.find_object(window.scene, "gem")
        self.assertEqual(gem["position"][0], 0.0)
        window.close()

    def test_mesh_casts_a_shadow(self):
        """The gem casts on the floor: rendering with its cast flag
        off is measurably brighter than with it on. Thresholds come
        from measured numbers at this window size (mean difference
        0.044, 75 pixels differing by >8), with wide margins;
        llvmpipe renders are deterministic (on-vs-on differs in 0
        pixels)."""
        from PySide6 import QtCore, QtGui, QtWidgets
        window = self._show_demo()
        vp = window._viewport

        def mean_brightness(img):
            total, count = 0.0, 0
            for y in range(0, img.height(), 3):
                for x in range(0, img.width(), 3):
                    rgb = img.pixel(x, y)
                    total += (QtGui.qRed(rgb) + QtGui.qGreen(rgb)
                              + QtGui.qBlue(rgb)) / 3.0
                    count += 1
            return total / max(1, count)

        def differing(img_a, img_b, threshold=8):
            diff = 0
            for y in range(0, img_a.height(), 2):
                for x in range(0, img_a.width(), 2):
                    a, b = img_a.pixel(x, y), img_b.pixel(x, y)
                    if (abs(QtGui.qRed(a) - QtGui.qRed(b)) > threshold
                            or abs(QtGui.qGreen(a) - QtGui.qGreen(b))
                            > threshold
                            or abs(QtGui.qBlue(a) - QtGui.qBlue(b))
                            > threshold):
                        diff += 1
            return diff

        img_on = vp.grabFramebuffer()
        mean_on = mean_brightness(img_on)
        gem = scene_mod.find_object(window.scene, "gem")
        gem["cast_shadow"] = False
        vp.update()
        self._pump()
        img_off = vp.grabFramebuffer()
        mean_off = mean_brightness(img_off)
        self.assertLess(mean_on, mean_off - 0.01,
                        "the gem's shadow did not darken the scene")
        self.assertGreater(differing(img_on, img_off), 10,
                           "too few pixels changed with the gem's "
                           "shadow off")
        window.close()

    def test_smooth_checkbox_disabled_for_mesh(self):
        """Meshes use the asset's stored normals as-is, so the Smooth
        shading checkbox is disabled for them (T7 decision) — but
        still works for a primitive next to the mesh."""
        window = self._show_demo()
        window._select("gem")
        self.assertFalse(window._smooth_check.isEnabled())
        window._select("crate")
        self.assertTrue(window._smooth_check.isEnabled())
        window.close()

    def test_missing_mesh_file_warns_and_skips(self):
        """A scene whose mesh file is missing loads the rest of the
        scene, drops the mesh object, and names the file in the
        status bar — never a faked mesh, never a crash."""
        from PySide6 import QtWidgets
        scene = scene_mod.new_scene("Missing mesh")
        box = scene_mod.new_object("box", name="Crate")
        scene["objects"].append(box)
        ghost = scene_mod.new_object("mesh", name="Ghost")
        ghost["mesh_file"] = "assets/ghost.mesh.json"
        scene["objects"].append(ghost)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.json")
            scene_mod.save_scene(scene, path)
            loaded = scene_mod.load_scene(path)
            window = self.MainWindow(scene=loaded, scene_path=path)
            self.assertIsNone(scene_mod.find_object(window.scene,
                                                    ghost["id"]))
            self.assertIsNotNone(scene_mod.find_object(window.scene,
                                                       box["id"]))
            self.assertEqual(len(window.scene["objects"]), 1)
            self.assertIn("assets/ghost.mesh.json",
                          window.statusBar().currentMessage())
            window.close()


@unittest.skipUnless(QT_TESTS, "Qt tests need LIGHT_ENGINE_QT_TESTS=1")
class QtObjImportTests(unittest.TestCase):
    """The .obj import flow in the live window (T8): assets written,
    objects created, one undo step, and the saved-scene guard. The
    dialog-free import_mesh_objects helper is driven directly — the
    file picker and the scale dialog need a human, so they are
    verified by the screenshot script instead."""

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets
        cls._app = QtWidgets.QApplication.instance()
        if cls._app is None:
            cls._app = QtWidgets.QApplication([])
        from light_engine.window import ImportDialog, MainWindow
        cls.MainWindow = MainWindow
        cls.ImportDialog = ImportDialog

    def _pump(self, rounds=10):
        from PySide6 import QtCore, QtWidgets
        for _ in range(rounds):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)

    def _show_saved_scene(self, tmp):
        """A window on a scene saved in tmp, so imports have a
        folder for their assets."""
        from PySide6 import QtWidgets
        scene = scene_mod.new_scene("Import test")
        floor = scene_mod.new_object("plane", name="Floor")
        scene["objects"].append(floor)
        path = os.path.join(tmp, "scene.json")
        scene_mod.save_scene(scene, path)
        loaded = scene_mod.load_scene(path)
        window = self.MainWindow(scene=loaded, scene_path=path)
        window.resize(800, 600)
        window.show()
        self._pump()
        return window, path

    def test_import_writes_assets_and_creates_objects(self):
        """The sample house imports: two assets land in assets/,
        two mesh objects are created with mesh_file set, the
        triangle readout counts the imported 16, and the scene
        renders with no GL errors."""
        with tempfile.TemporaryDirectory() as tmp:
            window, _path = self._show_saved_scene(tmp)
            parsed = meshio.parse(str(HOUSE_OBJ))
            created = window.import_mesh_objects(parsed, 100.0)
            self.assertEqual(len(created), 2)
            self.assertEqual(
                sorted(os.listdir(os.path.join(tmp, "assets"))),
                ["house-Roof.mesh.json", "house-Walls.mesh.json"])
            for obj in created:
                self.assertEqual(obj["shape"], "mesh")
                self.assertTrue(obj["mesh_file"].startswith("assets/"))
            self.assertEqual(len(window.scene["objects"]), 3)
            # Floor 2 + house 16.
            self.assertIn("18 triangles", window._tri_label.text())
            self._pump()
            vp = window._viewport
            self.assertIsNone(vp._shader_error)
            self.assertFalse(vp.grabFramebuffer().isNull())
            vp.makeCurrent()
            try:
                self.assertEqual(vp._gl.glGetError(), 0)
            finally:
                vp.doneCurrent()
            window.close()

    def test_imported_mesh_renders_indexed_and_selects(self):
        """An imported mesh renders through the indexed path and
        selects with its asset bounds, like the T7 gem."""
        with tempfile.TemporaryDirectory() as tmp:
            window, _path = self._show_saved_scene(tmp)
            parsed = meshio.parse(str(HOUSE_OBJ))
            created = window.import_mesh_objects(parsed, 100.0)
            walls = [o for o in created if o["name"] == "Walls"][0]
            self._pump()
            vp = window._viewport
            entry = vp._vbos.get(walls["id"])
            self.assertIsNotNone(entry, "walls have no GPU buffers")
            _vbo, vao, ibo, count, indexed, _key = entry
            self.assertTrue(indexed)
            self.assertIsNotNone(ibo)
            self.assertTrue(ibo.isCreated())
            self.assertTrue(vao.isCreated())
            self.assertEqual(count, 36)  # 12 triangles
            window._select(walls["id"])
            self._pump()
            self.assertFalse(vp.grabFramebuffer().isNull())
            window.close()

    def test_import_is_one_undo_step(self):
        """One undo after importing two objects removes both — the
        whole import is a single undo step."""
        with tempfile.TemporaryDirectory() as tmp:
            window, _path = self._show_saved_scene(tmp)
            parsed = meshio.parse(str(HOUSE_OBJ))
            window.import_mesh_objects(parsed, 100.0)
            self.assertEqual(len(window.scene["objects"]), 3)
            window.undo()
            self.assertEqual(len(window.scene["objects"]), 1)
            self.assertEqual(window.scene["objects"][0]["name"], "Floor")
            window.close()

    def test_import_needs_a_saved_scene(self):
        """A scene with no folder yet cannot import: the helper
        returns nothing and adds no objects (the menu slot tells
        the user to save first — no dialog is shown here, so this
        test cannot hang on one)."""
        window = self.MainWindow(scene=scene_mod.new_scene())
        window.show()
        self._pump()
        parsed = meshio.parse(str(HOUSE_OBJ))
        created = window.import_mesh_objects(parsed, 100.0)
        self.assertEqual(created, [])
        self.assertEqual(window.scene["objects"], [])
        window.close()

    def test_import_dialog_scale(self):
        """The import dialog shows the report and returns the scale
        the user picks. The dialog is accepted programmatically —
        it is never exec()'d here, so no modal loop runs."""
        from PySide6 import QtWidgets
        report = {"objects": 2, "triangles": 16,
                  "warnings": ["a warning"]}
        dialog = self.ImportDialog(
            None, report, scale=meshio.DEFAULT_SCALE)
        self.assertEqual(dialog.scale(), 100.0)
        dialog._scale_spin.setValue(2.5)
        dialog.accept()
        self.assertEqual(dialog.scale(), 2.5)
        dialog.close()


class GlbImportTests(unittest.TestCase):
    """glTF 2.0 binary (.glb) import (T9): container parsing,
    accessors, the Y-up -> Z-up conversion, node transforms, PBR
    colors, and the import_mesh entry point. Fixtures are built with
    tests/glb_fixture.py — real .glb files, not Reblend-only data."""

    def _write(self, tmp, builder, name="test.glb"):
        path = os.path.join(tmp, name)
        builder.write(path)
        return path

    def _triangle_builder(self, with_normals=True, with_uvs=True,
                          indexed=True, material=None, node=None,
                          indices_type=None):
        """One triangle: positions (1,0,0), (0,1,0), (0,0,1), all
        normals (0,0,1), uvs (0,0), (1,0), (0,1). `node` is a dict
        of add_node kwargs for the single node."""
        builder = glb_fixture.GlbBuilder()
        pos = builder.add_vec3([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                (0.0, 0.0, 1.0)])
        attrs = {"POSITION": pos}
        if with_normals:
            attrs["NORMAL"] = builder.add_vec3(
                [(0.0, 0.0, 1.0)] * 3)
        if with_uvs:
            attrs["TEXCOORD_0"] = builder.add_vec2(
                [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
        prim = {"attributes": attrs}
        if indexed:
            if indices_type is not None:
                prim["indices"] = builder.add_accessor(
                    [0, 1, 2], 1, indices_type)
            else:
                prim["indices"] = builder.add_indices([0, 1, 2])
        if material is not None:
            prim["material"] = material
        mesh = builder.add_mesh("Tri", [prim])
        kwargs = {"mesh": mesh}
        kwargs.update(node or {})
        node_index = builder.add_node(**kwargs)
        builder.set_scene([node_index])
        return builder

    def test_sample_imports(self):
        """The shipped sample: 2 objects, 18 triangles, no warnings.
        The crate is red, translated by its group + node and rotated
        30 degrees; the pyramid is teal, unrotated, its normals
        computed (16 verts after the weld splits the corners)."""
        objects, report = meshio.import_mesh(str(SAMPLE_GLB))
        self.assertEqual(report["objects"], 2)
        self.assertEqual(report["triangles"], 18)
        self.assertEqual(report["warnings"], [])
        by_name = {o["name"]: o for o in objects}
        crate = by_name["Crate"]
        self.assertEqual(crate["material"]["base_color"], "#cc3326")
        self.assertEqual(crate["material"]["metalness"], 0.0)
        self.assertEqual(crate["material"]["roughness"], 0.8)
        self.assertEqual(crate["position"], [-80.0, -50.0, 50.0])
        self.assertAlmostEqual(crate["rotation"][2], 30.0, places=5)
        self.assertEqual(crate["rotation"][0], 0.0)
        self.assertEqual(crate["rotation"][1], 0.0)
        self.assertEqual(len(crate["mesh"]["indices"]) // 3, 12)
        self.assertEqual(len(crate["mesh"]["positions"]) // 3, 24)
        pyramid = by_name["Pyramid"]
        self.assertEqual(pyramid["material"]["base_color"], "#1a998c")
        self.assertEqual(pyramid["material"]["metalness"], 0.1)
        self.assertEqual(pyramid["material"]["roughness"], 0.4)
        self.assertEqual(pyramid["position"], [90.0, 0.0, 0.0])
        self.assertEqual(pyramid["rotation"], [0.0, 0.0, 0.0])
        self.assertEqual(len(pyramid["mesh"]["indices"]) // 3, 6)
        self.assertEqual(len(pyramid["mesh"]["positions"]) // 3, 16)
        self.assertTrue(all(v == 0.0 for v in pyramid["mesh"]["uvs"]))

    def test_sample_b64_decodes_to_the_sample(self):
        """The repo carries only sample.glb.b64 (no binaries in the
        snapshot push). It must decode to the same sample — 2
        objects, 18 triangles — or a fresh clone runs a red suite."""
        b64 = HERE / "examples" / "sample.glb.b64"
        with tempfile.NamedTemporaryFile(
                delete=False, suffix=".glb") as tmp:
            tmp.write(base64.b64decode(b64.read_text()))
            tmp_path = tmp.name
        try:
            objects, report = meshio.import_mesh(tmp_path)
        finally:
            os.unlink(tmp_path)
        self.assertEqual(report["objects"], 2)
        self.assertEqual(report["triangles"], 18)
        self.assertEqual(report["warnings"], [])
        self.assertEqual({o["name"] for o in objects},
                         {"Crate", "Pyramid"})

    def test_coordinate_conversion(self):
        """glTF Y-up in meters -> Reblend Z-up in cm: (x, y, z) ->
        (x, -z, y), scaled by 100. Normals get the same rotation
        with no scale."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, self._triangle_builder())
            objects, _report = meshio.import_mesh(path)
        mesh = objects[0]["mesh"]
        self.assertEqual(mesh["positions"],
                         [100.0, 0.0, 0.0,
                          0.0, 0.0, 100.0,
                          0.0, -100.0, 0.0])
        self.assertEqual(mesh["normals"],
                         [0.0, -1.0, 0.0] * 3)
        self.assertEqual(mesh["uvs"], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    def test_node_translation_and_rotation(self):
        """A node rotation about X stays about X; about Z in glTF
        becomes about -Y in Reblend (the up-axis swap); a translation
        converts and scales like a position."""
        builder = glb_fixture.GlbBuilder()
        pos = builder.add_vec3([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                (0.0, 0.0, 1.0)])
        mesh = builder.add_mesh("Tri", [{"attributes": {"POSITION": pos},
                                         "indices": builder.add_indices(
                                             [0, 1, 2])}])
        rot_x = builder.add_node("RotX", mesh=mesh,
                                 rotation=glb_fixture.quat_x(90.0))
        rot_z = builder.add_node("RotZ", mesh=mesh,
                                 rotation=glb_fixture.quat_z(90.0))
        moved = builder.add_node("Moved", mesh=mesh,
                                 translation=[1.0, 2.0, 3.0])
        builder.set_scene([rot_x, rot_z, moved])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, builder)
            objects, _report = meshio.import_mesh(path)
        by_name = {o["name"]: o for o in objects}
        self.assertAlmostEqual(by_name["RotX"]["rotation"][0], 90.0,
                               places=5)
        self.assertAlmostEqual(by_name["RotZ"]["rotation"][1], -90.0,
                               places=5)
        self.assertEqual(by_name["Moved"]["position"],
                         [100.0, -300.0, 200.0])

    def test_parent_transforms_compose(self):
        """A child's world transform composes its parents': a
        translated parent plus a translated child add up, and a
        rotated parent turns the child's translation."""
        builder = glb_fixture.GlbBuilder()
        pos = builder.add_vec3([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                (0.0, 0.0, 1.0)])
        mesh = builder.add_mesh("Tri", [{"attributes": {"POSITION": pos},
                                         "indices": builder.add_indices(
                                             [0, 1, 2])}])
        child_a = builder.add_node("ChildA", mesh=mesh,
                                   translation=[0.0, 1.0, 0.0])
        parent_a = builder.add_node("ParentA", translation=[1.0, 0.0, 0.0],
                                    children=[child_a])
        child_b = builder.add_node("ChildB", mesh=mesh,
                                   translation=[1.0, 0.0, 0.0])
        parent_b = builder.add_node(
            "ParentB", rotation=glb_fixture.quat_y(90.0),
            children=[child_b])
        builder.set_scene([parent_a, parent_b])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, builder)
            objects, _report = meshio.import_mesh(path)
        by_name = {o["name"]: o for o in objects}
        # (1, 1, 0) in glTF -> (100, 0, 100) in Reblend.
        self.assertEqual(by_name["ChildA"]["position"],
                         [100.0, 0.0, 100.0])
        # Ry(90) turns (1, 0, 0) into (0, 0, -1) -> (0, 100, 0).
        got = by_name["ChildB"]["position"]
        for want, actual in zip((0.0, 100.0, 0.0), got):
            self.assertAlmostEqual(actual, want, places=5)

    def test_matrix_node_and_shear_bakes(self):
        """A node `matrix` (column-major) is honored like TRS. A
        sheared matrix cannot decompose, so it bakes into the
        vertices and the object sits at the origin."""
        builder = glb_fixture.GlbBuilder()
        pos = builder.add_vec3([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                (0.0, 0.0, 1.0)])
        mesh = builder.add_mesh("Tri", [{"attributes": {"POSITION": pos},
                                         "indices": builder.add_indices(
                                             [0, 1, 2])}])
        # Translation (2, 0, 0) as a column-major matrix.
        moved = builder.add_node(
            "Moved", mesh=mesh,
            matrix=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1])
        # Shear: x' = x + 0.5 * y, column-major.
        sheared = builder.add_node(
            "Sheared", mesh=mesh,
            matrix=[1, 0, 0, 0, 0.5, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        builder.set_scene([moved, sheared])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, builder)
            objects, _report = meshio.import_mesh(path)
        by_name = {o["name"]: o for o in objects}
        self.assertEqual(by_name["Moved"]["position"], [200.0, 0.0, 0.0])
        baked = by_name["Sheared"]
        self.assertEqual(baked["position"], [0.0, 0.0, 0.0])
        self.assertEqual(baked["rotation"], [0.0, 0.0, 0.0])
        self.assertEqual(baked["scale"], [1.0, 1.0, 1.0])
        # Vertex (0, 1, 0) shears to (0.5, 1, 0), converts to
        # (0.5, 0, 1), scales to (50, 0, 100).
        self.assertIn((50.0, 0.0, 100.0),
                      [tuple(baked["mesh"]["positions"][i:i + 3])
                       for i in range(0, 9, 3)])

    def test_decompose_round_trip(self):
        """Decompose inverts compose_matrix for TRS transforms —
        including non-uniform scale and the gimbal locks — and
        refuses shear, reflection, and zero scale (those bake)."""
        cases = [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            ([10.0, -20.0, 30.0], [30.0, 45.0, 60.0], [2.0, 3.0, 4.0]),
            ([0.0, 0.0, 0.0], [0.0, 90.0, 30.0], [1.0, 1.0, 1.0]),
            ([5.0, 5.0, 5.0], [20.0, -90.0, 10.0], [1.5, 1.5, 1.5]),
            ([0.0, 0.0, 0.0], [-45.0, 0.0, 120.0], [0.5, 2.0, 1.0]),
        ]
        for pos, rot, scl in cases:
            matrix = geometry.compose_matrix(pos, rot, scl)
            parts = meshio_glb._decompose(matrix)
            self.assertIsNotNone(parts, (pos, rot, scl))
            translation, euler, scale = parts
            rebuilt = geometry.compose_matrix(translation, euler, scale)
            for i in range(4):
                for j in range(4):
                    self.assertAlmostEqual(rebuilt[i][j], matrix[i][j],
                                           places=5, msg=(pos, rot, scl))
        shear = [[1.0, 0.5, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        self.assertIsNone(meshio_glb._decompose(shear))
        reflected = geometry.compose_matrix([0, 0, 0], [0, 0, 0],
                                            [-1.0, 1.0, 1.0])
        self.assertIsNone(meshio_glb._decompose(reflected))
        flat = geometry.compose_matrix([0, 0, 0], [0, 0, 0],
                                       [1.0, 0.0, 1.0])
        self.assertIsNone(meshio_glb._decompose(flat))

    def test_base_color_factor_and_pbr(self):
        """baseColorFactor maps to hex; metallic/roughness factors
        map to metalness/roughness. A material with no factors uses
        the glTF defaults (white, metallic 1, rough 1); a primitive
        with no material at all gets Reblend's default gray."""
        builder = glb_fixture.GlbBuilder()
        colored = builder.add_material(
            "Colored", base_color=[0.5, 0.25, 1.0, 1.0],
            metallic=0.7, roughness=0.25)
        bare = builder.add_material("Bare")
        pos = builder.add_vec3([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                (0.0, 0.0, 1.0)])
        idx = builder.add_indices([0, 1, 2])
        mesh = builder.add_mesh("Tri", [
            {"attributes": {"POSITION": pos}, "indices": idx,
             "material": colored},
            {"attributes": {"POSITION": pos}, "indices": idx,
             "material": bare},
            {"attributes": {"POSITION": pos}, "indices": idx},
        ])
        node = builder.add_node("Tri", mesh=mesh)
        builder.set_scene([node])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, builder)
            objects, _report = meshio.import_mesh(path)
        self.assertEqual(len(objects), 3)
        self.assertEqual(objects[0]["material"]["base_color"], "#8040ff")
        self.assertEqual(objects[0]["material"]["metalness"], 0.7)
        self.assertEqual(objects[0]["material"]["roughness"], 0.25)
        self.assertEqual(objects[1]["material"]["base_color"], "#ffffff")
        self.assertEqual(objects[1]["material"]["metalness"], 1.0)
        self.assertEqual(objects[1]["material"]["roughness"], 1.0)
        self.assertEqual(objects[2]["material"]["base_color"], "#b0b0b0")

    def test_missing_normals_are_computed(self):
        """No NORMAL in the file: the box's normals are computed with
        the T2 angle rule — every normal is unit length and
        axis-aligned, so the box keeps its crisp corners."""
        builder = glb_fixture.GlbBuilder()
        pos, _nor, uv, idx = glb_fixture.box_verts(0.5)
        mesh = builder.add_mesh("Box", [{
            "attributes": {"POSITION": builder.add_vec3(pos),
                           "TEXCOORD_0": builder.add_vec2(uv)},
            "indices": builder.add_indices(idx)}])
        node = builder.add_node("Box", mesh=mesh)
        builder.set_scene([node])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, builder)
            objects, _report = meshio.import_mesh(path)
        mesh_out = objects[0]["mesh"]
        self.assertEqual(len(mesh_out["indices"]) // 3, 12)
        normals = mesh_out["normals"]
        for i in range(0, len(normals), 3):
            nx, ny, nz = normals[i:i + 3]
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            self.assertAlmostEqual(length, 1.0, places=6)
            aligned = sum(1 for v in (nx, ny, nz) if abs(abs(v) - 1.0)
                          < 1e-6)
            self.assertEqual(aligned, 1,
                             f"normal {(nx, ny, nz)} is not axis-aligned")

    def test_missing_uvs_default_to_zero(self):
        """No TEXCOORD_0 in the file: every uv is 0."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp, self._triangle_builder(with_uvs=False))
            objects, _report = meshio.import_mesh(path)
        self.assertTrue(all(v == 0.0 for v in objects[0]["mesh"]["uvs"]))

    def test_missing_indices(self):
        """No index list: the positions are triangles in order."""
        builder = glb_fixture.GlbBuilder()
        pos = builder.add_vec3([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                (0.0, 1.0, 0.0), (0.0, 0.0, 0.0),
                                (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        mesh = builder.add_mesh("Pair", [{"attributes": {"POSITION": pos}}])
        node = builder.add_node("Pair", mesh=mesh)
        builder.set_scene([node])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, builder)
            objects, report = meshio.import_mesh(path)
        self.assertEqual(report["triangles"], 2)
        self.assertEqual(objects[0]["mesh"]["indices"], [0, 1, 2, 3, 4, 5])

    def test_uint_indices_and_interleaved(self):
        """UNSIGNED_INT indices read correctly, and interleaved
        POSITION + NORMAL data (one bufferView, byteStride) reads
        correctly too."""
        builder = glb_fixture.GlbBuilder()
        pos, nor = builder.add_interleaved([
            ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3,
             glb_fixture.F32),
            ([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 3,
             glb_fixture.F32),
        ])
        mesh = builder.add_mesh("Tri", [{
            "attributes": {"POSITION": pos, "NORMAL": nor},
            "indices": builder.add_accessor([0, 1, 2], 1,
                                            glb_fixture.U32)}])
        node = builder.add_node("Tri", mesh=mesh)
        builder.set_scene([node])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, builder)
            objects, _report = meshio.import_mesh(path)
        mesh_out = objects[0]["mesh"]
        self.assertEqual(mesh_out["positions"],
                         [100.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, -100.0,
                          0.0])
        self.assertEqual(mesh_out["normals"], [0.0, -1.0, 0.0] * 3)
        self.assertEqual(mesh_out["indices"], [0, 1, 2])

    def test_malformed_files_fail_plainly(self):
        """Bad magic, wrong version, truncation, bad JSON, no JSON
        chunk, no meshes, and bad accessors all raise ValueError
        with a plain message naming the problem — never a
        traceback-shaped error."""
        import struct as struct_mod

        def container(version=2, declared=None, chunks=()):
            body = bytearray()
            for chunk_type, payload in chunks:
                body.extend(struct_mod.pack("<II", len(payload),
                                            chunk_type))
                body.extend(payload)
            total = 12 + len(body) if declared is None else declared
            return (struct_mod.pack("<III", 0x46546C67, version, total)
                    + bytes(body))

        cases = []
        cases.append((b"this is not a glb file at all, no magic here",
                      "not a .glb file"))
        cases.append((container(version=1, chunks=[(0x4E4F534A, b"{}")]),
                      "version 1"))
        good = container(chunks=[(0x4E4F534A, b"{}")])
        cases.append((container(declared=len(good) + 100,
                                chunks=[(0x4E4F534A, b"{}")]),
                      "truncated"))
        cases.append((container(chunks=[(0x4E4F534A, b"{oops")]),
                      "not valid JSON"))
        cases.append((container(chunks=[(0x4E4F534A, b"[1, 2]")]),
                      "does not hold a glTF scene"))
        cases.append((container(chunks=[(0x004E4942, b"\x00\x00\x00\x00")]),
                      "no JSON chunk"))
        cases.append((container(chunks=[
            (0x4E4F534A, b'{"asset":{"version":"2.0"}}')]),
            "no meshes found"))
        # Bad accessor index: POSITION points at accessor 9.
        builder = glb_fixture.GlbBuilder()
        mesh = builder.add_mesh("Bad", [{"attributes": {"POSITION": 9}}])
        node = builder.add_node("Bad", mesh=mesh)
        builder.set_scene([node])
        cases.append((builder.build(), "accessor 9"))
        # Index outside the vertices.
        builder = glb_fixture.GlbBuilder()
        pos = builder.add_vec3([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                (0.0, 1.0, 0.0)])
        mesh = builder.add_mesh("Bad", [{
            "attributes": {"POSITION": pos},
            "indices": builder.add_indices([0, 1, 9])}])
        node = builder.add_node("Bad", mesh=mesh)
        builder.set_scene([node])
        cases.append((builder.build(), "outside its 3 vertices"))
        # Indices not a multiple of 3.
        builder = glb_fixture.GlbBuilder()
        pos = builder.add_vec3([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        mesh = builder.add_mesh("Bad", [{
            "attributes": {"POSITION": pos},
            "indices": builder.add_indices([0, 1, 2, 0])}])
        node = builder.add_node("Bad", mesh=mesh)
        builder.set_scene([node])
        cases.append((builder.build(), "not a multiple of 3"))
        for blob, needle in cases:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "bad.glb")
                with open(path, "wb") as fh:
                    fh.write(blob)
                with self.assertRaises(ValueError) as caught:
                    meshio_glb.parse_glb(path)
            self.assertIn(needle, str(caught.exception),
                          f"{blob[:40]!r} raised {caught.exception!r}")

    def test_skipped_features_warn(self):
        """Skins, animations, cameras, lights, textures, a
        non-triangle primitive, and a primitive with no positions
        all warn by name — and the one good mesh still imports."""
        builder = glb_fixture.GlbBuilder()
        builder.extras["skins"] = [{"joints": [0]}]
        builder.extras["animations"] = [{"channels": [], "samplers": []}]
        builder.extras["cameras"] = [{"type": "perspective",
                                      "perspective": {"yfov": 0.7,
                                                      "aspectRatio": 1.0,
                                                      "znear": 0.1,
                                                      "zfar": 100.0}}]
        builder.extras["extensions"] = {"KHR_lights_punctual": {
            "lights": [{"type": "point", "color": [1, 1, 1]}]}}
        textured = builder.add_material(
            "Textured", base_color=[1.0, 0.0, 0.0, 1.0],
            with_texture=True)
        pos = builder.add_vec3([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                (0.0, 0.0, 1.0)])
        idx = builder.add_indices([0, 1, 2])
        mesh = builder.add_mesh("Mixed", [
            {"attributes": {"POSITION": pos}, "indices": idx,
             "material": textured},
            {"attributes": {"POSITION": pos}, "indices": idx, "mode": 1},
            {"attributes": {}, "indices": idx},
        ])
        node = builder.add_node("Mixed", mesh=mesh)
        builder.set_scene([node])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, builder)
            objects, report = meshio.import_mesh(path)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["material"]["base_color"], "#ff0000")
        warnings = " ".join(report["warnings"])
        for needle in ("skin", "animation", "camera", "light", "texture",
                       "not triangles", "no positions"):
            self.assertIn(needle, warnings, report["warnings"])

    def test_import_mesh_entry_point(self):
        """import_mesh dispatches .glb and reports counts; an
        unsupported extension and a missing file fail plainly."""
        objects, report = meshio.import_mesh(str(SAMPLE_GLB))
        self.assertEqual(report["objects"], 2)
        self.assertEqual(report["triangles"], 18)
        self.assertEqual(len(objects), 2)
        with tempfile.TemporaryDirectory() as tmp:
            txt = os.path.join(tmp, "notes.txt")
            with open(txt, "w", encoding="utf-8") as fh:
                fh.write("hello")
            with self.assertRaises(ValueError) as caught:
                meshio.import_mesh(txt)
            self.assertIn("not a supported mesh file",
                          str(caught.exception))
            with self.assertRaises(FileNotFoundError):
                meshio.import_mesh(os.path.join(tmp, "missing.glb"))

    def test_scale_applied(self):
        """Default scale is x100 (meters to cm); a custom scale is
        honored exactly — for the vertices and the node translation
        alike."""
        default, _ = meshio.import_mesh(str(SAMPLE_GLB))
        custom, _ = meshio.import_mesh(str(SAMPLE_GLB), scale=2.5)
        by_name = {o["name"]: o for o in default}
        by_custom = {o["name"]: o for o in custom}
        self.assertEqual(by_name["Pyramid"]["position"], [90.0, 0.0, 0.0])
        self.assertEqual(by_custom["Pyramid"]["position"],
                         [2.25, 0.0, 0.0])
        # First crate vertex: the +X face corner (0.5, -0.5, 0.5)
        # converts to (0.5, -0.5, -0.5).
        self.assertEqual(by_name["Crate"]["mesh"]["positions"][:3],
                         [50.0, -50.0, -50.0])
        self.assertEqual(by_custom["Crate"]["mesh"]["positions"][:3],
                         [1.25, -1.25, -1.25])

    def test_duplicate_names(self):
        """Two nodes sharing one mesh and one name become "Rock"
        and "Rock 2" — the outliner never shows two identical
        names from one import."""
        builder = glb_fixture.GlbBuilder()
        pos = builder.add_vec3([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                (0.0, 0.0, 1.0)])
        mesh = builder.add_mesh("Rock", [{
            "attributes": {"POSITION": pos},
            "indices": builder.add_indices([0, 1, 2])}])
        first = builder.add_node("Rock", mesh=mesh)
        second = builder.add_node("Rock", mesh=mesh,
                                  translation=[2.0, 0.0, 0.0])
        builder.set_scene([first, second])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, builder)
            objects, _report = meshio.import_mesh(path)
        self.assertEqual([o["name"] for o in objects],
                         ["Rock", "Rock 2"])


@unittest.skipUnless(QT_TESTS, "Qt tests need LIGHT_ENGINE_QT_TESTS=1")
class QtGlbImportTests(unittest.TestCase):
    """The .glb import flow in the live window (T9): assets written,
    objects created with their node transforms, one undo step, and
    the saved-scene guard. Mirrors QtObjImportTests (T8)."""

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets
        cls._app = QtWidgets.QApplication.instance()
        if cls._app is None:
            cls._app = QtWidgets.QApplication([])
        from light_engine.window import MainWindow
        cls.MainWindow = MainWindow

    def _pump(self, rounds=10):
        from PySide6 import QtCore, QtWidgets
        for _ in range(rounds):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)

    def _show_saved_scene(self, tmp):
        """A window on a scene saved in tmp, so imports have a
        folder for their assets."""
        scene = scene_mod.new_scene("Import test")
        floor = scene_mod.new_object("plane", name="Floor")
        scene["objects"].append(floor)
        path = os.path.join(tmp, "scene.json")
        scene_mod.save_scene(scene, path)
        loaded = scene_mod.load_scene(path)
        window = self.MainWindow(scene=loaded, scene_path=path)
        window.resize(800, 600)
        window.show()
        self._pump()
        return window, path

    def test_import_writes_assets_and_creates_objects(self):
        """The sample imports: two assets land in assets/, two mesh
        objects are created with their node transforms, the triangle
        readout counts the imported 18, and the scene renders with
        no GL errors."""
        with tempfile.TemporaryDirectory() as tmp:
            window, _path = self._show_saved_scene(tmp)
            parsed = meshio.parse(str(SAMPLE_GLB))
            created = window.import_mesh_objects(parsed, 100.0)
            self.assertEqual(len(created), 2)
            self.assertEqual(
                sorted(os.listdir(os.path.join(tmp, "assets"))),
                ["sample-Crate.mesh.json", "sample-Pyramid.mesh.json"])
            for obj in created:
                self.assertEqual(obj["shape"], "mesh")
                self.assertTrue(obj["mesh_file"].startswith("assets/"))
            by_name = {o["name"]: o for o in created}
            self.assertEqual(by_name["Crate"]["position"],
                             [-80.0, -50.0, 50.0])
            self.assertAlmostEqual(by_name["Crate"]["rotation"][2], 30.0,
                                   places=5)
            self.assertEqual(by_name["Pyramid"]["position"],
                             [90.0, 0.0, 0.0])
            self.assertEqual(len(window.scene["objects"]), 3)
            # Floor 2 + sample 18.
            self.assertIn("20 triangles", window._tri_label.text())
            self._pump()
            vp = window._viewport
            self.assertIsNone(vp._shader_error)
            self.assertFalse(vp.grabFramebuffer().isNull())
            vp.makeCurrent()
            try:
                self.assertEqual(vp._gl.glGetError(), 0)
            finally:
                vp.doneCurrent()
            window.close()

    def test_imported_mesh_renders_indexed_and_selects(self):
        """An imported .glb mesh renders through the indexed path
        and selects with its asset bounds, like the T8 house."""
        with tempfile.TemporaryDirectory() as tmp:
            window, _path = self._show_saved_scene(tmp)
            parsed = meshio.parse(str(SAMPLE_GLB))
            created = window.import_mesh_objects(parsed, 100.0)
            crate = [o for o in created if o["name"] == "Crate"][0]
            self._pump()
            vp = window._viewport
            entry = vp._vbos.get(crate["id"])
            self.assertIsNotNone(entry, "crate has no GPU buffers")
            _vbo, vao, ibo, count, indexed, _key = entry
            self.assertTrue(indexed)
            self.assertIsNotNone(ibo)
            self.assertTrue(ibo.isCreated())
            self.assertTrue(vao.isCreated())
            self.assertEqual(count, 36)  # 12 triangles
            window._select(crate["id"])
            self._pump()
            self.assertFalse(vp.grabFramebuffer().isNull())
            window.close()

    def test_import_is_one_undo_step(self):
        """One undo after importing two objects removes both — the
        whole import is a single undo step."""
        with tempfile.TemporaryDirectory() as tmp:
            window, _path = self._show_saved_scene(tmp)
            parsed = meshio.parse(str(SAMPLE_GLB))
            window.import_mesh_objects(parsed, 100.0)
            self.assertEqual(len(window.scene["objects"]), 3)
            window.undo()
            self.assertEqual(len(window.scene["objects"]), 1)
            self.assertEqual(window.scene["objects"][0]["name"], "Floor")
            window.close()

    def test_import_needs_a_saved_scene(self):
        """A scene with no folder yet cannot import: the helper
        returns nothing and adds no objects."""
        window = self.MainWindow(scene=scene_mod.new_scene())
        window.show()
        self._pump()
        parsed = meshio.parse(str(SAMPLE_GLB))
        created = window.import_mesh_objects(parsed, 100.0)
        self.assertEqual(created, [])
        self.assertEqual(window.scene["objects"], [])
        window.close()


class ObjExportTests(unittest.TestCase):
    """The .obj writer (T10): de-indexed v/vt/vn/f + .mtl, world-space
    vertices, scale, and the round-trip acceptance — export the
    tavern, re-import it, same object and triangle counts."""

    def _read_obj(self, path):
        """(verts, uvs, normals, faces, groups, mtllib) parsed from
        an .obj file. Faces are tuples of (v, vt, vn) 1-based index
        triples; groups are the `o` names in order."""
        verts, uvs, normals, faces, groups = [], [], [], [], []
        mtllib = None
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "v":
                    verts.append(tuple(float(v) for v in parts[1:4]))
                elif parts[0] == "vt":
                    uvs.append(tuple(float(v) for v in parts[1:3]))
                elif parts[0] == "vn":
                    normals.append(tuple(float(v) for v in parts[1:4]))
                elif parts[0] == "f":
                    faces.append(tuple(
                        tuple(int(i) for i in corner.split("/"))
                        for corner in parts[1:4]))
                elif parts[0] == "o":
                    groups.append(parts[1])
                elif parts[0] == "mtllib":
                    mtllib = parts[1]
        return verts, uvs, normals, faces, groups, mtllib

    def _read_mtl(self, path):
        """{material name: (r, g, b)} from an .mtl file (Kd only)."""
        materials = {}
        current = None
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "newmtl" and len(parts) >= 2:
                    current = parts[1]
                elif parts[0] == "Kd" and current is not None:
                    materials[current] = tuple(float(v) for v in parts[1:4])
        return materials

    def test_round_trip_tavern_counts(self):
        """The acceptance test: export the tavern, re-import it
        through import_mesh, same object count AND same triangle
        count. The face and group counts are also read straight
        from the file, so the numbers are checked twice."""
        scene = scene_mod.load_scene(str(EXAMPLE))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tavern.obj")
            report = meshio_obj.export_obj(scene["objects"], path,
                                           scale=0.01)
            self.assertEqual(report["objects"], 19)
            self.assertEqual(report["triangles"], 1674)
            self.assertEqual(report["warnings"], [])
            _v, _vt, _vn, faces, groups, mtllib = self._read_obj(path)
            self.assertEqual(len(faces), 1674)
            self.assertEqual(len(groups), 19)
            self.assertEqual(mtllib, "tavern.mtl")
            self.assertTrue(os.path.isfile(
                os.path.join(tmp, "tavern.mtl")))
            objects, re_report = meshio.import_mesh(path, 100.0)
            self.assertEqual(len(objects), 19)
            self.assertEqual(re_report["objects"], 19)
            self.assertEqual(re_report["triangles"], 1674)
            self.assertEqual(re_report["warnings"], [])

    def test_round_trip_vertex_positions(self):
        """A re-imported vertex lands back at its original world
        position: export at 0.01, import at 100. Checked on the
        mug's bounds — the original world-space bounds (tessellated
        + transformed by hand) against the re-imported mesh."""
        scene = scene_mod.load_scene(str(EXAMPLE))
        mug = [o for o in scene["objects"] if o["id"] == "mug"][0]
        soup = geometry.tessellate_indexed(mug["shape"], mug["params"])
        model = geometry.compose_matrix(mug["position"], mug["rotation"],
                                        mug["scale"])
        xs, ys, zs = [], [], []
        for i in range(len(soup["positions"]) // 3):
            x, y, z = geometry.apply_matrix(
                model, tuple(soup["positions"][i * 3:i * 3 + 3]))
            xs.append(x)
            ys.append(y)
            zs.append(z)
        original = (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tavern.obj")
            meshio_obj.export_obj(scene["objects"], path, scale=0.01)
            objects, _report = meshio.import_mesh(path, 100.0)
        re_mug = [o for o in objects if o["name"] == "Mug"][0]["mesh"]
        pos = re_mug["positions"]
        reimported = (min(pos[0::3]), max(pos[0::3]),
                      min(pos[1::3]), max(pos[1::3]),
                      min(pos[2::3]), max(pos[2::3]))
        for a, b in zip(original, reimported):
            self.assertAlmostEqual(a, b, delta=1e-4)

    def test_transformed_object_exports_world_space(self):
        """A box moved, rotated 90 degrees about Z, and scaled
        non-uniformly exports world-space vertices: the bounds and
        one known corner are checked against hand-computed values.
        Without the transform every object would collapse to the
        origin."""
        obj = scene_mod.new_object("box", name="Crate")
        obj["position"] = [10.0, 20.0, 30.0]
        obj["rotation"] = [0.0, 0.0, 90.0]
        obj["scale"] = [2.0, 1.0, 1.0]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "crate.obj")
            meshio_obj.export_obj([obj], path, scale=1.0)
            verts, _uv, _vn, _f, _g, _m = self._read_obj(path)
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        zs = [v[2] for v in verts]
        # Local (50, 50, 50) -> scale -> (100, 50, 50) -> Rz 90 ->
        # (-50, 100, 50) -> translate -> (-40, 120, 80).
        self.assertAlmostEqual(min(xs), -40.0, delta=1e-6)
        self.assertAlmostEqual(max(xs), 60.0, delta=1e-6)
        self.assertAlmostEqual(min(ys), -80.0, delta=1e-6)
        self.assertAlmostEqual(max(ys), 120.0, delta=1e-6)
        self.assertAlmostEqual(min(zs), -20.0, delta=1e-6)
        self.assertAlmostEqual(max(zs), 80.0, delta=1e-6)
        self.assertTrue(
            any(abs(v[0] + 40.0) < 1e-6 and abs(v[1] - 120.0) < 1e-6
                and abs(v[2] - 80.0) < 1e-6 for v in verts),
            "the transformed corner (-40, 120, 80) is not in the file")

    def test_non_uniform_scale_normals_stay_perpendicular(self):
        """Normals go through the inverse-transpose: a flat cylinder
        scaled non-uniformly and rotated exports normals that are
        perpendicular to their own triangles. Rotation-only normals
        would fail this — the check is the invariant, not a
        comparison against one implementation."""
        obj = scene_mod.new_object("cylinder", name="Pipe")
        obj["smooth_shading"] = False
        obj["rotation"] = [0.0, 0.0, 30.0]
        obj["scale"] = [2.0, 1.0, 1.0]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pipe.obj")
            meshio_obj.export_obj([obj], path, scale=1.0)
            verts, _uv, normals, faces, _g, _m = self._read_obj(path)
        self.assertTrue(faces)
        for tri in faces:
            idx = [c[0] - 1 for c in tri]
            p = [verts[i] for i in idx]
            e1 = tuple(p[1][k] - p[0][k] for k in range(3))
            e2 = tuple(p[2][k] - p[0][k] for k in range(3))
            for i in idx:
                n = normals[i]
                dot1 = sum(n[k] * e1[k] for k in range(3))
                dot2 = sum(n[k] * e2[k] for k in range(3))
                self.assertAlmostEqual(dot1, 0.0, delta=1e-6)
                self.assertAlmostEqual(dot2, 0.0, delta=1e-6)
                length = math.sqrt(sum(c * c for c in n))
                self.assertAlmostEqual(length, 1.0, delta=1e-6)

    def test_mesh_object_exports_asset_geometry(self):
        """A mesh object exports its asset's geometry: the gem (8
        triangles) exports and re-imports with the same bounds as
        the asset file."""
        asset = meshfile.load_mesh(str(GEM_ASSET))
        obj = scene_mod.new_object("mesh", name="Gem")
        obj["mesh_file"] = str(GEM_ASSET)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gem.obj")
            report = meshio_obj.export_obj([obj], path, scale=1.0)
            self.assertEqual(report["objects"], 1)
            self.assertEqual(report["triangles"], 8)
            objects, re_report = meshio.import_mesh(path, 1.0)
            self.assertEqual(re_report["triangles"], 8)
        pos = asset["positions"]
        original = (min(pos[0::3]), max(pos[0::3]),
                    min(pos[1::3]), max(pos[1::3]),
                    min(pos[2::3]), max(pos[2::3]))
        re_pos = objects[0]["mesh"]["positions"]
        reimported = (min(re_pos[0::3]), max(re_pos[0::3]),
                      min(re_pos[1::3]), max(re_pos[1::3]),
                      min(re_pos[2::3]), max(re_pos[2::3]))
        for a, b in zip(original, reimported):
            self.assertAlmostEqual(a, b, delta=1e-4)

    def test_mtl_kd_matches_base_color(self):
        """The .mtl carries Kd for each distinct base color, and a
        re-import reads the colors back exactly."""
        red = scene_mod.new_object("box", name="Red")
        red["material"]["base_color"] = "#cc3326"
        teal = scene_mod.new_object("box", name="Teal")
        teal["material"]["base_color"] = "#1a998c"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "colors.obj")
            meshio_obj.export_obj([red, teal], path, scale=1.0)
            materials = self._read_mtl(os.path.join(tmp, "colors.mtl"))
            self.assertEqual(len(materials), 2)
            kds = sorted(materials.values())
            expected = sorted([(0xcc / 255.0, 0x33 / 255.0, 0x26 / 255.0),
                               (0x1a / 255.0, 0x99 / 255.0, 0x8c / 255.0)])
            for got, want in zip(kds, expected):
                for g, w in zip(got, want):
                    self.assertAlmostEqual(g, w, delta=0.001)
            objects, _report = meshio.import_mesh(path, 1.0)
            colors = sorted(o["material"]["base_color"] for o in objects)
            self.assertEqual(colors, ["#1a998c", "#cc3326"])

    def test_scale_applied(self):
        """Export scale multiplies the vertices: a 100 cm box at
        0.01 spans 1 unit, at 1.0 it spans 100."""
        obj = scene_mod.new_object("box", name="Box")
        with tempfile.TemporaryDirectory() as tmp:
            for scale, span in ((0.01, 1.0), (1.0, 100.0)):
                path = os.path.join(tmp, f"box-{scale}.obj")
                meshio_obj.export_obj([obj], path, scale=scale)
                verts, _uv, _vn, _f, _g, _m = self._read_obj(path)
                xs = [v[0] for v in verts]
                self.assertAlmostEqual(max(xs) - min(xs), span, delta=1e-9)

    def test_smooth_object_keeps_smooth_normals(self):
        """Smooth vs flat is written as-is: a smooth sphere's
        normals point radially (median deviation under 0.01), a flat
        sphere's are per-face (median deviation over 0.1). Medians,
        not means — the sphere's pole corners are outliers in both
        cases (a pre-existing tessellation characteristic, and the
        export writes the soup the viewport draws)."""
        for smooth, bound in ((True, 0.01), (False, 0.1)):
            obj = scene_mod.new_object("sphere", name="Ball")
            obj["smooth_shading"] = smooth
            obj["params"] = {"radius": 50.0, "segments": 20, "rings": 12}
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "ball.obj")
                meshio_obj.export_obj([obj], path, scale=1.0)
                verts, _uv, normals, _f, _g, _m = self._read_obj(path)
            deviations = []
            for (x, y, z), (nx, ny, nz) in zip(verts, normals):
                length = math.sqrt(x * x + y * y + z * z)
                rx, ry, rz = x / length, y / length, z / length
                deviations.append(math.sqrt((nx - rx) ** 2
                                            + (ny - ry) ** 2
                                            + (nz - rz) ** 2))
            deviations.sort()
            median = deviations[len(deviations) // 2]
            if smooth:
                self.assertLess(median, bound)
            else:
                self.assertGreater(median, bound)

    def test_missing_mesh_asset_warns_and_skips(self):
        """A mesh object whose asset file is missing warns naming
        the file and is skipped; the rest of the scene still
        exports."""
        box = scene_mod.new_object("box", name="Crate")
        ghost = scene_mod.new_object("mesh", name="Ghost")
        ghost["mesh_file"] = "assets/nope.mesh.json"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.obj")
            report = meshio_obj.export_obj([box, ghost], path, scale=1.0,
                                           base_path=tmp)
            self.assertEqual(report["objects"], 1)
            self.assertEqual(report["triangles"], 12)
            self.assertEqual(len(report["warnings"]), 1)
            self.assertIn("nope.mesh.json", report["warnings"][0])
            _v, _vt, _vn, faces, groups, _m = self._read_obj(path)
            self.assertEqual(groups, ["Crate"])
            self.assertEqual(len(faces), 12)

    def test_non_geometry_object_skipped(self):
        """An object whose kind is not 'object' (emitters are T16)
        is skipped with a plain note, never fatal."""
        emitter = scene_mod.new_object("box", name="Puffs")
        emitter["kind"] = "emitter"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.obj")
            report = meshio_obj.export_obj([emitter], path, scale=1.0)
            self.assertEqual(report["objects"], 0)
            self.assertEqual(report["triangles"], 0)
            self.assertEqual(len(report["warnings"]), 1)
            self.assertIn("not geometry", report["warnings"][0])

    def test_unwritable_path_raises(self):
        """An unwritable path raises OSError — the window turns it
        into a plain message, never a crash. The writer does not
        create folders."""
        obj = scene_mod.new_object("box", name="Box")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "no-such-dir", "out.obj")
            with self.assertRaises(OSError):
                meshio_obj.export_obj([obj], path, scale=1.0)

    def test_empty_export_writes_header_only(self):
        """Exporting no objects still writes a valid pair: a header
        and the mtllib line, no groups, an empty .mtl. The window
        guards the empty scene before the dialog; the writer handles
        it anyway."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.obj")
            report = meshio_obj.export_obj([], path, scale=1.0)
            self.assertEqual(report, {"objects": 0, "triangles": 0,
                                      "warnings": []})
            _v, _vt, _vn, faces, groups, mtllib = self._read_obj(path)
            self.assertEqual(faces, [])
            self.assertEqual(groups, [])
            self.assertEqual(mtllib, "empty.mtl")
            self.assertTrue(os.path.isfile(os.path.join(tmp, "empty.mtl")))


@unittest.skipUnless(QT_TESTS, "Qt tests need LIGHT_ENGINE_QT_TESTS=1")
class QtObjExportTests(unittest.TestCase):
    """The .obj export flow in the live window (T10): the dialog-free
    export_mesh_objects core, the export dialog's scale, and the
    round-tripped scene rendering. The file picker and the scale
    dialog need a human, so they are verified by the screenshot
    script instead."""

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets
        cls._app = QtWidgets.QApplication.instance()
        if cls._app is None:
            cls._app = QtWidgets.QApplication([])
        from light_engine.window import ExportDialog, MainWindow
        cls.MainWindow = MainWindow
        cls.ExportDialog = ExportDialog

    def _pump(self, rounds=10):
        from PySide6 import QtCore, QtWidgets
        for _ in range(rounds):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)

    def _show_tavern(self, tmp):
        """A window on the tavern scene saved in tmp."""
        scene = scene_mod.load_scene(str(EXAMPLE))
        path = os.path.join(tmp, "scene.json")
        scene_mod.save_scene(scene, path)
        loaded = scene_mod.load_scene(path)
        window = self.MainWindow(scene=loaded, scene_path=path)
        window.resize(800, 600)
        window.show()
        self._pump()
        return window

    def _show_saved_scene(self, tmp, name="Export test"):
        """A window on an empty scene saved in tmp."""
        scene = scene_mod.new_scene(name)
        path = os.path.join(tmp, "scene.json")
        scene_mod.save_scene(scene, path)
        loaded = scene_mod.load_scene(path)
        window = self.MainWindow(scene=loaded, scene_path=path)
        window.resize(800, 600)
        window.show()
        self._pump()
        return window

    def test_export_scene_writes_obj_and_mtl(self):
        """The tavern exports through the window core: an .obj +
        .mtl pair land on disk, and re-importing them gives back
        19 objects and 1,674 triangles."""
        with tempfile.TemporaryDirectory() as tmp:
            window = self._show_tavern(tmp)
            path = os.path.join(tmp, "tavern.obj")
            report = window.export_mesh_objects(window.scene["objects"],
                                                path, 0.01)
            self.assertEqual(report["objects"], 19)
            self.assertEqual(report["triangles"], 1674)
            self.assertEqual(report["warnings"], [])
            self.assertTrue(os.path.isfile(path))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "tavern.mtl")))
            objects, re_report = meshio.import_mesh(path, 100.0)
            self.assertEqual(len(objects), 19)
            self.assertEqual(re_report["triangles"], 1674)
            window.close()

    def test_export_mesh_writes_only_selected(self):
        """Exporting the selected mug writes one group with the
        mug's triangles — nothing else from the scene."""
        with tempfile.TemporaryDirectory() as tmp:
            window = self._show_tavern(tmp)
            window._select("mug")
            mug = scene_mod.find_object(window.scene, "mug")
            expected = geometry.tessellate(
                mug["shape"], mug["params"])["tri_count"]
            path = os.path.join(tmp, "mug.obj")
            report = window.export_mesh_objects([mug], path, 0.01)
            self.assertEqual(report["objects"], 1)
            self.assertEqual(report["triangles"], expected)
            with open(path, encoding="utf-8") as fh:
                groups = [line.split()[1] for line in fh
                          if line.startswith("o ")]
            self.assertEqual(groups, ["Mug"])
            window.close()

    def test_export_dialog_scale(self):
        """The export dialog shows the report and returns the scale
        the user picks. The dialog is accepted programmatically —
        it is never exec()'d here, so no modal loop runs."""
        report = {"objects": 2, "triangles": 16,
                  "warnings": ["a warning"]}
        dialog = self.ExportDialog(None, report)
        self.assertEqual(dialog.scale(), 0.01)
        dialog._scale_spin.setValue(2.5)
        dialog.accept()
        self.assertEqual(dialog.scale(), 2.5)
        dialog.close()

    def test_export_round_tripped_scene_renders(self):
        """The exported tavern re-imports into a fresh window and
        renders: 19 mesh objects, the 1,674-triangle readout, a
        real framebuffer, GL error 0."""
        with tempfile.TemporaryDirectory() as tmp:
            window = self._show_tavern(tmp)
            path = os.path.join(tmp, "tavern.obj")
            window.export_mesh_objects(window.scene["objects"], path, 0.01)
            window.close()
            scene = scene_mod.new_scene("Round trip")
            scene_path = os.path.join(tmp, "roundtrip.json")
            scene_mod.save_scene(scene, scene_path)
            loaded = scene_mod.load_scene(scene_path)
            window2 = self.MainWindow(scene=loaded, scene_path=scene_path)
            window2.resize(800, 600)
            window2.show()
            self._pump()
            parsed = meshio.parse(path)
            created = window2.import_mesh_objects(parsed, 100.0)
            self.assertEqual(len(created), 19)
            self.assertIn("1,674 triangles", window2._tri_label.text())
            self._pump()
            vp = window2._viewport
            self.assertIsNone(vp._shader_error)
            self.assertFalse(vp.grabFramebuffer().isNull())
            vp.makeCurrent()
            try:
                self.assertEqual(vp._gl.glGetError(), 0)
            finally:
                vp.doneCurrent()
            window2.close()

    def test_export_with_missing_mesh_warns(self):
        """A mesh object whose asset is missing warns naming the
        file and is skipped; the floor still exports. The object is
        added directly to the scene (the window drops missing meshes
        on load, so the normal path cannot hold one)."""
        with tempfile.TemporaryDirectory() as tmp:
            window = self._show_saved_scene(tmp)
            floor = scene_mod.new_object("plane", name="Floor")
            window.scene["objects"].append(floor)
            ghost = scene_mod.new_object("mesh", name="Ghost")
            ghost["mesh_file"] = "assets/nope.mesh.json"
            window.scene["objects"].append(ghost)
            path = os.path.join(tmp, "scene.obj")
            report = window.export_mesh_objects(window.scene["objects"],
                                                path, 0.01)
            self.assertEqual(report["objects"], 1)
            self.assertEqual(report["triangles"], 2)
            self.assertEqual(len(report["warnings"]), 1)
            self.assertIn("nope.mesh.json", report["warnings"][0])
            window.close()

    def test_export_mesh_needs_a_selection(self):
        """Export mesh with nothing selected says so in the status
        bar and opens no dialog — the slot returns before the file
        picker, so this test cannot hang on one."""
        with tempfile.TemporaryDirectory() as tmp:
            window = self._show_tavern(tmp)
            self.assertIsNone(window.selected_id)
            window._export_mesh()
            self.assertEqual(window.statusBar().currentMessage(),
                             "Select an object first")
            window.close()

    def test_export_scene_empty_shows_message(self):
        """Export scene on an empty scene says so in the status bar
        and opens no dialog."""
        window = self.MainWindow(scene=scene_mod.new_scene())
        window.show()
        self._pump()
        window._export_scene_obj()
        self.assertEqual(window.statusBar().currentMessage(),
                         "Nothing to export — the scene is empty")
        window.close()


class GlbExportTests(unittest.TestCase):
    """The .glb writer (T11): glTF 2.0 binary container, node TRS
    transforms, PBR materials, and the round-trip acceptance —
    export the tavern, re-import it through the T9 importer, same
    object and triangle counts, transforms and materials back."""

    def _read_container(self, path):
        """(magic, version, declared length, actual size, [(chunk
        length, chunk type)], json bytes, bin bytes) read straight
        from the file with struct — independent of the importer's
        own reader, so the header checks are not circular."""
        with open(path, "rb") as fh:
            data = fh.read()
        magic, version, length = struct.unpack_from("<III", data, 0)
        chunks = []
        offset = 12
        json_bytes = b""
        bin_bytes = b""
        while offset + 8 <= len(data):
            chunk_len, chunk_type = struct.unpack_from("<II", data,
                                                       offset)
            chunks.append((chunk_len, chunk_type))
            chunk = data[offset + 8:offset + 8 + chunk_len]
            if chunk_type == 0x4E4F534A:
                json_bytes = chunk
            elif chunk_type == 0x004E4942:
                bin_bytes = chunk
            offset += 8 + chunk_len
        return (magic, version, length, len(data), chunks,
                json_bytes, bin_bytes)

    def _assert_same_rotation(self, a_deg, b_deg, delta=1e-6):
        """Two Euler rotations describe the same rotation: compared
        via their matrices, not the angles — Euler has multiple
        representations of one rotation (and the importer's gimbal
        handling folds roll into yaw)."""
        ma = geometry.compose_matrix([0.0, 0.0, 0.0], a_deg,
                                     [1.0, 1.0, 1.0])
        mb = geometry.compose_matrix([0.0, 0.0, 0.0], b_deg,
                                     [1.0, 1.0, 1.0])
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(ma[i][j], mb[i][j], delta=delta)

    def test_header_and_container(self):
        """The container, checked with struct, not the module under
        test: magic 'glTF', version 2, declared length equal to the
        actual file size, exactly two chunks (JSON then BIN), both
        chunk lengths 4-byte aligned, JSON parses, asset version
        "2.0"."""
        obj = scene_mod.new_object("box", name="Box")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "box.glb")
            meshio_glb.export_glb([obj], path, scale=0.01)
            magic, version, length, size, chunks, json_bytes, _bin = \
                self._read_container(path)
        self.assertEqual(magic, 0x46546C67)
        self.assertEqual(version, 2)
        self.assertEqual(length, size)
        self.assertEqual([t for _l, t in chunks],
                         [0x4E4F534A, 0x004E4942])
        for chunk_len, _t in chunks:
            self.assertEqual(chunk_len % 4, 0)
        gltf = json.loads(json_bytes.decode("utf-8"))
        self.assertEqual(gltf["asset"]["version"], "2.0")
        self.assertEqual(size, 12 + 8 + len(json_bytes)
                         + 8 + chunks[1][0])

    def test_json_structure(self):
        """The JSON layout for the tavern: one scene with 19 root
        nodes, 19 nodes, 19 meshes, one primitive per mesh, every
        primitive with POSITION/NORMAL/TEXCOORD_0 + indices +
        material, POSITION accessors carrying min/max, 11 materials
        (the tavern's distinct color/metal/rough triples), all
        bufferView offsets 4-byte aligned, all indices
        UNSIGNED_SHORT (every tavern object is far under 65,536
        vertices)."""
        scene = scene_mod.load_scene(str(EXAMPLE))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tavern.glb")
            meshio_glb.export_glb(scene["objects"], path, scale=0.01)
            with open(path, "rb") as fh:
                data = fh.read()
        gltf, bin_bytes = meshio_glb._read_chunks(data)
        self.assertEqual(gltf["scene"], 0)
        self.assertEqual(len(gltf["scenes"]), 1)
        self.assertEqual(len(gltf["scenes"][0]["nodes"]), 19)
        self.assertEqual(len(gltf["nodes"]), 19)
        self.assertEqual(len(gltf["meshes"]), 19)
        self.assertEqual(len(gltf["materials"]), 11)
        self.assertEqual(gltf["buffers"][0]["byteLength"], len(bin_bytes))
        for view in gltf["bufferViews"]:
            self.assertEqual(view["byteOffset"] % 4, 0)
            self.assertEqual(view["buffer"], 0)
        for node in gltf["nodes"]:
            self.assertIn("mesh", node)
            self.assertIn("name", node)
        for mesh in gltf["meshes"]:
            self.assertEqual(len(mesh["primitives"]), 1)
            prim = mesh["primitives"][0]
            attrs = prim["attributes"]
            self.assertIn("POSITION", attrs)
            self.assertIn("NORMAL", attrs)
            self.assertIn("TEXCOORD_0", attrs)
            self.assertIn("indices", prim)
            self.assertIn("material", prim)
            pos = gltf["accessors"][attrs["POSITION"]]
            self.assertIn("min", pos)
            self.assertIn("max", pos)
            self.assertEqual(pos["type"], "VEC3")
            self.assertEqual(pos["componentType"], 5126)
            idx = gltf["accessors"][prim["indices"]]
            self.assertEqual(idx["componentType"], 5123)
            self.assertEqual(idx["type"], "SCALAR")

    def test_round_trip_tavern_counts(self):
        """The acceptance test: export the tavern, re-import it
        through import_mesh, same object count AND same triangle
        count, no warnings."""
        scene = scene_mod.load_scene(str(EXAMPLE))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tavern.glb")
            report = meshio_glb.export_glb(scene["objects"], path,
                                           scale=0.01)
            self.assertEqual(report["objects"], 19)
            self.assertEqual(report["triangles"], 1674)
            self.assertEqual(report["warnings"], [])
            objects, re_report = meshio.import_mesh(path, 100.0)
            self.assertEqual(len(objects), 19)
            self.assertEqual(re_report["objects"], 19)
            self.assertEqual(re_report["triangles"], 1674)
            self.assertEqual(re_report["warnings"], [])

    def test_round_trip_tavern_transforms(self):
        """Every re-imported tavern object sits where the original
        sat: positions back within float32 tolerance, scales exact,
        rotations compared via matrices. The tavern's non-uniform
        scales (Loaf, Flame) are the ones that would expose a
        conjugation mistake."""
        scene = scene_mod.load_scene(str(EXAMPLE))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tavern.glb")
            meshio_glb.export_glb(scene["objects"], path, scale=0.01)
            objects, _report = meshio.import_mesh(path, 100.0)
        by_name = {o["name"]: o for o in objects}
        self.assertEqual(len(by_name), 19)
        for original in scene["objects"]:
            reimported = by_name[original["name"]]
            for a, b in zip(original["position"],
                            reimported["position"]):
                self.assertAlmostEqual(a, b, delta=1e-3)
            for a, b in zip(original["scale"], reimported["scale"]):
                self.assertAlmostEqual(a, b, delta=1e-6)
            self._assert_same_rotation(original["rotation"],
                                       reimported["rotation"])

    def test_round_trip_tavern_materials(self):
        """Colors, metalness, and roughness all survive a .glb
        round-trip — the thing .obj could not do (its .mtl carries
        Kd only). Every tavern object's material comes back exact,
        including the polished plate (rough 0.3, metal 0.8)."""
        scene = scene_mod.load_scene(str(EXAMPLE))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tavern.glb")
            meshio_glb.export_glb(scene["objects"], path, scale=0.01)
            objects, _report = meshio.import_mesh(path, 100.0)
        by_name = {o["name"]: o for o in objects}
        for original in scene["objects"]:
            material = by_name[original["name"]]["material"]
            self.assertEqual(material["base_color"],
                             original["material"]["base_color"])
            self.assertAlmostEqual(
                material["metalness"],
                original["material"]["metalness"], delta=1e-9)
            self.assertAlmostEqual(
                material["roughness"],
                original["material"]["roughness"], delta=1e-9)

    def test_transformed_object_round_trips_transform(self):
        """A box moved, rotated on all three axes, and scaled
        non-uniformly round-trips its transform exactly: position,
        scale, and the rotation matrix all come back. The rotation
        is the hard part — Euler to quaternion to matrix to Euler
        — so it is compared as a matrix, not as angles."""
        obj = scene_mod.new_object("box", name="Crate")
        obj["position"] = [10.0, 20.0, 30.0]
        obj["rotation"] = [30.0, 45.0, 60.0]
        obj["scale"] = [2.0, 1.0, 0.5]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "crate.glb")
            meshio_glb.export_glb([obj], path, scale=1.0)
            objects, _report = meshio.import_mesh(path, 1.0)
        reimported = objects[0]
        for a, b in zip(obj["position"], reimported["position"]):
            self.assertAlmostEqual(a, b, delta=1e-6)
        for a, b in zip(obj["scale"], reimported["scale"]):
            self.assertAlmostEqual(a, b, delta=1e-6)
        self._assert_same_rotation(obj["rotation"],
                                   reimported["rotation"])

    def test_node_trs_in_file(self):
        """The node TRS is the conjugated transform, checked
        against hand-computed values: position (10, 20, 30) at
        scale 0.01 becomes translation (0.1, 0.3, -0.2); scale
        (2, 1, 0.5) becomes (2, 0.5, 1); a 90-degree rotation
        about Reblend's Z (up) becomes a 90-degree rotation about
        glTF's Y (up) — quaternion (0, sqrt(2)/2, 0, sqrt(2)/2).
        An untransformed object omits all three TRS fields."""
        spin = scene_mod.new_object("box", name="Spin")
        spin["position"] = [10.0, 20.0, 30.0]
        spin["rotation"] = [0.0, 0.0, 90.0]
        spin["scale"] = [2.0, 1.0, 0.5]
        plain = scene_mod.new_object("box", name="Plain")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "spin.glb")
            meshio_glb.export_glb([spin, plain], path, scale=0.01)
            with open(path, "rb") as fh:
                data = fh.read()
        gltf, _bin = meshio_glb._read_chunks(data)
        node = gltf["nodes"][0]
        self.assertEqual(node["name"], "Spin")
        for got, want in zip(node["translation"], [0.1, 0.3, -0.2]):
            self.assertAlmostEqual(got, want, delta=1e-9)
        self.assertEqual(node["scale"], [2.0, 0.5, 1.0])
        root2 = math.sqrt(2.0) / 2.0
        for got, want in zip(node["rotation"], [0.0, root2, 0.0, root2]):
            self.assertAlmostEqual(got, want, delta=1e-9)
        plain_node = gltf["nodes"][1]
        self.assertNotIn("translation", plain_node)
        self.assertNotIn("rotation", plain_node)
        self.assertNotIn("scale", plain_node)

    def test_quaternion_round_trip(self):
        """_matrix_to_quat and the importer's _quat_to_matrix are
        inverses: a rotation matrix survives the trip through a
        quaternion and back, across a spread of rotations including
        180-degree turns and a gimbal-adjacent one."""
        for rotation in ([0.0, 0.0, 0.0], [90.0, 0.0, 0.0],
                         [0.0, 90.0, 0.0], [0.0, 0.0, 90.0],
                         [30.0, 45.0, 60.0], [180.0, 0.0, 0.0],
                         [0.0, 0.0, 180.0], [45.0, 90.0, 30.0],
                         [-30.0, -60.0, 120.0]):
            matrix = geometry.compose_matrix(
                [0.0, 0.0, 0.0], rotation, [1.0, 1.0, 1.0])
            quat = meshio_glb._matrix_to_quat(
                [row[:3] for row in matrix[:3]])
            back = meshio_glb._quat_to_matrix(quat)
            for i in range(3):
                for j in range(3):
                    self.assertAlmostEqual(matrix[i][j], back[i][j],
                                           delta=1e-9)

    def test_mesh_object_exports_asset_geometry(self):
        """A mesh object exports its asset's geometry: the gem (8
        triangles) exports and re-imports with the same bounds as
        the asset file. The object has an identity transform, so
        the asset's local bounds are the re-imported bounds."""
        asset = meshfile.load_mesh(str(GEM_ASSET))
        obj = scene_mod.new_object("mesh", name="Gem")
        obj["mesh_file"] = str(GEM_ASSET)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gem.glb")
            report = meshio_glb.export_glb([obj], path, scale=1.0)
            self.assertEqual(report["objects"], 1)
            self.assertEqual(report["triangles"], 8)
            objects, re_report = meshio.import_mesh(path, 1.0)
            self.assertEqual(re_report["triangles"], 8)
        pos = asset["positions"]
        original = (min(pos[0::3]), max(pos[0::3]),
                    min(pos[1::3]), max(pos[1::3]),
                    min(pos[2::3]), max(pos[2::3]))
        re_pos = objects[0]["mesh"]["positions"]
        reimported = (min(re_pos[0::3]), max(re_pos[0::3]),
                      min(re_pos[1::3]), max(re_pos[1::3]),
                      min(re_pos[2::3]), max(re_pos[2::3]))
        for a, b in zip(original, reimported):
            self.assertAlmostEqual(a, b, delta=1e-4)

    def test_scale_applied(self):
        """Export scale multiplies the vertices in the file: a 100
        cm box at 0.01 spans 1 unit, at 1.0 it spans 100. Checked
        on the POSITION accessor's min/max — the values Blender
        would read."""
        obj = scene_mod.new_object("box", name="Box")
        with tempfile.TemporaryDirectory() as tmp:
            for scale, span in ((0.01, 1.0), (1.0, 100.0)):
                path = os.path.join(tmp, f"box-{scale}.glb")
                meshio_glb.export_glb([obj], path, scale=scale)
                with open(path, "rb") as fh:
                    data = fh.read()
                gltf, _bin = meshio_glb._read_chunks(data)
                pos = gltf["accessors"][0]
                self.assertAlmostEqual(pos["max"][0] - pos["min"][0],
                                       span, delta=1e-6)

    def test_no_winding_flip(self):
        """The axis swap is a rotation, so no winding flip is
        needed: every re-imported box triangle's geometric normal
        points away from the object's center. If the winding were
        wrong, the normals would point inward and the box would
        render inside-out."""
        obj = scene_mod.new_object("box", name="Box")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "box.glb")
            meshio_glb.export_glb([obj], path, scale=1.0)
            objects, _report = meshio.import_mesh(path, 1.0)
        mesh = objects[0]["mesh"]
        pos, indices = mesh["positions"], mesh["indices"]
        self.assertTrue(indices)
        for t in range(len(indices) // 3):
            tri = indices[t * 3:t * 3 + 3]
            pts = [(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2])
                   for i in tri]
            nx, ny, nz = geometry.face_normal(*pts)
            cx = sum(p[0] for p in pts) / 3.0
            cy = sum(p[1] for p in pts) / 3.0
            cz = sum(p[2] for p in pts) / 3.0
            self.assertGreater(nx * cx + ny * cy + nz * cz, 0.0)

    def test_missing_mesh_asset_warns_and_skips(self):
        """A mesh object whose asset file is missing warns naming
        the file and is skipped; the rest of the scene still
        exports."""
        box = scene_mod.new_object("box", name="Crate")
        ghost = scene_mod.new_object("mesh", name="Ghost")
        ghost["mesh_file"] = "assets/nope.mesh.json"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.glb")
            report = meshio_glb.export_glb([box, ghost], path, scale=1.0,
                                           base_path=tmp)
            self.assertEqual(report["objects"], 1)
            self.assertEqual(report["triangles"], 12)
            self.assertEqual(len(report["warnings"]), 1)
            self.assertIn("nope.mesh.json", report["warnings"][0])
            with open(path, "rb") as fh:
                data = fh.read()
            gltf, _bin = meshio_glb._read_chunks(data)
            self.assertEqual([n["name"] for n in gltf["nodes"]],
                             ["Crate"])

    def test_non_geometry_object_skipped(self):
        """An object whose kind is not 'object' (emitters are T16)
        is skipped with a plain note, never fatal."""
        emitter = scene_mod.new_object("box", name="Puffs")
        emitter["kind"] = "emitter"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.glb")
            report = meshio_glb.export_glb([emitter], path, scale=1.0)
            self.assertEqual(report["objects"], 0)
            self.assertEqual(report["triangles"], 0)
            self.assertEqual(len(report["warnings"]), 1)
            self.assertIn("not geometry", report["warnings"][0])

    def test_unwritable_path_raises(self):
        """An unwritable path raises OSError — the window turns it
        into a plain message, never a crash. The writer does not
        create folders."""
        obj = scene_mod.new_object("box", name="Box")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "no-such-dir", "out.glb")
            with self.assertRaises(OSError):
                meshio_glb.export_glb([obj], path, scale=1.0)

    def test_empty_export_writes_valid_glb(self):
        """Exporting no objects still writes a valid container: the
        header checks pass, the JSON parses, the scene has no
        nodes, the buffer is empty. The window guards the empty
        scene before the dialog; the writer handles it anyway.
        Re-importing raises plainly — there is nothing to import,
        and the importer says so instead of returning an empty
        scene."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.glb")
            report = meshio_glb.export_glb([], path, scale=1.0)
            self.assertEqual(report, {"objects": 0, "triangles": 0,
                                      "warnings": []})
            magic, version, length, size, _chunks, json_bytes, _bin = \
                self._read_container(path)
            self.assertEqual(magic, 0x46546C67)
            self.assertEqual(version, 2)
            self.assertEqual(length, size)
            gltf = json.loads(json_bytes.decode("utf-8"))
            self.assertEqual(gltf["nodes"], [])
            self.assertEqual(gltf["meshes"], [])
            self.assertEqual(gltf["buffers"][0]["byteLength"], 0)
            with self.assertRaises(meshio_glb.GlbError):
                meshio.import_mesh(path, 1.0)

    def test_importer_reads_unsigned_short_indices(self):
        """T11 found this in the T9 importer: component type 5123
        (UNSIGNED_SHORT) was mapped to 4 bytes, so every real-world
        16-bit index list — Blender writes them for small meshes —
        misread as garbage (the first T11 round-trip died with
        'index 65536 outside its 6 vertices'). A .glb built with
        5123 indices now imports exactly."""
        builder = glb_fixture.GlbBuilder()
        pos, nor, uv, idx = glb_fixture.box_verts(0.5)
        mesh = builder.add_mesh("Box", [{
            "attributes": {"POSITION": builder.add_vec3(pos),
                           "NORMAL": builder.add_vec3(nor),
                           "TEXCOORD_0": builder.add_vec2(uv)},
            "indices": builder.add_accessor(list(idx), 1, 5123),
        }])
        node = builder.add_node("Box", mesh=mesh)
        builder.set_scene([node])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "u16.glb")
            builder.write(path)
            objects, report = meshio.import_mesh(path, 1.0)
        self.assertEqual(report["objects"], 1)
        self.assertEqual(report["triangles"], 12)
        positions = objects[0]["mesh"]["positions"]
        self.assertAlmostEqual(min(positions[0::3]), -0.5, delta=1e-6)
        self.assertAlmostEqual(max(positions[0::3]), 0.5, delta=1e-6)


@unittest.skipUnless(QT_TESTS, "Qt tests need LIGHT_ENGINE_QT_TESTS=1")
class QtGlbExportTests(unittest.TestCase):
    """The .glb export flow in the live window (T11): the dialog-free
    export_glb_objects core, the export dialog's title and scale,
    and the round-tripped scene rendering. The file picker and the
    scale dialog need a human, so they are verified by the
    screenshot script instead."""

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets
        cls._app = QtWidgets.QApplication.instance()
        if cls._app is None:
            cls._app = QtWidgets.QApplication([])
        from light_engine.window import ExportDialog, MainWindow
        cls.MainWindow = MainWindow
        cls.ExportDialog = ExportDialog

    def _pump(self, rounds=10):
        from PySide6 import QtCore, QtWidgets
        for _ in range(rounds):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)

    def _show_tavern(self, tmp):
        """A window on the tavern scene saved in tmp."""
        scene = scene_mod.load_scene(str(EXAMPLE))
        path = os.path.join(tmp, "scene.json")
        scene_mod.save_scene(scene, path)
        loaded = scene_mod.load_scene(path)
        window = self.MainWindow(scene=loaded, scene_path=path)
        window.resize(800, 600)
        window.show()
        self._pump()
        return window

    def _show_saved_scene(self, tmp, name="Export test"):
        """A window on an empty scene saved in tmp."""
        scene = scene_mod.new_scene(name)
        path = os.path.join(tmp, "scene.json")
        scene_mod.save_scene(scene, path)
        loaded = scene_mod.load_scene(path)
        window = self.MainWindow(scene=loaded, scene_path=path)
        window.resize(800, 600)
        window.show()
        self._pump()
        return window

    def test_export_scene_writes_glb(self):
        """The tavern exports through the window core: a .glb file
        lands on disk, and re-importing it gives back 19 objects
        and 1,674 triangles."""
        with tempfile.TemporaryDirectory() as tmp:
            window = self._show_tavern(tmp)
            path = os.path.join(tmp, "tavern.glb")
            report = window.export_glb_objects(window.scene["objects"],
                                               path, 0.01)
            self.assertEqual(report["objects"], 19)
            self.assertEqual(report["triangles"], 1674)
            self.assertEqual(report["warnings"], [])
            self.assertTrue(os.path.isfile(path))
            objects, re_report = meshio.import_mesh(path, 100.0)
            self.assertEqual(len(objects), 19)
            self.assertEqual(re_report["triangles"], 1674)
            window.close()

    def test_export_mesh_writes_only_selected(self):
        """Exporting the selected mug writes one node with the
        mug's triangles — nothing else from the scene."""
        with tempfile.TemporaryDirectory() as tmp:
            window = self._show_tavern(tmp)
            window._select("mug")
            mug = scene_mod.find_object(window.scene, "mug")
            expected = geometry.tessellate(
                mug["shape"], mug["params"])["tri_count"]
            path = os.path.join(tmp, "mug.glb")
            report = window.export_glb_objects([mug], path, 0.01)
            self.assertEqual(report["objects"], 1)
            self.assertEqual(report["triangles"], expected)
            with open(path, "rb") as fh:
                data = fh.read()
            gltf, _bin = meshio_glb._read_chunks(data)
            self.assertEqual([n["name"] for n in gltf["nodes"]],
                             ["Mug"])
            window.close()

    def test_export_dialog_title_and_scale(self):
        """The export dialog shows the report, titles itself for
        the format, and returns the scale the user picks. The
        default title is still "Export .obj" — the T10 callers
        pass no title. The dialog is accepted programmatically —
        it is never exec()'d here, so no modal loop runs."""
        report = {"objects": 2, "triangles": 16,
                  "warnings": ["a warning"]}
        dialog = self.ExportDialog(None, report)
        self.assertEqual(dialog.windowTitle(), "Export .obj")
        self.assertEqual(dialog.scale(), 0.01)
        dialog.close()
        glb_dialog = self.ExportDialog(None, report, title="Export .glb")
        self.assertEqual(glb_dialog.windowTitle(), "Export .glb")
        self.assertEqual(glb_dialog.scale(), 0.01)
        glb_dialog._scale_spin.setValue(2.5)
        glb_dialog.accept()
        self.assertEqual(glb_dialog.scale(), 2.5)
        glb_dialog.close()

    def test_export_round_tripped_scene_renders(self):
        """The exported tavern re-imports into a fresh window and
        renders: 19 mesh objects, the 1,674-triangle readout, a
        real framebuffer, GL error 0."""
        with tempfile.TemporaryDirectory() as tmp:
            window = self._show_tavern(tmp)
            path = os.path.join(tmp, "tavern.glb")
            window.export_glb_objects(window.scene["objects"], path, 0.01)
            window.close()
            scene = scene_mod.new_scene("Round trip")
            scene_path = os.path.join(tmp, "roundtrip.json")
            scene_mod.save_scene(scene, scene_path)
            loaded = scene_mod.load_scene(scene_path)
            window2 = self.MainWindow(scene=loaded, scene_path=scene_path)
            window2.resize(800, 600)
            window2.show()
            self._pump()
            parsed = meshio.parse(path)
            created = window2.import_mesh_objects(parsed, 100.0)
            self.assertEqual(len(created), 19)
            self.assertIn("1,674 triangles", window2._tri_label.text())
            self._pump()
            vp = window2._viewport
            self.assertIsNone(vp._shader_error)
            self.assertFalse(vp.grabFramebuffer().isNull())
            vp.makeCurrent()
            try:
                self.assertEqual(vp._gl.glGetError(), 0)
            finally:
                vp.doneCurrent()
            window2.close()

    def test_export_with_missing_mesh_warns(self):
        """A mesh object whose asset is missing warns naming the
        file and is skipped; the floor still exports. The object is
        added directly to the scene (the window drops missing meshes
        on load, so the normal path cannot hold one)."""
        with tempfile.TemporaryDirectory() as tmp:
            window = self._show_saved_scene(tmp)
            floor = scene_mod.new_object("plane", name="Floor")
            window.scene["objects"].append(floor)
            ghost = scene_mod.new_object("mesh", name="Ghost")
            ghost["mesh_file"] = "assets/nope.mesh.json"
            window.scene["objects"].append(ghost)
            path = os.path.join(tmp, "scene.glb")
            report = window.export_glb_objects(window.scene["objects"],
                                               path, 0.01)
            self.assertEqual(report["objects"], 1)
            self.assertEqual(report["triangles"], 2)
            self.assertEqual(len(report["warnings"]), 1)
            self.assertIn("nope.mesh.json", report["warnings"][0])
            window.close()

    def test_export_mesh_needs_a_selection(self):
        """Export mesh as .glb with nothing selected says so in the
        status bar and opens no dialog — the slot returns before
        the file picker, so this test cannot hang on one."""
        with tempfile.TemporaryDirectory() as tmp:
            window = self._show_tavern(tmp)
            self.assertIsNone(window.selected_id)
            window._export_mesh_glb()
            self.assertEqual(window.statusBar().currentMessage(),
                             "Select an object first")
            window.close()

    def test_export_scene_empty_shows_message(self):
        """Export scene as .glb on an empty scene says so in the
        status bar and opens no dialog."""
        window = self.MainWindow(scene=scene_mod.new_scene())
        window.show()
        self._pump()
        window._export_scene_glb()
        self.assertEqual(window.statusBar().currentMessage(),
                         "Nothing to export — the scene is empty")
        window.close()


class GizmoGeometryTests(unittest.TestCase):
    """T12: the gizmo geometry is pure functions (no GL), so it is
    tested headless. Vertex counts are exact — move 78, rotate 288,
    scale 78 — from the segment counts in viewport.py."""

    def _verts(self, mode):
        from light_engine.viewport import gizmo_lines
        flat = gizmo_lines(mode)
        self.assertEqual(len(flat) % 6, 0)
        return [(flat[i], flat[i + 1], flat[i + 2],
                 flat[i + 3], flat[i + 4], flat[i + 5])
                for i in range(0, len(flat), 6)]

    def test_move_has_three_colored_arrows(self):
        verts = self._verts("move")
        self.assertEqual(len(verts), 78)
        # The X shaft runs from the origin to (0.75, 0, 0), in red.
        self.assertEqual(verts[0][:3], (0.0, 0.0, 0.0))
        self.assertEqual(verts[1][:3], (0.75, 0.0, 0.0))
        self.assertEqual(verts[0][3:], (0.85, 0.25, 0.25))
        # The Y shaft starts at vertex 26, the Z shaft at 52.
        self.assertEqual(verts[26][:3], (0.0, 0.0, 0.0))
        self.assertEqual(verts[27][:3], (0.0, 0.75, 0.0))
        self.assertEqual(verts[26][3:], (0.25, 0.75, 0.30))
        self.assertEqual(verts[52][:3], (0.0, 0.0, 0.0))
        self.assertEqual(verts[53][:3], (0.0, 0.0, 0.75))
        self.assertEqual(verts[52][3:], (0.30, 0.45, 0.9))
        # Every vertex is red, green, or blue — one color per axis.
        colors = {(v[3], v[4], v[5]) for v in verts}
        self.assertEqual(colors, {(0.85, 0.25, 0.25),
                                  (0.25, 0.75, 0.30),
                                  (0.30, 0.45, 0.9)})

    def test_move_cone_apex_and_base(self):
        verts = self._verts("move")
        # The X cone apex is (1, 0, 0): the farthest red X vertex.
        xs = [v[0] for v in verts if v[3:] == (0.85, 0.25, 0.25)]
        self.assertAlmostEqual(max(xs), 1.0)
        # The cone base circle sits at x = 0.75 with radius 0.09
        # (the shaft tip at (0.75, 0, 0) is not a base vertex).
        base = [v for v in verts
                if v[3:] == (0.85, 0.25, 0.25)
                and abs(v[0] - 0.75) < 1e-9
                and (v[1] != 0.0 or v[2] != 0.0)]
        self.assertTrue(base)
        for v in base:
            self.assertAlmostEqual(math.hypot(v[1], v[2]), 0.09)

    def test_rotate_rings_are_unit_circles(self):
        verts = self._verts("rotate")
        self.assertEqual(len(verts), 288)
        # The X ring lies in the YZ plane (x == 0), radius 1.
        ring_x = [v for v in verts if v[3:] == (0.85, 0.25, 0.25)]
        self.assertEqual(len(ring_x), 96)
        for v in ring_x:
            self.assertEqual(v[0], 0.0)
            self.assertAlmostEqual(math.hypot(v[1], v[2]), 1.0)
        # The Y ring lies in the XZ plane, the Z ring in the XY plane.
        ring_y = [v for v in verts if v[3:] == (0.25, 0.75, 0.30)]
        self.assertEqual(len(ring_y), 96)
        for v in ring_y:
            self.assertEqual(v[1], 0.0)
        ring_z = [v for v in verts if v[3:] == (0.30, 0.45, 0.9)]
        self.assertEqual(len(ring_z), 96)
        for v in ring_z:
            self.assertEqual(v[2], 0.0)

    def test_scale_has_lines_and_cubes(self):
        verts = self._verts("scale")
        self.assertEqual(len(verts), 78)
        # The X line runs from the origin to (0.7, 0, 0).
        self.assertEqual(verts[0][:3], (0.0, 0.0, 0.0))
        self.assertEqual(verts[1][:3], (0.7, 0.0, 0.0))
        # The X cube is the next 24 vertices: centred at (0.81, 0, 0)
        # with half-side 0.11, so x runs 0.70 to 0.92, y/z to +/-0.11.
        cube_x = verts[2:26]
        self.assertEqual(len(cube_x), 24)
        self.assertTrue(all(v[3:] == (0.85, 0.25, 0.25) for v in cube_x))
        xs = [v[0] for v in cube_x]
        self.assertAlmostEqual(min(xs), 0.70)
        self.assertAlmostEqual(max(xs), 0.92)
        self.assertAlmostEqual(max(abs(v[1]) for v in cube_x), 0.11)
        self.assertAlmostEqual(max(abs(v[2]) for v in cube_x), 0.11)

    def test_unknown_mode_raises(self):
        from light_engine.viewport import gizmo_lines
        with self.assertRaises(ValueError):
            gizmo_lines("spin")

    def test_gizmo_size_scales_with_distance(self):
        from light_engine.viewport import gizmo_size
        self.assertAlmostEqual(gizmo_size(400.0), 2.0 * gizmo_size(200.0))
        self.assertGreater(gizmo_size(400.0), 0.0)


@unittest.skipUnless(QT_TESTS, "Qt tests need LIGHT_ENGINE_QT_TESTS=1")
class QtGizmoTests(unittest.TestCase):
    """T12: gizmo modes, toolbar buttons, W/E/R shortcuts, and the
    render-exclusion rule."""

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets
        cls._app = QtWidgets.QApplication.instance()
        if cls._app is None:
            cls._app = QtWidgets.QApplication([])
        from light_engine.window import MainWindow
        cls.MainWindow = MainWindow

    def _open_tavern(self):
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        self._pump()
        return window

    def _pump(self, rounds=20):
        from PySide6 import QtCore, QtWidgets
        for _ in range(rounds):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)

    def _grab_png(self, window):
        """The viewport framebuffer as PNG bytes. PNG encoding is
        deterministic, so identical pixels give identical bytes —
        the same byte-identity check the tavern shots use."""
        from PySide6 import QtCore
        image = window._viewport.grabFramebuffer()
        buf = QtCore.QBuffer()
        buf.open(QtCore.QBuffer.ReadWrite)
        image.save(buf, "PNG")
        return bytes(buf.data())

    def test_default_mode_is_move(self):
        window = self._open_tavern()
        self.assertEqual(window._viewport.gizmo_mode, "move")
        self.assertTrue(window._gizmo_buttons["move"].isChecked())
        self.assertFalse(window._gizmo_buttons["rotate"].isChecked())
        self.assertFalse(window._gizmo_buttons["scale"].isChecked())
        window.close()

    def test_toolbar_buttons_switch_modes(self):
        window = self._open_tavern()
        window._gizmo_buttons["rotate"].click()
        self.assertEqual(window._viewport.gizmo_mode, "rotate")
        self.assertTrue(window._gizmo_buttons["rotate"].isChecked())
        self.assertFalse(window._gizmo_buttons["move"].isChecked())
        window._gizmo_buttons["scale"].click()
        self.assertEqual(window._viewport.gizmo_mode, "scale")
        self.assertTrue(window._gizmo_buttons["scale"].isChecked())
        window.close()

    def test_shortcuts_switch_modes(self):
        from PySide6 import QtCore
        from PySide6.QtTest import QTest
        window = self._open_tavern()
        vp = window._viewport
        # Xvfb has no window manager, so show() never activates the
        # window and setFocus() is a no-op until it is active — on a
        # real desktop, clicking the viewport focuses it.
        window.activateWindow()
        vp.setFocus()
        self._pump(5)
        QTest.keyClick(vp, QtCore.Qt.Key_E)
        self.assertEqual(vp.gizmo_mode, "rotate")
        QTest.keyClick(vp, QtCore.Qt.Key_R)
        self.assertEqual(vp.gizmo_mode, "scale")
        QTest.keyClick(vp, QtCore.Qt.Key_W)
        self.assertEqual(vp.gizmo_mode, "move")
        window.close()

    def test_bad_mode_raises_and_keeps_mode(self):
        window = self._open_tavern()
        with self.assertRaises(ValueError):
            window._viewport.set_gizmo_mode("spin")
        self.assertEqual(window._viewport.gizmo_mode, "move")
        window.close()

    def test_gizmo_draws_on_selected_object(self):
        window = self._open_tavern()
        self._pump(5)
        before = self._grab_png(window)
        window._select("mug")
        self._pump(10)
        after = self._grab_png(window)
        self.assertNotEqual(before, after)
        window.close()

    def test_mode_changes_the_gizmo_draw(self):
        """The selection box is mode-independent, so a mode change
        can only alter the grab through the gizmo — this proves the
        gizmo itself draws, in each mode."""
        window = self._open_tavern()
        window._select("mug")
        self._pump(10)
        move_png = self._grab_png(window)
        window._set_gizmo_mode("rotate")
        self._pump(10)
        rotate_png = self._grab_png(window)
        self.assertNotEqual(move_png, rotate_png)
        window._set_gizmo_mode("scale")
        self._pump(10)
        scale_png = self._grab_png(window)
        self.assertNotEqual(move_png, scale_png)
        self.assertNotEqual(rotate_png, scale_png)
        window.close()

    def test_no_gizmo_without_selection(self):
        window = self._open_tavern()
        self._pump(5)
        move_png = self._grab_png(window)
        window._set_gizmo_mode("rotate")
        self._pump(10)
        rotate_png = self._grab_png(window)
        # No selection, no gizmo: the mode must not change the view.
        self.assertEqual(move_png, rotate_png)
        window.close()

    def test_render_frame_to_excludes_gizmo(self):
        window = self._open_tavern()
        window._select("mug")
        self._pump(10)
        # A live grab with the gizmo suppressed is what a render
        # must look like; render_frame_to must match it exactly.
        window._viewport.set_gizmo_suppressed(True)
        self._pump(5)
        suppressed = self._grab_png(window)
        window._viewport.set_gizmo_suppressed(False)
        self._pump(5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "frame0.png")
            self.assertTrue(window.render_frame_to(0, path))
            with open(path, "rb") as handle:
                rendered = handle.read()
        self.assertEqual(rendered, suppressed)
        # The flag is cleared after the render: the live view still
        # shows the gizmo.
        self.assertFalse(window._viewport._gizmo_suppressed)
        live = self._grab_png(window)
        self.assertNotEqual(live, suppressed)
        window.close()


class GizmoPickTests(unittest.TestCase):
    """T13: the gizmo ray tests are pure functions (no GL), so they
    are tested headless — plus the proof that the pick geometry
    matches the drawn geometry, and the hover-highlight colors."""

    def _verts(self, mode, highlight=None):
        from light_engine.viewport import gizmo_lines
        flat = gizmo_lines(mode, highlight)
        self.assertEqual(len(flat) % 6, 0)
        return [(flat[i], flat[i + 1], flat[i + 2],
                 flat[i + 3], flat[i + 4], flat[i + 5])
                for i in range(0, len(flat), 6)]

    def test_ray_vs_plane_hit(self):
        t = geometry.ray_vs_plane((0.0, 0.0, -10.0), (0.0, 0.0, 1.0),
                                  (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        self.assertAlmostEqual(t, 10.0)

    def test_ray_vs_plane_parallel(self):
        t = geometry.ray_vs_plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                  (0.0, 0.0, 5.0), (0.0, 0.0, 1.0))
        self.assertIsNone(t)

    def test_ray_vs_plane_behind(self):
        t = geometry.ray_vs_plane((0.0, 0.0, 10.0), (0.0, 0.0, 1.0),
                                  (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        self.assertIsNone(t)

    def test_ray_vs_cylinder_side_hit(self):
        # Z-axis cylinder z=0..10, radius 1. A ray along +x at
        # (y=0, z=5) hits the side at x=-1, t=9 from x=-10.
        t = geometry.ray_vs_cylinder_segment(
            (-10.0, 0.0, 5.0), (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0), (0.0, 0.0, 10.0), 1.0)
        self.assertAlmostEqual(t, 9.0)

    def test_ray_vs_cylinder_miss(self):
        # Same cylinder, ray offset 2 in y: outside the radius.
        t = geometry.ray_vs_cylinder_segment(
            (-10.0, 2.0, 5.0), (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0), (0.0, 0.0, 10.0), 1.0)
        self.assertIsNone(t)

    def test_ray_vs_cylinder_past_the_end(self):
        # Ray at z=15, above the top cap: the infinite cylinder
        # would hit, the segment must not.
        t = geometry.ray_vs_cylinder_segment(
            (-10.0, 0.0, 15.0), (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0), (0.0, 0.0, 10.0), 1.0)
        self.assertIsNone(t)

    def test_ray_vs_cylinder_cap_hit(self):
        # Ray down the axis from above hits the top cap at z=10.
        t = geometry.ray_vs_cylinder_segment(
            (0.0, 0.0, 20.0), (0.0, 0.0, -1.0),
            (0.0, 0.0, 0.0), (0.0, 0.0, 10.0), 1.0)
        self.assertAlmostEqual(t, 10.0)

    def test_ray_vs_cylinder_inside(self):
        # A ray starting inside the cylinder exits the far side.
        t = geometry.ray_vs_cylinder_segment(
            (0.0, 0.0, 5.0), (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0), (0.0, 0.0, 10.0), 1.0)
        self.assertAlmostEqual(t, 1.0)

    def test_ray_vs_ring_hit(self):
        # Ring radius 5 in the XY plane (axis Z), thickness 0.5.
        # A ray down at (5, 0) lands on the circle.
        t = geometry.ray_vs_ring((5.0, 0.0, -10.0), (0.0, 0.0, 1.0),
                                 (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                                 5.0, 0.5)
        self.assertAlmostEqual(t, 10.0)

    def test_ray_vs_ring_inside_hole_miss(self):
        t = geometry.ray_vs_ring((0.0, 0.0, -10.0), (0.0, 0.0, 1.0),
                                 (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                                 5.0, 0.5)
        self.assertIsNone(t)

    def test_ray_vs_ring_outside_miss(self):
        t = geometry.ray_vs_ring((7.0, 0.0, -10.0), (0.0, 0.0, 1.0),
                                 (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                                 5.0, 0.5)
        self.assertIsNone(t)

    def test_ray_vs_ring_edge_of_band(self):
        # 5.4 is inside the 0.5 band, 5.6 is outside.
        self.assertIsNotNone(geometry.ray_vs_ring(
            (5.4, 0.0, -10.0), (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 5.0, 0.5))
        self.assertIsNone(geometry.ray_vs_ring(
            (5.6, 0.0, -10.0), (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 5.0, 0.5))

    def test_ray_vs_ring_parallel(self):
        # A ray running parallel to the ring plane never meets it.
        t = geometry.ray_vs_ring((5.0, 0.0, 1.0), (1.0, 0.0, 0.0),
                                 (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                                 5.0, 0.5)
        self.assertIsNone(t)

    def test_scale_cube_aabb(self):
        # The scale pick reuses the existing ray_hits_aabb on the
        # cube's box: a ray through the cube hits, a ray past it
        # misses.
        t = geometry.ray_hits_aabb((0.0, 0.0, -10.0), (0.0, 0.0, 1.0),
                                   (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(t, 9.0)
        self.assertIsNone(geometry.ray_hits_aabb(
            (5.0, 0.0, -10.0), (0.0, 0.0, 1.0),
            (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))

    def test_pick_constants_match_drawn_move(self):
        # The pick cylinder runs centre -> GIZMO_MOVE_TIP; the drawn
        # arrow's farthest vertex must be exactly there, and the cone
        # base radius must fit inside the pick tolerance (else the
        # cone would stick out of its own pick cylinder).
        from light_engine.viewport import (
            GIZMO_CONE_BASE_RADIUS, GIZMO_MOVE_TIP, GIZMO_PICK_TOLERANCE)
        verts = self._verts("move")
        xs = [v[0] for v in verts if v[3:] == (0.85, 0.25, 0.25)]
        self.assertAlmostEqual(max(xs), GIZMO_MOVE_TIP)
        self.assertLess(GIZMO_CONE_BASE_RADIUS, GIZMO_PICK_TOLERANCE)

    def test_pick_constants_match_drawn_rotate(self):
        from light_engine.viewport import GIZMO_RING_RADIUS
        verts = self._verts("rotate")
        ring_x = [v for v in verts if v[3:] == (0.85, 0.25, 0.25)]
        for v in ring_x:
            self.assertAlmostEqual(math.hypot(v[1], v[2]),
                                   GIZMO_RING_RADIUS)

    def test_pick_constants_match_drawn_scale(self):
        from light_engine.viewport import (
            GIZMO_SCALE_CUBE_CENTER, GIZMO_SCALE_CUBE_HALF)
        verts = self._verts("scale")
        cube_x = verts[2:26]
        xs = [v[0] for v in cube_x]
        self.assertAlmostEqual(
            min(xs), GIZMO_SCALE_CUBE_CENTER - GIZMO_SCALE_CUBE_HALF)
        self.assertAlmostEqual(
            max(xs), GIZMO_SCALE_CUBE_CENTER + GIZMO_SCALE_CUBE_HALF)

    def test_highlight_brightens_one_axis(self):
        from light_engine.viewport import GIZMO_AXIS_COLORS, _gizmo_bright
        verts = self._verts("move", highlight="x")
        bright = _gizmo_bright(GIZMO_AXIS_COLORS["x"])
        self.assertTrue(any(v[3:] == bright for v in verts))
        # Y and Z keep their plain colors; X has no plain verts left.
        colors = {(v[3], v[4], v[5]) for v in verts}
        self.assertEqual(colors, {bright, GIZMO_AXIS_COLORS["y"],
                                  GIZMO_AXIS_COLORS["z"]})

    def test_bright_is_brighter(self):
        from light_engine.viewport import GIZMO_AXIS_COLORS, _gizmo_bright
        for color in GIZMO_AXIS_COLORS.values():
            bright = _gizmo_bright(color)
            self.assertTrue(all(b >= c for b, c in zip(bright, color)))
            self.assertTrue(any(b > c for b, c in zip(bright, color)))

    def test_bad_highlight_raises(self):
        from light_engine.viewport import gizmo_lines
        with self.assertRaises(ValueError):
            gizmo_lines("move", "w")


@unittest.skipUnless(QT_TESTS, "Qt tests need LIGHT_ENGINE_QT_TESTS=1")
class QtGizmoDragTests(unittest.TestCase):
    """T13: handle picking, hover highlight, and dragging — driven
    through real mouse events on the tavern mug. Pixels come from
    projecting known world points (the projection helper inverts
    nothing: it runs the viewport's own matrices forward), so a
    press lands on the handle it is meant to land on. Drags in the
    default tavern view assert the right axis, the right direction,
    and the other components exactly unchanged; the exact-value
    tests use a camera with the axis in the drag plane, where the
    drag math recovers the offset exactly."""

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets
        cls._app = QtWidgets.QApplication.instance()
        if cls._app is None:
            cls._app = QtWidgets.QApplication([])
        from light_engine.window import MainWindow
        cls.MainWindow = MainWindow

    def _open_tavern(self):
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        self._pump()
        return window

    def _pump(self, rounds=20):
        from PySide6 import QtCore, QtWidgets
        for _ in range(rounds):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)

    def _reload(self, window):
        """Fresh tavern scene in the same window (resets the mug's
        base transform AND its keys between drag cases)."""
        window.scene = scene_mod.load_scene(str(EXAMPLE))
        window._apply_scene_to_ui()
        window._select("mug")
        self._pump(5)

    def _grab_png(self, window):
        from PySide6 import QtCore
        image = window._viewport.grabFramebuffer()
        buf = QtCore.QBuffer()
        buf.open(QtCore.QBuffer.ReadWrite)
        image.save(buf, "PNG")
        return bytes(buf.data())

    def _screen_point(self, vp, world):
        """Viewport pixel for a world point (forward projection
        through the viewport's own matrices)."""
        from PySide6 import QtCore, QtGui
        proj, view = vp._matrices()
        mapped = (proj * view).map(QtGui.QVector3D(*world))
        x = (mapped.x() + 1.0) / 2.0 * vp.width()
        y = (1.0 - mapped.y()) / 2.0 * vp.height()
        return QtCore.QPoint(int(round(x)), int(round(y)))

    def _mug(self, window):
        return scene_mod.find_object(window.scene, "mug")

    def _handle_point(self, vp, mode, axis):
        """A world point on the given handle: the move shaft midpoint,
        the rotate ring at 45 degrees (midway between the points
        where the rings cross, so the ray cannot hit a second ring),
        or the scale cube centre."""
        from light_engine.viewport import (
            GIZMO_SCALE_CUBE_CENTER, _gizmo_axis_frame, gizmo_size)
        center = vp._gizmo_center()
        size = gizmo_size(vp.distance)
        u, p, q = _gizmo_axis_frame(axis)
        if mode == "rotate":
            return tuple(center[i] + size * (p[i] + q[i]) / math.sqrt(2.0)
                         for i in range(3))
        at = 0.5 if mode == "move" else GIZMO_SCALE_CUBE_CENTER
        return tuple(center[i] + at * size * u[i] for i in range(3))

    def _ring_point(self, vp, axis, degrees):
        """A world point on the rotate ring at the given angle in the
        ring's (p, q) frame."""
        from light_engine.viewport import _gizmo_axis_frame, gizmo_size
        center = vp._gizmo_center()
        size = gizmo_size(vp.distance)
        _u, p, q = _gizmo_axis_frame(axis)
        a = math.radians(degrees)
        return tuple(center[i] + size * (math.cos(a) * p[i]
                                         + math.sin(a) * q[i])
                     for i in range(3))

    def _drag(self, vp, start_px, end_px, shift=False):
        """Press, move, release through QTest. The small delay after
        each event lets the event loop deliver it before the next
        one is posted. With shift, the move goes out as a hand-built
        QMouseEvent carrying ShiftModifier: QTest.keyPress(Key_Shift)
        does not reach QTest's mouse events (they read the real
        keyboard, which holds nothing in Xvfb), so a simulated key
        press would silently test the unsnapped path."""
        from PySide6 import QtCore, QtGui
        from PySide6.QtTest import QTest
        QTest.mousePress(vp, QtCore.Qt.LeftButton,
                         QtCore.Qt.NoModifier, start_px, 30)
        if shift:
            event = QtGui.QMouseEvent(
                QtCore.QEvent.MouseMove, QtCore.QPointF(end_px),
                QtCore.Qt.NoButton, QtCore.Qt.LeftButton,
                QtCore.Qt.ShiftModifier)
            QtCore.QCoreApplication.sendEvent(vp, event)
        else:
            QTest.mouseMove(vp, end_px, 30)
        QTest.mouseRelease(vp, QtCore.Qt.LeftButton,
                           QtCore.Qt.NoModifier, end_px, 30)
        self._pump(5)

    def _press(self, vp, px):
        """Press the left button and return the drag state (or None
        when the press did not start a drag). The exact-value tests
        build their move target from the ACTUAL grab point in the
        state: the press pixel is rounded to an integer, so the ray
        meets the drag plane up to a pixel off the ideal handle
        point — building the target from the ideal point leaves a
        ~1.5 cm error, building it from the actual grab point leaves
        only the target pixel's rounding."""
        from PySide6 import QtCore
        from PySide6.QtTest import QTest
        QTest.mousePress(vp, QtCore.Qt.LeftButton,
                         QtCore.Qt.NoModifier, px, 30)
        self._pump(5)
        return vp._gizmo_drag

    def _move_release(self, vp, px, shift=False):
        """Move to px, then release. With shift the move carries
        ShiftModifier (see _drag for why it is hand-built)."""
        from PySide6 import QtCore, QtGui
        from PySide6.QtTest import QTest
        if shift:
            event = QtGui.QMouseEvent(
                QtCore.QEvent.MouseMove, QtCore.QPointF(px),
                QtCore.Qt.NoButton, QtCore.Qt.LeftButton,
                QtCore.Qt.ShiftModifier)
            QtCore.QCoreApplication.sendEvent(vp, event)
        else:
            QTest.mouseMove(vp, px, 30)
        QTest.mouseRelease(vp, QtCore.Qt.LeftButton,
                           QtCore.Qt.NoModifier, px, 30)
        self._pump(5)

    def _face_camera(self, vp, axis):
        """Put the camera on a side perpendicular to the axis, so the
        axis lies in the drag plane and the drag math recovers the
        offset exactly. The eye sits along the frame's p vector with
        a slight q tilt — both are perpendicular to the axis, so the
        view direction is exactly perpendicular to it, and the tilt
        keeps the view off the world up vector (lookAt is degenerate
        there). (A camera ON the axis is the worst case, not the
        best: the axis then points at the camera and the drag plane
        reads almost no movement along it.)"""
        from light_engine.viewport import _gizmo_axis_frame
        center = vp._gizmo_center()
        _u, p, q = _gizmo_axis_frame(axis)
        d = vp.distance
        eye = tuple(center[i] + d * (p[i] + 0.15 * q[i])
                    for i in range(3))
        vp.set_camera_position(eye, center)
        self._pump(5)

    # -- picking -------------------------------------------------------

    def test_pick_gizmo_move_handles(self):
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        self._pump(10)
        for axis in ("x", "y", "z"):
            px = self._screen_point(
                vp, self._handle_point(vp, "move", axis))
            self.assertEqual(vp.pick_gizmo(px.x(), px.y()), (axis, "move"))
        window.close()

    def test_pick_gizmo_rotate_handles(self):
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        window._set_gizmo_mode("rotate")
        self._pump(10)
        for axis in ("x", "y", "z"):
            px = self._screen_point(
                vp, self._handle_point(vp, "rotate", axis))
            self.assertEqual(vp.pick_gizmo(px.x(), px.y()),
                             (axis, "rotate"))
        window.close()

    def test_pick_gizmo_scale_handles(self):
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        window._set_gizmo_mode("scale")
        self._pump(10)
        for axis in ("x", "y", "z"):
            px = self._screen_point(
                vp, self._handle_point(vp, "scale", axis))
            self.assertEqual(vp.pick_gizmo(px.x(), px.y()),
                             (axis, "scale"))
        window.close()

    def test_pick_gizmo_misses_empty_space(self):
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        self._pump(10)
        # The top-left corner is empty sky: no handle, no object.
        self.assertIsNone(vp.pick(5, 5))
        self.assertIsNone(vp.pick_gizmo(5, 5))
        window.close()

    def test_pick_gizmo_none_without_selection(self):
        window = self._open_tavern()
        vp = window._viewport
        self._pump(10)
        self.assertIsNone(vp.pick_gizmo(vp.width() // 2,
                                        vp.height() // 2))
        window.close()

    def test_pick_gizmo_none_when_suppressed(self):
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        self._pump(10)
        px = self._screen_point(vp, self._handle_point(vp, "move", "x"))
        self.assertIsNotNone(vp.pick_gizmo(px.x(), px.y()))
        vp.set_gizmo_suppressed(True)
        self.assertIsNone(vp.pick_gizmo(px.x(), px.y()))
        vp.set_gizmo_suppressed(False)
        window.close()

    # -- hover ---------------------------------------------------------

    def test_hover_highlights_handle(self):
        from PySide6.QtTest import QTest
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        self._pump(10)
        plain = self._grab_png(window)
        px = self._screen_point(vp, self._handle_point(vp, "move", "x"))
        QTest.mouseMove(vp, px, 30)
        self._pump(10)
        self.assertEqual(vp._gizmo_hover, "x")
        bright = self._grab_png(window)
        self.assertNotEqual(plain, bright)
        window.close()

    def test_hover_clears_when_leaving(self):
        from PySide6.QtTest import QTest
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        self._pump(10)
        px = self._screen_point(vp, self._handle_point(vp, "move", "x"))
        QTest.mouseMove(vp, px, 30)
        self._pump(5)
        self.assertEqual(vp._gizmo_hover, "x")
        from PySide6 import QtCore
        QTest.mouseMove(vp, QtCore.QPoint(5, 5), 30)
        self._pump(5)
        self.assertIsNone(vp._gizmo_hover)
        window.close()

    # -- the nine drag cases -------------------------------------------
    # In the default view the recovered offset is the target offset
    # times (1 - (axis . view_dir)^2), between 0.5x and 0.8x for the
    # three axes — so the range 0.2x..1.2x is generous, and the other
    # two components must be exactly unchanged (a drag writes one
    # component only).

    def _drag_move_case(self, window, axis):
        from light_engine.viewport import _gizmo_axis_frame
        vp = window._viewport
        self._reload(window)
        mug = self._mug(window)
        start = list(mug["position"])
        idx = ("x", "y", "z").index(axis)
        u = _gizmo_axis_frame(axis)[0]
        grab_world = self._handle_point(vp, "move", axis)
        px0 = self._screen_point(vp, grab_world)
        self.assertEqual(vp.pick_gizmo(px0.x(), px0.y()), (axis, "move"))
        target = tuple(grab_world[i] + 50.0 * u[i] for i in range(3))
        self._drag(vp, px0, self._screen_point(vp, target))
        delta = mug["position"][idx] - start[idx]
        self.assertGreater(delta, 10.0)
        self.assertLess(delta, 60.0)
        for j in range(3):
            if j != idx:
                self.assertEqual(mug["position"][j], start[j])
        return mug

    def test_drag_move_x(self):
        window = self._open_tavern()
        self._drag_move_case(window, "x")
        window.close()

    def test_drag_move_y(self):
        window = self._open_tavern()
        self._drag_move_case(window, "y")
        window.close()

    def test_drag_move_z(self):
        window = self._open_tavern()
        self._drag_move_case(window, "z")
        window.close()

    def _drag_rotate_case(self, window, axis):
        vp = window._viewport
        self._reload(window)
        window._set_gizmo_mode("rotate")
        self._pump(5)
        mug = self._mug(window)
        idx = ("x", "y", "z").index(axis)
        start = list(mug["rotation"])
        px0 = self._screen_point(vp, self._ring_point(vp, axis, 45.0))
        self.assertEqual(vp.pick_gizmo(px0.x(), px0.y()),
                         (axis, "rotate"))
        px1 = self._screen_point(vp, self._ring_point(vp, axis, 135.0))
        self._drag(vp, px0, px1)
        delta = mug["rotation"][idx] - start[idx]
        self.assertAlmostEqual(delta, 90.0, delta=3.0)
        for j in range(3):
            if j != idx:
                self.assertEqual(mug["rotation"][j], start[j])
        return mug

    def test_drag_rotate_x(self):
        window = self._open_tavern()
        self._drag_rotate_case(window, "x")
        window.close()

    def test_drag_rotate_y(self):
        window = self._open_tavern()
        self._drag_rotate_case(window, "y")
        window.close()

    def test_drag_rotate_z(self):
        window = self._open_tavern()
        self._drag_rotate_case(window, "z")
        window.close()

    def _drag_scale_case(self, window, axis):
        from light_engine.viewport import (
            GIZMO_SCALE_CUBE_CENTER, _gizmo_axis_frame, gizmo_size)
        vp = window._viewport
        self._reload(window)
        window._set_gizmo_mode("scale")
        self._pump(5)
        mug = self._mug(window)
        idx = ("x", "y", "z").index(axis)
        u = _gizmo_axis_frame(axis)[0]
        center = vp._gizmo_center()
        size = gizmo_size(vp.distance)
        grab_world = self._handle_point(vp, "scale", axis)
        px0 = self._screen_point(vp, grab_world)
        self.assertEqual(vp.pick_gizmo(px0.x(), px0.y()),
                         (axis, "scale"))
        # Twice the grab distance along the axis: the recovered ratio
        # is 2 - (axis . view_dir)^2, between 1.5x and 1.8x.
        target = tuple(center[i] + 2.0 * GIZMO_SCALE_CUBE_CENTER
                       * size * u[i] for i in range(3))
        self._drag(vp, px0, self._screen_point(vp, target))
        ratio = mug["scale"][idx]
        self.assertGreater(ratio, 1.3)
        self.assertLess(ratio, 2.1)
        for j in range(3):
            if j != idx:
                self.assertEqual(mug["scale"][j], 1.0)
        return mug

    def test_drag_scale_x(self):
        window = self._open_tavern()
        self._drag_scale_case(window, "x")
        window.close()

    def test_drag_scale_y(self):
        window = self._open_tavern()
        self._drag_scale_case(window, "y")
        window.close()

    def test_drag_scale_z(self):
        window = self._open_tavern()
        self._drag_scale_case(window, "z")
        window.close()

    # -- exact drag math (axis in the drag plane) ----------------------

    def test_drag_move_x_exact(self):
        from light_engine.viewport import _gizmo_axis_frame
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        self._pump(10)
        self._face_camera(vp, "x")
        mug = self._mug(window)
        start = mug["position"][0]
        grab_world = self._handle_point(vp, "move", "x")
        px0 = self._screen_point(vp, grab_world)
        self.assertEqual(vp.pick_gizmo(px0.x(), px0.y()), ("x", "move"))
        state = self._press(vp, px0)
        self.assertIsNotNone(state)
        u = _gizmo_axis_frame("x")[0]
        grab = state["grab_point"]
        target = tuple(grab[i] + 50.0 * u[i] for i in range(3))
        self._move_release(vp, self._screen_point(vp, target))
        self.assertAlmostEqual(mug["position"][0] - start, 50.0, delta=1.0)
        window.close()

    def test_drag_scale_x_exact(self):
        from light_engine.viewport import _gizmo_axis_frame
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        window._set_gizmo_mode("scale")
        self._pump(10)
        self._face_camera(vp, "x")
        mug = self._mug(window)
        grab_world = self._handle_point(vp, "scale", "x")
        px0 = self._screen_point(vp, grab_world)
        self.assertEqual(vp.pick_gizmo(px0.x(), px0.y()), ("x", "scale"))
        state = self._press(vp, px0)
        self.assertIsNotNone(state)
        # Twice the ACTUAL grab distance (from the drag state, not the
        # ideal cube centre): the ratio is exactly 2 up to the target
        # pixel's rounding.
        center = state["center"]
        u = _gizmo_axis_frame("x")[0]
        target = tuple(center[i] + 2.0 * state["grab_distance"] * u[i]
                       for i in range(3))
        self._move_release(vp, self._screen_point(vp, target))
        self.assertAlmostEqual(mug["scale"][0], 2.0, delta=0.05)
        window.close()

    # -- Shift snap -----------------------------------------------------

    def test_snap_move(self):
        from light_engine.viewport import _gizmo_axis_frame
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        self._pump(10)
        self._face_camera(vp, "x")
        mug = self._mug(window)
        start = mug["position"][0]
        grab_world = self._handle_point(vp, "move", "x")
        px0 = self._screen_point(vp, grab_world)
        state = self._press(vp, px0)
        self.assertIsNotNone(state)
        u = _gizmo_axis_frame("x")[0]
        grab = state["grab_point"]
        # 48 cm with Shift held snaps to 50 (the 45 boundary is 3 cm
        # away, far outside pixel-rounding error).
        target = tuple(grab[i] + 48.0 * u[i] for i in range(3))
        self._move_release(vp, self._screen_point(vp, target), shift=True)
        self.assertAlmostEqual(mug["position"][0] - start, 50.0, delta=0.5)
        window.close()

    def test_snap_rotate(self):
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        window._set_gizmo_mode("rotate")
        self._pump(10)
        mug = self._mug(window)
        px0 = self._screen_point(vp, self._ring_point(vp, "z", 45.0))
        px1 = self._screen_point(vp, self._ring_point(vp, "z", 125.0))
        # 80 degrees with Shift held snaps to 75.
        self._drag(vp, px0, px1, shift=True)
        self.assertAlmostEqual(mug["rotation"][2], 75.0, delta=2.0)
        window.close()

    def test_snap_scale(self):
        from light_engine.viewport import _gizmo_axis_frame
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        window._set_gizmo_mode("scale")
        self._pump(10)
        self._face_camera(vp, "x")
        mug = self._mug(window)
        grab_world = self._handle_point(vp, "scale", "x")
        px0 = self._screen_point(vp, grab_world)
        state = self._press(vp, px0)
        self.assertIsNotNone(state)
        center = state["center"]
        u = _gizmo_axis_frame("x")[0]
        # 1.79x the ACTUAL grab distance with Shift held snaps to 1.8
        # (the 1.75 boundary is 0.04 away, far outside the target
        # pixel's rounding error).
        target = tuple(center[i] + 1.79 * state["grab_distance"] * u[i]
                       for i in range(3))
        self._move_release(vp, self._screen_point(vp, target), shift=True)
        self.assertAlmostEqual(mug["scale"][0], 1.8, delta=0.02)
        window.close()

    # -- live inspector, priority, orbit, keys ------------------------

    def test_inspector_updates_live_during_drag(self):
        from PySide6 import QtCore
        from PySide6.QtTest import QTest
        from light_engine.viewport import _gizmo_axis_frame
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        self._pump(10)
        mug = self._mug(window)
        grab_world = self._handle_point(vp, "move", "x")
        px0 = self._screen_point(vp, grab_world)
        u = _gizmo_axis_frame("x")[0]
        target = tuple(grab_world[i] + 50.0 * u[i] for i in range(3))
        px1 = self._screen_point(vp, target)
        QTest.mousePress(vp, QtCore.Qt.LeftButton,
                         QtCore.Qt.NoModifier, px0, 30)
        QTest.mouseMove(vp, px1, 30)
        self._pump(5)
        # Mid-drag, before the release: the spin box already shows
        # the new value (1 decimal, like the inspector always shows).
        self.assertAlmostEqual(
            window._spins[("position", "x")].value(),
            round(mug["position"][0], 1), delta=0.05)
        self.assertNotEqual(mug["position"][0], -60.0)
        QTest.mouseRelease(vp, QtCore.Qt.LeftButton,
                           QtCore.Qt.NoModifier, px1, 30)
        self._pump(5)
        window.close()

    def test_handle_wins_over_orbit(self):
        from light_engine.viewport import _gizmo_axis_frame
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        self._pump(10)
        mug = self._mug(window)
        azimuth = vp.azimuth
        grab_world = self._handle_point(vp, "move", "x")
        px0 = self._screen_point(vp, grab_world)
        u = _gizmo_axis_frame("x")[0]
        target = tuple(grab_world[i] + 50.0 * u[i] for i in range(3))
        self._drag(vp, px0, self._screen_point(vp, target))
        self.assertEqual(vp.azimuth, azimuth)
        self.assertNotEqual(mug["position"][0], -60.0)
        window.close()

    def test_orbit_still_works_on_empty_space(self):
        from PySide6 import QtCore
        from PySide6.QtTest import QTest
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        self._pump(10)
        self.assertIsNone(vp.pick(5, 5))
        azimuth = vp.azimuth
        QTest.mousePress(vp, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                         QtCore.QPoint(5, 5), 30)
        QTest.mouseMove(vp, QtCore.QPoint(105, 5), 30)
        QTest.mouseRelease(vp, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                           QtCore.QPoint(105, 5), 30)
        self._pump(5)
        self.assertNotEqual(vp.azimuth, azimuth)
        window.close()

    def test_press_release_on_handle_keeps_selection(self):
        from PySide6 import QtCore
        from PySide6.QtTest import QTest
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        self._pump(10)
        mug = self._mug(window)
        px = self._screen_point(vp, self._handle_point(vp, "move", "x"))
        QTest.mousePress(vp, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                         px, 30)
        QTest.mouseRelease(vp, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                           px, 30)
        self._pump(5)
        self.assertEqual(window.selected_id, "mug")
        self.assertEqual(mug["position"], [-60.0, -20.0, 91.0])
        self.assertIsNone(vp._gizmo_drag)
        window.close()

    def test_drag_moves_exact_frame_key(self):
        # The mug has a position key at frame 0. A drag at frame 0
        # moves the key with the base, so the object visibly follows
        # the pointer (keys override the base at their frames) — and
        # the gizmo centre moves with it.
        from light_engine.viewport import _gizmo_axis_frame
        window = self._open_tavern()
        vp = window._viewport
        window._select("mug")
        self._pump(10)
        mug = self._mug(window)
        grab_world = self._handle_point(vp, "move", "x")
        px0 = self._screen_point(vp, grab_world)
        u = _gizmo_axis_frame("x")[0]
        target = tuple(grab_world[i] + 50.0 * u[i] for i in range(3))
        self._drag(vp, px0, self._screen_point(vp, target))
        keys = anim_mod.get_keys(window.scene, "mug", "position")
        key0 = next(v for f, v in keys if f == 0)
        self.assertEqual(key0, mug["position"])
        center = vp._gizmo_center()
        self.assertAlmostEqual(center[0], mug["position"][0], delta=0.01)
        window.close()


@unittest.skipUnless(QT_TESTS, "needs LIGHT_ENGINE_QT_TESTS=1 + a display")
class QtGizmoUndoTests(unittest.TestCase):
    """T14: one undo step per gizmo drag, via the window's live-edit
    pattern (arm on drag start, commit on release). Driven through
    real mouse events on the tavern mug, like the T13 drag tests.
    Undo restores the pre-drag values EXACTLY (the snapshot is a
    scene copy, so assertEqual — not AlmostEqual — is the right
    check), and redo restores the dragged values exactly. Edge
    cases: a press with no movement pushes no step, a drag that
    ends where it started pushes no step, and a keyed-frame drag
    undoes the key with the base (the snapshot covers the whole
    scene, keys included)."""

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets
        cls._app = QtWidgets.QApplication.instance()
        if cls._app is None:
            cls._app = QtWidgets.QApplication([])
        from light_engine.window import MainWindow
        cls.MainWindow = MainWindow

    def _open_tavern(self):
        scene = scene_mod.load_scene(str(EXAMPLE))
        window = self.MainWindow(scene=scene)
        window.resize(800, 600)
        window.show()
        self._pump()
        return window

    def _pump(self, rounds=20):
        from PySide6 import QtCore, QtWidgets
        for _ in range(rounds):
            QtWidgets.QApplication.processEvents()
            QtCore.QThread.msleep(50)

    def _reload(self, window):
        """Fresh tavern scene in the same window (resets the mug's
        base transform AND its keys between drag cases)."""
        window.scene = scene_mod.load_scene(str(EXAMPLE))
        window._apply_scene_to_ui()
        window._select("mug")
        self._pump(5)

    def _screen_point(self, vp, world):
        """Viewport pixel for a world point (forward projection
        through the viewport's own matrices)."""
        from PySide6 import QtCore, QtGui
        proj, view = vp._matrices()
        mapped = (proj * view).map(QtGui.QVector3D(*world))
        x = (mapped.x() + 1.0) / 2.0 * vp.width()
        y = (1.0 - mapped.y()) / 2.0 * vp.height()
        return QtCore.QPoint(int(round(x)), int(round(y)))

    def _mug(self, window):
        return scene_mod.find_object(window.scene, "mug")

    def _handle_point(self, vp, mode, axis):
        """A world point on the given handle: the move shaft midpoint,
        the rotate ring at 45 degrees, or the scale cube centre."""
        from light_engine.viewport import (
            GIZMO_SCALE_CUBE_CENTER, _gizmo_axis_frame, gizmo_size)
        center = vp._gizmo_center()
        size = gizmo_size(vp.distance)
        u, p, q = _gizmo_axis_frame(axis)
        if mode == "rotate":
            return tuple(center[i] + size * (p[i] + q[i]) / math.sqrt(2.0)
                         for i in range(3))
        at = 0.5 if mode == "move" else GIZMO_SCALE_CUBE_CENTER
        return tuple(center[i] + at * size * u[i] for i in range(3))

    def _ring_point(self, vp, axis, degrees):
        """A world point on the rotate ring at the given angle in the
        ring's (p, q) frame."""
        from light_engine.viewport import _gizmo_axis_frame, gizmo_size
        center = vp._gizmo_center()
        size = gizmo_size(vp.distance)
        _u, p, q = _gizmo_axis_frame(axis)
        a = math.radians(degrees)
        return tuple(center[i] + size * (math.cos(a) * p[i]
                                         + math.sin(a) * q[i])
                     for i in range(3))

    def _drag(self, vp, start_px, end_px):
        """Press, move, release through QTest."""
        from PySide6 import QtCore
        from PySide6.QtTest import QTest
        QTest.mousePress(vp, QtCore.Qt.LeftButton,
                         QtCore.Qt.NoModifier, start_px, 30)
        QTest.mouseMove(vp, end_px, 30)
        QTest.mouseRelease(vp, QtCore.Qt.LeftButton,
                           QtCore.Qt.NoModifier, end_px, 30)
        self._pump(5)

    def _drag_points(self, vp, points):
        """Press at the first point, move through the rest in order,
        release at the last one — one gesture, many mouse moves."""
        from PySide6 import QtCore
        from PySide6.QtTest import QTest
        QTest.mousePress(vp, QtCore.Qt.LeftButton,
                         QtCore.Qt.NoModifier, points[0], 30)
        for px in points[1:]:
            QTest.mouseMove(vp, px, 30)
        QTest.mouseRelease(vp, QtCore.Qt.LeftButton,
                           QtCore.Qt.NoModifier, points[-1], 30)
        self._pump(5)

    def _drag_targets(self, vp, mode, axis):
        """Press and release pixels for one drag of the given handle
        (the T13 case geometry: 50 cm along the axis, a 90-degree
        ring sweep, or twice the grab distance)."""
        from light_engine.viewport import (
            GIZMO_SCALE_CUBE_CENTER, _gizmo_axis_frame, gizmo_size)
        u = _gizmo_axis_frame(axis)[0]
        if mode == "move":
            grab_world = self._handle_point(vp, mode, axis)
            px0 = self._screen_point(vp, grab_world)
            target = tuple(grab_world[i] + 50.0 * u[i] for i in range(3))
            return px0, self._screen_point(vp, target)
        if mode == "rotate":
            return (self._screen_point(vp, self._ring_point(vp, axis, 45.0)),
                    self._screen_point(vp, self._ring_point(vp, axis, 135.0)))
        center = vp._gizmo_center()
        size = gizmo_size(vp.distance)
        px0 = self._screen_point(vp, self._handle_point(vp, mode, axis))
        target = tuple(center[i] + 2.0 * GIZMO_SCALE_CUBE_CENTER
                       * size * u[i] for i in range(3))
        return px0, self._screen_point(vp, target)

    def _drag_undo_case(self, window, mode, axis):
        """One drag of the given handle, then undo, then redo. The
        drag must move the axis (a no-op drag pushes no step, which
        would make the undo assertions vacuous), the undo must
        restore the pre-drag values EXACTLY, and the redo must
        restore the dragged values EXACTLY. Undo replaces the
        window's scene object, so the mug is re-fetched after each
        undo/redo."""
        vp = window._viewport
        self._reload(window)
        window._set_gizmo_mode(mode)
        self._pump(5)
        group = {"move": "position", "rotate": "rotation",
                 "scale": "scale"}[mode]
        idx = ("x", "y", "z").index(axis)
        mug = self._mug(window)
        before = list(mug[group])
        px0, px1 = self._drag_targets(vp, mode, axis)
        self.assertEqual(vp.pick_gizmo(px0.x(), px0.y()), (axis, mode))
        stack_before = len(window._undo_stack)
        self._drag(vp, px0, px1)
        mug = self._mug(window)
        dragged = list(mug[group])
        self.assertNotEqual(dragged[idx], before[idx],
                            "the drag must move the axis")
        for j in range(3):
            if j != idx:
                self.assertEqual(dragged[j], before[j])
        # One drag is one undo step, however many mouse moves it took.
        self.assertEqual(len(window._undo_stack), stack_before + 1)
        window.undo()
        mug = self._mug(window)
        self.assertEqual(list(mug[group]), before)
        self.assertEqual(list(anim_mod.sample_object(
            window.scene, mug, window._frame)[group]), before)
        window.redo()
        mug = self._mug(window)
        self.assertEqual(list(mug[group]), dragged)
        self.assertEqual(list(anim_mod.sample_object(
            window.scene, mug, window._frame)[group]), dragged)
        return mug

    # -- the nine drag-undo cases --------------------------------------

    def test_drag_move_x_undo(self):
        window = self._open_tavern()
        self._drag_undo_case(window, "move", "x")
        window.close()

    def test_drag_move_y_undo(self):
        window = self._open_tavern()
        self._drag_undo_case(window, "move", "y")
        window.close()

    def test_drag_move_z_undo(self):
        window = self._open_tavern()
        self._drag_undo_case(window, "move", "z")
        window.close()

    def test_drag_rotate_x_undo(self):
        window = self._open_tavern()
        self._drag_undo_case(window, "rotate", "x")
        window.close()

    def test_drag_rotate_y_undo(self):
        window = self._open_tavern()
        self._drag_undo_case(window, "rotate", "y")
        window.close()

    def test_drag_rotate_z_undo(self):
        window = self._open_tavern()
        self._drag_undo_case(window, "rotate", "z")
        window.close()

    def test_drag_scale_x_undo(self):
        window = self._open_tavern()
        self._drag_undo_case(window, "scale", "x")
        window.close()

    def test_drag_scale_y_undo(self):
        window = self._open_tavern()
        self._drag_undo_case(window, "scale", "y")
        window.close()

    def test_drag_scale_z_undo(self):
        window = self._open_tavern()
        self._drag_undo_case(window, "scale", "z")
        window.close()

    # -- one step per gesture, however many moves ----------------------

    def test_drag_is_one_undo_step(self):
        window = self._open_tavern()
        vp = window._viewport
        self._reload(window)
        mug = self._mug(window)
        before = list(mug["position"])
        px0, px1 = self._drag_targets(vp, "move", "x")
        # Five moves in one gesture: the stack must grow by one.
        mid = [(px0.x() + (px1.x() - px0.x()) * k // 4,
                px0.y() + (px1.y() - px0.y()) * k // 4)
               for k in range(1, 4)]
        from PySide6 import QtCore
        points = [px0] + [QtCore.QPoint(x, y) for x, y in mid] + [px1]
        self._drag_points(vp, points)
        mug = self._mug(window)
        self.assertNotEqual(mug["position"][0], before[0])
        self.assertEqual(len(window._undo_stack), 1)
        window.undo()
        self.assertEqual(list(self._mug(window)["position"]), before)
        window.close()

    # -- no change, no step --------------------------------------------

    def test_press_without_moving_pushes_no_step(self):
        from PySide6 import QtCore
        from PySide6.QtTest import QTest
        window = self._open_tavern()
        vp = window._viewport
        self._reload(window)
        mug = self._mug(window)
        before = list(mug["position"])
        px = self._screen_point(vp, self._handle_point(vp, "move", "x"))
        QTest.mousePress(vp, QtCore.Qt.LeftButton,
                         QtCore.Qt.NoModifier, px, 30)
        QTest.mouseRelease(vp, QtCore.Qt.LeftButton,
                           QtCore.Qt.NoModifier, px, 30)
        self._pump(5)
        self.assertEqual(len(window._undo_stack), 0)
        window.undo()  # nothing to undo: a no-op, values untouched
        self.assertEqual(list(self._mug(window)["position"]), before)
        window.close()

    def test_drag_out_and_back_pushes_no_step(self):
        window = self._open_tavern()
        vp = window._viewport
        self._reload(window)
        mug = self._mug(window)
        before = list(mug["position"])
        px0, px1 = self._drag_targets(vp, "move", "x")
        # Out to the target and back to the exact press pixel: the
        # drag math is deterministic, so the value lands back on the
        # start exactly and the commit finds nothing to push.
        self._drag_points(vp, [px0, px1, px0])
        self.assertEqual(list(self._mug(window)["position"]), before)
        self.assertEqual(len(window._undo_stack), 0)
        window.close()

    # -- keys -----------------------------------------------------------

    def test_keyed_frame_drag_undo_restores_key(self):
        # The mug has a position key at frame 0. A drag there moves
        # the key with the base (T13); undo must restore BOTH, and
        # redo must move both back — the snapshot covers the keys.
        window = self._open_tavern()
        vp = window._viewport
        self._reload(window)
        mug = self._mug(window)
        before = list(mug["position"])
        keys = anim_mod.get_keys(window.scene, "mug", "position")
        key_before = list(next(v for f, v in keys if f == 0))
        self.assertEqual(key_before, before)
        px0, px1 = self._drag_targets(vp, "move", "x")
        self._drag(vp, px0, px1)
        mug = self._mug(window)
        dragged = list(mug["position"])
        self.assertNotEqual(dragged[0], before[0])
        keys = anim_mod.get_keys(window.scene, "mug", "position")
        self.assertEqual(list(next(v for f, v in keys if f == 0)),
                         dragged)
        window.undo()
        mug = self._mug(window)
        self.assertEqual(list(mug["position"]), before)
        keys = anim_mod.get_keys(window.scene, "mug", "position")
        self.assertEqual(list(next(v for f, v in keys if f == 0)), before)
        window.redo()
        mug = self._mug(window)
        self.assertEqual(list(mug["position"]), dragged)
        keys = anim_mod.get_keys(window.scene, "mug", "position")
        self.assertEqual(list(next(v for f, v in keys if f == 0)),
                         dragged)
        window.close()

    def test_interpolated_frame_drag_undo(self):
        # Frame 24 sits between the mug's position keys at 0 and 48:
        # a drag there edits the base only (no key at 24, and a drag
        # never creates one). Undo must restore the base exactly and
        # leave both keys untouched.
        window = self._open_tavern()
        vp = window._viewport
        self._reload(window)
        window.set_frame(24)
        self._pump(5)
        mug = self._mug(window)
        before = list(mug["position"])
        keys = anim_mod.get_keys(window.scene, "mug", "position")
        key0 = list(next(v for f, v in keys if f == 0))
        key48 = list(next(v for f, v in keys if f == 48))
        px0, px1 = self._drag_targets(vp, "move", "x")
        self._drag(vp, px0, px1)
        mug = self._mug(window)
        self.assertNotEqual(mug["position"][0], before[0])
        keys = anim_mod.get_keys(window.scene, "mug", "position")
        self.assertEqual(list(next(v for f, v in keys if f == 0)), key0)
        self.assertEqual(list(next(v for f, v in keys if f == 48)), key48)
        window.undo()
        mug = self._mug(window)
        self.assertEqual(list(mug["position"]), before)
        keys = anim_mod.get_keys(window.scene, "mug", "position")
        self.assertEqual(list(next(v for f, v in keys if f == 0)), key0)
        self.assertEqual(list(next(v for f, v in keys if f == 48)), key48)
        window.close()

    def test_no_key_track_drag_undo(self):
        # The mug's scale track has no keys at all: a scale drag
        # edits the base only, and undo restores it exactly.
        window = self._open_tavern()
        vp = window._viewport
        self._reload(window)
        window._set_gizmo_mode("scale")
        self._pump(5)
        self.assertEqual(
            anim_mod.get_keys(window.scene, "mug", "scale"), [])
        mug = self._mug(window)
        before = list(mug["scale"])
        px0, px1 = self._drag_targets(vp, "scale", "x")
        self._drag(vp, px0, px1)
        mug = self._mug(window)
        self.assertNotEqual(mug["scale"][0], before[0])
        window.undo()
        self.assertEqual(list(self._mug(window)["scale"]), before)
        window.close()

    # -- redo discipline -------------------------------------------------

    def test_redo_cleared_by_new_edit(self):
        # Drag, undo, then drag again: the second drag is a new edit,
        # so the redo stack (which held the first drag) is cleared —
        # redo after it is a no-op and the second drag's result stays.
        window = self._open_tavern()
        vp = window._viewport
        self._reload(window)
        px0, px1 = self._drag_targets(vp, "move", "x")
        self._drag(vp, px0, px1)
        window.undo()
        self.assertEqual(len(window._redo_stack), 1)
        px0, px1 = self._drag_targets(vp, "move", "x")
        self._drag(vp, px0, px1)
        second = list(self._mug(window)["position"])
        self.assertEqual(len(window._redo_stack), 0)
        window.redo()
        self.assertEqual(list(self._mug(window)["position"]), second)
        window.close()

    def test_undo_redo_undo(self):
        # Drag, undo, redo, undo: the object ends back where it
        # started, and redo is available again after the second undo.
        window = self._open_tavern()
        vp = window._viewport
        self._reload(window)
        mug = self._mug(window)
        before = list(mug["position"])
        px0, px1 = self._drag_targets(vp, "move", "x")
        self._drag(vp, px0, px1)
        dragged = list(self._mug(window)["position"])
        window.undo()
        self.assertEqual(list(self._mug(window)["position"]), before)
        window.redo()
        self.assertEqual(list(self._mug(window)["position"]), dragged)
        window.undo()
        self.assertEqual(list(self._mug(window)["position"]), before)
        window.redo()
        self.assertEqual(list(self._mug(window)["position"]), dragged)
        window.close()

    def test_undo_keeps_selection_and_gizmo(self):
        # Undo replaces the scene, and set_scene clears the
        # viewport's selection — undo/redo must re-sync it, or the
        # gizmo vanishes after an undo even though the outliner
        # still shows the object selected. Found by the
        # redo-cleared test, which drags again after an undo.
        window = self._open_tavern()
        vp = window._viewport
        self._reload(window)
        px0, px1 = self._drag_targets(vp, "move", "x")
        self._drag(vp, px0, px1)
        window.undo()
        self.assertEqual(window.selected_id, "mug")
        self.assertEqual(vp.selected_id, "mug")
        self.assertIsNotNone(vp._gizmo_center())
        window.redo()
        self.assertEqual(vp.selected_id, "mug")
        self.assertIsNotNone(vp._gizmo_center())
        window.close()

    def test_undo_keeps_camera(self):
        # Undo/redo restore the scene, not the view: set_scene
        # re-fits the camera (frame() re-centers on the scene
        # centroid and resets the orbit angles), so undo/redo must
        # preserve the camera across the re-apply — otherwise
        # undoing a move yanks the view a pixel or two (the moved
        # object shifts the centroid) and resets an orbited view
        # to the defaults. Found by the T14 shot check: the redone
        # move-x shot differed from the dragged shot by a ~2px
        # global shift until this fix.
        window = self._open_tavern()
        vp = window._viewport
        self._reload(window)
        vp.target = (10.0, 20.0, 30.0)
        vp.distance = 500.0
        vp.azimuth = -100.0
        vp.elevation = 45.0
        self._pump(5)
        camera = (vp.target, vp.distance, vp.azimuth, vp.elevation)
        px0, px1 = self._drag_targets(vp, "move", "x")
        self._drag(vp, px0, px1)
        self.assertEqual((vp.target, vp.distance, vp.azimuth,
                          vp.elevation), camera)
        window.undo()
        self.assertEqual((vp.target, vp.distance, vp.azimuth,
                          vp.elevation), camera)
        window.redo()
        self.assertEqual((vp.target, vp.distance, vp.azimuth,
                          vp.elevation), camera)
        window.close()


if __name__ == "__main__":
    unittest.main()
