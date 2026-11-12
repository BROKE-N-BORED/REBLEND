"""Unit tests for Light Engine v0.1. Standard library unittest only.

Run from the light-engine folder:  sh test.sh
Qt tests run only when LIGHT_ENGINE_QT_TESTS=1 (they need a display;
under Xvfb + xcb, QOpenGLWidget works — it does not under offscreen).
"""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from light_engine import animation as anim_mod  # noqa: E402
from light_engine import geometry  # noqa: E402
from light_engine import scene as scene_mod  # noqa: E402
from light_engine import textures  # noqa: E402

EXAMPLE = HERE / "examples" / "tavern-tabletop.json"
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


if __name__ == "__main__":
    unittest.main()
