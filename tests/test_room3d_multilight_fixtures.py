"""Reusable wall fixtures expose separate shades and preserve finite-wall contact."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from batch_room3d import generate_mesh
from room3d_connections import support_mesh
from room3d_fixtures import anchor_wall_lamp, inset_wall_lamp, wall_plate_contact
from room3d_shapes import generate


class MultilightFixtureTests(unittest.TestCase):
    def params(self, arms=2):
        return {"profile": "wall_candelabra", "arms": arms, "height": 600,
                "radius": 250, "thickness": 20, "shade": 100,
                "span": (arms - 1) * 200, "arm_rise": .2}

    def test_separate_lights_transform_with_anchor_and_have_explicit_roles(self):
        for arms in (2, 3, 4):
            for yaw in (0, .7, math.pi / 2):
                with self.subTest(arms=arms, yaw=yaw):
                    normal = [math.sin(yaw), 0, math.cos(yaw)]
                    prior = {"point": [300, -2000, 500], "normal": normal, "group": "wall"}
                    obj = anchor_wall_lamp(generate("lamp", self.params(arms)), prior)
                    self.assertEqual(obj["shape"]["parameters"]["arms"], arms)
                    self.assertEqual(obj["part_roles"].count("shade"), arms)
                    self.assertEqual(len(obj["part_roles"]), len(obj["parts"]))
                    anchor = np.asarray(obj["parts"][0])[:4].mean(0)
                    np.testing.assert_allclose(anchor, prior["point"], atol=1e-10)
                    shades = np.asarray(obj["parts"])[2:2 + arms]
                    tangent = np.array([math.cos(yaw), 0, -math.sin(yaw)])
                    extent = shades @ tangent
                    self.assertTrue((extent[1:].min(1) > extent[:-1].max(1)).all())
                    for shade in shades:
                        self.assertTrue(((shade - prior["point"]) @ np.asarray(normal) > 0).all())

    def test_inset_preserves_multilight_topology_and_actual_plate_contact(self):
        room = {"cameras": [], "boundaries": [{"index": 0, "xmin": -2000, "xmax": 2000,
                                "zmin": -100, "zmax": 0, "flags": 256, "type": 1}]}
        prior = {"point": [0, -1000, 0], "normal": [0, 0, 1], "group": "boundary_0"}
        obj = anchor_wall_lamp(generate("lamp", self.params()), prior)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wall.obj"
            generate_mesh(room, path, {"wall_height": 1000, "primitives": []})
            triangles, groups = support_mesh(path)
            self.assertFalse(wall_plate_contact(obj, triangles, groups))
            inset = inset_wall_lamp(obj, triangles, groups)
            self.assertIsNotNone(inset)
            self.assertTrue(wall_plate_contact(inset, triangles, groups))
            self.assertEqual(inset["shape"]["parameters"], self.params())

    def test_invalid_counts_and_intersecting_shades_are_rejected(self):
        integral_count = generate("lamp", {**self.params(), "arms": 2.0})
        self.assertEqual(integral_count["part_roles"].count("shade"), 2)
        for changes in ({"arms": 1}, {"arms": 5}, {"span": 100}, {"span": float("inf")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                generate("lamp", {**self.params(), **changes})

    def test_export_labels_every_shade_and_keeps_branches_as_support(self):
        room = {"cameras": [], "boundaries": [{"index": 0, "xmin": -2000,
                "xmax": 2000, "zmin": -1000, "zmax": 4000, "flags": 0, "type": 4}]}
        obj = generate("lamp", self.params(3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.obj"
            recipe = {"primitives": [obj], "local_fixture_appearance": True}
            generate_mesh(room, path, recipe)
            text = path.read_text()
            self.assertEqual(text.count("usemtl fixture_shade\n"), 3 * 6)
            self.assertEqual(text.count("usemtl fixture_support\n"), (len(obj["parts"]) - 3) * 6)
            obj.pop("part_roles")
            with self.assertRaises(ValueError):
                generate_mesh(room, path, recipe)


if __name__ == "__main__":
    unittest.main()
