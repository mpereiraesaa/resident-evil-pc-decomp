"""Finite wall plates must fit inside their support, including wall-top seeds."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from batch_room3d import generate_mesh
from room3d_connections import support_mesh
from room3d_fixtures import anchor_wall_lamp, inset_wall_lamp, wall_plate_contact
from room3d_preview import render
from room3d_shape_fit import fit_group
from room3d_shapes import generate


class FixtureInsetTests(unittest.TestCase):
    def scene(self):
        room = {"boundaries": [
            {"index": 0, "xmin": -2000, "xmax": 2000, "zmin": -100,
             "zmax": 0, "flags": 256, "type": 1},
            {"index": 1, "xmin": -2000, "xmax": 2000, "zmin": -1000,
             "zmax": 4000, "flags": 0, "type": 4}], "cameras": []}
        params = {"height": 1000, "radius": 200, "thickness": 20,
                  "arms": 1, "shade": 100, "profile": "wall_sconce"}
        prior = {"group": "boundary_0", "point": [0, -1000, 0], "normal": [0, 0, 1]}
        clipped = anchor_wall_lamp(generate("lamp", params), prior)
        return room, clipped

    def test_top_and_corner_plate_insets_have_real_finite_surface_contact(self):
        room, reference = self.scene()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wall.obj"
            generate_mesh(room, path, {"wall_height": 1000, "primitives": []})
            triangles, groups = support_mesh(path)
            for point in ([0, -1000, 0], [2000, -1000, 0]):
                original = anchor_wall_lamp(reference, {**reference["attachment_prior"], "point": point})
                frozen = copy.deepcopy(original)
                self.assertFalse(wall_plate_contact(original, triangles, groups))
                inset = inset_wall_lamp(original, triangles, groups)
                self.assertIsNotNone(inset)
                self.assertTrue(wall_plate_contact(inset, triangles, groups))
                self.assertEqual(original, frozen)
                self.assertLess(inset["attachment_prior"]["inset_displacement"], 150)
                self.assertIsNone(inset_wall_lamp(inset, triangles, groups))
            # A plate cannot be rescued when the designated wall is absent.
            self.assertIsNone(inset_wall_lamp(reference, triangles, ["other"] * len(groups)))

    def test_single_view_opt_in_selects_visible_attached_solution_over_floating_fit(self):
        room, clipped = self.scene()
        camera = {"index": 0, "origin": [1200, -500, 4000],
                  "target": [0, -500, 0], "fov": 240, "roll": 0}
        room["cameras"] = [camera]
        recipe = {"wall_height": 1000, "primitives": [], "creative_fallback": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "wall.obj"
            generate_mesh(room, path, recipe)
            triangles, groups = support_mesh(path)
            inset = inset_wall_lamp(clipped, triangles, groups)
            inset["initialization"] = "wall_inset"
            floating = generate("lamp", clipped["shape"]["parameters"], [0, 0, 100])
            floating["initialization"] = "mask_depth"
            generate_mesh(room, path, {**recipe, "primitives": [floating]})
            _, mask, depth = render(path, camera, return_depth=True, only_groups={"primitive_0"})
            depths = [(np.where(np.isfinite(depth), depth, 10000), np.isfinite(depth))]
            group = [{"kind": "lamp", "camera": 0, "mask": mask}]
            off, _ = fit_group(room, recipe, group, depths, root / "off", generate_mesh,
                               initial_candidates=[floating, inset], rounds=0)
            on, report = fit_group(room, {**recipe, "single_view_wall_fixtures": True,
                                         "inset_fixture_anchors": True},
                                   group, depths, root / "on", generate_mesh,
                                   initial_candidates=[floating, inset], rounds=0)
        self.assertNotIn("attachment_prior", off)
        self.assertEqual(on["attachment_prior"]["group"], "boundary_0")
        self.assertTrue(report["after"][0]["attachment_valid"])
        self.assertGreaterEqual(report["after"][0]["visible_pixels"], 16)


if __name__ == "__main__":
    unittest.main()
