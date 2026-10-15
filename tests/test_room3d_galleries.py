"""Check procedural galleries against exported mesh, not just recipe bounds."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from batch_room3d import generate_mesh
from room3d_assemblies import gallery_for_stairs
from room3d_connections import stair_connections, stair_headroom, support_mesh
from room3d_shapes import generate


class GalleryTests(unittest.TestCase):
    def test_export_connects_both_ends_without_covering_treads(self):
        bounds = [-9000, -9000, 9000, 9000]
        room = {"boundaries": [{"index": 0, "xmin": bounds[0], "zmin": bounds[1],
                               "xmax": bounds[2], "zmax": bounds[3], "type": 4, "flags": 0}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gallery.obj"
            for yaw in (0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 3):
                with self.subTest(yaw=yaw):
                    stairs = generate("staircase", {"width": 1200, "run": 5000,
                                      "rise": 3000, "steps": 15, "structure": "suspended",
                                      "slab_depth": 100}, [0, 0, 0], yaw)
                    gallery = gallery_for_stairs(stairs, bounds, walkway_width=1500)
                    generate_mesh(room, path, {"primitives": [stairs, gallery]})
                    triangles, groups = support_mesh(path)
                    contacts = stair_connections(stairs, triangles, groups=groups,
                                                 walkable_groups={"floor", "primitive_1"},
                                                 excluded_groups={"primitive_0"})
                    self.assertTrue(contacts["valid"], contacts)
                    self.assertTrue(contacts["walkable_valid"], contacts)
                    clearance = stair_headroom(stairs, triangles, groups, {"primitive_1"})
                    self.assertEqual(clearance["obstructed_samples"], 0)
                    self.assertEqual(clearance["unknown_samples"], 45)
                    for part in gallery["parts"]:
                        vertices = np.asarray(part)
                        self.assertEqual(vertices.shape, (8, 3))
                        self.assertTrue(np.isfinite(vertices).all())
                        self.assertTrue((vertices[:, [0, 2]] >= bounds[:2]).all())
                        self.assertTrue((vertices[:, [0, 2]] <= bounds[2:]).all())
                    # Probe strictly inside exported top quads. Overlapping
                    # connector/perimeter faces must not survive partitioning.
                    grid = np.linspace(-8957, 8941, 57)
                    points = np.array(np.meshgrid(grid, grid)).reshape(2, -1).T
                    coverage = np.zeros(len(points), dtype=int)
                    for part in gallery["parts"]:
                        polygon = np.asarray(part)[:4, [0, 2]]
                        edges = np.roll(polygon, -1, axis=0) - polygon
                        offsets = points[None, :, :] - polygon[:, None, :]
                        sides = edges[:, None, 0] * offsets[:, :, 1] - edges[:, None, 1] * offsets[:, :, 0]
                        coverage += (sides > 1e-6).all(axis=0)
                    self.assertLessEqual(int(coverage.max()), 1)
                    self.assertGreater(int(coverage.sum()), 100)
                    for triangle, group in zip(triangles, groups, strict=True):
                        if group == "primitive_1":
                            xyz = np.asarray(triangle)[:, :3]
                            self.assertGreater(np.linalg.norm(np.cross(xyz[1] - xyz[0],
                                                                      xyz[2] - xyz[0])), 1e-6)

    def test_invalid_gallery_parameters_are_rejected(self):
        stairs = generate("staircase", {"width": 1200, "run": 5000,
                          "rise": 3000, "steps": 15})
        for options in ({"walkway_width": 0}, {"walkway_width": 10000},
                        {"thickness": -1}, {"minimum_landing_depth": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                gallery_for_stairs(stairs, [-9000, -9000, 9000, 9000], **options)
        with self.assertRaises(ValueError):
            gallery_for_stairs(stairs, [-100, -100, 100, 100], walkway_width=50)


if __name__ == "__main__":
    unittest.main()
