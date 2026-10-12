"""Procedural shape invariants, independent of fitting and proprietary assets."""

import sys
import unittest
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from room3d_shapes import beam, generate


class PrimitiveTests(unittest.TestCase):
    def test_suspended_stairs_preserve_treads_and_leave_space_under_flight(self):
        p = {'width': 2000, 'run': 5000, 'rise': 3000, 'steps': 15, 'landing': 1000}
        solid = generate('staircase', p)
        explicit = generate('staircase', {**p, 'structure': 'solid'})
        suspended = generate('staircase', {**p, 'structure': 'suspended', 'slab_depth': 100})
        np.testing.assert_equal(solid['parts'], explicit['parts'])
        a, b = np.asarray(solid['parts']), np.asarray(suspended['parts'])
        # Top quad vertices and every X/Z coordinate retain exact positions.
        np.testing.assert_equal(a[:, [0, 1, 4, 5]], b[:, [0, 1, 4, 5]])
        np.testing.assert_equal(a[:, :, [0, 2]], b[:, :, [0, 2]])
        self.assertEqual(b[:, :, 1].max(), 0)
        self.assertEqual(b[-1, :, 1].max(), -2900)
        self.assertEqual(b[-2, :, 1].max(), -2700)
        # Adjacent step solids overlap vertically at their shared edge.
        for previous, current in pairwise(b[:15]):
            self.assertGreaterEqual(current[:, 1].max(), previous[:, 1].min())
        self.assertEqual(suspended['shape']['parameters']['structure'], 'suspended')

    def test_suspended_stairs_validate_structure_and_slab_depth(self):
        p = {'width': 2000, 'run': 5000, 'rise': 3000, 'steps': 15}
        for slab in (0, -1, 241, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                generate('staircase', {**p, 'structure': 'suspended', 'slab_depth': slab})
        with self.assertRaises(ValueError):
            generate('staircase', {**p, 'structure': 'unknown'})

    def test_column_has_closed_shaft_base_capital_and_ground_contact(self):
        column = generate("column", {"height": 4000, "width": 400}, [100, -50, 200], 0.4)
        parts = np.asarray(column["parts"])
        self.assertEqual(parts.shape, (8, 8, 3))
        self.assertEqual(parts[:, :, 1].max(), -50)
        self.assertEqual(parts[:, :, 1].min(), -4050)
        for part in parts:
            for face in ((0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1),
                         (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)):
                points = part[list(face)]
                self.assertGreater(np.linalg.norm(np.cross(points[1]-points[0], points[2]-points[0])), 0)
        for parameters in ({"height": 4000, "width": -1},
                           {"height": 4000, "width": 400, "base_height": 3800},
                           {"height": 4000, "width": 400, "capital_height": float("nan")}):
            with self.assertRaises(ValueError):
                generate("column", parameters)

    def test_stairs_have_regular_rises_and_shared_tread_edges(self):
        obj = generate(
            "staircase",
            {"width": 1200, "run": 3600, "rise": 1800, "steps": 9, "landing": 900},
        )
        parts = np.array(obj["parts"])
        np.testing.assert_allclose(parts[:9, :, 1].min(1), -200 * np.arange(1, 10))
        np.testing.assert_allclose(parts[:-1, :, 2].max(1), parts[1:, :, 2].min(1))
        self.assertEqual(parts[-1, :, 1].min(), -1800)
        self.assertEqual(parts[-1, :, 2].max(), 3600)

    def test_rigid_pose_preserves_lengths(self):
        params = {"run": 3000, "height": 900, "thickness": 40, "posts": 7, "rise": 1500}
        a = np.array(generate("railing", params)["parts"])
        b = np.array(generate("railing", params, [300, -600, 700], 1.2)["parts"])
        np.testing.assert_allclose(
            np.linalg.norm(a[:, 0] - a[:, 6], axis=1),
            np.linalg.norm(b[:, 0] - b[:, 6], axis=1),
        )

    def test_rail_posts_meet_handrail_centerline(self):
        parts = np.array(
            generate(
                "railing",
                {"run": 3000, "height": 900, "thickness": 40, "posts": 7, "rise": 1500},
            )["parts"]
        )
        tops = parts[:-1, 4:].mean(1)
        start, end = parts[-1, :4].mean(0), parts[-1, 4:].mean(0)
        np.testing.assert_allclose(tops, np.linspace(start, end, 7))

    def test_lamp_repeated_arms_are_symmetric(self):
        parts = np.array(
            generate(
                "lamp",
                {
                    "height": 800,
                    "radius": 400,
                    "thickness": 30,
                    "arms": 4,
                    "shade": 150,
                },
            )["parts"]
        )
        tips = parts[1::2, 4:].mean(1)
        np.testing.assert_allclose(tips[:, [0, 2]].mean(0), [0, 0], atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(tips[:, [0, 2]], axis=1), 400)

    def test_invalid_shapes_fail_before_export(self):
        for updates in (
            {"steps": 0},
            {"steps": 3.5},
            {"run": -1},
            {"landing": 4000},
            {"rise": float("nan")},
        ):
            p = {"width": 1200, "run": 3000, "rise": 1800, "steps": 9}
            p.update(updates)
            with self.assertRaises(ValueError):
                generate("staircase", p)
        with self.assertRaises(ValueError):
            beam([0, 0, 0], [0, 0, 0], 1)



if __name__ == "__main__":
    unittest.main()
