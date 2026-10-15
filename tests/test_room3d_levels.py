"""Camera geometry checks for shared horizontal-line hypotheses."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from room3d_levels import (
    base_lines,
    corroborate,
    floor_anchors,
    shared_line,
    stair_level_starts,
)
from room3d_reconstruct import project


class LevelTests(unittest.TestCase):
    def test_shared_line_recovers_height_and_rejects_same_view(self):
        a = {
            "origin": [-2000, -1000, -4000],
            "target": [0, -1000, 2000],
            "roll": 0,
            "fov": 240,
        }
        b = {**a, "origin": [2500, -1800, -2500]}
        points = np.array([[-1000, -1500, 2000], [1000, -1500, 2000]])
        la = project(points, a)[0]
        lb = project(points, b)[0]
        result = shared_line(a, la, b, lb)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["height"], -1500, places=5)
        self.assertAlmostEqual(result["length"], 2000, places=5)
        self.assertIsNone(shared_line(a, la, a, la))
        third = {**a, "origin": [0, -2300, -3000]}
        projected = project(points, third)[0]
        self.assertTrue(corroborate(result, third, projected)["valid"])
        self.assertFalse(corroborate(result, third, projected + [0, 20])["valid"])

    def test_floor_anchors_reproject_to_the_mask_without_depth(self):
        camera = {
            "origin": [0, -1000, 0],
            "target": [0, 0, 2000],
            "roll": 0,
            "fov": 240,
        }
        mask = np.zeros((240, 320), bool)
        mask[140:180, 140:180] = True
        anchors = floor_anchors(camera, mask)
        self.assertEqual(len(anchors), 3)
        for anchor in anchors:
            self.assertAlmostEqual(anchor["point"][1], 0)
            uv, _ = project(np.array([anchor["point"]]), camera)
            np.testing.assert_allclose(uv[0], anchor["pixel"], atol=1e-6)
        self.assertEqual(floor_anchors(camera, np.zeros_like(mask)), [])

    def test_reserved_camera_cannot_supply_a_height_prior(self):
        camera = {
            "origin": [0, -1000, 0],
            "target": [0, 0, 2000],
            "roll": 0,
            "fov": 240,
        }
        mask = np.zeros((240, 320), bool)
        mask[140:180, 140:180] = True
        group = [{"camera_model": camera, "mask": mask}]
        report = {
            "hypotheses": [
                {
                    "height": -3000,
                    "cameras": [0, 1],
                    "corroborating_cameras": [2],
                    "corroborated": True,
                }
            ]
        }
        starts = list(stair_level_starts(group, report))
        self.assertEqual(len(starts), 24)
        for obj in starts:
            p = obj["shape"]["parameters"]
            self.assertAlmostEqual(p["rise"], 3000)
            self.assertIn(round(p["rise"] / p["run"], 2), (0.5, 0.7))
        self.assertEqual(
            list(stair_level_starts(group, report, excluded_cameras=[2])), []
        )

    def test_mask_lower_edge_produces_horizontal_segment(self):
        mask = np.zeros((240, 320), bool)
        mask[40:80, 40:240] = True
        lines = base_lines(mask)
        self.assertTrue(lines)
        for line in lines:
            np.testing.assert_allclose(line[:, 1], 79)


if __name__ == "__main__":
    unittest.main()
