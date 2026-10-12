"""Monocular depth keeps relative shape without pretending to measure scale."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from room3d_depth import camera_inputs, single_view_scale


class SingleViewDepthTests(unittest.TestCase):
    def test_prior_is_invariant_to_arbitrary_model_scale_and_preserves_camera(self):
        camera = {"index": 0, "origin": [100, -3000, 200],
                  "target": [100, 0, 4200], "fov": 230, "roll": 0}
        ext, intr = camera_inputs({"cameras": [camera]})
        source = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
        outputs = []
        for scale in (1, 100):
            prediction = SimpleNamespace(depth=source * scale)
            evidence = single_view_scale(prediction, camera, ext, intr)
            self.assertFalse(evidence["metric_scale_verified"])
            self.assertEqual(evidence["target_distance"], 5000)
            self.assertAlmostEqual(float(np.median(prediction.depth)), 5000)
            np.testing.assert_array_equal(prediction.extrinsics, ext[:, :3])
            np.testing.assert_array_equal(prediction.intrinsics, intr)
            np.testing.assert_allclose(prediction.depth / prediction.depth.min(), source / source.min())
            outputs.append(prediction.depth)
        np.testing.assert_allclose(*outputs)
        np.testing.assert_array_equal(source, [[[1, 2], [3, 4]]])

    def test_invalid_depth_or_prior_cannot_silently_become_a_metric_map(self):
        camera = {"origin": [0, 0, 0], "target": [0, 0, 100]}
        for array in (np.zeros((1, 2, 2)), np.full((1, 2, 2), np.nan), np.ones((2, 2, 2))):
            with self.subTest(array=array), self.assertRaises(ValueError):
                single_view_scale(SimpleNamespace(depth=array), camera, np.eye(4)[None], np.eye(3)[None])
        with self.assertRaises(ValueError):
            single_view_scale(SimpleNamespace(depth=np.ones((1, 2, 2))),
                              {"origin": [0, 0, 0], "target": [0, 0, 0]},
                              np.eye(4)[None], np.eye(3)[None])


if __name__ == "__main__":
    unittest.main()
