"""Camera convention tests independent of GenRecon/CUDA installation."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_room3d_genrecon import WORLD_ROTATION, colmap_pose


class CameraConversionTest(unittest.TestCase):
    def test_forward_camera_and_world_up(self):
        camera = {
            "origin": [1000, -2000, 3000],
            "target": [1000, -2000, 4000],
            "roll": 0,
        }
        q, t = colmap_pose(camera, 0.001)
        rotation = Rotation.from_quat(q, scalar_first=True).as_matrix()
        # Known point one metre right, half a metre down, two metres forward.
        point = WORLD_ROTATION @ np.array([2000, -1500, 5000]) * 0.001
        np.testing.assert_allclose(rotation @ point + t, [1, 0.5, 2], atol=1e-12)
        np.testing.assert_allclose(WORLD_ROTATION @ [0, -1, 0], [0, 0, 1])
        self.assertAlmostEqual(np.linalg.det(rotation), 1)

    def test_oblique_camera_target_projects_to_center(self):
        camera = {
            "origin": [2214, -1116, 15084],
            "target": [5094, -3060, 2178],
            "roll": 0,
        }
        q, t = colmap_pose(camera, 0.001)
        point = WORLD_ROTATION @ np.array(camera["target"]) * 0.001
        view = Rotation.from_quat(q, scalar_first=True).apply(point) + t
        np.testing.assert_allclose(view[:2], 0, atol=1e-12)
        self.assertGreater(view[2], 0)


if __name__ == "__main__":
    unittest.main()
