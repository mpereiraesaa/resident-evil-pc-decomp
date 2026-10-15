"""Known-plane checks for the dense reconstruction experiment (no game assets)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from probe_room3d_tsdf import fuse, project
from probe_room3d_multiview import view_surface


class FusionTests(unittest.TestCase):
    def camera(self, x=0):
        return {
            "index": 0,
            "origin": [x, 0, 0],
            "target": [x, 0, 1000],
            "fov": 100,
            "roll": 0,
        }

    def test_reprojection_preserves_known_camera_and_depth(self):
        camera = self.camera(300)
        depth = np.full((240, 320), 2000.0)
        vertices, uv, faces = view_surface(depth, camera, stride=40)
        z, xy, _, valid = project(vertices, camera)
        np.testing.assert_allclose(z, 2000)
        np.testing.assert_allclose(xy, uv * [319, -239] + [0, 239], atol=1e-6)
        self.assertTrue(valid.all())
        self.assertGreater(len(faces), 0)

    def test_two_cameras_fuse_world_plane_without_duplicate_depth_layers(self):
        room = {
            "cameras": [self.camera(), self.camera(300)],
            "boundaries": [{"xmin": -1000, "xmax": 1000, "zmin": 1000, "zmax": 3000}],
        }
        depth = np.full((240, 320), 2000.0)
        depths = [(depth, np.ones(depth.shape, bool)) for _ in room["cameras"]]
        vertices, faces, _ = fuse(room, depths, voxel=200)
        central = vertices[
            (np.abs(vertices[:, 0]) < 600) & (np.abs(vertices[:, 1]) < 600)
        ]
        self.assertGreater(len(central), 0)
        np.testing.assert_allclose(central[:, 2], 2000, atol=1e-3)
        self.assertGreater(len(faces), 0)


if __name__ == "__main__":
    unittest.main()
