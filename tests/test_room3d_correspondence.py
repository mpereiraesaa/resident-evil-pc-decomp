"""Known-camera matching must distinguish ray evidence from collision priors."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_reconstruct import feature_cloud, unique_correspondences


class CorrespondenceTests(unittest.TestCase):
    def test_duplicate_orientations_do_not_inflate_evidence(self):
        a = {'pixels': [[10, 20], [30, 40]], 'views': [0, 1], 'reprojection_error': .8}
        b = {**a, 'reprojection_error': .2}
        c = {**a, 'pixels': [[20, 20], [40, 40]]}
        d = {**a, 'views': [0, 2]}
        self.assertEqual(unique_correspondences([a, b, c, d]), [b, c, d])

    def test_sparse_collisions_do_not_silently_erase_ray_evidence(self):
        rng = np.random.default_rng(42)
        texture = rng.integers(0, 256, (240, 320), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), .5)
        shifted = cv2.warpAffine(texture, np.float32([[1, 0, -50], [0, 1, 0]]), (320, 240))
        images = [Image.fromarray(texture), Image.fromarray(shifted)]
        room = {'cameras': [
            {'index': 0, 'origin': [0, 0, 0], 'target': [0, 0, 1000], 'fov': 200, 'roll': 0},
            {'index': 1, 'origin': [500, 0, 0], 'target': [500, 0, 1000], 'fov': 200, 'roll': 0}],
            'boundaries': [{'xmin': 0, 'xmax': 36, 'zmin': 0, 'zmax': 18}]}
        bounded = feature_cloud(room, images)
        unbounded = feature_cloud(room, images, use_collision_bounds=False)
        self.assertGreater(len(unbounded['points']), 20)
        self.assertEqual(bounded['points'], [])
        self.assertEqual(bounded['pairs'][0]['bounds_rejected'], unbounded['pairs'][0]['accepted'])
        self.assertFalse(unbounded['collision_bounds_applied'])
        self.assertAlmostEqual(np.median([p['xyz'][2] for p in unbounded['points']]), 2000, delta=5)
        room['boundaries'] = []
        self.assertGreater(len(feature_cloud(room, images, use_collision_bounds=False)['points']), 20)
        with self.assertRaises(ValueError):
            feature_cloud(room, images)


if __name__ == '__main__':
    unittest.main()
