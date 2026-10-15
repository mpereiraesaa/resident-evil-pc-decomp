"""Multi-view wall reconciliation must respect distinct walls and scene scale."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_enclosure import (
    compare_view_evidence,
    extract_wall_planes,
    fit_wall,
    reconcile_walls,
    supported_wall_tiles,
    wall_view_evidence,
)


def wall(camera, x, zmin=0, zmax=2000, scale=1, count=25):
    y, z = np.meshgrid(np.linspace(-2000, 0, count), np.linspace(zmin, zmax, count))
    points = np.column_stack([np.full(y.size, x), y.ravel(), z.ravel()]) * scale
    return {'camera': camera, 'points': points, 'median_depth': 5000 * scale}


class EnclosureTests(unittest.TestCase):
    def test_shared_plane_is_balanced_and_scale_invariant(self):
        for scale in (.01, 1, 100):
            a, b = wall(0, 1000, scale=scale, count=80), wall(1, 1040, scale=scale)
            before = a['points'].copy()
            planes, decisions = reconcile_walls([b, a])
            self.assertEqual(len(planes), 1)
            self.assertEqual(planes[0]['cameras'], [0, 1])
            self.assertTrue(decisions[0]['accepted'])
            self.assertAlmostEqual(abs(planes[0]['offset']) / scale, 1020, delta=2)
            np.testing.assert_array_equal(a['points'], before)

    def test_separate_parallel_walls_and_disjoint_patches_are_not_merged(self):
        for other in (wall(1, 1400), wall(1, 1000, 3000, 5000)):
            planes, decisions = reconcile_walls([wall(0, 1000), other])
            self.assertEqual(len(planes), 2)
            self.assertFalse(decisions[0]['accepted'])

    def test_compatible_chain_cannot_drift_away_from_first_view(self):
        planes, decisions = reconcile_walls([wall(0, 1000), wall(1, 1080), wall(2, 1160)])
        self.assertEqual(len(planes), 2)
        self.assertTrue(decisions[0]['accepted'])
        self.assertFalse(decisions[1]['accepted'])
        self.assertTrue(any(not d['passes'] for d in decisions[1]['views']))

    def test_extraction_separates_corner_without_reusing_samples(self):
        first = wall(0, 1000)['points']
        second = first[:, [2, 1, 0]].copy()
        second[:, 2] = 2000
        cloud = np.concatenate([first, second, [[np.nan, 0, 0]]])
        for scale in (0.1, 10):
            planes, remaining = extract_wall_planes(cloud * scale, 5000 * scale,
                                                    min_samples=100)
            self.assertEqual(len(planes), 2)
            self.assertLess(abs(np.dot(planes[0]['normal'], planes[1]['normal'])), .05)
            indices = [index for plane in planes for index in plane['indices']]
            self.assertEqual(len(indices), len(set(indices)))
            self.assertGreater(len(indices), 1100)
            self.assertNotIn(len(cloud) - 1, indices + remaining)
            again, _ = extract_wall_planes(cloud * scale, 5000 * scale, min_samples=100)
            self.assertEqual(planes, again)

    def test_extraction_does_not_split_a_clean_plane_or_claim_tiny_cloud(self):
        cloud = wall(0, 1000)['points']
        planes, remaining = extract_wall_planes(cloud, 5000, min_samples=100)
        self.assertEqual(len(planes), 1)
        self.assertEqual(remaining, [])
        planes, remaining = extract_wall_planes(cloud[:50], 5000, min_samples=100)
        self.assertEqual(planes, [])
        self.assertEqual(len(remaining), 50)

    def test_supported_tiles_preserve_unobserved_gap(self):
        left = wall(0, 1000, 0, 800)['points']
        right = wall(0, 1000, 1600, 2400)['points']
        points = np.concatenate([left, right])
        plane = fit_wall(points)
        tiles = supported_wall_tiles(plane, points, 200, min_samples=4)
        self.assertTrue(tiles)
        for tile in tiles:
            p = np.array(tile['points'])
            self.assertFalse(p[:, 2].min() < 1500 and p[:, 2].max() > 1100)
            np.testing.assert_allclose(p[:, 0], 1000)
        self.assertEqual(supported_wall_tiles(plane, np.empty((0, 3)), 200), [])

    def test_view_evidence_distinguishes_occlusion_and_unknown(self):
        camera = {'index': 0, 'origin': [0, 0, 0], 'target': [0, 0, 1000], 'fov': 100, 'roll': 0}
        depth = np.full((240, 320), 1000.)
        valid = np.ones(depth.shape, bool)
        wall_mask = valid.copy()
        points = np.array([[0, 0, 1000], [0, 0, 1500], [0, 0, 500], [0, 0, -100]])
        np.testing.assert_array_equal(wall_view_evidence(points, camera, depth, valid, wall_mask),
                                      [1, 2, 3, 0])
        wall_mask[:] = False
        self.assertEqual(wall_view_evidence(points, camera, depth, valid, wall_mask)[0], 4)
        valid[120, 161] = False
        np.testing.assert_array_equal(wall_view_evidence(points, camera, depth, valid, wall_mask), 0)
        valid[:] = True
        depth[120, 161] = 2000
        np.testing.assert_array_equal(wall_view_evidence(points, camera, depth, valid, wall_mask), 0)

    def test_lost_visibility_is_not_resolved_conflict(self):
        report = compare_view_evidence([3, 3, 3, 3, 3, 1, 0], [0, 1, 2, 3, 4, 0, 3])
        self.assertEqual(report['samples'], 7)
        for key, value in report.items():
            if key != 'samples':
                self.assertEqual(value, 1, key)
        hidden = compare_view_evidence([3] * 122, [0] * 122)
        self.assertEqual(hidden['conflict_to_wall_support'], 0)
        self.assertEqual(hidden['conflict_to_unknown'], 122)
        with self.assertRaises(ValueError):
            compare_view_evidence([1, 3], [1])

    def test_invalid_and_degenerate_clouds_fail_explicitly(self):
        for cloud in (np.zeros((2, 3)), np.zeros((20, 3)), np.full((20, 3), np.nan)):
            with self.assertRaises(ValueError):
                fit_wall(cloud)
        invalid = wall(0, 1000)
        invalid['median_depth'] = float('nan')
        with self.assertRaises(ValueError):
            reconcile_walls([invalid])


if __name__ == '__main__':
    unittest.main()
