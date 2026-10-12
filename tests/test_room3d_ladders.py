"""Physical open ladder topology and actual exported multi-camera visibility."""
import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from batch_room3d import generate_mesh
from room3d_preview import camera_basis, render
from room3d_shapes import generate


class LadderTests(unittest.TestCase):
    def setUp(self):
        self.params = {'width': 800, 'height': 3600, 'thickness': 50,
                       'stand_off': 180, 'rungs': 9}

    def test_closed_connected_parts_leave_regular_open_gaps(self):
        obj = generate('ladder', self.params)
        parts = np.asarray(obj['parts'])
        self.assertEqual(obj['part_roles'], ['rail'] * 2 + ['rung'] * 9 + ['bracket'] * 4)
        self.assertEqual(parts.shape, (15, 8, 3))
        for part in parts:
            self.assertGreater(abs(np.linalg.det([part[1]-part[0], part[3]-part[0], part[4]-part[0]])), 0)
        rails, rungs, brackets = parts[:2], parts[2:11], parts[11:]
        np.testing.assert_allclose(np.diff(rungs[:, :, 1].mean(axis=1)), -360)
        self.assertAlmostEqual(parts[:, :, 1].min(), -3600)
        self.assertAlmostEqual(parts[:, :, 1].max(), 0)
        for rung in rungs:
            for rail in rails:
                overlap = np.minimum(rung.max(0), rail.max(0)) - np.maximum(rung.min(0), rail.min(0))
                self.assertTrue((overlap > 0).all())
        for bracket in brackets:
            self.assertAlmostEqual(bracket[:, 2].min(), 0)
            self.assertAlmostEqual(bracket[:, 2].max(), 180)
            self.assertTrue(any((np.minimum(bracket.max(0), rail.max(0))
                                 - np.maximum(bracket.min(0), rail.min(0)) > 0).all()
                                for rail in rails))

    def test_native_mesh_preserves_holes_in_three_camera_views(self):
        room = {'boundaries': [{'index': 0, 'xmin': -2000, 'xmax': 2000,
                               'zmin': -100, 'zmax': 0, 'type': 1, 'flags': 256}],
                'cameras': [{'index': i, 'origin': [x, -1800, 6000],
                             'target': [0, -1800, 180], 'roll': 0, 'fov': 230}
                            for i, x in enumerate((-1000, 0, 1000))]}
        obj = generate('ladder', self.params)
        with tempfile.TemporaryDirectory() as temp:
            mesh = Path(temp) / 'ladder.obj'
            generate_mesh(room, mesh, {'primitives': [obj]})
            for camera in room['cameras']:
                _, mask = render(mesh, camera, only_groups={'primitive_0'})
                self.assertGreater(mask.sum(), 150)
                for point, expected in [([0, -1800, 180], True),
                                        ([0, -1620, 180], False),
                                        ([375, -1620, 180], True)]:
                    local = (np.asarray(point) - camera['origin']) @ camera_basis(camera).T
                    x, y = np.rint(local[:2] * camera['fov'] / local[2] + [160, 120]).astype(int)
                    self.assertEqual(bool(mask[y, x]), expected)

    def test_pose_and_parameters_are_serializable_without_mutation(self):
        saved = copy.deepcopy(self.params)
        origin, yaw = np.array([300, -100, 700]), .7
        source = generate('ladder', self.params)
        posed = generate('ladder', self.params, origin, yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        np.testing.assert_allclose(posed['parts'], np.asarray(source['parts']) @ rotation + origin)
        self.assertEqual(self.params, saved)
        self.assertEqual(posed['shape']['parameters'], saved)

    def test_invalid_dimensions_and_counts_abstain(self):
        for changes in ({'thickness': 0}, {'width': 200}, {'height': 1000},
                        {'stand_off': 25}, {'rungs': True}, {'rungs': 1},
                        {'rungs': 65}, {'rungs': 2.5}, {'height': float('nan')},
                        {'stand_off': float('inf')}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                generate('ladder', {**self.params, **changes})


if __name__ == '__main__':
    unittest.main()
