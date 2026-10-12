"""Finite bracket caps, pose preservation and failed-attachment rollback."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from batch_room3d import generate_mesh
from room3d_connections import closest_triangle_point, support_mesh
from room3d_ladder_supports import (
    anchor_brackets,
    closest_points,
    finite_contact,
    wall_proposals,
)
from room3d_shapes import generate


class LadderSupportTests(unittest.TestCase):
    def scene(self, root):
        room = {'boundaries': [{'index': 0, 'xmin': -2000, 'xmax': 2000,
                               'zmin': -100, 'zmax': 0, 'type': 1, 'flags': 256}],
                'cameras': [{'index': 0, 'origin': [0, -1500, 5000],
                             'target': [0, -1500, 0], 'fov': 230, 'roll': 0}]}
        mesh = root/'wall.obj'
        generate_mesh(room, mesh, {'wall_height': 3000})
        obj = generate('ladder', {'width': 800, 'height': 3600, 'thickness': 50,
                                 'stand_off': 180, 'rungs': 9, 'rung_phase': -.1},
                       [300, 0, 400], .25)
        return room, obj, support_mesh(mesh)

    def test_vector_closest_points_matches_existing_finite_reference(self):
        rng = np.random.default_rng(7)
        triangles = rng.normal(size=(20, 3, 3))*100
        triangles[0, 1] = triangles[0, 0]
        triangles[1] = triangles[1, 0]
        for point in rng.normal(size=(4, 3))*150:
            actual = closest_points(point, triangles)
            expected = np.array([closest_triangle_point(point, t) for t in triangles])
            np.testing.assert_allclose(np.linalg.norm(actual-point, axis=1),
                                       np.linalg.norm(expected-point, axis=1), atol=1e-8)

    def test_finite_caps_preserve_rails_rungs_and_regenerate_from_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            _, obj, (triangles, groups) = self.scene(Path(temp))
            saved = copy.deepcopy(obj)
            attached, report = anchor_brackets(obj, triangles, groups, {'boundary_0'})
            self.assertEqual(report['status'], 'anchored', report)
            self.assertTrue(finite_contact(attached, triangles, groups))
            self.assertEqual(obj, saved)
            np.testing.assert_array_equal(attached['parts'][:-4], obj['parts'][:-4])
            np.testing.assert_allclose(np.asarray(attached['parts'][-4:])[:, 4:].mean(1),
                                       np.asarray(obj['parts'][-4:])[:, 4:].mean(1))
            self.assertEqual(attached['shape']['origin'], obj['shape']['origin'])
            self.assertEqual(attached['shape']['yaw'], obj['shape']['yaw'])
            self.assertEqual(attached['part_roles'], obj['part_roles'])
            regenerated = generate('ladder', attached['shape']['parameters'],
                                   attached['shape']['origin'], attached['shape']['yaw'])
            np.testing.assert_array_equal(regenerated['parts'], attached['parts'])
            shifted = copy.deepcopy(attached)
            shifted['parts'][-1][0][0] += 10000
            self.assertFalse(finite_contact(shifted, triangles, groups))

    def test_unsupported_and_too_distant_caps_roll_back_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            room, obj, (triangles, groups) = self.scene(Path(temp))
            for allowed, limit in [({'floor'}, 1000), ({'boundary_0'}, 50)]:
                result, report = anchor_brackets(obj, triangles, groups, allowed, maximum_distance=limit)
                self.assertIs(result, obj)
                self.assertEqual(report['status'], 'no_finite_bracket_support')
            with self.assertRaises(ValueError):
                anchor_brackets(obj, triangles, groups, {'boundary_0'}, maximum_distance=float('nan'))
            short_wall = Path(temp)/'short-wall.obj'
            generate_mesh(room, short_wall, {'wall_height': 1000})
            result, report = anchor_brackets(obj, *support_mesh(short_wall), {'boundary_0'})
            self.assertIs(result, obj)
            self.assertEqual(report['failed_bracket'], 1)
            self.assertEqual(len(report['brackets']), 1)

    def test_wall_aligned_alternatives_adjust_finite_bracket_height(self):
        with tempfile.TemporaryDirectory() as temp:
            room, obj, (triangles, groups) = self.scene(Path(temp))
            candidates = list(wall_proposals(obj, triangles, groups, {'boundary_0'},
                                             [room['cameras'][0]['origin']]))
            self.assertTrue(candidates)
            for candidate in candidates:
                self.assertTrue(finite_contact(candidate, triangles, groups))
                self.assertLess(candidate['shape']['parameters']['bracket_levels'][1], .9)
                self.assertEqual(candidate['shape']['origin'][1], 0)

    def test_invalid_bracket_overrides_and_levels_are_rejected(self):
        p = {'width': 800, 'height': 3600, 'thickness': 50, 'stand_off': 180, 'rungs': 9}
        for changes in ({'bracket_levels': [0, .9]}, {'bracket_levels': [.9, .1]},
                        {'bracket_levels': [.1, float('nan')]}, {'bracket_parts': []},
                        {'bracket_parts': np.zeros((4, 8, 3)).tolist()},
                        {'bracket_parts': np.full((4, 8, 3), np.nan).tolist()}):
            with self.assertRaises(ValueError):
                generate('ladder', {**p, **changes})


if __name__ == '__main__':
    unittest.main()
