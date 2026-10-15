"""Visual attachment generation preserves independent geometry across orientations."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_wall_attachment import attachment_recipe


class AttachmentRecipeTests(unittest.TestCase):
    def test_axes_sides_preserve_outside_region_and_join_back_plane(self):
        for axis in (0, 2):
            for side in (-1, 1):
                def transform(p, side=side, axis=axis):
                    q = list(p)
                    q[0] *= side
                    return q if axis == 0 else [q[2], q[1], q[0]]
                main = [[200, 0, 0], [200, -600, 0], [200, -600, 1000], [200, 0, 1000]]
                patch = [[150, 0, 200], [150, -300, 200], [150, -300, 800], [150, 0, 800]]
                bound = {'index': 42, 'xmin': 0, 'xmax': 300, 'zmin': 300, 'zmax': 600}
                if side < 0:
                    bound['xmin'], bound['xmax'] = -300, 0
                if axis == 2:
                    bound['xmin'], bound['zmin'] = bound['zmin'], bound['xmin']
                    bound['xmax'], bound['zmax'] = bound['zmax'], bound['xmax']
                room = {'boundaries': [{'index': 17}, bound]}
                untouched = {'group': 'other', 'material': 'wall', 'points': main}
                recipe = {'structural_surfaces': [{'group': 'boundary_17', 'material': 'wall',
                                                   'points': list(map(transform, main))}, untouched],
                          'hide': [[3, 'other']], 'floor_y': 50}
                before = copy.deepcopy((room, recipe))
                normal = [0., 0.]
                normal[axis // 2] = side
                plane = {'normal': normal, 'offset': 150, 'points': list(map(transform, patch))}
                candidate = {'normal_axis': 'x' if axis == 0 else 'z', 'protrusion_side': side,
                             'backing': 17, 'attachment': 42}
                result = attachment_recipe(room, recipe, plane, candidate, 150)
                self.assertEqual((room, recipe), before)
                self.assertIn(untouched, result['structural_surfaces'])
                self.assertEqual(result['hide'], recipe['hide'])
                remnants = [s for s in result['structural_surfaces'] if s['group'] == 'boundary_17']
                self.assertEqual(len(remnants), 2)
                vertices = np.array([p for s in result['structural_surfaces'] if s['group'] == 'boundary_42' for p in s['points']])
                self.assertEqual(set(vertices[:, axis]), {150 * side, 300 * side})
                self.assertEqual(set(vertices[:, 1]), {-100, 50})
                self.assertEqual(result['skip_boundaries'], [42])
                for face in result['structural_surfaces']:
                    a, b, c, d = np.array(face['points'])
                    np.testing.assert_allclose(a + c, b + d)
                with self.assertRaises(ValueError):
                    attachment_recipe(room, recipe, plane, candidate, 1000)

    def test_tilted_plane_keeps_attachment_faces_planar_and_rejects_partial_extent(self):
        room = {'boundaries': [{'index': 1}, {'index': 2, 'xmin': 0, 'xmax': 300,
                                             'zmin': 300, 'zmax': 600}]}
        recipe = {'structural_surfaces': [{'group': 'boundary_1', 'material': 'wall',
                  'points': [[200, 0, 0], [200, -600, 0], [200, -600, 1000], [200, 0, 1000]]}]}
        normal = np.array([1., .01]); normal /= np.linalg.norm(normal)
        def point(z, y):
            return [(150 - normal[1]*z)/normal[0], y, z]
        plane = {'normal': normal.tolist(), 'offset': 150,
                 'points': [point(200, 0), point(200, -300), point(800, -300), point(800, 0)]}
        candidate = {'normal_axis': 'x', 'protrusion_side': 1, 'backing': 1, 'attachment': 2}
        result = attachment_recipe(room, recipe, plane, candidate, 150)
        for surface in result['structural_surfaces']:
            a, b, c, d = np.array(surface['points'])
            np.testing.assert_allclose(a + c, b + d, atol=1e-10)
        room['boundaries'][1]['zmax'] = 900
        with self.assertRaisesRegex(ValueError, 'outside observed'):
            attachment_recipe(room, recipe, plane, candidate, 150)


if __name__ == '__main__':
    unittest.main()
