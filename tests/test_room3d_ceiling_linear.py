"""Exported geometry, finite support and visible-body invariants."""
import copy
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from batch_room3d import generate_mesh
from room3d_ceiling_linear import anchor, contact, proposals
from room3d_connections import support_mesh
from room3d_preview import load_obj, render
from room3d_shapes import generate


class CeilingLinearTests(unittest.TestCase):
    def setUp(self):
        self.params = {'profile': 'ceiling_linear', 'height': 220, 'radius': 900,
                       'housing_width': 280, 'thickness': 20, 'shade': 60, 'arms': 2}
        self.room = {'boundaries': [{'index': 0, 'xmin': -3000, 'xmax': 3000,
                                    'zmin': -3000, 'zmax': 3000, 'type': 4, 'flags': 0}],
                     'cameras': []}

    @staticmethod
    def roof(half=1500, y=-2000, group='roof'):
        a, b, c, d = [[-half, y, -half], [half, y, -half],
                      [half, y, half], [-half, y, half]]
        return [np.array([a, b, c]), np.array([a, c, d])], [group, group]

    def test_exported_closed_parts_and_roles_keep_rendered_depth(self):
        camera = {'index': 0, 'origin': [0, -700, -3000],
                  'target': [0, -1800, 0], 'fov': 230, 'roll': 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for count in (1, 2, 3):
                obj = generate('lamp', {**self.params, 'arms': count}, [0, -1780, 0], .37)
                self.assertEqual(obj['part_roles'], ['support']*4 + ['shade']*count)
                before = copy.deepcopy(obj)
                plain, roles = root/f'plain{count}.obj', root/f'roles{count}.obj'
                generate_mesh(self.room, plain, {'primitives': [obj]})
                generate_mesh(self.room, roles, {'primitives': [obj], 'local_fixture_appearance': True})
                all_triangles, groups = support_mesh(roles)
                triangles = [t for t, g in zip(all_triangles, groups, strict=True) if g == 'primitive_0']
                self.assertEqual(len(triangles), 12 * (4 + count))
                for start in range(0, len(triangles), 12):
                    edges, volume = Counter(), 0
                    for xyz in np.asarray(triangles[start:start+12])[:, :, :3]:
                        volume += np.dot(xyz[0], np.cross(xyz[1], xyz[2]))/6
                        for a, b in zip(xyz, np.roll(xyz, -1, axis=0), strict=True):
                            edges[tuple(sorted((tuple(a), tuple(b))))] += 1
                    self.assertTrue(all(n == 2 for n in edges.values()))
                    self.assertGreater(abs(volume), 1)
                names = Counter(material for _, material in load_obj(roles, -1, only_groups={'primitive_0'})[0])
                self.assertEqual(names['fixture_shade'], count*12)
                self.assertEqual(names['fixture_support'], 48)
                a, am, az = render(plain, camera, return_depth=True, only_groups={'primitive_0'})
                b, bm, bz = render(roles, camera, return_depth=True, only_groups={'primitive_0'})
                self.assertGreater(int(am.sum()), 100)
                np.testing.assert_array_equal(am, bm)
                np.testing.assert_array_equal(az, bz)
                self.assertFalse(np.array_equal(a, b))
                self.assertEqual(obj, before)

    def test_finite_rotated_mount_rejects_center_only_and_wrong_support(self):
        obj = generate('lamp', self.params, [0, -1780, 0], .6)
        anchored = anchor(obj, {'support': 'ceiling', 'group': 'roof', 'point': [0, -2000, 0]})
        self.assertTrue(contact(anchored, *self.roof()))
        self.assertFalse(contact(anchored, *self.roof(half=20)))
        self.assertFalse(contact(anchored, *self.roof(group='wall')))
        self.assertFalse(contact(anchored, *self.roof(y=-2011)))
        for tolerance in (-1, float('nan')):
            self.assertFalse(contact(anchored, *self.roof(), tolerance=tolerance))
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory)/'actual.obj'
            ceiling = {'points': [[-1500, -2000, -1500], [1500, -2000, -1500],
                                  [1500, -2000, 1500], [-1500, -2000, 1500]],
                       'material': 'ceiling', 'group': 'roof'}
            generate_mesh(self.room, mesh, {'structural_surfaces': [ceiling]})
            self.assertTrue(contact(anchored, *support_mesh(mesh)))

    def test_extension_preserves_body_and_never_invents_ceiling(self):
        obj = generate('lamp', self.params, [0, -1780, 0], .45)
        obj['evidence'] = {'camera': 7, 'mask': 'own.png', 'confidence': .41}
        saved = copy.deepcopy(obj)
        triangles, groups = self.roof(y=-2500)
        candidates = list(proposals(obj, triangles, groups, {'roof'}))
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertTrue(contact(candidate, triangles, groups))
        np.testing.assert_array_equal(candidate['parts'][3:], obj['parts'][3:])
        self.assertEqual(candidate['shape']['origin'], obj['shape']['origin'])
        self.assertEqual(candidate['shape']['yaw'], obj['shape']['yaw'])
        self.assertEqual(candidate['evidence'], obj['evidence'])
        candidate['evidence']['confidence'] = 0
        self.assertEqual(obj, saved)
        for roof, allowed in ((self.roof(half=20, y=-2500), {'roof'}),
                              (self.roof(y=-5000), {'roof'}),
                              (self.roof(y=-1900), {'roof'}),
                              (self.roof(y=-2500), {'other'}), (([], []), {'roof'})):
            self.assertEqual(list(proposals(obj, *roof, allowed)), [])
        sloped, groups = self.roof(y=-2500)
        sloped[0][0, 1] -= 100
        sloped[1][0, 1] -= 100
        self.assertEqual(list(proposals(obj, sloped, groups, {'roof'})), [])
        with self.assertRaises(ValueError):
            list(proposals(obj, triangles, groups, {'roof'}, maximum_extension=float('inf')))

    def test_invalid_dimensions_or_anchors_reject_degenerate_lights(self):
        for update in ({'arms': 4}, {'arms': True}, {'housing_width': 20},
                       {'radius': 200}, {'thickness': 60}, {'height': 80},
                       {'shade': float('nan')}, {'housing_width': float('inf')}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                generate('lamp', {**self.params, **update})
        obj = generate('lamp', self.params, [0, -1780, 0])
        for prior in ({'support': 'wall', 'group': 'roof', 'point': [0, -2000, 0]},
                      {'support': 'ceiling', 'group': '', 'point': [0, -2000, 0]},
                      {'support': 'ceiling', 'group': 'roof', 'point': [50, -2000, 0]},
                      {'support': 'ceiling', 'group': 'roof', 'point': [0, -1800, 0]}):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                anchor(obj, prior)


if __name__ == '__main__':
    unittest.main()
