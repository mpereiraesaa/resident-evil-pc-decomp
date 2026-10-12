"""Actual-mesh checks for elongated wall fixtures and mask-driven initialization."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from batch_room3d import generate_mesh
from room3d_connections import support_mesh
from room3d_fixtures import (
    anchor_wall_lamp,
    linear_wall_proposals,
    multilight_wall_proposals,
    wall_plate_contact,
)
from room3d_preview import render
from room3d_shapes import generate


class LinearFixtureTests(unittest.TestCase):
    def setUp(self):
        self.params = {'profile': 'wall_linear', 'height': 275, 'radius': 900,
                       'thickness': 20, 'arms': 2, 'shade': 100}
        self.prior = {'group': 'boundary_0', 'point': [0, -1000, 0], 'normal': [0, 0, 1]}
        self.room = {'cameras': [{'index': i, 'origin': [x, -1000, 4000],
                     'target': [0, -1000, 0], 'fov': 230, 'roll': 0}
                    for i, x in [(4, 0), (7, 500)]],
                     'boundaries': [{'index': 0, 'xmin': -3000, 'xmax': 3000,
                       'zmin': -100, 'zmax': 0, 'type': 1, 'flags': 256}]}

    def test_tubes_roles_closed_parts_and_wall_contact(self):
        for arms in (1, 2, 3):
            obj = anchor_wall_lamp(generate('lamp', {**self.params, 'arms': arms, 'height': (arms+.75)*100}), self.prior)
            self.assertEqual(obj['shape']['parameters']['profile'], 'wall_linear')
            self.assertEqual(obj['part_roles'], ['support', 'support']+['shade']*arms)
            self.assertEqual(len(obj['parts']), arms+2)
            local = generate('lamp', {**self.params, 'arms': arms, 'height': (arms+.75)*100})
            for tube in np.asarray(local['parts'][2:]):
                self.assertAlmostEqual(tube[:, 2].min(), 0)
                self.assertAlmostEqual(np.asarray(local['parts'][1])[:, 2].max(), 0)
            for part in np.asarray(obj['parts']):
                self.assertEqual(part.shape, (8, 3))
                self.assertTrue(np.isfinite(part).all())
                self.assertGreater(abs(np.linalg.det(np.array([part[1]-part[0], part[3]-part[0], part[4]-part[0]]))), 0)
            with tempfile.TemporaryDirectory() as temp:
                mesh = Path(temp)/'wall.obj'
                generate_mesh(self.room, mesh, {})
                self.assertTrue(wall_plate_contact(obj, *support_mesh(mesh)))
                self.assertEqual(list(multilight_wall_proposals(obj, *support_mesh(mesh))), [])

    def test_mask_plane_proposals_reproduce_shared_extent_without_mutation(self):
        obj = anchor_wall_lamp(generate('lamp', self.params), self.prior)
        seed = anchor_wall_lamp(generate('lamp', {**self.params, 'profile': 'wall_sconce',
                                                'arms': 1, 'shade': 100}), self.prior)
        saved = copy.deepcopy(seed)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, wall = root/'source.obj', root/'wall.obj'
            generate_mesh(self.room, source, {'primitives': [obj]})
            generate_mesh(self.room, wall, {})
            observations = [{'camera': c['index'], 'camera_model': c,
                             'mask': render(source, c, only_groups={'primitive_0'})[1]}
                            for c in self.room['cameras']]
            masks = [o['mask'].copy() for o in observations]
            proposals = list(linear_wall_proposals(seed, observations, *support_mesh(wall)))
            self.assertEqual(len(proposals), 2)
            for candidate in proposals:
                self.assertTrue(wall_plate_contact(candidate, *support_mesh(wall)))
                self.assertEqual(candidate['attachment_prior']['source_cameras'], [4, 7])
                self.assertLess(abs(candidate['shape']['parameters']['radius']-900), 100)
                self.assertLess(abs(candidate['attachment_prior']['point'][1]+1000), 50)
            self.assertEqual(seed, saved)
            for observation, mask in zip(observations, masks, strict=True):
                np.testing.assert_array_equal(observation['mask'], mask)
            square = copy.deepcopy(observations)
            for observation in square:
                observation['mask'][:] = False
                observation['mask'][100:140, 140:180] = True
            self.assertEqual(list(linear_wall_proposals(seed, square, *support_mesh(wall))), [])
            # An otherwise plausible long light cannot extend off finite support.
            short = copy.deepcopy(self.room)
            short['boundaries'][0].update(xmin=-100, xmax=100)
            generate_mesh(short, wall, {})
            self.assertEqual(list(linear_wall_proposals(seed, observations, *support_mesh(wall))), [])

    def test_invalid_dimensions_and_existing_family(self):
        for changed in ({'arms': 4}, {'radius': 100}, {'thickness': 100}, {'shade': float('nan')}):
            with self.assertRaises(ValueError):
                generate('lamp', {**self.params, **changed})
        old = {**self.params, 'profile': 'wall_sconce', 'arms': 1}
        obj = anchor_wall_lamp(generate('lamp', old), self.prior)
        self.assertEqual(obj['shape']['parameters'], old)


if __name__ == '__main__':
    unittest.main()
