"""Actual-mesh checks for elongated wall fixtures and mask-driven initialization."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from batch_room3d import generate_mesh
from room3d_connections import support_mesh
from room3d_fixtures import (
    anchor_wall_lamp,
    linear_highlights,
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

    def test_broad_dark_housing_uses_tube_seed_without_replacing_masks(self):
        mask = np.zeros((240, 320), bool)
        mask[80:160, 100:220] = True
        rgb = np.full((240, 320, 3), 12, dtype='uint8')
        rgb[105:113, 100:220] = 255
        rgb[125:133, 100:220] = 255
        image = Image.fromarray(rgb)
        core, evidence = linear_highlights(image, mask)
        self.assertEqual(evidence['components'], 2)
        self.assertFalse(evidence['clipped'])
        self.assertTrue((core <= mask).all())
        saved_mask = mask.copy()
        seed = anchor_wall_lamp(generate('lamp', {**self.params, 'profile': 'wall_sconce',
                                                'arms': 1}), self.prior)
        observations = [{'camera': c['index'], 'camera_model': c, 'mask': mask.copy()}
                        for c in self.room['cameras']]
        with tempfile.TemporaryDirectory() as temp:
            wall = Path(temp)/'wall.obj'
            generate_mesh(self.room, wall, {})
            support = support_mesh(wall)
            self.assertEqual(list(linear_wall_proposals(seed, observations, *support)), [])
            proposals = list(linear_wall_proposals(seed, observations, *support,
                             images={4: image, 7: image}))
            self.assertEqual(len(proposals), 1)
            for obj in proposals:
                self.assertEqual(obj['shape']['parameters']['arms'], 2)
                self.assertTrue(obj['attachment_prior']['tube_count_consensus'])
                self.assertEqual(obj['attachment_prior']['region'], 'elongated_training_highlights')
                self.assertEqual([e['camera'] for e in obj['attachment_prior']['highlight_evidence']], [4, 7])
        for obs in observations:
            np.testing.assert_array_equal(obs['mask'], saved_mask)
        np.testing.assert_array_equal(np.asarray(image), rgb)
        self.assertIsNone(linear_highlights(Image.new('RGB', (320, 240), 'white'), mask))
        glints = np.full_like(rgb, 12)
        glints[95:125, 145:175] = 255
        self.assertIsNone(linear_highlights(Image.fromarray(glints), mask))
        # Bright strips touching the image edge remain partial evidence.
        shifted = np.roll(rgb, -100, axis=1)
        shifted_mask = np.roll(mask, -100, axis=1)
        self.assertTrue(linear_highlights(Image.fromarray(shifted), shifted_mask)[1]['clipped'])

    def test_export_and_bake_keep_bright_tubes_separate_from_dark_housing(self):
        from room3d_reconstruct import bake_projected

        obj = anchor_wall_lamp(generate('lamp', self.params), self.prior)
        recipe = {'primitives': [obj], 'local_fixture_appearance': True}
        pixels = np.full((240, 320, 3), 25, dtype='uint8')
        pixels[:60] = 255
        images = [Image.fromarray(pixels)]*2
        mask = np.ones((240, 320), bool)
        item = {'mask': 'lamp.png', 'label': 'wall lamp', 'score': .8, 'predicted_iou': .9}
        with tempfile.TemporaryDirectory() as temp:
            source, baked = Path(temp)/'source.obj', Path(temp)/'baked.obj'
            generate_mesh(self.room, source, recipe)
            self.assertIn('usemtl fixture_shade', source.read_text())
            bake_projected(source, self.room, images, baked,
                           segments=[[(item, mask)], [(item, mask)]],
                           group_kinds={'primitive_0': 'lamp'},
                           interpreted_groups={'primitive_0'},
                           fixture_observations={'primitive_0': [{'camera': 4, 'mask': 'lamp.png'}]})
            self.assertIn('usemtl fixture_tint_0_shade', baked.read_text())
            image, visible, tube_depth = render(baked, self.room['cameras'][0],
                        return_depth=True, only_materials={'fixture_tint_0_shade'})
            scene_depth = render(baked, self.room['cameras'][0], return_depth=True)[2]
            exposed = visible & np.isfinite(scene_depth) & (np.abs(
                np.where(visible, tube_depth, 0)-np.where(np.isfinite(scene_depth), scene_depth, 0)) < .01)
            self.assertGreater(exposed.sum(), 100)
            self.assertGreater(np.asarray(image)[exposed].mean(), 240)

    def test_continuous_fitting_keeps_initialized_tube_count(self):
        from room3d_shape_fit import variants

        obj = anchor_wall_lamp(generate('lamp', self.params), self.prior)
        candidates = list(variants(obj))
        self.assertTrue(candidates)
        self.assertEqual({c['shape']['parameters']['arms'] for c in candidates}, {2})

    def test_invalid_dimensions_and_existing_family(self):
        for changed in ({'arms': 4}, {'radius': 100}, {'thickness': 100}, {'shade': float('nan')}):
            with self.assertRaises(ValueError):
                generate('lamp', {**self.params, **changed})
        old = {**self.params, 'profile': 'wall_sconce', 'arms': 1}
        obj = anchor_wall_lamp(generate('lamp', old), self.prior)
        self.assertEqual(obj['shape']['parameters'], old)


if __name__ == '__main__':
    unittest.main()
