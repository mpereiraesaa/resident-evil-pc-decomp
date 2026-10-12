"""Training-only ceiling-light proposals and actual finite mounting checks."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from batch_room3d import generate_mesh
from room3d_ceiling_linear import anchor, contact, mask_starts
from room3d_connections import support_mesh
from room3d_fixtures import anchor_fixture, fixture_contact
from room3d_preview import render
from room3d_shape_fit import fit_group, observations, variants
from room3d_shapes import generate


class CeilingLightFitTests(unittest.TestCase):
    def test_panel_dimensions_and_continuous_search_preserve_light_count(self):
        params = {'profile': 'ceiling_panel', 'height': 100, 'radius': 500,
                  'housing_width': 700, 'thickness': 10, 'shade': 10, 'arms': 1}
        panel = generate('lamp', params)
        self.assertEqual(panel['part_roles'], ['support']*4+['shade'])
        self.assertTrue(all(c['shape']['parameters']['arms']==1 for c in variants(panel)))
        for update in ({'arms': 2}, {'shade': 20}, {'housing_width': 1100},
                       {'housing_width': 30}, {'height': 30}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                generate('lamp', {**params, **update})

    def setup_scene(self, root, half_depth=1500):
        room = {'boundaries': [{'index': 0, 'xmin': -2500, 'xmax': 2500,
                               'zmin': -2500, 'zmax': 2500, 'type': 4, 'flags': 0}],
                'cameras': [{'index': i, 'origin': [x, -900, -3500],
                             'target': [0, -1900, 0], 'fov': 230, 'roll': 0}
                            for i, x in enumerate((-1000, 1000, 200))]}
        recipe = {'primitives': [{'kind': 'ceiling', 'box': [-1500, -half_depth, 1500, half_depth, -2050, -2000]}]}
        params = {'profile': 'ceiling_panel', 'height': 100, 'radius': 500,
                  'housing_width': 700, 'thickness': 10, 'shade': 10, 'arms': 1}
        target = anchor(generate('lamp', params, [0, -1900, 0]),
                        {'support': 'ceiling', 'group': 'primitive_0', 'point': [0, -2000, 0]})
        path = root/'source.obj'
        generate_mesh(room, path, {**recipe, 'primitives': [*recipe['primitives'], target]})
        segments, depths = [], []
        for camera in room['cameras']:
            _, mask, _ = render(path, camera, return_depth=True, only_groups={'primitive_1'})
            depth = render(path, camera, return_depth=True)[2]
            self.assertGreater(int(mask.sum()), 24)
            segments.append([({'label': 'wall lamp chandelier', 'mask': f"own{camera['index']}.png",
                              'score': .41, 'predicted_iou': .91}, mask)])
            depths.append((np.where(np.isfinite(depth), depth, 10000), np.ones_like(mask)))
        group = observations(room, segments, depths)
        support = root/'support.obj'
        generate_mesh(room, support, recipe)
        return room, recipe, target, group, depths, support

    def test_training_mask_proposals_are_finite_and_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, group, _, support = self.setup_scene(root)
            saved = copy.deepcopy(group)
            triangles, groups = support_mesh(support)
            candidates = list(mask_starts(group[:2], triangles, groups, {'primitive_0'}))
            self.assertTrue(candidates)
            self.assertLessEqual(len(candidates), 12)
            self.assertTrue(all(contact(c, triangles, groups) for c in candidates))
            self.assertTrue(all(c['attachment_prior']['source_cameras'] == [0, 1] for c in candidates))
            self.assertTrue(any(c['shape']['parameters']['profile']=='ceiling_panel' for c in candidates))
            self.assertEqual(list(mask_starts(group[:1], triangles, groups, {'primitive_0'})), [])
            self.assertEqual(list(mask_starts(group[:2], triangles, groups, set())), [])
            for before, after in zip(saved, group, strict=True):
                np.testing.assert_array_equal(before['mask'], after['mask'])
                self.assertEqual(before['confidence_evidence'], after['confidence_evidence'])

    def test_actual_fitter_reserves_final_mask_and_default_does_not_add_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            room, recipe, _, group, depths, _ = self.setup_scene(root)
            _, off = fit_group(room, recipe, group, depths, root/'off', generate_mesh, rounds=0)
            enabled = {**recipe, 'ceiling_light_fixtures': True}
            obj, on = fit_group(room, enabled, group, depths, root/'on', generate_mesh, rounds=0)
            changed = copy.deepcopy(group)
            changed[-1]['mask'][:] = False
            repeated, alternate = fit_group(room, enabled, changed, depths, root/'changed', generate_mesh, rounds=0)
            self.assertFalse(any(i['shape']['parameters'].get('profile') in ('ceiling_panel','ceiling_linear')
                                 for i in off['initializations']))
            self.assertTrue(any(i['shape']['parameters'].get('profile')=='ceiling_panel'
                                for i in on['initializations']))
            self.assertEqual(on['initializations'], alternate['initializations'])
            self.assertEqual(obj['shape'], repeated['shape'])
            self.assertEqual(on['heldout_cameras'], [2])
            self.assertNotEqual(on['heldout'], alternate['heldout'])

    def test_contact_dispatch_and_cache_follow_mount_plate_not_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            room, recipe, small, group, depths, support = self.setup_scene(root, half_depth=250)
            prior = small['attachment_prior']
            large = anchor(generate('lamp', {**small['shape']['parameters'], 'housing_width': 1000},
                                    small['shape']['origin']), prior)
            np.testing.assert_array_equal(small['parts'][1], large['parts'][1])
            triangles, groups = support_mesh(support)
            self.assertTrue(fixture_contact(small, triangles, groups))
            self.assertFalse(fixture_contact(large, triangles, groups))
            self.assertEqual(anchor_fixture(small, prior), anchor(small, prior))
            _, report = fit_group(room, recipe, group[:2], depths, root/'fit', generate_mesh,
                                  rounds=0, initial_candidates=[small, large])
            self.assertTrue(all(s['attachment_valid'] for s in report['initializations'][0]['scores']))
            self.assertTrue(all(not s['attachment_valid'] for s in report['initializations'][1]['scores']))


if __name__ == '__main__':
    unittest.main()
