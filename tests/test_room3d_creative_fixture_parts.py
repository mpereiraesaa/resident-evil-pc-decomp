"""Exercise actual final-scene safety and rollback independently of part optimization."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from batch_room3d import generate_mesh
from room3d_creative_fixture_parts import refine
from room3d_fixtures import anchor_wall_lamp
from room3d_preview import render
from room3d_shapes import generate


class CreativeFixtureSafetyTests(unittest.TestCase):
    def test_selected_only_reserved_depth_veto_and_input_preservation(self):
        cameras = [{'index': i, 'origin': [0, -1000, 2000], 'target': [0, -1000, 0],
                    'fov': 230, 'roll': 0} for i in (4, 7, 9)]
        room = {'cameras': cameras, 'boundaries': [{'index': 0, 'xmin': -3000, 'xmax': 3000,
                'zmin': -100, 'zmax': 0, 'type': 1, 'flags': 256}]}
        params = {'height': 800, 'radius': 300, 'thickness': 40, 'shade': 300,
                  'arms': 1, 'profile': 'wall_sconce'}
        prior = {'point': [0, -1000, 0], 'normal': [0, 0, 1], 'group': 'boundary_0'}
        original = anchor_wall_lamp(generate('lamp', params), prior)
        original['interpretation'] = {'source': 'test_creative_prior'}
        selected = anchor_wall_lamp(generate('lamp', {**params, 'shade': 120}), prior)
        selected['interpretation'] = original['interpretation'].copy()
        recipe = {'primitives': [original], 'wall_height': 3000}
        baseline_recipe = copy.deepcopy(recipe)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before, after = root/'before.obj', root/'after.obj'
            generate_mesh(room, before, recipe)
            generate_mesh(room, after, {**recipe, 'primitives': [selected]})
            old = render(before, cameras[0], return_depth=True)[2]
            new = render(after, cameras[0], return_depth=True)[2]
            changed = np.isfinite(old) & np.isfinite(new) & (np.abs(np.where(np.isfinite(old), old, 0)
                       - np.where(np.isfinite(new), new, 0)) > 50)
            self.assertGreater(changed.sum(), 20)
            target = np.where(np.isfinite(old), old, 5000)
            unchanged = np.isfinite(old) & np.isfinite(new) & ~changed
            selected_mask = np.isfinite(render(after, cameras[0], return_depth=True,
                                              only_groups={'primitive_0'})[2])
            original_mask = np.isfinite(render(before, cameras[0], return_depth=True,
                                              only_groups={'primitive_0'})[2])
            parts = {'status': 'selected', 'heldout_cameras': [9],
                     'after': [{'creative_placement': {'valid': True}, 'attachment_valid': True}]*3}
            for veto in ('none', 'depth', 'training_silhouette', 'reserved_silhouette'):
                group = [{'camera': c['index'], 'mask':
                          original_mask if (veto == 'training_silhouette' and i == 0
                                            or veto == 'reserved_silhouette' and i == 2)
                          else selected_mask} for i, c in enumerate(cameras)]
                depths = [(target.copy(), (changed if veto == 'depth' and i == 2 else unchanged).copy()) for i in range(3)]
                snapshot = [(d.copy(), v.copy()) for d,v in depths]
                with patch('room3d_creative_fixture_parts.refine_wall_parts', return_value=(selected, parts)) as optimizer:
                    result, fitted, report = refine(room, recipe, 0, group, depths, [], [[], [], []],
                                                    root/f'check-{veto}', generate_mesh)
                self.assertEqual(optimizer.call_count, 1)
                self.assertEqual(report['accepted'], veto == 'none')
                if veto != 'none':
                    self.assertIs(result, recipe)
                    self.assertIs(fitted, original)
                    self.assertEqual(report['reason'], {
                        'depth': 'reserved_raw_depth_regressed',
                        'training_silhouette': 'training_whole_silhouette_regressed',
                        'reserved_silhouette': 'reserved_whole_silhouette_regressed'}[veto])
                else:
                    self.assertEqual(result['primitives'][0], selected)
                    self.assertTrue(report['gates']['floor_reference'])
                self.assertEqual(recipe, baseline_recipe)
                for (d,v),(saved_d,saved_v) in zip(depths,snapshot,strict=True):
                    np.testing.assert_array_equal(d,saved_d)
                    np.testing.assert_array_equal(v,saved_v)


if __name__ == '__main__':
    unittest.main()
