"""Physical fixture docking and training-only local-support selection."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from batch_room3d import generate_mesh
from room3d_connections import support_mesh
from room3d_fixture_pedestals import contact, propose
from room3d_fixtures import fit_fixture_supports, fixture_contact
from room3d_preview import render
from room3d_shapes import generate


class PedestalTests(unittest.TestCase):
    def example(self, y=-1750):
        room = {'boundaries': [
            {'index': 0, 'type': 1, 'flags': 0, 'xmin': -900, 'xmax': 900, 'zmin': 0, 'zmax': 3000},
            {'index': 1, 'type': 4, 'flags': 0, 'xmin': -5000, 'xmax': 5000, 'zmin': -2000, 'zmax': 8000}],
            'cameras': [{'index': i, 'origin': origin, 'target': [0, -1500, 1500], 'fov': 240, 'roll': 0}
                        for i, origin in enumerate(([3500, -2500, -1500], [-3500, -2500, -1500], [0, -2800, 6000]))]}
        lamp = generate('lamp', {'height': 600, 'shade': 200, 'radius': 100,
                                'thickness': 20, 'arms': 1, 'profile': 'table_lamp'},
                        [0, y, 1500], yaw=.3)
        lamp['attachment_prior'] = {'support': 'horizontal', 'group': 'boundary_0', 'point': [0, y, 1500]}
        return room, {'primitives': [], 'local_fixture_pedestals': True}, lamp

    def test_generated_pedestal_contacts_parent_and_fixture_without_furniture_edits(self):
        room, recipe, lamp = self.example()
        before = copy.deepcopy((room, recipe, lamp))
        candidate, fixture = propose(room, recipe, lamp)
        self.assertEqual(candidate.get('boundary_heights', {}), {})
        self.assertEqual(fixture['attachment_prior']['group'], 'primitive_0')
        np.testing.assert_array_equal(fixture['parts'], lamp['parts'])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'assembly.obj'
            generate_mesh(room, path, {**candidate, 'primitives': [*candidate['primitives'], fixture]})
            triangles, groups = support_mesh(path)
            self.assertTrue(contact(candidate, triangles, groups))
            self.assertTrue(fixture_contact(fixture, triangles, groups))
            shifted = copy.deepcopy(candidate)
            shifted['primitives'][0]['parts'] = (np.asarray(shifted['primitives'][0]['parts']) + [0, 50, 0]).tolist()
            self.assertFalse(contact(shifted, triangles, groups))
        self.assertEqual((room, recipe, lamp), before)

    def test_wall_skipped_outside_footprint_and_nonpositive_lift_are_excluded(self):
        room, recipe, lamp = self.example()
        for extra in ({'skip_boundaries': [0]}, {'shared_wall_boundaries': [0]}):
            self.assertIsNone(propose(room, {**recipe, **extra}, lamp))
        wall = copy.deepcopy(room)
        wall['boundaries'][0]['flags'] = 256
        self.assertIsNone(propose(wall, recipe, lamp))
        narrow = copy.deepcopy(room)
        narrow['boundaries'][0]['xmin'] = 0
        self.assertIsNone(propose(narrow, recipe, lamp))
        for y in (-1500, -1400, -2600):
            r, p, f = self.example(y)
            self.assertIsNone(propose(r, p, f))
        candidate, _ = propose(room, {**recipe, 'boundary_heights': {'0': 1200}}, lamp)
        self.assertEqual(np.asarray(candidate['primitives'][0]['parts'])[..., 1].max(), -1200)

    def test_local_support_selection_does_not_use_reserved_mask_or_depth(self):
        room, recipe, lamp = self.example()
        local, fixture = propose(room, recipe, lamp)
        raised = {**recipe, 'boundary_heights': {'0': 1750}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            truth = root / 'truth.obj'
            generate_mesh(room, truth, {**local, 'primitives': [*local['primitives'], fixture]})
            depths, group = [], []
            for camera in room['cameras']:
                scene = render(truth, camera, return_depth=True)[2]
                part = render(truth, camera, return_depth=True, only_groups={'primitive_1'})[2]
                mask = np.isfinite(part) & (part <= scene + 1)
                depths.append((np.where(np.isfinite(scene), scene, 10000), np.isfinite(scene)))
                group.append({'camera': camera['index'], 'kind': 'lamp', 'mask': mask})

            def fit_stub(*args, **kwargs):
                loss = .1 if args[4].name == 'fixture-support0' else 1.
                return copy.deepcopy(lamp), {'after': [{'loss': loss, 'attachment_valid': True}] * 2,
                                             'training_cameras': [0, 1], 'heldout_cameras': [2]}

            def run(path, observations, evidence):
                with patch('room3d_shape_fit.fit_group', side_effect=fit_stub), \
                        patch('room3d_shape_fit.initial_shapes', side_effect=lambda *args: iter([lamp])), \
                        patch('room3d_fixtures.ray_table_lamp_proposals', return_value=[lamp]), \
                        patch('room3d_fixtures.lamp_support_hypotheses', return_value=[
                            (raised, {'boundary': 0, 'height': 1750})]):
                    path.mkdir()
                    return fit_fixture_supports(room, recipe, observations, evidence, path, generate_mesh)

            selected, report, selected_recipe = run(root / 'first', group, depths)
            changed_group = [*group[:2], {**group[2], 'mask': np.zeros_like(group[2]['mask'])}]
            changed_depths = [*depths[:2], (np.full_like(depths[2][0], 10000), depths[2][1])]
            other, other_report, other_recipe = run(root / 'changed-reserved', changed_group, changed_depths)
            self.assertEqual(selected, other)
            self.assertEqual(selected_recipe, other_recipe)
            self.assertEqual(report['fixture_support_trials'], other_report['fixture_support_trials'])
            self.assertTrue(report.get('localized_support'))
            self.assertEqual(selected_recipe.get('boundary_heights', {}), {})
            self.assertNotEqual(report['heldout'], other_report['heldout'])


if __name__ == '__main__':
    unittest.main()
