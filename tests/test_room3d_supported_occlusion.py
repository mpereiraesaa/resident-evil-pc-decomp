"""Training-only scene proposals must not be mistaken for heldout approval."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_supported_occlusion import (
    refine_raised_occlusion,
    refine_support_candidates,
)


class RaisedOcclusionTests(unittest.TestCase):
    def test_raised_alternative_can_beat_initial_floor_winner(self):
        floor = (.4, {}, {}, {'name': 'floor'}, {'kind': 'floor'})
        raised = (.6, {}, {'original': True}, {'name': 'raised'}, {'kind': 'collision_box'})
        with patch('room3d_supported_occlusion.refine_raised_occlusion', return_value=(
                {'name': 'corrected'}, [], {'selected': True, 'selection_loss': .2})):
            expanded, searches = refine_support_candidates(
                {}, [floor, raised], [], [], Path('unused'), None,
            )
        best = min(expanded, key=lambda s: s[0])
        self.assertEqual(best[3]['name'], 'corrected')
        self.assertEqual(best[4]['kind'], 'collision_box')
        self.assertTrue(searches[0]['selected'])
        self.assertEqual(raised[2], {'original': True})

    def test_search_is_bounded_and_skips_floor(self):
        solutions = [(loss, {}, {}, {}, {'kind': kind, 'id': i})
                     for i, (loss, kind) in enumerate([
                         (.1, 'floor'), (.8, 'collision_box'),
                         (.3, 'collision_box'), (.5, 'collision_box')])]
        with patch('room3d_supported_occlusion.refine_raised_occlusion', return_value=(
                {}, [], {'selected': False, 'selection_loss': .3})) as refine:
            expanded, searches = refine_support_candidates(
                {}, solutions, [], [], Path('unused'), None,
            )
        self.assertEqual(refine.call_count, 2)
        self.assertEqual([s['support']['id'] for s in searches], [2, 3])
        self.assertEqual(expanded, solutions)

    def run_trial(self, views, foreground=False):
        mask = np.zeros((8, 8), bool)
        mask[2:6, 2:6] = True
        target = np.full((8, 8), 800. if foreground else 1000.)
        room = {'cameras': [{'index': i} for i in range(views)]}
        group = [{'camera': i, 'mask': mask} for i in range(views)]
        recipe = {'primitives': []}
        obj = {'kind': 'staircase'}
        def mesh(_room, path, candidate):
            path.write_text(json.dumps(candidate))
        def render(path, camera, return_depth=False, only_groups=None):
            candidate = json.loads(path.read_text())
            if only_groups:
                depth = np.where(mask, 1000., np.inf)
            else:
                depth = np.full((8, 8), 1000.)
                if not candidate.get('alternative'):
                    depth[mask] = 800.
                elif camera['index'] == 2:
                    depth[:] = 2000.  # A bad reserved view must not select a proposal.
            return None, np.isfinite(depth), depth
        alternatives = [({'primitives': [], 'alternative': True}, {'groups': ['boundary_0']})]
        with tempfile.TemporaryDirectory() as directory, \
                patch('room3d_supports.occlusion_hypotheses', return_value=alternatives), \
                patch('room3d_preview.render', side_effect=render):
            return refine_raised_occlusion(
                room, recipe, obj, {}, group,
                [(target, np.ones((8, 8), bool))] * views, Path(directory), mesh,
            )

    def test_training_selects_but_reserved_depth_failure_remains_visible(self):
        recipe, scores, report = self.run_trial(3)
        self.assertTrue(recipe['alternative'])
        self.assertTrue(report['selected'])
        self.assertEqual(report['training_cameras'], [0, 1])
        self.assertEqual(report['heldout_cameras'], [2])
        self.assertGreater(report['trials'][0]['depth'][2], .5)
        self.assertEqual(len(scores), 3)
        self.assertNotIn('accepted', report)

    def test_observed_foreground_prevents_false_visibility_improvement(self):
        recipe, _, report = self.run_trial(1, foreground=True)
        self.assertNotIn('alternative', recipe)
        self.assertFalse(report['selected'])
        self.assertFalse(report['trials'][0]['eligible'])

    def test_selection_uses_same_tread_objective_as_initial_fit(self):
        with patch('room3d_shape_fit.tread_edge_error', return_value=.5):
            _, scores, report = self.run_trial(1)
        self.assertAlmostEqual(scores[0]['loss'], .1)
        self.assertEqual(scores[0]['tread_edge_error'], .5)
        self.assertAlmostEqual(report['selection_loss'], .1)


if __name__ == '__main__':
    unittest.main()
