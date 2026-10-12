"""Actual open-mask spacing, clipped views and heldout exact rollback."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from batch_room3d import generate_mesh
from room3d_ladder_spacing import refine_spacing, spacing_evidence
from room3d_preview import render
from room3d_shapes import generate


class LadderSpacingTests(unittest.TestCase):
    def scene(self, root, missing=False):
        room = {'boundaries': [{'index': 0, 'xmin': -2000, 'xmax': 2000,
                               'zmin': -100, 'zmax': 0, 'type': 1, 'flags': 256}],
                'cameras': [{'index': i, 'origin': origin, 'target': [0, -1800, 180],
                             'roll': 0, 'fov': 230} for i, origin in enumerate(
                                 ([800, -1800, 6000], [-900, -1000, 2200], [1500, -1500, 5000]))]}
        params = {'width': 800, 'height': 3600, 'thickness': 50,
                  'stand_off': 180, 'rungs': 9, 'rung_phase': -.1}
        truth = generate('ladder', params)
        if missing:
            truth['parts'].pop(6)
            truth['part_roles'].pop(6)
        mesh = root/'truth.obj'
        generate_mesh(room, mesh, {'primitives': [truth]})
        observations = [{'camera': c['index'], 'camera_model': c,
                         'mask': render(mesh, c, only_groups={'primitive_0'})[1]}
                        for c in room['cameras']]
        y, x = np.nonzero(observations[0]['mask'])
        envelope = np.zeros((240, 320), 'uint8')
        cv2.fillConvexPoly(envelope, cv2.convexHull(np.column_stack((x, y)).astype('int32')), 1)
        observations[0]['mask'] = envelope.astype(bool)
        baseline = generate('ladder', {**params, 'rungs': 17, 'rung_phase': 0})
        return room, baseline, observations

    def test_filled_mask_supplies_no_count_and_clipped_open_mask_supplies_spacing(self):
        with tempfile.TemporaryDirectory() as temp:
            _room, baseline, observations = self.scene(Path(temp))
            evidence, report = spacing_evidence(baseline, observations[0])
            self.assertIsNone(evidence)
            self.assertEqual(report['status'], 'insufficient_open_rungs')
            self.assertTrue(observations[1]['mask'][0].any())
            evidence, report = spacing_evidence(baseline, observations[1])
            self.assertIsNotNone(evidence, report)
            self.assertEqual(report['inferred_rungs'], 9)

    def test_actual_raster_refinement_recovers_count_phase_and_preserves_pose(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            room, baseline, observations = self.scene(root)
            saved = copy.deepcopy(baseline)
            masks = [o['mask'].copy() for o in observations]
            selected, report = refine_spacing(room, baseline, observations, root/'fit', generate_mesh)
            self.assertEqual(report['status'], 'selected', report)
            self.assertEqual(selected['shape']['parameters']['rungs'], 9)
            self.assertAlmostEqual(selected['shape']['parameters']['rung_phase'], -.1)
            self.assertEqual(report['training_cameras'], [0, 1])
            self.assertEqual(report['heldout_cameras'], [2])
            self.assertEqual(baseline, saved)
            self.assertEqual(selected['shape']['origin'], baseline['shape']['origin'])
            self.assertEqual(selected['shape']['yaw'], baseline['shape']['yaw'])
            np.testing.assert_array_equal(selected['parts'][:2], baseline['parts'][:2])
            np.testing.assert_array_equal(selected['parts'][-4:], baseline['parts'][-4:])
            for o, mask in zip(observations, masks, strict=True):
                np.testing.assert_array_equal(o['mask'], mask)

    def test_reserved_mask_is_a_veto_without_candidate_reselection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            room, baseline, observations = self.scene(root)
            _, good = refine_spacing(room, baseline, observations, root/'good', generate_mesh)
            observations[-1]['mask'][:] = False
            selected, veto = refine_spacing(room, baseline, observations, root/'veto', generate_mesh)
            self.assertIs(selected, baseline)
            self.assertEqual(veto['status'], 'reserved_spacing_unavailable')
            self.assertEqual(good['trials'], veto['trials'])
            self.assertEqual(good['selected_parameters'], veto['selected_parameters'])

    def test_one_missing_rung_does_not_double_spacing(self):
        with tempfile.TemporaryDirectory() as temp:
            _, baseline, observations = self.scene(Path(temp), missing=True)
            evidence, report = spacing_evidence(baseline, observations[-1])
            self.assertIsNotNone(evidence, report)
            self.assertEqual(report['inferred_rungs'], 9)
            self.assertIn(2, report['gap_multiples'])

    def test_invalid_phase_is_rejected_and_default_geometry_is_unchanged(self):
        params = {'width': 800, 'height': 3600, 'thickness': 50, 'stand_off': 180, 'rungs': 9}
        np.testing.assert_array_equal(generate('ladder', params)['parts'],
                                      generate('ladder', {**params, 'rung_phase': 0})['parts'])
        for phase in (-.46, .46, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                generate('ladder', {**params, 'rung_phase': phase})


if __name__ == '__main__':
    unittest.main()
