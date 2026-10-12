"""Shared rail triangulation with partial masks and a genuinely reserved view."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from batch_room3d import generate_mesh
from room3d_ladder_fit import proposals, rail_lines
from room3d_preview import render
from room3d_shapes import generate


class SharedLadderTests(unittest.TestCase):
    def observations(self):
        room = {'boundaries': [{'index': 0, 'xmin': -2000, 'xmax': 2000,
                               'zmin': -100, 'zmax': 0, 'type': 1, 'flags': 256}],
                'cameras': [{'index': i, 'origin': origin, 'target': [0, -1800, 180],
                             'roll': 0, 'fov': 230} for i, origin in enumerate(
                                 ([800, -1800, 6000], [-900, -1000, 2200], [1500, -1500, 5000]))]}
        obj = generate('ladder', {'width': 800, 'height': 3600, 'thickness': 50,
                                 'stand_off': 180, 'rungs': 9})
        with tempfile.TemporaryDirectory() as temp:
            mesh = Path(temp) / 'ladder.obj'
            generate_mesh(room, mesh, {'primitives': [obj]})
            observations = [{'camera': c['index'], 'camera_model': c,
                             'mask': render(mesh, c, only_groups={'primitive_0'})[1]}
                            for c in room['cameras']]
        # First mask mimics SAM's filled envelope, while the close view keeps
        # physical gaps and clips the top of the ladder at the viewport.
        mask = observations[0]['mask'].astype('uint8')
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(mask, contours, 1)
        observations[0]['mask'] = mask.astype(bool)
        return observations

    def test_shared_axes_survive_filled_and_clipped_observations(self):
        observations = self.observations()
        self.assertTrue(observations[1]['mask'][0].any())
        saved = copy.deepcopy(observations)
        candidates, report = proposals(observations)
        self.assertEqual(report['training_cameras'], [0, 1])
        self.assertEqual(report['heldout_cameras'], [2])
        self.assertTrue(candidates, report)
        plausible = [o for o in candidates if abs(o['shape']['parameters']['width']-800)<250
                     and abs(o['shape']['parameters']['height']-3600)<500]
        self.assertTrue(plausible, [o['shape'] for o in candidates])
        for original, current in zip(saved, observations, strict=True):
            np.testing.assert_array_equal(original['mask'], current['mask'])
            self.assertEqual(original['camera_model'], current['camera_model'])
        self.assertTrue(any(e['border_clipped'] for e in plausible[0]['ladder_evidence']['observed_extents']))

    def test_reserved_mask_cannot_change_proposals_or_training_evidence(self):
        observations = self.observations()
        baseline, report = proposals(observations)
        observations[-1]['mask'][:] = False
        changed, changed_report = proposals(observations)
        self.assertTrue(baseline)
        self.assertEqual(baseline, changed)
        self.assertEqual(report, changed_report)

    def test_single_view_duplicate_camera_and_parallel_baseline_abstain(self):
        observations = self.observations()
        self.assertEqual(proposals(observations[:1])[0], [])
        duplicate = copy.deepcopy(observations[:2])
        duplicate[1]['camera'] = duplicate[0]['camera']
        with self.assertRaises(ValueError):
            proposals(duplicate)
        same = copy.deepcopy(observations[:1]) * 2
        same[1] = {**same[1], 'camera': 7}
        self.assertEqual(proposals(same)[0], [])

    def test_invalid_masks_and_nonvertical_object_abstain(self):
        observation = self.observations()[0]
        bad = {**observation, 'mask': np.zeros((240, 320), 'uint8')}
        with self.assertRaises(ValueError):
            rail_lines(bad)
        bad = {**observation, 'mask': np.zeros((240, 320), bool)}
        bad['mask'][110:130, 40:280] = True
        self.assertIsNone(rail_lines(bad))
        for value in (0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                proposals([observation], rung_spacing=value)


if __name__ == '__main__':
    unittest.main()
