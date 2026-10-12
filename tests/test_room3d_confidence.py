"""Source-mask score provenance through merging, fitting and failure summaries."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from batch_room3d import generate_mesh
from evaluate_room3d_details import object_evidence
from room3d_confidence import failure_confidence, observation_confidence
from room3d_shape_fit import fit_objects, observations


class ConfidenceTests(unittest.TestCase):
    def test_merged_masks_retain_scores_in_rejected_fit_and_evaluation(self):
        room = {'cameras': [{'index': 7, 'origin': [0, -1000, -2000],
                            'target': [0, -1000, 0], 'fov': 230, 'roll': 0}],
                'boundaries': [{'index': 0, 'xmin': -3000, 'xmax': 3000,
                                'zmin': 0, 'zmax': 4000, 'flags': 0, 'type': 4}]}
        mask = np.zeros((240, 320), bool)
        mask[110:120, 110:210] = True
        entries = [({'label': 'railing', 'mask': f'mask{i}.png', 'score': score,
                     'predicted_iou': sam}, mask.copy())
                   for i, (score, sam) in enumerate(((.31, .81), (.88, .97)))]
        saved = copy.deepcopy(entries)
        depths = [(np.full((240, 320), 1000.), np.ones((240, 320), bool))]
        merged = observations(room, [entries], depths)
        self.assertEqual(len(merged), 1)
        self.assertEqual([v['detector_score'] for v in merged[0]['confidence_evidence']], [.31, .88])
        with tempfile.TemporaryDirectory() as temp:
            _, report = fit_objects(room, {'primitives': []}, [entries], depths,
                                    Path(temp), generate_mesh)
            obj = report['objects'][0]
            self.assertFalse(obj['accepted'])  # too little world extent for a rail
            confidence = object_evidence(obj)['model_confidence']
            self.assertEqual(confidence['status'], 'recorded')
            self.assertTrue(all(v['mask_pixels_sha256'] == hashlib.sha256(mask.tobytes()).hexdigest()
                                for v in confidence['records']))
            self.assertEqual([(v['camera'], v['mask']) for v in confidence['records']],
                             [(7, 'mask0.png'), (7, 'mask1.png')])
            bucket = failure_confidence(report['objects'])['categories']['railing']
            self.assertEqual(bucket['detector_score_range'], [.31, .88])
            self.assertEqual(bucket['sam_predicted_iou_range'], [.81, .97])
            self.assertEqual(bucket['recorded_masks'], 2)
            self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
        for (a, mask_a), (b, mask_b) in zip(entries, saved, strict=True):
            self.assertEqual(a, b)
            np.testing.assert_array_equal(mask_a, mask_b)
        for field in ('score', 'predicted_iou'):
            for value in (np.nan, np.inf, -np.inf, -1., 1.01):
                invalid = [({**entries[0][0], field: value}, mask)]
                self.assertEqual(observations(room, [invalid], depths), [])
        refined = {**entries[0][0], 'label': 'staircase', 'predicted_iou': .3,
                   'original_predicted_iou': .91, 'structural_refinement':
                   {'accepted_proposal': True, 'structural_evidence': {'tread_edges': 5}}}
        proposal = observations(room, [[(refined, mask)]], depths)[0]['confidence_evidence'][0]
        self.assertEqual(proposal['predicted_iou'], .3)
        self.assertEqual(proposal['original_predicted_iou'], .91)
        self.assertEqual(proposal['sam_prediction_source'], 'refined_mask')

    def test_legacy_and_invalid_scores_are_explicit_without_invented_confidence(self):
        objects = [
            {'kind': 'lamp', 'accepted': False, 'observations': [{'camera': 0, 'mask': 'old.png'}]},
            {'kind': 'lamp', 'accepted': False, 'observations': [
                {'camera': 1, 'mask': 'partial.png', 'predicted_iou': .76}]},
            {'kind': 'lamp', 'accepted': True, 'observations': [{'camera': 2, 'mask': 'accepted.png',
                'detector_score': .99, 'predicted_iou': .99}]},
            {'kind': 'railing', 'accepted': False, 'observations': [{'camera': 3, 'mask': 'bad.png',
                'detector_score': float('nan'), 'predicted_iou': float('inf')}]}]
        self.assertEqual(observation_confidence(objects[0])['status'], 'not_recorded')
        self.assertEqual(observation_confidence(objects[1])['status'], 'partial')
        report = failure_confidence(objects)
        self.assertEqual(report['categories']['lamp']['rejected_objects'], 2)
        self.assertIsNone(report['categories']['lamp']['detector_score_range'])
        self.assertEqual(report['categories']['lamp']['sam_predicted_iou_range'], [.76, .76])
        self.assertIsNone(report['categories']['railing']['sam_predicted_iou_range'])
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
