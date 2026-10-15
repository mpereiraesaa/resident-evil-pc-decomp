"""Focused recovery must retain original inputs and abstain without evidence."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_checkpoint import inventory
from room3d_stair_recovery import retry_cameras, run, select_candidates


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (320, 240), 'white')
        self.mask = np.zeros((240, 320), bool)
        self.mask[50:100, 50:100] = True
        self.item = {'label': 'staircase', 'score': .6, 'predicted_iou': .8}

    def test_retry_requires_existing_stair_hypothesis_and_uses_camera_ids(self):
        room = {'cameras': [{'index': 4}, {'index': 7}]}
        self.assertEqual(retry_cameras(room, [[(self.item, self.mask)], []]), [7])
        self.assertEqual(retry_cameras(room, [[], []]), [])
        ambiguous = {**self.item, 'label': 'staircase railing'}
        self.assertEqual(retry_cameras(room, [[(ambiguous, self.mask)], []]), [])
        self.assertEqual(retry_cameras(room, [[({**self.item, 'score': .2}, self.mask)], []]), [])

    def test_selection_needs_geometry_and_deduplicates_while_retaining_better_retry(self):
        proposal = (self.item, self.mask)
        args = (self.image, {}, np.ones((240, 320)))
        self.assertEqual(select_candidates(*args, [proposal], [], region_finder=lambda *a: []), [])
        regions = lambda *a: [{'tread_edges': 5, 'correlation': .95, 'mask': self.mask}]
        selected = select_candidates(*args, [proposal, proposal], [], region_finder=regions)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][0]['predicted_iou'], .8)
        self.assertNotIn('mask', selected[0][0]['stair_recovery']['tread_regions'][0])
        self.assertEqual(select_candidates(*args, [proposal], [proposal], region_finder=regions), [])
        weak = ({**self.item, 'score': .2}, self.mask)
        self.assertEqual(len(select_candidates(*args, [proposal], [weak], region_finder=regions)), 1)
        for change in ({'score': .2}, {'predicted_iou': .6}, {'score': float('nan')}, {'label': 'railing'}):
            self.assertEqual(select_candidates(*args, [({**self.item, **change}, self.mask)], [],
                                               region_finder=regions), [])

    def test_negative_room_copies_hash_verified_cache_without_loading_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / 'images'
            segmentation = root / 'masks'
            depth = root / 'depth'
            for path in (images, segmentation, depth):
                path.mkdir()
            self.image.save(images / 'camera0-original.png')
            rgb = hashlib.sha256(self.image.tobytes()).hexdigest()
            survey = root / 'inputs/survey.json'
            survey.parent.mkdir()
            survey.write_text(json.dumps({'cameras': [{'index': 0}]}))
            record = {'rgb_sha256': rgb, 'segments': [{**self.item, 'label': 'wall', 'mask': 'wall.png'}]}
            (segmentation / 'camera0-original.json').write_text(json.dumps(record))
            Image.fromarray(self.mask.astype('uint8') * 255).save(segmentation / 'wall.png')
            np.save(depth / 'camera0-depth.npy', np.ones((240, 320)))
            np.save(depth / 'camera0-confidence.npy', np.ones((240, 320)))
            (depth / 'depth.json').write_text(json.dumps({
                'survey_sha256': hashlib.sha256(survey.read_bytes()).hexdigest(),
                'views': [{'camera': 0, 'rgb_sha256': rgb}]}))
            before = inventory(segmentation)
            out = root / 'recovered'
            report = run(survey, images, segmentation, depth, out)
            self.assertEqual(report['retry_cameras'], [])
            self.assertEqual(inventory(out / 'segmentation'), before)
            self.assertEqual(inventory(segmentation), before)
            record['rgb_sha256'] = 'invalid'
            (segmentation / 'camera0-original.json').write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                run(survey, images, segmentation, depth, root / 'invalid')
            self.assertFalse((root / 'invalid').exists())


if __name__ == '__main__':
    unittest.main()
