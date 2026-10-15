"""Part reviews are exact-input proposals, never geometric authority."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_part_review import annotate_segments, read_review, record_path, signature
from room3d_part_review import main as review_main
from room3d_shape_fit import fit_objects


class PartReviewTests(unittest.TestCase):
    def sample(self):
        room = {'source': 'ROOM1010.RDT', 'cameras': [{'index': 0, 'origin': [0, 0, 0],
                 'target': [0, 0, 1000], 'fov': 100, 'roll': 0}]}
        image = Image.new('RGB', (12, 12), 'gray')
        mask = np.zeros((12, 12), bool)
        mask[4:8, 4:8] = True
        item = {'mask': 'stairs.png', 'label': 'staircase', 'score': .4, 'predicted_iou': .9}
        return room, image, mask, item

    def save(self, root, room, image, mask, item, response='B', category='railing'):
        identity = signature(room, room['cameras'][0], item, image, mask)
        path = record_path(root, identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {'signature': identity, 'response': response, 'category': category,
                  'status': 'unverified_part_category'}
        path.write_text(json.dumps(record))
        return path, identity

    def test_annotation_preserves_detector_label_and_input(self):
        room, image, mask, item = self.sample()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save(root, room, image, mask, item)
            annotated, proposals = annotate_segments(room, [image], [[(item, mask)]], root)
        self.assertEqual(annotated[0][0][0]['label'], 'staircase')
        self.assertEqual(proposals[0]['category'], 'railing')
        self.assertNotIn('part_review', item)

    def test_changed_pixels_mask_model_or_prompt_cannot_reuse_review(self):
        room, image, mask, item = self.sample()
        with tempfile.TemporaryDirectory() as directory:
            path, identity = self.save(Path(directory), room, image, mask, item)
            for key in ('image_rgb_sha256', 'mask_sha256', 'model', 'revision', 'prompt'):
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    read_review(path, {**identity, key: 'changed'})
            changed = mask.copy()
            changed[0, 0] = True
            with self.assertRaises(ValueError):
                read_review(path, signature(room, room['cameras'][0], item, image, changed))

    def test_invalid_category_mapping_and_unsafe_mask_fail(self):
        room, image, mask, item = self.sample()
        with tempfile.TemporaryDirectory() as directory:
            path, identity = self.save(Path(directory), room, image, mask, item,
                                       response='A', category='railing')
            with self.assertRaises(ValueError):
                read_review(path, identity)
        with self.assertRaises(ValueError):
            signature(room, room['cameras'][0], {**item, 'mask': '../x.png'}, image, mask)

    def test_wrong_visible_tread_proposal_cannot_override_door_guard(self):
        room, image, mask, item = self.sample()
        door = {'mask': 'door.png', 'label': 'door', 'score': .7, 'predicted_iou': .95}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save(root, room, image, mask, item, response='A', category='visible_treads')
            _, report = fit_objects(
                room, {'primitives': [], 'stair_part_reviews': str(root), 'semantic_stair_guard': True},
                [[(item, mask), (door, np.ones_like(mask))]],
                [(np.full(mask.shape, 1000.), np.ones_like(mask))], root / 'fit', None,
                images=[image],
            )
        self.assertEqual(report['objects'], [])
        self.assertEqual(report['part_reviews'][0]['category'], 'visible_treads')
        self.assertEqual(len(report['semantic_rejections']), 1)

    def test_cli_reuses_exact_cache_without_loading_model_and_rejects_changed_image(self):
        room, image, mask, item = self.sample()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch, reviews = root / 'batch', root / 'reviews'
            source = batch / 'room1010'
            segmentation = batch / 'segmentation/room1010'
            source.mkdir(parents=True)
            segmentation.mkdir(parents=True)
            (source / 'survey.json').write_text(json.dumps(room))
            image.save(source / 'camera0-original.png')
            Image.fromarray(mask.astype('uint8') * 255).save(segmentation / item['mask'])
            def save_segments():
                (segmentation / 'camera0-original.json').write_text(json.dumps({
                    'rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                    'segments': [item],
                }))
            save_segments()
            cache, _ = self.save(reviews, room, image, mask, item)
            original_record = cache.read_bytes()
            argv = ['review', '--batch', str(batch), '--rooms', 'room1010', '--out', str(reviews)]
            with patch.object(sys, 'argv', argv), patch('room3d_part_review.Reviewer') as model:
                model.side_effect = AssertionError('Cache reuse must not load model')
                review_main()
                self.assertEqual(cache.read_bytes(), original_record)
                image.putpixel((0, 0), (255, 0, 0))
                image.save(source / 'camera0-original.png')
                save_segments()
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    review_main()
                model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
