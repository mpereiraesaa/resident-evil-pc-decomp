"""Deferred reviews bind to exact images and never run inference on stale inputs."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_review import MODELS, PROMPT
from room3d_review_queue import enqueue, review_index, run_queue


class ReviewQueueTests(unittest.TestCase):
    def test_identical_variant_reuses_review_but_changed_prompt_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            room = root / 'room1001'
            cached = root / 'old/room1000'
            room.mkdir()
            cached.mkdir(parents=True)
            paths = [room / ('camera0-' + suffix + '.png') for suffix in ('original', 'assisted')]
            for path in paths:
                Image.new('RGB', (2, 2), 'blue').save(path)
            enqueue(room, [{'index': 0}])
            model, revision = MODELS['4b']
            record = {'model': model, 'revision': revision, 'prompt': PROMPT, 'response': 'NO_CLEAR_DIFFERENCE',
                      'inputs': [{'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]}
            review = cached / 'camera5-review.json'
            review.write_text(json.dumps(record))
            index = review_index([root / 'old'])
            self.assertEqual(len(index), 1)
            with patch('room3d_review_queue.subprocess.run') as inference:
                results = run_queue(room / 'review-queue.json', root / 'new', Path(sys.executable), shared_cache=index)
                inference.assert_not_called()
                self.assertTrue(results[0]['reused'])
            record['prompt'] = 'different prompt'
            review.write_text(json.dumps(record))
            self.assertEqual(review_index([root / 'old']), {})

    def test_cached_review_and_stale_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            room, cache, out = (root / name for name in ('room1000', 'cache', 'reviews'))
            room.mkdir()
            cache.mkdir()
            paths = [room / ('camera0-' + suffix + '.png') for suffix in ('original', 'assisted')]
            for path in paths:
                Image.new('RGB', (2, 2), 'blue').save(path)
            enqueue(room, [{'index': 0}])
            model, revision = MODELS['4b']
            record = {'model': model, 'revision': revision, 'prompt': PROMPT, 'response': 'NO_CLEAR_DIFFERENCE',
                      'inputs': [{'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]}
            (cache / 'camera0-review.json').write_text(json.dumps(record))
            with patch('room3d_review_queue.subprocess.run') as inference:
                result = run_queue(room / 'review-queue.json', out, Path(sys.executable), cache)
                inference.assert_not_called()
                self.assertEqual(result[0]['status'], 'reviewed')
                self.assertTrue(result[0]['reused'])
                Image.new('RGB', (2, 2), 'red').save(paths[1])
                result = run_queue(room / 'review-queue.json', out, Path(sys.executable), cache)
                self.assertEqual(result[0]['status'], 'error')
                inference.assert_not_called()
            Image.new('RGB', (2, 2), 'blue').save(paths[1])
            (out / 'camera0-review.json').write_text('{')
            def finish_review(*args, **kwargs):
                (out / 'camera0-review.json').write_text(json.dumps(record))
            with patch('room3d_review_queue.subprocess.run', side_effect=finish_review) as inference:
                result = run_queue(room / 'review-queue.json', out, Path(sys.executable), cache)
                inference.assert_called_once()
                self.assertEqual(result[0]['status'], 'reviewed')
                self.assertFalse(result[0]['reused'])

    def test_cannot_write_reviews_into_room_export(self):
        with tempfile.TemporaryDirectory() as folder:
            room = Path(folder)
            enqueue(room, [])
            with self.assertRaises(ValueError):
                run_queue(room / 'review-queue.json', room / 'reviews', Path(sys.executable))


if __name__ == '__main__':
    unittest.main()
