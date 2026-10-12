"""Explicit stair phrases preserve detector provenance through refinement."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_stair_edges import load_refinements, refine_segments


class StairLabelTests(unittest.TestCase):
    def test_supported_phrases_are_refined_without_relabeling_ambiguous_categories(self):
        labels = ['staircase', 'carpeted staircase', 'wooden staircase', 'stone staircase',
                  'staircase railing', 'stair treads staircase', 'wooden floor', 'shelf']
        room = {'cameras': [{'index': 7}]}
        image = Image.new('RGB', (16, 16), 'gray')
        mask = np.ones((16, 16), bool)
        items = [{'label': label, 'mask': f'mask{i}.png', 'score': .4, 'predicted_iou': .7}
                 for i, label in enumerate(labels)]
        segments = [[(item, mask) for item in items]]
        depths = [(np.ones_like(mask, float), mask)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('room3d_stair_edges.flight_regions', return_value=[]) as search:
                result = refine_segments(room, [image], segments, depths, root)
            self.assertEqual(search.call_count, 4)
            self.assertEqual([item['label'] for item, _ in result[0]], labels)
            self.assertEqual([item['label'] for item in items], labels)
            records = json.loads((root / 'refinements.json').read_text())
            self.assertEqual([record['source_label'] for record in records], labels[:4])
            self.assertTrue(all(not record['accepted_proposal'] for record in records))
            loaded = load_refinements(root, room, [image], segments, depths)
            self.assertEqual([item['label'] for item, _ in loaded[0]], labels)
            changed = [[({**item, 'label': 'door'} if i == 1 else item, original)
                        for i, (item, original) in enumerate(segments[0])]]
            with self.assertRaisesRegex(ValueError, 'source label'):
                load_refinements(root, room, [image], changed, depths)


if __name__ == '__main__':
    unittest.main()
