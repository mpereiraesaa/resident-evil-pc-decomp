"""Category conflict abstention must not remove independent stair observations."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_semantics import filter_stair_conflicts
from room3d_shape_fit import fit_objects


class StairConflictTests(unittest.TestCase):
    def sample(self):
        mask = np.zeros((12, 12), bool)
        mask[4:8, 4:8] = True
        door = np.zeros_like(mask)
        door[2:10, 2:10] = True
        stair = {'label': 'staircase', 'mask': 'stairs.png', 'score': .4, 'predicted_iou': .9}
        item = {'label': 'door', 'mask': 'door.png', 'score': .7, 'predicted_iou': .95}
        return [(stair, mask), (item, door)]

    def test_contained_door_panel_abstains_without_mutation(self):
        entries = self.sample()
        before = copy.deepcopy(entries)
        kept, reasons = filter_stair_conflicts({'cameras': [{'index': 7}]}, [entries])
        self.assertEqual([d['label'] for d, _ in kept[0]], ['door'])
        self.assertEqual(reasons[0]['camera'], 7)
        self.assertEqual(reasons[0]['contained_fraction'], 1)
        for (a, am), (b, bm) in zip(entries, before, strict=True):
            self.assertEqual(a, b)
            np.testing.assert_array_equal(am, bm)

    def test_uncertain_door_does_not_override_stair(self):
        for change in ({'predicted_iou': .8}, {'score': .5}, {'label': 'wall lamp door'},
                       {'score': float('nan')}, {'predicted_iou': float('inf')}):
            entries = self.sample()
            entries[1][0].update(change)
            kept, reasons = filter_stair_conflicts({'cameras': [{'index': 0}]}, [entries])
            self.assertEqual(len(kept[0]), 2)
            self.assertEqual(reasons, [])

    def test_partial_overlap_and_railing_label_are_preserved(self):
        for railing in (False, True):
            entries = self.sample()
            if railing:
                entries[0][0]['label'] = 'staircase railing'
            else:
                entries[1][1][6:] = False
            kept, reasons = filter_stair_conflicts({'cameras': [{'index': 0}]}, [entries])
            self.assertEqual(len(kept[0]), 2)
            self.assertEqual(reasons, [])

    def test_pipeline_reports_abstention_without_fitting_false_stair(self):
        room = {'cameras': [{'index': 0, 'origin': [0, 0, 0],
                            'target': [0, 0, 1000], 'fov': 100, 'roll': 0}]}
        with tempfile.TemporaryDirectory() as directory:
            recipe, report = fit_objects(
                room, {'primitives': [], 'semantic_stair_guard': True}, [self.sample()],
                [(np.full((12, 12), 1000.), np.ones((12, 12), bool))],
                Path(directory), None,
            )
        self.assertEqual(recipe['primitives'], [])
        self.assertEqual(report['objects'], [])
        self.assertEqual(len(report['semantic_rejections']), 1)


if __name__ == '__main__':
    unittest.main()
