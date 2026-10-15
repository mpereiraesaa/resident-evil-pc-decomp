"""Attachment proposals require geometry and independent rays, never room IDs."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_enclosure import identify_wall_attachment


def fixture(scale=1, rotated=False):
    bounds = [{'index': 17, 'xmin': 0, 'xmax': 200, 'zmin': 0, 'zmax': 1000},
              {'index': 42, 'xmin': 0, 'xmax': 300, 'zmin': 300, 'zmax': 600}]
    points = np.array([[150, 0, 200], [150, -300, 200], [150, -300, 800], [150, 0, 800]], float)
    for b in bounds:
        for k in ['xmin', 'xmax', 'zmin', 'zmax']:
            b[k] *= scale
        if rotated:
            b['xmin'], b['zmin'] = b['zmin'], b['xmin']
            b['xmax'], b['zmax'] = b['zmax'], b['xmax']
    points *= scale
    if rotated:
        points = points[:, [2, 1, 0]]
    plane = {'normal': [0, 1] if rotated else [1, 0], 'points': points.tolist()}
    records = [{'camera': view, 'target': [1, n, 3],
                'occluders': [{'group': 'boundary_17'}, {'group': 'boundary_42'}]}
               for n in range(3) for view in (0, 1)]
    return bounds, plane, records


class AttachmentTests(unittest.TestCase):
    def test_axis_scale_and_ids_do_not_change_topology(self):
        for scale in (.01, 1, 100):
            for rotated in (False, True):
                bounds, plane, records = fixture(scale, rotated)
                before = copy.deepcopy(bounds)
                result = identify_wall_attachment(bounds, plane, records)
                self.assertEqual(result['status'], 'unique_hypothesis')
                self.assertEqual(result['candidates'][0]['backing'], 17)
                self.assertEqual(result['candidates'][0]['attachment'], 42)
                self.assertEqual(bounds, before)

    def test_missing_views_and_duplicate_points_cannot_supply_evidence(self):
        bounds, plane, records = fixture()
        self.assertEqual(identify_wall_attachment(bounds, plane, records[:1] * 20)['status'], 'no_hypothesis')
        one_view = [r for r in records if r['camera'] == 0]
        self.assertEqual(identify_wall_attachment(bounds, plane, one_view)['status'], 'no_hypothesis')

    def test_multiple_plausible_attachments_remain_ambiguous(self):
        bounds, plane, records = fixture()
        bounds.append({**bounds[1], 'index': 99})
        for record in records:
            record['occluders'].append({'group': 'boundary_99'})
        result = identify_wall_attachment(bounds, plane, records)
        self.assertEqual(result['status'], 'ambiguous')
        self.assertEqual(len(result['candidates']), 2)


if __name__ == '__main__':
    unittest.main()
