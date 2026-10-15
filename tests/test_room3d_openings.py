"""Script door panels must not be buried behind reconstructed visual walls."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_structure import door_openings


class DoorOpeningTests(unittest.TestCase):
    def test_subtracts_only_panel_rectangle_preserving_area_winding_and_inputs(self):
        surface = {'points': [[0, 0, 0], [0, -4000, 0], [0, -4000, 5000], [0, 0, 5000]],
                   'group': 'boundary_1', 'material': 'wall'}
        door = {'kind': 'door_panel', 'box': [20, 1500, 40, 3500, -2500, 0],
                'hide_with': 'boundary_1', 'source_offset': 123}
        original = copy.deepcopy(surface)
        result = door_openings([surface], [door])
        self.assertEqual(len(result), 3)
        area = 0
        for patch in result:
            p = np.array(patch['points'])
            normal = np.cross(p[1]-p[0], p[2]-p[0])
            self.assertLess(normal[0], 0)
            area += np.linalg.norm(normal)
            self.assertFalse(p[:, 2].min() < 3500 and p[:, 2].max() > 1500
                             and p[:, 1].max() > -2500)
        self.assertEqual(area, 4000*5000-2000*2500)
        self.assertEqual(surface, original)
        self.assertEqual(door_openings(result, [door]), result)
        for changed in ({**door, 'source_offset': None, 'hide_with': 'boundary_2'},
                        {k: v for k, v in door.items() if k != 'source_offset'},
                        {**door, 'box': [300, 1500, 340, 3500, -2500, 0]}):
            self.assertEqual(door_openings([surface], [changed]), [surface])

    def test_only_nearest_parallel_surface_is_opened(self):
        near = {'points': [[0, 0, 0], [0, -4000, 0], [0, -4000, 5000], [0, 0, 5000]],
                'group': 'boundary_1', 'material': 'wall'}
        far = {**near, 'points': (np.array(near['points'])+[90, 0, 0]).tolist()}
        door = {'kind': 'door_panel', 'box': [20, 1500, 40, 3500, -2500, 0],
                'hide_with': 'boundary_1', 'source_offset': 123}
        result = door_openings([near, far], [door])
        self.assertIn(far, result)
        self.assertNotIn(near, result)


if __name__ == '__main__':
    unittest.main()
