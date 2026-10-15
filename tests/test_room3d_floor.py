"""Floor cuts preserve exterior area and leave no floor covering a stair opening."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_floor import area, floor_patches


class FloorTests(unittest.TestCase):
    def test_rotated_opening_preserves_area_and_winding_across_tiles(self):
        openings = [{'polygon': [[5, 2], [8, 5], [5, 8], [2, 5]], 'source_object': 'stairs'}]
        before = copy.deepcopy(openings)
        patches = floor_patches((0, 0, 10, 10), openings, tile_size=3)
        self.assertAlmostEqual(sum(abs(area(p)) for p in patches), 82)
        self.assertTrue(all(area(p) < 0 for p in patches))
        self.assertEqual(openings, before)
        for patch in patches:
            # Every residual patch must lie outside at least one side of the diamond.
            self.assertTrue(any(all(sign_x*(x-5)+sign_z*(z-5) >= 3-1e-7 for x,z in patch)
                                for sign_x,sign_z in [(1,1),(1,-1),(-1,1),(-1,-1)]))

    def test_overlapping_outside_and_repeated_openings(self):
        a = {'polygon': [[2,2],[6,2],[6,6],[2,6]]}
        b = {'polygon': [[4,4],[8,4],[8,8],[4,8]]}
        self.assertAlmostEqual(sum(abs(area(p)) for p in floor_patches((0,0,10,10),[a,b,a])),72)
        outside = {'polygon': [[20,20],[30,20],[30,30],[20,30]]}
        self.assertAlmostEqual(sum(abs(area(p)) for p in floor_patches((0,0,10,10),[outside])),100)

    def test_invalid_openings_fail_explicitly(self):
        for polygon in ([[0,3],[1,-1],[-3,1],[3,1],[-1,-1]], [[0,0],[1,1],[0,1],[1,0]], [[0,0],[0,0],[1,1]],
                        [[0,0],[2,0],[1,.5],[2,2],[0,2]]):
            with self.assertRaises(ValueError):
                floor_patches((0,0,10,10),[{'polygon':polygon}])


if __name__ == '__main__':
    unittest.main()
