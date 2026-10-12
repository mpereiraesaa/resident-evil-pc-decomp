"""Fast clipping must preserve polygon order and all interpolated attributes."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_preview import clip_near


class ClipTests(unittest.TestCase):
    def test_random_polygons_match_reference_including_boundary(self):
        rng = np.random.default_rng(17)
        for count in (0, 3, 4, 7):
            for _ in range(100):
                points = rng.uniform(-3, 3, (count, 8))
                if count:
                    points[0, 2] = 1  # Vertices on the near plane stay inside.
                before = points.copy()
                expected = []
                for a, b in zip(points, np.roll(points, -1, axis=0), strict=True):
                    if a[2] >= 1:
                        expected.append(a)
                    if (a[2] >= 1) != (b[2] >= 1):
                        expected.append(a + (b - a) * ((1 - a[2]) / (b[2] - a[2])))
                np.testing.assert_array_equal(clip_near(points, 1), expected)
                np.testing.assert_array_equal(points, before)


if __name__ == '__main__':
    unittest.main()
