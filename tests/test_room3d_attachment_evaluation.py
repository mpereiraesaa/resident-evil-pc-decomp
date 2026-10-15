"""A better average cannot hide a camera regression or missing evidence."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_attachment_evaluation import rank_alternatives


def view(camera, error, pixels=100):
    return {'camera': camera, 'pixels': pixels, 'rgb_mae': error, 'edge_distance': 1., 'uncovered': 0}


class RankingTests(unittest.TestCase):
    def test_rejects_average_win_with_single_camera_regression(self):
        report = rank_alternatives({'baseline': [view(0,.2), view(1,.2)],
                                   'bad': [view(0,.05), view(1,.22)],
                                   'good': [view(0,.18), view(1,.19)]})
        self.assertEqual(report['recommended_candidate'], 'good')
        self.assertIn('camera1_regression', report['decisions'][0]['reasons'])

    def test_no_changed_pixels_cannot_win(self):
        report = rank_alternatives({'baseline': [view(0,.2,0)], 'empty': [view(0,0,0)]})
        self.assertIsNone(report['recommended_candidate'])

    def test_nonfinite_scores_cannot_be_recommended(self):
        with self.assertRaises(ValueError):
            rank_alternatives({'baseline': [view(0,.2)], 'invalid': [view(0,float('nan'))]})

    def test_new_holes_fail_even_with_better_color(self):
        candidate = {**view(0,.1), 'uncovered': 1}
        report = rank_alternatives({'baseline': [view(0,.2)], 'hole': [candidate]})
        self.assertIsNone(report['recommended_candidate'])


if __name__ == '__main__':
    unittest.main()
