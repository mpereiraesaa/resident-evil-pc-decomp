"""Projective layout priors must remain explicit and allow rotated flights."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_preview import camera_basis
from room3d_stair_edges import horizontal_axis_edges


class TreadAxisTests(unittest.TestCase):
    def setUp(self):
        self.camera = {'origin': [300, -1700, -2500], 'target': [0, -500, 1000],
                       'fov': 230, 'roll': 0}

    def project(self, endpoints):
        view = (np.asarray(endpoints) - self.camera['origin']) @ camera_basis(self.camera).T
        return view[:, :2] / view[:, 2:] * self.camera['fov'] + [160, 120]

    def test_layout_rejects_diagonal_but_explicit_rotated_axis_accepts_it(self):
        lines = np.array([
            self.project([[-500, -300, 1200], [500, -300, 1200]]),
            self.project([[0, -700, 800], [0, -700, 1800]]),
            self.project([[-400, -500, 800], [400, -500, 1600]]),
        ])
        self.assertEqual(horizontal_axis_edges(lines, self.camera, [[1, 0], [0, 1]]).tolist(),
                         [True, True, False])
        self.assertEqual(horizontal_axis_edges(lines, self.camera, [[1, 1]]).tolist(),
                         [False, False, True])
        self.assertEqual(horizontal_axis_edges(lines[:, ::-1], self.camera,
                                              [[-5, 0], [0, -2]]).tolist(),
                         [True, True, False])

    def test_scale_position_height_and_endpoint_order_do_not_change_axis(self):
        lines = []
        for y, z, width in [(-800, 1200, 400), (-300, 2100, 800), (0, 3000, 1600)]:
            lines.append(self.project([[-width/2, y, z], [width/2, y, z]]))
        self.assertTrue(horizontal_axis_edges(lines, self.camera, [[1, 0]]).all())

    def test_degenerate_lines_abstain_and_bad_priors_raise(self):
        self.assertFalse(horizontal_axis_edges([[[100, 100], [100, 100]]],
                                               self.camera, [[1, 0]])[0])
        self.assertEqual(len(horizontal_axis_edges(np.empty((0, 2, 2)),
                                                  self.camera, [[1, 0]])), 0)
        for axes in [[], [1, 0], [[0, 0]], [[1, np.nan]], [[1, 0, 0]]]:
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                horizontal_axis_edges([[[0, 0], [1, 1]]], self.camera, axes)

    def test_refinement_cache_binds_prior_and_preserves_original_confidence(self):
        from PIL import Image
        from room3d_stair_edges import load_refinements, refine_segments

        camera = {**self.camera, 'index': 0}
        room = {'cameras': [camera]}
        image = Image.new('RGB', (320, 240))
        mask = np.ones((240, 320), bool)
        depth = np.ones(mask.shape)
        detection = {'label': 'staircase', 'mask': 'source.png',
                     'score': .4, 'predicted_iou': .7}
        segments = [[(detection, mask)]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('room3d_stair_edges.flight_regions', return_value=[]) as search:
                refined = refine_segments(room, [image], segments, [(depth, mask)],
                                          root, horizontal_axes=[[2, 0], [0, 3]])
            self.assertEqual(search.call_args.kwargs['horizontal_axes'], [[1., 0.], [0., 1.]])
            self.assertIs(refined[0][0][0], detection)
            self.assertIs(refined[0][0][1], mask)
            self.assertEqual(detection['predicted_iou'], .7)
            self.assertTrue(mask.all())
            record = json.loads((root / 'refinements.json').read_text())[0]
            self.assertFalse(record['accepted_proposal'])
            self.assertEqual(record['horizontal_axes'], [[1., 0.], [0., 1.]])
            loaded = load_refinements(root, room, [image], segments, [(depth, mask)],
                                      horizontal_axes=[[1, 0], [0, 1]])
            self.assertIs(loaded[0][0][1], mask)
            for axes in (None, [[1, 1]]):
                with self.subTest(axes=axes), self.assertRaisesRegex(ValueError, 'prior differs'):
                    load_refinements(root, room, [image], segments, [(depth, mask)],
                                     horizontal_axes=axes)
            # CLI/recipe forwarding may normalize a diagonal axis twice. Only
            # roundoff is tolerated, not a different physical layout prior.
            diagonal = np.array([[1., 1.]]) / np.sqrt(2)
            record['horizontal_axes'] = diagonal.tolist()
            (root / 'refinements.json').write_text(json.dumps([record]))
            load_refinements(root, room, [image], segments, [(depth, mask)],
                             horizontal_axes=diagonal)


if __name__ == '__main__':
    unittest.main()
