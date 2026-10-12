"""Actual raster fitting of rung counts with a reserved-image veto."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from batch_room3d import generate_mesh
from room3d_ladder_rungs import refine_count
from room3d_preview import render
from room3d_shapes import generate


class LadderRungTests(unittest.TestCase):
    def scene(self, root, noisy=False):
        room = {'boundaries': [{'index': 0, 'xmin': -2000, 'xmax': 2000,
                               'zmin': -100, 'zmax': 0, 'type': 1, 'flags': 256}],
                'cameras': [{'index': i, 'origin': origin, 'target': [0, -1800, 180],
                             'roll': 0, 'fov': 230} for i, origin in enumerate(
                                 ([800, -1800, 6000], [-1200, -1800, 5000], [1000, -1600, 4500]))]}
        p = {'width': 800, 'height': 3600, 'thickness': 50, 'stand_off': 180, 'rungs': 9}
        truth, baseline = generate('ladder', p), generate('ladder', {**p, 'rungs': 17})
        mesh = root/'truth.obj'
        generate_mesh(room, mesh, {'primitives': [truth]})
        observations, images = [], {}
        for c in room['cameras']:
            mask = render(mesh, c, only_groups={'primitive_0'})[1]
            image = (np.random.default_rng(42+c['index']).integers(120, 220, (240, 320, 1), dtype='uint8').repeat(3, axis=2)
                     if noisy else np.full((240, 320, 3), 180, 'uint8'))
            image[mask] = 30
            envelope = np.zeros(mask.shape, 'uint8')
            y, x = np.nonzero(mask)
            cv2.fillConvexPoly(envelope, cv2.convexHull(np.column_stack((x, y)).astype('int32')), 1)
            observations.append({'camera': c['index'], 'camera_model': c, 'mask': envelope.astype(bool)})
            images[c['index']] = Image.fromarray(image)
        return room, baseline, observations, images

    def test_rendered_training_edges_recover_count_and_preserve_pose(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            room, baseline, observations, images = self.scene(root)
            saved = copy.deepcopy(baseline)
            masks = [o['mask'].copy() for o in observations]
            pixels = {i: im.tobytes() for i, im in images.items()}
            selected, report = refine_count(room, baseline, observations, images,
                                            root/'fit', generate_mesh, counts=(6, 8, 9, 10, 12))
            self.assertEqual(report['status'], 'selected', report)
            self.assertEqual(selected['shape']['parameters']['rungs'], 9)
            self.assertEqual(report['training_cameras'], [0, 1])
            self.assertEqual(report['heldout_cameras'], [2])
            self.assertEqual(baseline, saved)
            for key in ('origin', 'yaw'):
                self.assertEqual(selected['shape'][key], baseline['shape'][key])
            for key in ('width', 'height', 'thickness', 'stand_off'):
                self.assertEqual(selected['shape']['parameters'][key], baseline['shape']['parameters'][key])
            np.testing.assert_array_equal(selected['parts'][:2], baseline['parts'][:2])
            np.testing.assert_array_equal(selected['parts'][-4:], baseline['parts'][-4:])
            for o, m in zip(observations, masks, strict=True):
                np.testing.assert_array_equal(o['mask'], m)
            self.assertEqual(pixels, {i: im.tobytes() for i, im in images.items()})

    def test_reserved_image_cannot_reselect_training_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            room, baseline, observations, images = self.scene(root)
            _, good = refine_count(room, baseline, observations, images, root/'good', generate_mesh, counts=(8, 9, 12))
            images[2] = Image.new('RGB', (320, 240), 'gray')
            result, veto = refine_count(room, baseline, observations, images, root/'veto', generate_mesh, counts=(8, 9, 12))
            self.assertIs(result, baseline)
            self.assertEqual(veto['status'], 'reserved_edges_unavailable')
            self.assertEqual(good['trials'], veto['trials'])
            self.assertEqual(good['selected_training_rungs'], veto['selected_training_rungs'])

    def test_background_grain_does_not_favor_dense_rungs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            room, baseline, observations, images = self.scene(root, noisy=True)
            result, report = refine_count(room, baseline, observations, images,
                                          root/'fit', generate_mesh, counts=(9, 17, 24, 30))
            if report['status'] == 'selected':
                self.assertEqual(result['shape']['parameters']['rungs'], 9)
            else:
                # Strong grain may erase coherent evidence. Exact rollback is
                # acceptable; selecting dense rungs to match noise is not.
                self.assertIs(result, baseline)
                self.assertEqual(result['shape']['parameters']['rungs'], 17)

    def test_flat_training_images_abstain_without_modifying_ladder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            room, baseline, observations, images = self.scene(root)
            images[0] = Image.new('RGB', (320, 240), 'gray')
            result, report = refine_count(room, baseline, observations, images, root/'fit', generate_mesh)
            self.assertIs(result, baseline)
            self.assertEqual(report['status'], 'insufficient_training_edges')
            self.assertFalse((root/'fit').exists())


if __name__ == '__main__':
    unittest.main()
