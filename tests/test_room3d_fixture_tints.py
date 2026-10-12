"""Object-local colors must not leak across fixtures or heldout cameras."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_complete import fixture_tints


class FixtureTintTests(unittest.TestCase):
    def test_bake_preserves_parts_and_excludes_reserved_palette(self):
        from room3d_reconstruct import bake_projected

        camera = {'index': 0, 'origin': [0, 0, 0], 'target': [0, 0, 1000],
                  'roll': 0, 'fov': 100}
        image = np.full((240, 320, 3), 30, dtype=np.uint8)
        image[:60] = 250
        mask = np.ones((240, 320), dtype=bool)
        item = {'mask': 'lamp.png', 'label': 'lamp', 'score': .8, 'predicted_iou': .9}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / 'shape.obj'
            mesh.write_text('g lamp\nusemtl fixture_shade\n'
                            'v -50 -50 1000\nv 50 -50 1000\nv 0 50 1000\nf 1 2 3\n'
                            'usemtl fixture_support\nf 3 2 1\n')
            mesh.with_suffix('.cameras').write_text('')
            for excluded in (None, 0):
                out = root / f'bake-{excluded}.obj'
                report = bake_projected(
                    mesh, {'cameras': [camera]}, [Image.fromarray(image)], out,
                    segments=[[(item, mask)]], group_kinds={'lamp': 'lamp'},
                    interpreted_groups={'lamp'}, exclude_camera=excluded,
                    fixture_observations={'lamp': [{'camera': 0, 'mask': 'lamp.png'}]},
                )
                contents = out.read_text()
                if excluded is None:
                    self.assertIn('usemtl fixture_tint_0_shade', contents)
                    self.assertIn('usemtl fixture_tint_0_support', contents)
                    roles = report['fixture_tints']['lamp']['roles']
                    self.assertGreater(roles['shade'][0], .9)
                    self.assertLess(roles['support'][0], .15)
                else:
                    self.assertEqual(report['fixture_tints'], {})
                    self.assertNotIn('usemtl fixture_tint', contents)
                self.assertNotIn('usemtl camera', contents)

    def test_observed_bright_shade_and_dark_support(self):
        room = {'cameras': [{'index': 4}]}
        pixels = np.full((8, 8, 3), [30, 40, 50], dtype=np.uint8)
        pixels[:2] = [245, 250, 255]
        # A brighter unrelated region outside the exact object mask is excluded.
        pixels[:, 7] = [255, 0, 0]
        mask = np.ones((8, 8), dtype=bool)
        mask[:, 7] = False
        item = {'mask': 'a.png', 'label': 'wall lamp', 'predicted_iou': .9, 'score': .4}
        obs = {'primitive_0': [{'camera': 4, 'mask': 'a.png'}]}
        result = fixture_tints(room, [Image.fromarray(pixels)], [[(item, mask)]], obs)
        roles = result['primitive_0']['roles']
        np.testing.assert_allclose(roles['support'], np.array([30, 40, 50]) / 255)
        np.testing.assert_allclose(roles['shade'], np.array([245, 250, 255]) / 255)

    def test_exact_object_and_reserved_camera(self):
        room = {'cameras': [{'index': 5}, {'index': 9}]}
        images = [Image.new('RGB', (8, 8), color) for color in ['red', 'blue']]
        mask = np.ones((8, 8), dtype=bool)
        item = {'mask': 'lamp.png', 'label': 'wall lamp', 'predicted_iou': .9, 'score': .35}
        segments = [[(item, mask)], [(item, mask)]]
        observations = {'primitive_0': [{'camera': 5, 'mask': 'lamp.png'}],
                        'primitive_1': [{'camera': 9, 'mask': 'lamp.png'}]}
        result = fixture_tints(room, images, segments, observations)
        self.assertEqual(result['primitive_0']['rgb'], [1, 0, 0])
        self.assertEqual(result['primitive_0']['roles'],
                         {'support': [1, 0, 0], 'shade': [1, 0, 0]})
        self.assertEqual(result['primitive_1']['rgb'], [0, 0, 1])
        withheld = fixture_tints(room, images, segments, observations, exclude_camera=0)
        self.assertNotIn('primitive_0', withheld)
        self.assertEqual(withheld['primitive_1']['rgb'], [0, 0, 1])
        wrong = {'primitive_0': [{'camera': 5, 'mask': 'different.png'}]}
        self.assertEqual(fixture_tints(room, images, segments, wrong), {})

    def test_no_unreliable_or_tiny_mask_samples(self):
        room = {'cameras': [{'index': 0}]}
        images = [Image.new('RGB', (4, 4), 'white')]
        item = {'mask': 'a.png', 'label': 'wall lamp', 'predicted_iou': .9, 'score': .4}
        obs = {'primitive_0': [{'camera': 0, 'mask': 'a.png'}]}
        mask = np.ones((4, 4), dtype=bool)
        for bad in ({**item, 'predicted_iou': .6}, {**item, 'label': 'wall'}):
            self.assertEqual(fixture_tints(room, images, [[(bad, mask)]], obs), {})
        mask[0, 0] = False
        self.assertEqual(fixture_tints(room, images, [[(item, mask)]], obs), {})


if __name__ == '__main__':
    unittest.main()
