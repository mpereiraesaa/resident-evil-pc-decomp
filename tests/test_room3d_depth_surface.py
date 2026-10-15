"""Exercise actual draft geometry, projection and discontinuity handling."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_depth_surface import build_surface
from room3d_preview import camera_basis, load_obj, render


class DepthSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.camera = {'index': 0, 'origin': [100, -300, 200],
                       'target': [600, 100, 4200], 'fov': 230, 'roll': 0}
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        yy, xx = np.mgrid[:240, :320]
        self.image = Image.fromarray(np.stack([xx % 256, yy, (xx + yy) % 256], -1).astype('uint8'))

    def build(self, depth, reliable=None, name='draft', **kwargs):
        out = Path(self.folder.name) / name
        report = build_surface(self.camera, depth,
                               np.ones((240, 320), bool) if reliable is None else reliable,
                               self.image, out, step=16, **kwargs)
        return out, report

    def test_plane_reprojects_texture_and_depth_in_rotated_camera(self):
        out, report = self.build(np.full((240, 320), 3000.))
        preview, visible, depth = render(out / 'draft.obj', self.camera, return_depth=True)
        self.assertTrue(visible.all())
        np.testing.assert_allclose(depth, 3000, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(preview), np.asarray(self.image))
        self.assertEqual(report['hidden_geometry'], 'not_reconstructed')
        self.assertEqual(report['navigation'], 'not_validated')
        self.assertEqual(report['uncertain_faces'], 0)
        self.assertIn('reliable_proposal', (out / 'draft.obj').read_text())
        with self.assertRaises(FileExistsError):
            self.build(np.full((240, 320), 3000.))

    def test_depth_step_is_not_bridged_and_confidence_is_not_geometry_truth(self):
        depth = np.full((240, 320), 2000.)
        depth[:, 160:] = 4000
        reliable = np.ones((240, 320), bool)
        reliable[:, 160:] = False
        out, report = self.build(depth, reliable)
        self.assertGreater(report['rejected_discontinuity_faces'], 0)
        self.assertGreater(report['uncertain_faces'], 0)
        triangles, _ = load_obj(out / 'draft.obj', 0)
        basis = camera_basis(self.camera)
        for triangle, _ in triangles:
            z = ((triangle[:, :3] - self.camera['origin']) @ basis.T)[:, 2]
            self.assertLess(z.max() / z.min() - 1, .15)
        # A displaced camera exposes missing coverage; this is not a closed room.
        moved = {**self.camera, 'origin': (np.asarray(self.camera['origin']) + basis[0] * 400).tolist()}
        _, mask = render(out / 'draft.obj', moved)
        self.assertLess(mask.mean(), .95)

    def test_invalid_inputs_leave_no_output(self):
        for name, depth, kwargs in (
            ('nan', np.full((240, 320), np.nan), {}),
            ('zero', np.zeros((240, 320)), {}),
            ('jump', np.ones((240, 320)), {'maximum_jump': .8}),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.build(depth, name=name, **kwargs)
            self.assertFalse((Path(self.folder.name) / name).exists())


if __name__ == '__main__':
    unittest.main()
