"""Cached depths must be invalidated when geometry, cutaways or cameras change."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_reconstruct import BakeDepthCache


class BakeCacheTests(unittest.TestCase):
    def test_reuse_and_invalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            mesh = Path(folder) / 'room.obj'
            mesh.write_text('first geometry')
            camera = {'index': 0, 'origin': np.array([1., 2., 3.]), 'fov': 240}
            cache = BakeDepthCache()
            with patch('room3d_reconstruct.render', side_effect=lambda *a, **k: (None, None, np.ones((2, 2)))) as renderer:
                initial = cache.render_depth(mesh, camera)
                self.assertIs(cache.render_depth(mesh, dict(camera)), initial)
                self.assertEqual(renderer.call_count, 1)
                with self.assertRaises(ValueError):
                    initial[0, 0] = 4
                cache.render_depth(mesh, camera, {'floor'})
                self.assertEqual(renderer.call_count, 2)
                cache.render_depth(mesh, {**camera, 'fov': 250})
                self.assertEqual(renderer.call_count, 3)
                mesh.with_suffix('.cameras').write_text('visibility change')
                cache.render_depth(mesh, camera)
                self.assertEqual(renderer.call_count, 4)
                mesh.write_text('other geometry')
                cache.render_depth(mesh, camera)
                self.assertEqual(renderer.call_count, 5)
                mesh.with_suffix('.cameras').unlink()
                cache.render_depth(mesh, camera)
                self.assertEqual(renderer.call_count, 6)


if __name__ == '__main__':
    unittest.main()
