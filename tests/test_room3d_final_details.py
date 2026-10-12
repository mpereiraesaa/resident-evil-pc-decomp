"""Final metrics must follow exported triangles and visibility, not stale fit shapes."""

import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from batch_room3d import generate_mesh
from room3d_final_details import measure
from room3d_preview import render
from room3d_shapes import generate


class FinalDetailsTests(unittest.TestCase):
    def test_actual_group_geometry_and_camera_cutaway_control_measurements(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            camera = {'index': 7, 'origin': [0, -2000, -5000],
                      'target': [0, -700, 0], 'fov': 230, 'roll': 0}
            room = {'cameras': [camera], 'boundaries': [
                {'index': 0, 'xmin': -4000, 'xmax': 4000, 'zmin': -2000,
                 'zmax': 4000, 'type': 1, 'flags': 0}]}
            obj = generate('staircase', {'width': 1500, 'run': 2500, 'rise': 1500,
                                       'steps': 6, 'landing': 0}, [0, 0, 0], 0)
            obj['evidence'] = {'shared_observations': [{'camera': 7, 'mask': 'stairs.png'}]}
            recipe = {'skip_boundaries': [0], 'primitives': [obj]}
            path = root/'final.obj'
            generate_mesh(room, path, recipe)
            part = render(path, camera, return_depth=True, only_groups={'primitive_0'})[2]
            mask = np.isfinite(part)
            self.assertGreater(mask.sum(), 100)
            segments = [[({'mask': 'stairs.png'}, mask)]]
            target = np.where(mask, part, 10000.)
            depths = [(target, np.ones_like(mask))]
            original_mask, original_depth = mask.copy(), target.copy()
            mesh_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            # Deliberately stale proposal coordinates must not supply rendered metrics.
            stale_recipe = copy.deepcopy(recipe)
            stale_recipe['primitives'][0]['parts'] = []
            measured = measure(room, stale_recipe, segments, depths, path)
            self.assertEqual(measured['mesh_sha256'], mesh_hash)
            self.assertEqual(measured['objects'][0]['views'][0]['scores']['silhouette_iou'], 1)
            path.with_suffix('.cameras').write_text('hide 7 primitive_0\n')
            hidden = measure(room, recipe, segments, depths, path)
            self.assertEqual(hidden['objects'][0]['views'][0]['scores']['silhouette_iou'], 0)
            self.assertNotEqual(measured['visibility_sha256'], hidden['visibility_sha256'])
            segments[0] = []
            missing = measure(room, recipe, segments, depths, path)['objects'][0]
            self.assertEqual(missing['status'], 'missing_observations')
            self.assertEqual(missing['missing_observations'], [{'camera': 7, 'mask': 'stairs.png'}])
            self.assertEqual(missing['views'], [])
            np.testing.assert_array_equal(mask, original_mask)
            np.testing.assert_array_equal(target, original_depth)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), mesh_hash)


if __name__ == '__main__':
    unittest.main()
