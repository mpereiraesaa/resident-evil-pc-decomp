"""Rendered tests for shared surfaces outside invisible collision footprints."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from batch_room3d import generate_mesh
from room3d_preview import camera_basis, render
from room3d_visual_surfaces import propose, recover


class ReservedEntries:
    def __iter__(self):
        raise AssertionError('Reserved semantic masks must not fit geometry')


class VisualSurfaceTests(unittest.TestCase):
    def test_shared_geometry_reserved_independence_acceptance_and_rollback(self):
        cameras = [{'index': i, 'origin': origin, 'target': [20000, -1500, 15000],
                    'fov': 230, 'roll': 0} for i, origin in enumerate(
                        ([10000, -2000, 15000], [12000, -3000, 17000], [14000, -2000, 14000]))]
        room = {'cameras': cameras, 'boundaries': [{'index': 0, 'xmin': 0, 'xmax': 36,
                'zmin': 0, 'zmax': 18, 'type': 1, 'flags': 256}]}
        recipe = {'primitives': [], 'wall_height': 4000}
        truth = {**recipe, 'skip_boundaries': [0], 'structural_surfaces': [
            {'group': 'test_wall', 'material': 'wall', 'points':
             [[20000, 0, 8000], [20000, -4000, 8000], [20000, -4000, 22000], [20000, 0, 22000]]},
            {'group': 'test_floor', 'material': 'floor', 'points':
             [[10000, 0, 8000], [10000, 0, 22000], [20000, 0, 22000], [20000, 0, 8000]]}]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mesh = root/'truth.obj'
            generate_mesh(room, mesh, truth)
            depths, segments = [], []
            for c in cameras:
                scene = render(mesh, c, return_depth=True)[2]
                valid = np.isfinite(scene)
                depths.append((np.where(valid, scene, 50000), valid))
                entries = []
                for kind in ('floor', 'wall'):
                    part = render(mesh, c, return_depth=True, only_groups={f'test_{kind}'})[2]
                    mask = np.isfinite(part) & (part <= scene+50)
                    self.assertGreater(mask.sum(), 1200)
                    entries.append(({'label': kind, 'score': .9, 'predicted_iou': .95}, mask))
                segments.append(entries)
            snapshot = copy.deepcopy(recipe)
            saved_depths = [(d.copy(), v.copy()) for d, v in depths]
            # Deliberately inaccessible heldout masks prove they never choose
            # wall planes, floor height, or spatial extents.
            segments[2] = ReservedEntries()
            candidate, proposed = propose(room, recipe, depths, segments)
            self.assertEqual(proposed['status'], 'proposed')
            self.assertEqual(proposed['heldout_cameras'], [2])
            self.assertGreaterEqual(proposed['shared_walls'], 1)
            result, report = recover(room, recipe, depths, segments, root/'accepted', generate_mesh)
            self.assertTrue(report['accepted'])
            self.assertEqual(result, candidate)
            self.assertEqual(json.loads(json.dumps(report)), report)
            self.assertEqual(json.loads(json.dumps(candidate)), candidate)
            self.assertTrue(all(v['geometry_pixels'] > 150 for v in report['views']))
            self.assertEqual(recipe, snapshot)
            for (d, v), (old_d, old_v) in zip(depths, saved_depths, strict=True):
                np.testing.assert_array_equal(d, old_d)
                np.testing.assert_array_equal(v, old_v)
            # Disagreeing monocular heights may retain the existing editing
            # plane only when both training estimates admit that same prior.
            for offset, expected in ((250, 'proposed'), (1000, 'inconsistent_training_floor')):
                biased = [(d.copy(), v.copy()) for d, v in depths]
                for i, camera in enumerate(cameras[:2]):
                    y, x = np.indices((240, 320))
                    ray_y = (np.stack(((x-160)/camera['fov'], (y-120)/camera['fov'],
                                      np.ones_like(x)), -1) @ camera_basis(camera))[..., 1]
                    mask = segments[i][0][1]
                    height = offset if i else -offset
                    biased[i][0][mask] = (height-camera['origin'][1])/ray_y[mask]
                _, biased_report = propose(room, recipe, biased, segments)
                self.assertEqual(biased_report['status'], expected)
                if expected == 'proposed':
                    self.assertEqual(biased_report['floor']['policy'], 'estimated_existing_floor_prior')
                    self.assertEqual(biased_report['floor_y'], 0)
            # A heldout camera looking away vetoes the same training proposal;
            # it must not cause the fitter to extend surfaces toward that view.
            adverse = copy.deepcopy(room)
            adverse['cameras'][2]['target'] = [0, -10000, 14000]
            same, _ = propose(adverse, recipe, depths, segments)
            self.assertEqual(same, candidate)
            rolled_back, rejected = recover(adverse, recipe, depths, segments,
                                           root/'rejected', generate_mesh)
            self.assertIs(rolled_back, recipe)
            self.assertFalse(rejected['accepted'])
            self.assertEqual(rejected['reason'], 'reserved_depth_or_visibility_failed')
            visible, untouched = recover(room, truth, depths, segments,
                                         root/'already-visible', generate_mesh)
            self.assertIs(visible, truth)
            self.assertEqual(untouched['reason'], 'existing_visible_geometry')
            abstained, insufficient = propose(room, recipe, depths, [[], [], ReservedEntries()])
            self.assertIs(abstained, recipe)
            self.assertEqual(insufficient['status'], 'insufficient_shared_evidence')


if __name__ == '__main__':
    unittest.main()
