"""Handrail geometry and actual multiview fit, including reserved-point isolation."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from batch_room3d import generate_mesh
from room3d_preview import camera_basis, render
from room3d_railings import handrail_seed, surface_handrail_seed
from room3d_shape_fit import fit_group
from room3d_shapes import generate


class ReservedPoints:
    def __array__(self, *args, **kwargs):
        raise AssertionError('Reserved depth points must not initialize a family')


class RailingTests(unittest.TestCase):
    def test_open_baluster_control_keeps_original_family(self):
        cameras = [{'index': i, 'origin': origin, 'target': [0, -500, 1500],
                    'fov': 230, 'roll': 0} for i, origin in enumerate(
                        ([0, -1300, -4000], [2500, -1400, -2500]))]
        room = {'cameras': cameras, 'boundaries': [{'index': 0, 'xmin': -6000, 'xmax': 6000,
                'zmin': -5000, 'zmax': 6000, 'type': 4, 'flags': 0}]}
        true = generate('railing', {'run': 3000, 'rise': 0, 'height': 900,
                                   'thickness': 80, 'posts': 8})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mesh = root/'true.obj'
            generate_mesh(room, mesh, {'primitives': [true]})
            groups, depths = [], []
            for c in cameras:
                part = render(mesh, c, return_depth=True, only_groups={'primitive_0'})[2]
                y, x = np.nonzero(np.isfinite(part))
                z = part[y, x]
                points = np.column_stack(((x-160)*z/c['fov'], (y-120)*z/c['fov'], z))
                groups.append({'kind': 'railing', 'camera': c['index'], 'mask': np.isfinite(part),
                               'points': points @ camera_basis(c)+c['origin'], 'camera_model': c})
                target = render(mesh, c, return_depth=True)[2]
                depths.append((np.where(np.isfinite(target), target, 20000), np.isfinite(target)))
            old, _ = fit_group(room, {'primitives': []}, groups, depths, root/'off', generate_mesh)
            new, _ = fit_group(room, {'primitives': [], 'handrail_candidates': True}, groups,
                               depths, root/'on', generate_mesh)
            self.assertEqual(old['shape'], new['shape'])
            self.assertEqual(new['shape']['parameters'].get('profile', 'balusters'), 'balusters')

    def test_visible_corner_has_multiple_rail_segments(self):
        cameras = [{'index': i, 'origin': origin, 'target': [500, -900, 3500],
                    'fov': 230, 'roll': 0} for i, origin in enumerate(
                        ([2000, -1500, 0], [3000, -1600, 1000]))]
        room = {'cameras': cameras, 'boundaries': [
            {'index': 0, 'xmin': -100, 'xmax': 0, 'zmin': 0, 'zmax': 4100, 'type': 1, 'flags': 256},
            {'index': 1, 'xmin': 0, 'xmax': 4000, 'zmin': 4000, 'zmax': 4100, 'type': 1, 'flags': 256}]}
        recipe = {'primitives': [], 'wall_height': 3000}
        rail = generate('railing', {'profile': 'handrail_segments', 'thickness': 70,
                        'segments': [[[35, -900, 500], [35, -900, 3965]],
                                     [[35, -900, 3965], [3500, -900, 3965]]]})
        self.assertEqual(len(rail['parts']), 2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene, truth = root/'scene.obj', root/'truth.obj'
            generate_mesh(room, scene, recipe)
            generate_mesh(room, truth, {**recipe, 'primitives': [rail]})
            group = [{'kind': 'railing', 'camera': c['index'], 'points': ReservedPoints(),
                      'mask': np.isfinite(render(truth, c, return_depth=True,
                                         only_groups={'primitive_0'})[2])} for c in cameras]
            candidate = surface_handrail_seed(room, group, scene)
            self.assertIsNotNone(candidate)
            self.assertEqual(candidate['shape']['parameters']['profile'], 'handrail_segments')
            self.assertGreaterEqual(len(candidate['parts']), 2)
            self.assertTrue(np.isfinite(candidate['parts']).all())

    def test_visible_wall_prior_places_rail_at_the_surface(self):
        cameras = [{'index': i, 'origin': origin, 'target': [0, -900, 2000],
                    'fov': 230, 'roll': 0} for i, origin in enumerate(
                        ([2500, -1500, 0], [2500, -1600, 4000]))]
        room = {'cameras': cameras, 'boundaries': [{'index': 0, 'xmin': -100, 'xmax': 0,
                'zmin': 0, 'zmax': 4000, 'type': 1, 'flags': 256}]}
        recipe = {'primitives': [], 'wall_height': 3000}
        rail = generate('railing', {'profile': 'handrail', 'run': 3000, 'rise': 0,
                        'height': 900, 'thickness': 70}, [35, 0, 500])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene, truth = root/'scene.obj', root/'truth.obj'
            generate_mesh(room, scene, recipe)
            generate_mesh(room, truth, {**recipe, 'primitives': [rail]})
            group = [{'kind': 'railing', 'camera': c['index'], 'points': ReservedPoints(),
                      'mask': np.isfinite(render(truth, c, return_depth=True,
                                         only_groups={'primitive_0'})[2])} for c in cameras]
            candidate = surface_handrail_seed(room, group, scene)
            self.assertIsNotNone(candidate)
            vertices = np.asarray(candidate['parts'][0])
            # Native-depth rasterization is approximate; contact must remain
            # within the same 25-unit plane-residual allowance used by the seed.
            self.assertLess(abs(float(vertices[:, 0].min())), 25)
            self.assertLess(float(vertices[:, 0].max()), 120.01)
            self.assertGreaterEqual(float(vertices[:, 2].min()), 0)
            self.assertLessEqual(float(vertices[:, 2].max()), 4000)
            self.assertEqual(candidate['initialization'], 'visible_wall_handrail')
            self.assertIsNone(surface_handrail_seed(room, [{**group[0], 'mask':
                np.zeros((240, 320), bool)}], scene))

    def test_profiles_and_rotated_horizontal_seed(self):
        p = {'run': 3000, 'height': 900, 'thickness': 60, 'posts': 8, 'rise': 0}
        self.assertEqual(len(generate('railing', p)['parts']), 9)
        rail = generate('railing', {**p, 'profile': 'handrail'}, [0, 0, 0], .7)
        self.assertEqual(len(rail['parts']), 1)
        vertices = np.array(rail['parts'][0])
        self.assertTrue(np.isfinite(vertices).all())
        t = np.linspace(0, 1, 1000)
        points = vertices[:4].mean(0) + t[:, None]*(vertices[4:].mean(0)-vertices[:4].mean(0))
        seed = handrail_seed([{'kind': 'railing', 'points': points}])
        self.assertIsNotNone(seed)
        self.assertAlmostEqual(seed['shape']['parameters']['rise'], 0, places=7)
        self.assertAlmostEqual(seed['shape']['yaw'], .7, places=7)
        self.assertEqual(seed['shape']['parameters']['profile'], 'handrail')
        with self.assertRaises(ValueError):
            generate('railing', {**p, 'profile': 'unsupported'})
        with self.assertRaises(ValueError):
            handrail_seed([{'kind': 'railing', 'points': np.zeros((20, 2))}])
        self.assertIsNone(handrail_seed([{'kind': 'railing', 'points': np.full((24, 3), np.nan)}]))

    def test_rendered_profile_selection_does_not_read_reserved_points(self):
        cameras = [{'index': i, 'origin': origin, 'target': [1000, -900, 1000],
                    'fov': 230, 'roll': 0} for i, origin in enumerate(
                        ([0, -1600, -4000], [3000, -2000, -2500], [-3000, -1800, -2000]))]
        room = {'cameras': cameras, 'boundaries': [{'index': 0, 'xmin': -8000, 'xmax': 8000,
                'zmin': -4000, 'zmax': 8000, 'type': 4, 'flags': 0}]}
        true = generate('railing', {'profile': 'handrail', 'run': 3000, 'rise': 0,
                        'height': 900, 'thickness': 80}, [0, 0, 0], .7)
        recipe = {'primitives': []}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mesh = root/'true.obj'
            generate_mesh(room, mesh, {'primitives': [true]})
            groups, depths = [], []
            for camera in cameras:
                part = render(mesh, camera, return_depth=True, only_groups={'primitive_0'})[2]
                mask = np.isfinite(part)
                y, x = np.nonzero(mask)
                self.assertGreater(len(x), 24)
                z = part[y, x]
                points = np.column_stack(((x-160)*z/camera['fov'], (y-120)*z/camera['fov'], z))
                points = points @ camera_basis(camera) + camera['origin']
                groups.append({'kind': 'railing', 'camera': camera['index'], 'mask': mask,
                               'points': points, 'camera_model': camera})
                target = render(mesh, camera, return_depth=True)[2]
                valid = np.isfinite(target)
                depths.append((np.where(valid, target, 20000), valid))
            groups[2]['points'] = ReservedPoints()
            _, old = fit_group(room, recipe, groups, depths, root/'old', generate_mesh, rounds=0)
            fitted, new = fit_group(room, {**recipe, 'handrail_candidates': True}, groups,
                                    depths, root/'new', generate_mesh, rounds=0)
            self.assertEqual(fitted['shape']['parameters']['profile'], 'handrail')
            self.assertIn('hidden brackets', fitted['estimated'])
            self.assertEqual(new['heldout_cameras'], [2])
            for a, b in zip(new['after'], old['after'], strict=True):
                self.assertGreater(a['silhouette_iou'], b['silhouette_iou']+.1)


if __name__ == '__main__':
    unittest.main()
