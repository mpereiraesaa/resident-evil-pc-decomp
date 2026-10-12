"""Fan geometry, material roles and training-only family selection."""
import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from batch_room3d import generate_mesh
from room3d_complete import fixture_tints
from room3d_connections import lamp_attachment, support_mesh
from room3d_fixtures import (
    anchor_ceiling_fan,
    ceiling_fan_proposals,
    fixture_contact,
    slide_fixture_anchor,
)
from room3d_preview import camera_basis, render
from room3d_segment import texture_support
from room3d_shape_fit import fit_group, fit_objects, initial_shapes
from room3d_shapes import generate


class CeilingFanTests(unittest.TestCase):
    params: ClassVar[dict] = {'profile': 'ceiling_fan', 'height': 600, 'radius': 800,
                             'thickness': 15, 'blade_width': 140, 'arms': 4, 'shade': 100}

    def room(self):
        return {'boundaries': [{'index': 0, 'xmin': -2500, 'xmax': 2500,
                               'zmin': -2500, 'zmax': 2500, 'type': 4, 'flags': 0}],
                'cameras': [{'index': i, 'origin': origin, 'target': [0, -1200, 0],
                            'fov': 220, 'roll': 0}
                            for i, origin in enumerate(([0, -600, -3000],
                                                       [3000, -800, 0],
                                                       [-2500, -700, -2000]))]}

    def test_blades_closed_and_mount_contacts_ceiling_after_pose_transform(self):
        room = self.room()
        for arms in (3, 4, 5, 6):
            obj = generate('lamp', {**self.params, 'arms': arms}, [200, -1400, 100], .7)
            parts = np.asarray(obj['parts'])
            self.assertEqual(parts.shape, (3+arms, 8, 3))
            self.assertEqual(obj['part_roles'], ['support', 'support', 'shade'] + ['support']*arms)
            np.testing.assert_allclose(parts[0, :4].mean(0), [200, -2000, 100])
            for blade in parts[3:]:
                # Opposite rings enclose a nonzero volume after world rotation.
                edges = [blade[1]-blade[0], blade[3]-blade[0], blade[4]-blade[0]]
                self.assertGreater(abs(np.linalg.det(edges)), 1)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'fan.obj'
                ceiling = {'kind': 'ceiling', 'box': [-2500, -2500, 2500, 2500, -2050, -2000]}
                recipe = {'primitives': [ceiling, obj], 'local_fixture_appearance': True}
                generate_mesh(room, path, recipe)
                triangles, groups = support_mesh(path)
                self.assertTrue(lamp_attachment(obj, triangles, groups, {'primitive_0'}, tolerance=1e-6)['within_tolerance'])
                text = path.read_text()
                self.assertEqual(text.count('usemtl fixture_shade\n'), 6)
                self.assertEqual(text.count('usemtl fixture_support\n'), (2+arms)*6)

    def test_invalid_blades_and_intersecting_dimensions_rejected(self):
        for changes in ({'arms': 2}, {'arms': 7}, {'arms': True},
                        {'blade_width': 25}, {'blade_width': 600},
                        {'blade_width': math.inf}, {'shade': 400},
                        {'thickness': 80}, {'height': 100}, {'light_drop': 0},
                        {'light_drop': 590}, {'light_drop': 90}, {'light_drop': math.nan}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                generate('lamp', {**self.params, **changes})

    def test_ceiling_proposal_preserves_body_and_checks_finite_stem_cap(self):
        room = self.room()
        fan = generate('lamp', self.params, [200, -1000, 100], .7)
        original = copy.deepcopy(fan)
        roof = {'kind': 'ceiling', 'box': [-2500, -2500, 2500, 2500, -2450, -2400]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'roof.obj'
            recipe = {'primitives': [roof]}
            generate_mesh(room, path, recipe)
            triangles, groups = support_mesh(path)
            proposals = list(ceiling_fan_proposals(fan, triangles, groups, {'primitive_0'}))
            self.assertEqual(len(proposals), 1)
            attached = proposals[0]
            self.assertEqual(attached['attachment_prior']['point'], [200, -2400, 100])
            np.testing.assert_allclose(attached['parts'][1:], fan['parts'][1:], atol=1e-10)
            self.assertTrue(fixture_contact(attached, triangles, groups, tolerance=1e-6))
            moved = generate('lamp', attached['shape']['parameters'], [250, -950, 180], .8)
            slid = slide_fixture_anchor(moved, attached)
            self.assertEqual(slid['shape']['parameters']['height'], 1450)
            self.assertEqual(slid['shape']['parameters']['light_drop'], 210)
            self.assertTrue(fixture_contact(slid, triangles, groups, tolerance=1e-6))
            self.assertEqual(list(ceiling_fan_proposals(fan, triangles, groups, {'absent'})), [])
            self.assertEqual(list(ceiling_fan_proposals(fan, triangles, groups, {'primitive_0'}, maximum_extension=700)), [])
            # Center on a ceiling edge is insufficient: the full stem cap must fit.
            edge = generate('lamp', {**self.params, 'thickness': 35}, [2500, -1000, 100])
            self.assertEqual(list(ceiling_fan_proposals(edge, triangles, groups, {'primitive_0'})), [])
            # A roof marked for omission cannot supply invisible support geometry.
            generate_mesh(room, path, {'primitives': [{**roof, 'omit_geometry': True}]})
            self.assertEqual(list(ceiling_fan_proposals(fan, *support_mesh(path), {'primitive_0'})), [])
        self.assertEqual(fan, original)

    def test_anchored_fitting_retains_contact_and_reserved_independence(self):
        room = self.room()
        floating = generate('lamp', self.params, [0, -1000, 0])
        ceiling = {'kind': 'ceiling', 'box': [-2500, -2500, 2500, 2500, -2450, -2400]}
        prior = {'support': 'ceiling', 'point': [0, -2400, 0], 'group': 'primitive_0', 'normal': [0, 1, 0]}
        target = anchor_ceiling_fan(floating, prior)
        recipe = {'primitives': [ceiling], 'ceiling_fan_anchors': True, 'refine_fixture_anchor': True}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'target.obj'
            generate_mesh(room, path, {'primitives': [target]})
            group, depths = [], []
            for camera in room['cameras']:
                _, mask, depth = render(path, camera, return_depth=True, only_groups={'primitive_0'})
                yy, xx = np.nonzero(mask)
                z = depth[yy, xx]
                points = np.column_stack(((xx-160)*z/camera['fov'], (yy-120)*z/camera['fov'], z))
                points = points @ camera_basis(camera) + camera['origin']
                group.append({'kind': 'lamp', 'camera': camera['index'], 'camera_model': camera,
                              'mask': mask, 'points': points, 'label': 'ceiling fan'})
                depths.append((np.full((240, 320), 10000.), np.zeros((240, 320), bool)))
            before = copy.deepcopy((recipe, floating))
            fitted, report = fit_group(room, recipe, group, depths, root/'first', generate_mesh,
                                       rounds=1, initial_candidates=[floating])
            self.assertEqual(fitted['shape']['parameters']['profile'], 'ceiling_fan')
            triangles, groups = support_mesh(root/'first/scene.obj')
            self.assertTrue(fixture_contact(fitted, triangles, groups, tolerance=1e-6))
            self.assertTrue(all(s['attachment_valid'] for s in report['after']))
            changed = copy.deepcopy(group)
            changed[-1]['mask'] = ~changed[-1]['mask']
            changed_depths = copy.deepcopy(depths)
            changed_depths[-1] = (np.ones((240, 320)), np.ones((240, 320), bool))
            second, other = fit_group(room, recipe, changed, changed_depths, root/'second', generate_mesh,
                                      rounds=1, initial_candidates=[floating])
            self.assertEqual(fitted['shape'], second['shape'])
            self.assertEqual(report['after'], other['after'])
            self.assertEqual((recipe, floating), before)

    def test_final_scene_rejects_floating_fan_and_accepts_contacting_fan(self):
        room = self.room()
        floating = generate('lamp', self.params, [0, -1000, 0])
        ceiling = {'kind': 'ceiling', 'box': [-2500, -2500, 2500, 2500, -2450, -2400]}
        anchored = anchor_ceiling_fan(floating, {
            'support': 'ceiling', 'point': [0, -2400, 0], 'group': 'primitive_0', 'normal': [0, 1, 0]})
        recipe = {'primitives': [ceiling]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene, target = root/'scene.obj', root/'target.obj'
            generate_mesh(room, scene, recipe)
            generate_mesh(room, target, {'primitives': [ceiling, anchored]})
            segments, depths = [], []
            for camera in room['cameras']:
                _, mask, _ = render(target, camera, return_depth=True, only_groups={'primitive_1'})
                background = render(scene, camera, return_depth=True)[2]
                valid = np.isfinite(background) & ~mask
                self.assertTrue(valid.any())
                # Association sees correct object depth; final scene checks see
                # reliable background only, so attachment is an independent gate.
                object_depth = render(target, camera, return_depth=True)[2]
                depth = np.where(np.isfinite(object_depth), object_depth, 10000)
                segments.append([({'label': 'ceiling fan', 'score': .9, 'predicted_iou': .95,
                                   'mask': f"fan-{camera['index']}.png"}, mask)])
                depths.append((depth, valid))
            for name, obj, expected in [('floating', floating, False), ('anchored', anchored, True)]:
                fit = {'status': 'fitted', 'training_cameras': [0, 1], 'heldout_cameras': [2],
                       'after': [{'attachment_valid': True}]*2}
                # Only the expensive parameter optimizer is controlled; grouping,
                # scene rendering and every final acceptance gate are real.
                with patch('room3d_shape_fit.fit_group', return_value=(copy.deepcopy(obj), fit)):
                    _, report = fit_objects(room, recipe, segments, depths, root/name, generate_mesh)
                self.assertEqual(len(report['objects']), 1)
                result = report['objects'][0]
                self.assertEqual(result['accepted'], expected)
                self.assertEqual(result['gates']['fixture_attachment'], expected)
                self.assertEqual(result['ceiling_attachment']['stem_cap_contact'], expected)

    def test_own_fan_palette_and_architecture_foreground_guard(self):
        mask = np.ones((20, 20), bool)
        rgb = np.full((20, 20, 3), [15, 25, 35], dtype=np.uint8)
        rgb[8:12, 8:12] = [230, 240, 250]
        entry = {'label': 'ceiling fan', 'mask': 'fan.png', 'score': .8, 'predicted_iou': .9}
        segments = [[(entry, mask)]]
        room = {'cameras': [{'index': 0}]}
        palette = fixture_tints(room, [Image.fromarray(rgb)], segments,
                                {'primitive_0': [{'camera': 0, 'mask': 'fan.png'}]})
        self.assertIn('primitive_0', palette)
        self.assertEqual(palette['primitive_0']['sources'][0]['mask'], 'fan.png')
        self.assertGreater(np.mean(palette['primitive_0']['roles']['shade']), .8)
        self.assertLess(np.mean(palette['primitive_0']['roles']['support']), .2)
        xy = np.array([[10, 10]])
        self.assertTrue(texture_support(segments[0], 'lamp', xy, True)[0])
        for architecture in ('wall', 'floor', 'ceiling'):
            self.assertFalse(texture_support(segments[0], architecture, xy, True)[0])
        reserved = fixture_tints(room, [Image.fromarray(rgb)], segments,
                                 {'primitive_0': [{'camera': 0, 'mask': 'fan.png'}]}, exclude_camera=0)
        self.assertEqual(reserved, {})

    def test_rendered_family_selection_ignores_reserved_mask_and_depth(self):
        room = self.room()
        fan = generate('lamp', self.params, [0, -1000, 0])
        old = generate('lamp', {**self.params, 'profile': 'shade'}, [0, -1000, 0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'target.obj'
            generate_mesh(room, path, {'primitives': [fan]})
            group, depths = [], []
            for camera in room['cameras']:
                _, mask, depth = render(path, camera, return_depth=True, only_groups={'primitive_0'})
                self.assertGreater(mask.sum(), 100)
                yy, xx = np.nonzero(mask)
                z = depth[yy, xx]
                points = np.column_stack(((xx-160)*z/camera['fov'], (yy-120)*z/camera['fov'], z))
                points = points @ camera_basis(camera) + camera['origin']
                group.append({'kind': 'lamp', 'camera': camera['index'], 'camera_model': camera,
                              'mask': mask, 'points': points, 'label': 'lamp'})
                depths.append((np.full((240, 320), 10000.), np.zeros((240, 320), bool)))
            snapshot = copy.deepcopy(group)
            recipe = {'primitives': [], 'ceiling_fan_fixtures': True}
            selected, report = fit_group(room, recipe, group, depths, root/'first',
                                         generate_mesh, rounds=0, initial_candidates=[old, fan])
            self.assertEqual(selected['shape']['parameters']['profile'], 'ceiling_fan')
            changed = copy.deepcopy(group)
            changed[-1]['mask'] = ~changed[-1]['mask']
            other_depths = copy.deepcopy(depths)
            other_depths[-1] = (np.ones((240, 320)), np.ones((240, 320), bool))
            second, other = fit_group(room, recipe, changed, other_depths, root/'second',
                                      generate_mesh, rounds=0, initial_candidates=[old, fan])
            self.assertEqual(selected['shape'], second['shape'])
            self.assertEqual(report['after'], other['after'])
            for before, after in zip(snapshot, group, strict=True):
                np.testing.assert_array_equal(before['mask'], after['mask'])
                np.testing.assert_array_equal(before['points'], after['points'])
            legacy = list(initial_shapes(room, group[:2]))
            experimental = list(initial_shapes(room, group[:2], ceiling_fans=True))
            self.assertFalse(any(o['shape']['parameters'].get('profile') == 'ceiling_fan' for o in legacy))
            self.assertTrue(any(o['shape']['parameters'].get('profile') == 'ceiling_fan' for o in experimental))


if __name__ == '__main__':
    unittest.main()
