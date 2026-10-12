"""Approximate stair masks keep their confidence and obey explicit design limits."""

import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_assemblies import stair_proportions
from room3d_creative_stairs import replacement_starts, replacement_valid
from room3d_shape_fit import observations
from room3d_shapes import generate


class CreativeStairTests(unittest.TestCase):
    def test_scene_visibility_retains_real_occlusion_and_depth_penalty(self):
        from room3d_shape_fit import compare

        part = np.full((8, 8), 1000.)
        scene = np.full((8, 8), 2000.)
        target = np.ones((8, 8), bool)
        depth = np.full((8, 8), 500.)
        confidence = np.ones((8, 8), bool)
        self.assertEqual(compare(part, scene, target, depth, confidence)['silhouette_iou'], 0)
        actual = compare(part, scene, target, depth, confidence, model_occlusion=False)
        self.assertEqual(actual['silhouette_iou'], 1)
        self.assertAlmostEqual(actual['depth_error'], math.log(2))
        scene[:] = 500
        self.assertEqual(compare(part, scene, target, depth, confidence,
                                 model_occlusion=False)['silhouette_iou'], 0)

    def test_approximate_masks_are_explicit_exact_category_and_confidence_unchanged(self):
        camera = {'index': 0, 'origin': [0, -1000, 0], 'target': [0, -1000, 1000],
                  'roll': 0, 'fov': 230}
        mask = np.zeros((240, 320), bool)
        mask[100:140, 100:140] = True
        entries = [({'label': label, 'score': .4, 'predicted_iou': score, 'mask': f'{i}.png'}, mask)
                   for i, (label, score) in enumerate([
                       ('staircase', .7), ('carpeted staircase', .64),
                       ('staircase railing', .7), ('lamp', .7)])]
        original = copy.deepcopy(entries)
        args = ({'cameras': [camera]}, [entries], [(np.full(mask.shape, 2000.), mask)])
        self.assertEqual(observations(*args), [])
        found = observations(*args, creative_stair_replacement=True)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['kind'], 'staircase')
        self.assertEqual(found[0]['predicted_iou'], .7)
        self.assertTrue(found[0]['approximate_stair_mask'])
        self.assertIsNone(found[0]['structural_refinement'])
        for (old, a), (new, b) in zip(original, entries, strict=True):
            self.assertEqual(old, new)
            np.testing.assert_array_equal(a, b)
        supported = {**entries[0][0], 'structural_refinement': {
            'accepted_proposal': True, 'structural_evidence': {'tread_edges': 5}}}
        actual = observations(args[0], [[(supported, mask)]], args[2],
                              creative_stair_replacement=True)
        self.assertEqual(len(actual), 1)
        self.assertNotIn('approximate_stair_mask', actual[0])
        self.assertEqual(actual[0]['predicted_iou'], .7)

    def test_bounded_alternatives_preserve_center_height_and_explore_both_ascent_signs(self):
        seed = generate('staircase', {'width': 9000, 'run': 7000, 'rise': 4000,
                                     'steps': 17, 'landing': 1000}, [100, 80, 200], .6)
        original = copy.deepcopy(seed)
        candidates = list(replacement_starts([seed], [[1, 0], [0, 1]]))
        self.assertEqual(len(candidates), 8)
        center = np.array(seed['shape']['origin']) + 3500*np.array([math.sin(.6), 0, math.cos(.6)])
        directions = []
        for candidate in candidates:
            shape = candidate['shape']
            p = shape['parameters']
            direction = np.array([math.sin(shape['yaw']), 0, math.cos(shape['yaw'])])
            np.testing.assert_allclose(np.array(shape['origin']) + p['run']/2*direction, center)
            self.assertEqual(shape['origin'][1], 80)
            self.assertIn(p['width'], (2000, 4000))
            self.assertTrue(stair_proportions(candidate)['plausible'])
            self.assertTrue(replacement_valid(candidate, [[1, 0], [0, 1]]))
            self.assertTrue(candidate['interpretation'])
            directions.append(tuple(np.round(direction[[0, 2]], 6)))
        self.assertEqual(len(set(directions)), 4)
        self.assertEqual(seed, original)
        self.assertEqual(len(list(replacement_starts([seed, seed], [[1, 0], [0, 1]]))), 8)
        self.assertFalse(replacement_valid(seed, [[1, 0], [0, 1]]))
        rotated = generate('staircase', {**seed['shape']['parameters'], 'width': 2000}, yaw=.6)
        self.assertFalse(replacement_valid(rotated, [[1, 0], [0, 1]]))
        self.assertTrue(replacement_valid(rotated, [[math.cos(.6), -math.sin(.6)]]))

    def test_actual_fit_keeps_bounds_during_polish_and_ignores_heldout_mask_qualification(self):
        from batch_room3d import generate_mesh
        from room3d_preview import render
        from room3d_shape_fit import fit_group

        truth = generate('staircase', {'width': 2000, 'run': 3600, 'rise': 1800, 'steps': 9})
        wide = generate('staircase', {**truth['shape']['parameters'], 'width': 6000})
        cameras = [{'index': i, 'origin': [i*300, -3500, -5000],
                    'target': [0, -900, 1800], 'fov': 280, 'roll': 0} for i in range(3)]
        room = {'cameras': cameras, 'boundaries': [
            {'index': 0, 'xmin': -5000, 'xmax': 5000,
             'zmin': -5000, 'zmax': 5000, 'type': 4, 'flags': 0}]}
        recipe = {'creative_stair_replacement': True, 'tread_horizontal_axes': [[1, 0], [0, 1]]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_mesh(room, root/'truth.obj', {'primitives': [truth]})
            group, depths = [], []
            for camera in cameras:
                _, mask, depth = render(root/'truth.obj', camera, return_depth=True,
                                        only_groups={'primitive_0'})
                group.append({'kind': 'staircase', 'camera': camera['index'],
                              'camera_model': camera, 'mask': mask,
                              'approximate_stair_mask': camera['index'] < 2})
                depths.append((np.where(mask, depth, 10000), mask))
            reports = []
            for run in range(3):
                changed = copy.deepcopy(group)
                changed[-1]['approximate_stair_mask'] = run > 0
                if run == 2:
                    for observation in changed[:-1]:
                        observation['approximate_stair_mask'] = False
                fitted, report = fit_group(room, recipe, changed, depths, root/f'run{run}',
                                          generate_mesh, rounds=0, initial_candidates=[wide, truth])
                reports.append(report)
                if run < 2:
                    self.assertTrue(replacement_valid(fitted, recipe['tread_horizontal_axes']))
                    self.assertTrue(all(s['loss'] == 2 for s in report['initializations'][0]['scores']))
            self.assertEqual(reports[0]['after'], reports[1]['after'])
            self.assertEqual(reports[0]['creative_layout_prior'], reports[1]['creative_layout_prior'])
            self.assertNotIn('creative_layout_prior', reports[2])
            # Simulate unusable depth seeds: actual camera/mask initialization
            # must still generate a bounded flight, without reading heldout masks.
            volume_reports = []
            for run in range(2):
                changed = copy.deepcopy(group)
                if run:
                    changed[-1]['mask'][:] = False
                with patch('room3d_shape_fit.initial_shapes', return_value=[]):
                    fitted, report = fit_group(room, recipe, changed, depths,
                                              root/f'volume{run}', generate_mesh, rounds=0)
                self.assertIsNotNone(fitted)
                self.assertTrue(replacement_valid(fitted, recipe['tread_horizontal_axes']))
                self.assertTrue(all(x['initialization'].startswith('creative_mask_volume_')
                                    for x in report['initializations']))
                volume_reports.append(report)
            self.assertEqual(volume_reports[0]['initializations'], volume_reports[1]['initializations'])
            self.assertEqual(volume_reports[0]['after'], volume_reports[1]['after'])


if __name__ == '__main__':
    unittest.main()
