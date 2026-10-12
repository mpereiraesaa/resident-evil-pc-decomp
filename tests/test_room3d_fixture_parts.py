"""Brightness hypotheses must abstain without clear, contained training evidence."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_fixture_parts import (
    bright_region,
    bright_regions,
    joint_size_part_variants,
    light_assignment_score,
    refine_wall_parts,
    size_part_variants,
)


class FixturePartTests(unittest.TestCase):
    def test_creative_search_is_explicit_and_joint_variants_are_bounded(self):
        from room3d_fixtures import anchor_wall_lamp
        from room3d_shapes import generate

        obj = anchor_wall_lamp(generate('lamp', {'height': 600, 'radius': 300,
            'thickness': 30, 'shade': 160, 'arms': 1, 'profile': 'wall_sconce'}),
            {'point': [0, -1000, 0], 'normal': [0, 0, 1], 'group': 'boundary_0'})
        with self.assertRaises(ValueError):
            refine_wall_parts({}, {}, obj, [], [], [], Path('unused'), None, creative=True)
        variants = list(joint_size_part_variants(obj))
        self.assertLessEqual(len(variants), 168)
        self.assertEqual({round(c['shape']['parameters']['shade']/160, 1) for c, _ in variants}, {.4, .6, .8})
        self.assertTrue(all(c['attachment_prior']['point'][0::2] == [0, 0] for c, _ in variants))
        self.assertTrue(all(p['stage'] == 'joint_shade_size_anchor' for _, p in variants))

    def test_size_variants_preserve_anchor_centres_and_distinct_shades(self):
        from room3d_fixtures import anchor_wall_lamp
        from room3d_shapes import generate

        params = {'height': 600, 'radius': 200, 'thickness': 20, 'shade': 160,
                  'arms': 2, 'span': 260, 'profile': 'wall_candelabra', 'arm_rise': .2}
        anchor = {'point': [300, -1200, 200], 'normal': [0, 0, 1], 'group': 'wall'}
        obj = anchor_wall_lamp(generate('lamp', params), anchor)
        before = np.asarray(obj['parts'][2:4]).mean(axis=(0, 1))[1]
        candidates = list(size_part_variants(obj))
        self.assertTrue(candidates)
        for candidate, _ in candidates:
            self.assertEqual(candidate['attachment_prior'], obj['attachment_prior'])
            self.assertAlmostEqual(np.asarray(candidate['parts'][2:4]).mean(axis=(0, 1))[1], before)
            shades = np.asarray(candidate['parts'][2:4])
            self.assertLess(shades[0, :, 0].max(), shades[1, :, 0].min())
            self.assertEqual(candidate['part_roles'].count('shade'), 2)

    def test_rendered_size_fit_improves_parts_without_using_reserved_image(self):
        from batch_room3d import generate_mesh
        from room3d_fixtures import anchor_wall_lamp
        from room3d_preview import render
        from room3d_shape_fit import isolated_recipe
        from room3d_shapes import generate

        class ReservedImage:
            def convert(self, *_):
                raise AssertionError('Reserved image entered size fitting')

        room = {'cameras': [{'index': i, 'origin': [i*100, -1000, 2000],
                             'target': [0, -1000, 0], 'fov': 230, 'roll': 0} for i in range(3)],
                'boundaries': [{'index': 0, 'xmin': -3000, 'xmax': 3000,
                                'zmin': -100, 'zmax': 0, 'flags': 256, 'type': 1}]}
        params = {'height': 1200, 'radius': 250, 'thickness': 40, 'shade': 160,
                  'arms': 2, 'span': 400, 'profile': 'wall_candelabra', 'arm_rise': .2}
        obj = anchor_wall_lamp(generate('lamp', params),
                              {'point': [0, -1000, 0], 'normal': [0, 0, 1], 'group': 'boundary_0'})
        target = next(c for c, p in size_part_variants(obj) if p['shade_scale'] == .4 and p['span_scale'] == 1)
        recipe = {'primitives': [], 'wall_height': 3000, 'fixture_size_part_fit': True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'target.obj'
            generate_mesh(room, path, isolated_recipe(room, recipe, target))
            images, depths, group = [], [], []
            for camera in room['cameras']:
                _, mask, depth = render(path, camera, return_depth=True, only_groups={'primitive_0'})
                pixels = np.full((240, 320, 3), 30, dtype='uint8')
                for part in target['parts'][2:4]:
                    shape = {**target, 'parts': [part]}
                    shade_path = root / 'shade.obj'
                    generate_mesh(room, shade_path, isolated_recipe(room, recipe, shape))
                    bright = render(shade_path, camera, only_groups={'primitive_0'})[1]
                    pixels[bright] = 250
                images.append(Image.fromarray(pixels))
                depths.append((np.where(np.isfinite(depth), depth, 2000), np.ones_like(mask)))
                group.append({'camera': camera['index'], 'mask': mask})
            images[-1] = ReservedImage()
            for creative in (False, True):
                candidate = {**obj, 'interpretation': {'source': 'synthetic_creative_prior'}} if creative else obj
                fitted, report = refine_wall_parts(room, recipe, candidate, group, depths, images,
                                                  root / f'fit-{creative}', generate_mesh, creative=creative)
                self.assertEqual(report['status'], 'selected')
                stage = 'joint_shade_size_anchor' if creative else 'shade_size_spacing'
                self.assertTrue(any(t.get('stage') == stage for t in report['trials']))
                self.assertLess(fitted['shape']['parameters']['shade'], params['shade'])
                self.assertGreater(min(report['selected_part_iou']), min(report['baseline_part_iou']) + .05)
                self.assertEqual(report['heldout_cameras'], [2])
                self.assertEqual(report['support_criterion'],
                                 'creative_placement_and_bounded_anchor' if creative else 'whole_mask_iou')
                if creative:
                    self.assertTrue(all(s['creative_placement']['valid'] for s in report['after']))

    def test_multilight_components_preserve_single_shade_abstention(self):
        pixels = np.full((40, 40, 3), 30, dtype='uint8')
        mask = np.zeros((40, 40), bool)
        mask[10:30, 10:30] = True
        pixels[10:14, 10:18] = 250
        pixels[25:29, 22:30] = 250
        pixels[15, 25] = 250  # One isolated glint cannot become a third bulb.
        image = Image.fromarray(pixels)
        self.assertIsNone(bright_region(image, mask))
        regions = bright_regions(image, mask)
        self.assertEqual(len(regions), 2)
        self.assertEqual([int(r.sum()) for r in regions], [32, 32])
        self.assertFalse((regions[0] & regions[1]).any())
        self.assertIsNone(bright_regions(Image.new('RGB', (40, 40), 'white'), mask))

    def test_part_assignment_cannot_reuse_one_shade_for_two_lights(self):
        self.assertAlmostEqual(light_assignment_score([[.8, 0], [.8, 0]]), .2)
        self.assertAlmostEqual(light_assignment_score([[0, .7], [.7, 0]]), .7)
        self.assertAlmostEqual(light_assignment_score([[.3]]), .3)
        with self.assertRaises(ValueError):
            light_assignment_score([[.9], [.9]])

    def sample(self):
        pixels = np.full((40, 40, 3), 30, dtype='uint8')
        pixels[15:19, 15:19] = 250
        mask = np.zeros((40, 40), bool)
        mask[10:30, 10:30] = True
        return pixels, mask

    def test_exact_contained_highlight(self):
        pixels, mask = self.sample()
        # Keep highlight population above the top15% threshold.
        pixels[14:22, 14:22] = 250
        pixels[:5] = 255  # Unrelated background never joins the component.
        region = bright_region(Image.fromarray(pixels), mask)
        self.assertEqual(int(region.sum()), 64)
        self.assertFalse(region[~mask].any())

    def test_flat_fragmented_and_clipped_regions_abstain(self):
        pixels, mask = self.sample()
        self.assertIsNone(bright_region(Image.new('RGB', (40, 40), 'white'), mask))
        pixels[10:14, 10:18] = 250
        pixels[25:29, 22:30] = 250
        self.assertIsNone(bright_region(Image.fromarray(pixels), mask))
        pixels = np.full((40, 40, 3), 30, dtype='uint8')
        pixels[:8, 10:18] = 250
        mask[:10, 10:30] = True
        self.assertIsNone(bright_region(Image.fromarray(pixels), mask))

    def test_reserved_image_is_not_consulted(self):
        class ReservedImage:
            def convert(self, *_):
                raise AssertionError('Reserved image entered training')
        room = {'cameras': [{'index': i} for i in range(3)]}
        mask = np.ones((40, 40), bool)
        group = [{'camera': i, 'mask': mask} for i in range(3)]
        obj = {'shape': {'parameters': {'profile': 'wall_sconce'}}}
        fitted, report = refine_wall_parts(
            room, {}, obj, group, [(None, None)] * 3,
            [Image.new('RGB', (40, 40), 'gray')] * 2 + [ReservedImage()],
            Path('unused'), None,
        )
        self.assertIs(fitted, obj)
        self.assertEqual(report['status'], 'no_reliable_bright_region')

    def test_invalid_mask_fails_explicitly(self):
        with self.assertRaises(ValueError):
            bright_region(Image.new('RGB', (40, 40)), np.ones((2, 2), bool))

    def test_enabled_pipeline_requires_original_images(self):
        from room3d_shape_fit import fit_objects

        with self.assertRaisesRegex(ValueError, 'original camera images'):
            fit_objects({}, {'fixture_part_fit': True}, [], [], Path('unused'), None)


if __name__ == '__main__':
    unittest.main()
