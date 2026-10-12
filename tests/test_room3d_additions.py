"""Inserted objects must be real editable geometry and preserve existing assets."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_additions import (
    LIBRARY,
    ROLE_MATERIALS,
    append_objects,
    associate_fixture_masks,
)
from room3d_preview import render
from room3d_retouch import catalog, read_mesh, run, write_mesh


class AdditionsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.room = self.root / 'room'
        self.room.mkdir()
        (self.room / 'assisted.obj').write_text(
            'mtllib original.mtl\nv 10000 0 1000\nv 11000 0 1000\nv 10000 -500 1000\n'
            'vt .1 .2\nvt .3 .4\nvt .5 .6\ng fixed\nusemtl original\nf 1/1 2/2 3/3\n')
        (self.room / 'original.mtl').write_text('newmtl original\nKd 1 1 1\nmap_Kd texture.png\n')
        (self.room / 'assisted.cameras').write_text('# No camera cutaways\n')
        Image.new('RGB', (320, 240), (120, 80, 60)).save(self.room / 'camera0-original.png')
        Image.new('RGB', (2, 2), (80, 90, 100)).save(self.room / 'texture.png')
        self.camera = {'index': 0, 'origin': [0, -500, 0],
                       'target': [0, -500, 2000], 'fov': 230, 'roll': 0}
        (self.room / 'survey.json').write_text(json.dumps({'cameras': [self.camera], 'boundaries': []}))
        (self.room / 'assisted.json').write_text(json.dumps({'fitted_recipe': {'primitives': []}}))
        self.additions = [{'name': 'stairs', 'kind': 'staircase', 'origin': [0, 0, 2000],
                           'parameters': {'width': 600, 'run': 1000, 'rise': 500, 'steps': 4}}]

    def fingerprints(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.room.iterdir()}

    def test_addition_renders_and_can_be_transformed_without_moving_source(self):
        before = self.fingerprints()
        out = self.root / 'variant'
        record = run(self.room, {'added_stairs': {'translate': [200, 0, 0]}},
                     out, additions=self.additions)
        self.assertEqual(before, self.fingerprints())
        source = read_mesh(self.room / 'assisted.obj')
        edited = read_mesh(out / 'assisted.obj')
        self.assertEqual(source[1], edited[1])
        for old_ref, new_ref in zip(source[2][0][2], edited[2][0][2], strict=True):
            self.assertEqual(source[0][old_ref[0]], edited[0][new_ref[0]])
        self.assertEqual(len(edited[2]) - len(source[2]), 4 * 6)
        self.assertEqual(catalog(edited)['added_stairs']['min'][0], -100)
        self.assertEqual(record['additions'][0]['fidelity'], 'unverified')
        self.assertEqual((out / 'texture.png').read_bytes(), (self.room / 'texture.png').read_bytes())
        self.assertFalse(render(self.room / 'assisted.obj', self.camera)[1].any())
        self.assertGreater(render(out / 'assisted.obj', self.camera)[1].mean(), .01)

    def test_reprojection_supports_added_group_and_preserves_inputs(self):
        before = self.fingerprints()
        out = self.root / 'projected'
        record = run(self.room, {}, out, reproject=True, additions=self.additions)
        self.assertEqual(record['appearance'], 'reprojected')
        self.assertEqual(self.fingerprints(), before)
        self.assertIn('added_stairs', catalog(read_mesh(out / 'assisted.obj')))
        image, mask = render(out / 'assisted.obj', self.camera)
        self.assertGreater(mask.mean(), .01)
        self.assertGreater(np.asarray(image)[mask].mean(), 0)
        self.assertTrue((out / 'reprojection.json').is_file())

    def test_invalid_additions_are_rejected_before_output_creation(self):
        source = read_mesh(self.room / 'assisted.obj')
        unchanged = copy.deepcopy(source)
        for specs in ({}, self.additions * 2, [{**self.additions[0], 'name': '../escape'}],
                      [{**self.additions[0], 'kind': 'unknown'}],
                      [{**self.additions[0], 'origin': [float('nan'), 0, 0]}],
                      [{**self.additions[0], 'parameters': {'width': -1}}]):
            with self.subTest(specs=specs), self.assertRaises((ValueError, KeyError)):
                run(self.room, {}, self.root / 'invalid', additions=specs)
            self.assertFalse((self.root / 'invalid').exists())
        self.assertEqual(source, unchanged)
        append_objects(source, self.additions)
        self.assertEqual(source, unchanged)

    def test_successive_insertions_preserve_previous_objects_and_reuse_neutral_library(self):
        from room3d_checkpoint import inventory

        first = self.root / 'first-addition'
        run(self.room, {}, first, additions=self.additions)
        before = inventory(first)
        rail = {'name': 'rail', 'kind': 'railing', 'origin': [300, 0, 2000],
                'parameters': {'run': 1000, 'height': 800, 'thickness': 40, 'posts': 5}}
        second = self.root / 'second-addition'
        run(first, {}, second, additions=[rail])
        old, new = read_mesh(first / 'assisted.obj'), read_mesh(second / 'assisted.obj')
        self.assertEqual(new[3].count(LIBRARY), 1)
        self.assertEqual(catalog(old)['added_stairs'], catalog(new)['added_stairs'])
        for old_face, new_face in zip(old[2], new[2][:len(old[2])], strict=True):
            self.assertEqual(old_face[:2], new_face[:2])
            for a, b in zip(old_face[2], new_face[2], strict=True):
                self.assertEqual(old[0][a[0]], new[0][b[0]])
        self.assertEqual((first / LIBRARY).read_bytes(), (second / LIBRARY).read_bytes())
        self.assertEqual(inventory(first), before)
        third = self.root / 'third-reprojected'
        run(second, {'added_rail': {'translate': [50, 0, 0]}}, third, reproject=True)
        context = json.loads((third / 'retouch-appearance.json').read_text())
        self.assertEqual(context['added_groups'], [
            {'group': 'added_stairs', 'kind': 'staircase'}, {'group': 'added_rail', 'kind': 'railing'}])
        with self.assertRaisesRegex(ValueError, 'Duplicate object group'):
            run(second, {}, self.root / 'duplicate', additions=self.additions)
        self.assertFalse((self.root / 'duplicate').exists())

    def test_successive_insertions_do_not_overwrite_customized_materials(self):
        from room3d_checkpoint import inventory

        first = self.root / 'customized'
        run(self.room, {}, first, additions=self.additions)
        material = first / LIBRARY
        material.write_text(material.read_text().replace('Kd .55 .55 .55', 'Kd .2 .3 .4'))
        before = inventory(first)
        addition = {**self.additions[0], 'name': 'other_stairs'}
        target = self.root / 'rejected-customized'
        with self.assertRaisesRegex(ValueError, 'materials were customized'):
            run(first, {}, target, additions=[addition])
        self.assertFalse(target.exists())
        self.assertEqual(inventory(first), before)

    def test_all_supported_families_export_separate_editable_groups(self):
        source = read_mesh(self.room / 'assisted.obj')
        specs = self.additions + [
            {'name': 'rail', 'kind': 'railing', 'origin': [0, 0, 2000],
             'parameters': {'run': 1000, 'height': 800, 'thickness': 40, 'posts': 5}},
            {'name': 'light', 'kind': 'lamp', 'origin': [500, 0, 2200],
             'parameters': {'height': 800, 'radius': 200, 'thickness': 30, 'arms': 1,
                            'shade': 200, 'profile': 'table_lamp'}},
            {'name': 'pillar', 'kind': 'column', 'origin': [-500, 0, 2200],
             'parameters': {'height': 1600, 'width': 200}},
        ]
        mesh, generated = append_objects(source, specs)
        self.assertEqual(set(catalog(mesh)), {'fixed', 'added_stairs', 'added_rail',
                                             'added_light', 'added_pillar'})
        self.assertEqual(len(generated), 4)
        self.assertTrue(np.isfinite(np.asarray(mesh[0])).all())
        for group, _, refs in mesh[2]:
            for vi, ti in refs:
                self.assertLess(vi, len(mesh[0]))
                if group.startswith('added_'):
                    self.assertIsNone(ti)

    def lamp_scene(self, duplicate=False, occluded=False):
        spec = {'name': 'light', 'kind': 'lamp', 'origin': [0, 0, 2000],
                'parameters': {'height': 800, 'radius': 120, 'thickness': 30,
                               'arms': 1, 'shade': 200, 'profile': 'table_lamp'}}
        mesh, _ = append_objects(([], [], [], []), [spec])
        if duplicate:
            # Same shape under two groups must not share ownership of one mask.
            mesh = (*mesh[:2], mesh[2] + [('other_light', m, refs) for _, m, refs in mesh[2]], mesh[3])
        if occluded:
            start = len(mesh[0])
            mesh[0].extend([[-2000, -2000, 1000], [2000, -2000, 1000],
                            [2000, 2000, 1000], [-2000, 2000, 1000]])
            mesh[2].append(('occluder', 'retouch_added_support', [(start+i, None) for i in range(4)]))
        path = self.root / 'lamp.obj'
        write_mesh(mesh, {}, path)
        (self.root / LIBRARY).write_text(''.join(f'newmtl {name}\nKd .55 .55 .55\n'
                                               for name in ROLE_MATERIALS.values()))
        mask = render(path, self.camera, only_groups={'added_light'})[1]
        item = {'mask': 'light.png', 'label': 'table lamp', 'score': .8, 'predicted_iou': .9}
        return spec, path, item, mask

    def test_fixture_association_abstains_for_ambiguity_unreliability_and_occlusion(self):
        room = {'cameras': [self.camera]}
        _, path, item, mask = self.lamp_scene()
        groups = ['added_light']
        found = associate_fixture_masks(path, room, [[(item, mask)]], groups)
        self.assertEqual(found['added_light'][0]['mask'], 'light.png')
        for entries in ([(item, mask), ({**item, 'mask': 'duplicate.png'}, mask)],
                        [({**item, 'predicted_iou': .6}, mask)],
                        [({**item, 'score': .2}, mask)]):
            self.assertEqual(associate_fixture_masks(path, room, [entries], groups), {'added_light': []})
        _, path, item, mask = self.lamp_scene(duplicate=True)
        self.assertEqual(associate_fixture_masks(path, room, [[(item, mask)]], groups + ['other_light']),
                         {'added_light': [], 'other_light': []})
        _, path, item, mask = self.lamp_scene(occluded=True)
        self.assertEqual(associate_fixture_masks(path, room, [[(item, mask)]], groups), {'added_light': []})

    def test_added_lamp_bakes_only_its_associated_mask_palette_with_part_roles(self):
        spec, _, item, mask = self.lamp_scene()
        pixels = np.full((240, 320, 3), [255, 0, 0], dtype=np.uint8)
        ys, xs = np.nonzero(mask)
        pixels[mask] = [20, 30, 50]
        pixels[ys[:len(ys)//4], xs[:len(xs)//4]] = [230, 245, 255]
        image = Image.fromarray(pixels)
        image.save(self.room / 'camera0-original.png')
        segmentation = self.root / 'segments'
        segmentation.mkdir()
        Image.fromarray(mask.astype('uint8') * 255).save(segmentation / 'light.png')
        (segmentation / 'camera0-original.json').write_text(json.dumps({
            'rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(), 'segments': [item]}))
        before = self.fingerprints()
        out = self.root / 'lamp-tinted'
        run(self.room, {}, out, reproject=True, segmentation=segmentation, additions=[spec])
        self.assertEqual(before, self.fingerprints())
        report = json.loads((out / 'reprojection.json').read_text())
        colors = report['fixture_tints']['added_light']['roles']
        np.testing.assert_allclose(colors['shade'], np.array([230, 245, 255]) / 255)
        np.testing.assert_allclose(colors['support'], np.array([20, 30, 50]) / 255)
        text = (out / 'assisted.obj').read_text()
        self.assertIn('usemtl fixture_tint_0_shade', text)
        self.assertIn('usemtl fixture_tint_0_support', text)
        second = self.root / 'lamp-tinted-second'
        run(out, {}, second, reproject=True, segmentation=segmentation)
        context = json.loads((second / 'retouch-appearance.json').read_text())
        self.assertEqual(context['added_groups'], [{'group': 'added_light', 'kind': 'lamp'}])
        second_report = json.loads((second / 'reprojection.json').read_text())
        self.assertEqual(len(second_report['preserved_fixture_materials']), 2)
        before_view = render(out / 'assisted.obj', self.camera, return_depth=True)
        after_view = render(second / 'assisted.obj', self.camera, return_depth=True)
        np.testing.assert_array_equal(np.asarray(before_view[0]), np.asarray(after_view[0]))
        np.testing.assert_array_equal(before_view[1], after_view[1])
        np.testing.assert_allclose(before_view[2], after_view[2])


if __name__ == '__main__':
    unittest.main()
