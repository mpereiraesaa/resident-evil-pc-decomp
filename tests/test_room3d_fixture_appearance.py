"""Bright shade cores preserve geometry and use only own training observations."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_fixture_appearance import baked_fixture_role, core_patch
from room3d_preview import load_obj, render
from room3d_reconstruct import bake_projected


class FixtureAppearanceTests(unittest.TestCase):
    def setUp(self):
        self.camera = {'index': 7, 'origin': [0, 0, 0], 'target': [0, 0, 1000],
                       'roll': 0, 'fov': 100}
        pixels = np.full((240, 320, 3), [255, 0, 0], dtype=np.uint8)
        pixels[100:140, 140:180] = 20
        pixels[110:130, 150:170] = 250
        self.image = Image.fromarray(pixels)
        self.mask = np.zeros((240, 320), dtype=bool)
        self.mask[100:140, 140:180] = True
        self.item = {'mask': 'lamp.png', 'label': 'lamp', 'score': .8, 'predicted_iou': .9}
        self.palette = {'sources': [{'camera': 7, 'mask': 'lamp.png'}],
                        'roles': {'support': [20/255]*3, 'shade': [250/255]*3}}

    def patch(self, image=None, area=1600, excluded=None):
        return core_patch(self.palette, {'cameras': [self.camera]},
                          [self.image if image is None else image],
                          [[(self.item, self.mask)]], {7: area}, excluded)

    def test_own_pixels_area_and_abstention(self):
        texture, report = self.patch()
        self.assertEqual(report['fraction'], .25)
        self.assertEqual(report['sources'][0]['bright_pixels'], 400)
        a = np.asarray(texture)
        np.testing.assert_array_equal(a[32, 32], [250]*3)
        np.testing.assert_array_equal(a[0, 0], [20]*3)
        self.assertAlmostEqual(self.patch(area=3200)[1]['fraction'], .125)
        self.assertIsNone(self.patch(image=Image.new('RGB', (320, 240), 'white')))
        self.assertIsNone(self.patch(area=0))

    def test_excluded_image_is_never_read(self):
        class Unreadable:
            def convert(self, *_):
                raise AssertionError('Reserved image accessed')
        self.assertIsNone(self.patch(image=Unreadable(), excluded=0))
        self.palette['sources'][0]['mask'] = 'other.png'
        self.assertIsNone(self.patch(image=Unreadable()))

    def test_retouch_reprojection_preserves_baked_roles_and_texture_charts(self):
        from room3d_checkpoint import inventory
        from room3d_retouch import read_mesh, run, write_mesh

        for name, role in [('fixture_tint_9_shade', 'shade'),
                           ('preserved_fixture_2_support', 'support'),
                           ('fixture_tint_9', None), ('wall_shade', None),
                           ('fixture_tint_9_shade_extra', None)]:
            self.assertEqual(baked_fixture_role(name), role)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'
            source.mkdir()
            mesh = source/'assisted.obj'
            mesh.write_text('mtllib shade.mtl\ng primitive_0\n'
                            'usemtl fixture_tint_9_shade\n'
                            'v -200 -200 1000 .8 .7 .6\nv 200 -200 1000 .8 .7 .6\n'
                            'v 200 200 1000 .8 .7 .6\nv -200 200 1000 .8 .7 .6\n'
                            'vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3 4/4\n'
                            'usemtl fixture_tint_9_support\n'
                            'v -50 200 900\nv 50 200 900\nv 0 350 900\nf 5 6 7\n')
            mesh.with_suffix('.cameras').write_text('')
            pixels = np.full((16, 16, 3), [20, 40, 60], dtype='uint8')
            pixels[4:12, 6:10] = [230, 220, 210]
            Image.fromarray(pixels).save(source/'core.png')
            (source/'shade.mtl').write_text('newmtl fixture_tint_9_shade\n'
                                           'Kd .9 .8 .7\nmap_Kd core.png\n'
                                           'newmtl fixture_tint_9_support\nKd .1 .2 .3\n')
            survey = {'cameras': [self.camera], 'boundaries': []}
            (source/'survey.json').write_text(json.dumps(survey))
            (source/'assisted.json').write_text(json.dumps({'fitted_recipe': {
                'primitives': [{'kind': 'lamp', 'interpretation': {'source': 'test'}}],
                'local_fixture_appearance': True, 'fixture_core_appearance': True}}))
            Image.new('RGB', (320, 240), 'red').save(source/'camera7-original.png')
            before = render(mesh, self.camera, return_depth=True)
            out = root/'retouched'
            run(source, {}, out, reproject=True)
            report = json.loads((out/'reprojection.json').read_text())
            self.assertEqual(len(report['preserved_fixture_materials']), 2)
            self.assertEqual(report['fixture_cores'], {})
            for path in (out/'assisted.obj', root/'rebaked.obj'):
                if path.parent == root:
                    report = bake_projected(out/'assisted.obj', survey, [self.image], path,
                                            interpreted_groups={'primitive_0'},
                                            group_kinds={'primitive_0': 'lamp'},
                                            preserve_fixture_appearance=True)
                    self.assertEqual(len(report['preserved_fixture_materials']), 2)
                after = render(path, self.camera, return_depth=True)
                np.testing.assert_array_equal(np.asarray(before[0]), np.asarray(after[0]))
                np.testing.assert_array_equal(before[1], after[1])
                np.testing.assert_array_equal(before[2], after[2])
            self.assertTrue((source/'shade.mtl').exists())
            # A second and third CLI-level operation must work from the previous
            # variant alone; it intentionally has no original assisted.json.
            original_export = inventory(source)
            first_variant = inventory(out)
            second = root/'second'
            run(out, {}, second, reproject=True)
            self.assertFalse((second/'assisted.json').exists())
            np.testing.assert_array_equal(np.asarray(before[0]),
                                          np.asarray(render(second/'assisted.obj', self.camera)[0]))
            expected = root/'expected'
            shutil.copytree(second, expected)
            edits = {'primitive_0': {'translate': [40, 0, 0]}}
            write_mesh(read_mesh(second/'assisted.obj'), edits, expected/'assisted.obj')
            third = root/'third'
            second_variant = inventory(second)
            run(second, edits, third, reproject=True)
            expected_view = render(expected/'assisted.obj', self.camera, return_depth=True)
            third_view = render(third/'assisted.obj', self.camera, return_depth=True)
            np.testing.assert_array_equal(np.asarray(expected_view[0]), np.asarray(third_view[0]))
            np.testing.assert_array_equal(expected_view[1], third_view[1])
            np.testing.assert_allclose(expected_view[2], third_view[2])
            self.assertEqual(inventory(source), original_export)
            self.assertEqual(inventory(out), first_variant)
            self.assertEqual(inventory(second), second_variant)

    def test_prebaked_appearance_cannot_leak_into_heldout_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'unused.obj'
            with self.assertRaisesRegex(ValueError, 'heldout appearance'):
                bake_projected(Path(directory)/'missing.obj', {'cameras': []}, [], target,
                               preserve_fixture_appearance=True, exclude_camera=0)
            self.assertFalse(target.exists())

    def test_old_variant_without_semantics_keeps_uv_edits_but_rejects_reprojection_early(self):
        from room3d_checkpoint import inventory
        from room3d_retouch import run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'old-variant'
            source.mkdir()
            (source/'assisted.obj').write_text('g object\nv 0 0 1\nv 1 0 1\nv 0 1 1\nf 1 2 3\n')
            (source/'survey.json').write_text(json.dumps({'cameras': [], 'boundaries': []}))
            original = inventory(source)
            run(source, {}, root/'uv-variant')
            rejected = root/'rejected'
            with self.assertRaisesRegex(ValueError, 'appearance semantics'):
                run(source, {}, rejected, reproject=True)
            self.assertFalse(rejected.exists())
            self.assertEqual(inventory(source), original)

    def test_baked_core_changes_appearance_only_and_quad_has_shared_chart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / 'shape.obj'
            mesh.write_text('g lamp\nusemtl fixture_shade\n'
                            'v -200 -200 1000\nv 200 -200 1000\n'
                            'v 200 200 1000\nv -200 200 1000\nf 1 2 3 4\n')
            mesh.with_suffix('.cameras').write_text('')
            plain = load_obj(mesh, -1)
            charted = load_obj(mesh, -1, with_face_charts=True)
            self.assertEqual(len(plain), 2)
            self.assertEqual(len(charted), 3)
            self.assertIs(charted[2][0], charted[2][1])
            for a, b in zip(plain[0], charted[0], strict=True):
                np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(load_obj(mesh, -1, only_materials={'other'})[0], [])
            renders = []
            for enabled, excluded in [(False, None), (True, None), (True, 0)]:
                out = root / f'baked-{enabled}-{excluded}.obj'
                report = bake_projected(
                    mesh, {'cameras': [self.camera]}, [self.image], out,
                    segments=[[(self.item, self.mask)]], group_kinds={'lamp': 'lamp'},
                    interpreted_groups={'lamp'}, exclude_camera=excluded,
                    fixture_observations={'lamp': [{'camera': 7, 'mask': 'lamp.png'}]},
                    fixture_core_appearance=enabled,
                )
                renders.append(render(out, self.camera, return_depth=True))
                if enabled:
                    self.assertEqual(bool(report['fixture_cores']), excluded is None)
                else:
                    self.assertNotIn('fixture_cores', report)
            flat, core, reserved = renders
            for r in (core, reserved):
                np.testing.assert_array_equal(flat[1], r[1])
                np.testing.assert_allclose(flat[2], r[2])
            flat_bright = (np.asarray(flat[0])[..., 0] > 200) & flat[1]
            core_bright = (np.asarray(core[0])[..., 0] > 200) & core[1]
            self.assertGreater(core_bright.sum(), flat_bright.sum()*.15)
            self.assertLess(core_bright.sum(), flat_bright.sum()*.4)
            # Both source triangles meet without repeating the patch at the diagonal.
            a = np.asarray(core[0])
            self.assertGreater(a[120, 160, 0], 240)
            self.assertLess(a[102, 142, 0], 30)


if __name__ == '__main__':
    unittest.main()
