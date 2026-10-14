"""Observed panel detail must preserve geometry, ownership and heldout isolation."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from batch_room3d import generate_mesh
from room3d_fixture_appearance import baked_fixture_role
from room3d_preview import render
from room3d_reconstruct import bake_projected
from room3d_shapes import generate


class PanelAppearanceTests(unittest.TestCase):
    def scene(self, root):
        room = {'boundaries': [{'index': 0, 'xmin': -3000, 'xmax': 3000,
                               'zmin': -3000, 'zmax': 3000, 'type': 4, 'flags': 0}],
                'cameras': [{'index': i, 'origin': [x, -700, -3000],
                             'target': [0, -1900, 0], 'fov': 230, 'roll': 0}
                            for i, x in enumerate((-500, 500))]}
        obj = generate('lamp', {'profile': 'ceiling_panel', 'height': 100, 'radius': 700,
            'housing_width': 700, 'thickness': 10, 'shade': 10, 'arms': 1}, [0, -1900, 0])
        mesh = root/'source.obj'
        generate_mesh(room, mesh, {'primitives':[obj], 'local_fixture_appearance':True})
        images, segments, refs = [], [], []
        for c in room['cameras']:
            _, mask = render(mesh, c, only_groups={'primitive_0'})
            _, xx = np.indices(mask.shape)
            pixels = np.full((240, 320, 3), [0, 255, 0], dtype='uint8')
            pixels[mask] = np.where((xx[mask] % 10 < 5)[:, None], [245, 180, 20], [30, 40, 160])
            images.append(Image.fromarray(pixels))
            item = {'label':'wall lamp chandelier','mask':f'own{c["index"]}.png','score':.34,'predicted_iou':.91}
            segments.append([(item, mask)])
            refs.append({'camera':c['index'],'mask':item['mask']})
        args = {'segments':segments, 'group_kinds':{'primitive_0':'lamp'},
                'interpreted_groups':{'primitive_0'}, 'fixture_observations':{'primitive_0':refs}}
        return mesh, room, images, args

    def test_actual_render_gains_only_own_detail_with_identical_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);mesh,room,images,args=self.scene(root)
            saved=[im.tobytes() for im in images];masks=copy.deepcopy(args['segments'])
            off,on=root/'off.obj',root/'on.obj'
            bake_projected(mesh,room,images,off,**args)
            result=bake_projected(mesh,room,images,on,**args,observed_panel_groups={'primitive_0'})
            self.assertIn('primitive_0',result['observed_panels'])
            evidence=result['observed_panels']['primitive_0']
            self.assertGreater(evidence['observed_texels'],100)
            self.assertTrue(all(s['unique_visible_source_pixels']>=16 for s in evidence['sources']))
            texture=np.asarray(Image.open(root/evidence['file']))
            self.assertFalse(((texture[:,:,1]>240)&(texture[:,:,0]<10)).any())
            for camera in room['cameras']:
                a,am,az=render(off,camera,return_depth=True);b,bm,bz=render(on,camera,return_depth=True)
                np.testing.assert_array_equal(am,bm);np.testing.assert_array_equal(az,bz)
                self.assertFalse(np.array_equal(a,b))
            self.assertEqual(saved,[im.tobytes() for im in images])
            for old,new in zip(masks,args['segments'],strict=True):
                self.assertEqual(old[0][0],new[0][0]);np.testing.assert_array_equal(old[0][1],new[0][1])

    def test_excluded_camera_and_low_confidence_never_supply_detail(self):
        class Unreadable:
            def convert(self,*_):
                raise AssertionError('Excluded source sampled')
            def save(self,*_):
                pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);mesh,room,images,args=self.scene(root)
            images[0]=Unreadable()
            # The existing baker exports arrays for all source images; use a
            # real image there and poison only the new sampler below.
            from room3d_complete import fixture_tints
            from room3d_panel_appearance import textures
            from room3d_preview import load_obj
            actual=[Image.new('RGB',(320,240),'red'),images[1]]
            palette=fixture_tints(room,actual,args['segments'],args['fixture_observations'],0)
            loaded=load_obj(mesh,-1,with_face_charts=True)
            groups=['primitive_0' if material.startswith('fixture_') else 'floor' for _,material in loaded[0]]
            depths=[render(mesh,c,return_depth=True)[2] for c in room['cameras']]
            result=textures(mesh,loaded[0],groups,loaded[2],room,images,args['segments'],
                args['fixture_observations'],palette,{'primitive_0'},depths,exclude_camera=0)
            self.assertEqual([s['camera'] for s in result['primitive_0'][1]['sources']],[1])
            weak=copy.deepcopy(args['segments']);weak[1][0][0]['predicted_iou']=.79
            self.assertEqual(textures(mesh,loaded[0],groups,loaded[2],room,images,weak,
                args['fixture_observations'],palette,{'primitive_0'},depths,exclude_camera=0),{})

    def test_foreign_masks_and_actual_scene_occlusion_cannot_supply_detail(self):
        from room3d_complete import fixture_tints
        from room3d_panel_appearance import textures
        from room3d_preview import load_obj

        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);mesh,room,images,args=self.scene(root)
            loaded=load_obj(mesh,-1,with_face_charts=True)
            groups=['primitive_0' if material.startswith('fixture_') else 'floor' for _,material in loaded[0]]
            depths=[render(mesh,c,return_depth=True)[2] for c in room['cameras']]
            palette=fixture_tints(room,images,args['segments'],args['fixture_observations'])
            foreign=copy.deepcopy(args['segments'])
            for entries in foreign:
                item,mask=entries[0]
                entries.append(({**item,'mask':'other.png','score':.9},mask.copy()))
            self.assertEqual(textures(mesh,loaded[0],groups,loaded[2],room,images,foreign,
                args['fixture_observations'],palette,{'primitive_0'},depths),{})
            obj=generate('lamp',{'profile':'ceiling_panel','height':100,'radius':700,
                'housing_width':700,'thickness':10,'shade':10,'arms':1},[0,-1900,0])
            occluded=root/'occluded.obj'
            generate_mesh(room,occluded,{'primitives':[obj,
                {'kind':'chest_body','box':[-2000,-2000,2000,2000,-1850,-1700]}],
                'local_fixture_appearance':True})
            scene=[render(occluded,c,return_depth=True)[2] for c in room['cameras']]
            self.assertEqual(textures(occluded,loaded[0],groups,loaded[2],room,images,args['segments'],
                args['fixture_observations'],palette,{'primitive_0'},scene),{})

    def test_observed_material_survives_repeated_reprojection_without_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);mesh,room,images,args=self.scene(root)
            current=root/'first.obj'
            report=bake_projected(mesh,room,images,current,**args,observed_panel_groups={'primitive_0'})
            material=report['observed_panels']['primitive_0']['material']
            self.assertEqual(baked_fixture_role(material),'shade')
            expected=[render(current,c,return_depth=True) for c in room['cameras']]
            for i in range(3):
                target=root/f'retouch{i}.obj'
                report=bake_projected(current,room,images,target,group_kinds=args['group_kinds'],
                    interpreted_groups=args['interpreted_groups'],preserve_fixture_appearance=True)
                self.assertTrue(report['preserved_fixture_materials'])
                for c,(image,mask,depth) in zip(room['cameras'],expected,strict=True):
                    actual,am,az=render(target,c,return_depth=True)
                    np.testing.assert_array_equal(image,actual);np.testing.assert_array_equal(mask,am);np.testing.assert_array_equal(depth,az)
                current=target
            with self.assertRaises(ValueError):
                bake_projected(current,room,images,root/'leak.obj',exclude_camera=0,preserve_fixture_appearance=True)


if __name__=='__main__':
    unittest.main()
