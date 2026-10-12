"""Local source sampling, real baked color/depth and repeated retouch preservation."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from room3d_checkpoint import inventory
from room3d_ladder_appearance import baked_ladder_material, palettes
from room3d_preview import render
from room3d_reconstruct import bake_projected
from room3d_retouch import run


class LadderAppearanceTests(unittest.TestCase):
    def scene(self,root):
        camera={'index':7,'origin':[0,0,0],'target':[0,0,1000],'fov':100,'roll':0}
        mesh=root/'mesh.obj'
        mesh.write_text('g primitive_0\nv -200 -200 1000\nv 200 -200 1000\n'
                        'v 200 200 1000\nv -200 200 1000\nf 1 2 3 4\n')
        mesh.with_suffix('.cameras').write_text('')
        _,part,depth=render(mesh,camera,return_depth=True)
        pixels=np.full((240,320,3),[255,0,0],dtype='uint8');pixels[part]=[20,30,40]
        source=np.zeros((240,320),bool);source[60:180,100:220]=True
        item={'mask':'object.png','label':'staircase','score':.34,'predicted_iou':.81}
        return mesh,{'cameras':[camera],'boundaries':[]},Image.fromarray(pixels),[(item,source)],part,depth

    def test_only_own_visible_pixels_and_foreground_veto_are_sampled(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);mesh,room,image,entries,part,depth=self.scene(root)
            fixture=part.copy();fixture[:,:160]=False
            pixels=np.asarray(image).copy();pixels[fixture]=[0,255,0];image=Image.fromarray(pixels)
            entries.append(({'mask':'lamp.png','label':'wall lamp','score':.35,'predicted_iou':.9},fixture))
            old=copy.deepcopy(entries);before=image.tobytes()
            sources={'primitive_0':[{'camera':7,'mask':'object.png'}]}
            result=palettes(mesh,room,[image],[entries],sources,{'primitive_0':'ladder'},[depth])
            np.testing.assert_allclose(result['primitive_0']['rgb'],np.array([20,30,40])/255)
            evidence=result['primitive_0']['sources'][0]
            self.assertEqual(evidence['matched_visible_pixels'],int((part&~fixture).sum()))
            self.assertEqual(evidence['source_confidence'][0]['score'],.34)
            self.assertEqual(evidence['source_confidence'][0]['predicted_iou'],.81)
            self.assertEqual(image.tobytes(),before)
            for (a,b),(c,d) in zip(entries,old,strict=True):
                self.assertEqual(a,c);np.testing.assert_array_equal(b,d)

    def test_excluded_unknown_and_occluded_sources_are_never_sampled(self):
        class Unreadable:
            def convert(self,*_):raise AssertionError('Excluded/unsupported source accessed')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);mesh,room,_,entries,_,depth=self.scene(root)
            sources={'primitive_0':[{'camera':7,'mask':'object.png'}]}
            for kwargs,observations,scene in [({'exclude_camera':0},sources,depth),
                    ({},{'primitive_0':[{'camera':7,'mask':'missing.png'}]},depth),
                    ({},sources,np.zeros(depth.shape))]:
                result=palettes(mesh,room,[Unreadable()],[entries],observations,
                                {'primitive_0':'ladder'},[scene],**kwargs)
                self.assertEqual(result,{})
            entries[0][0]['score']=.29
            self.assertEqual(palettes(mesh,room,[Unreadable()],[entries],sources,{'primitive_0':'ladder'},[depth]),{})

    def test_actual_bake_changes_only_ladder_color_and_preserves_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);mesh,room,image,entries,part,_=self.scene(root)
            views=[]
            for enabled in (False,True):
                out=root/str(enabled);out.mkdir()
                report=bake_projected(mesh,room,[image],out/'baked.obj',segments=[entries],
                    group_kinds={'primitive_0':'ladder'},interpreted_groups={'primitive_0'},
                    reconstructed_observations={'primitive_0':[{'camera':7,'mask':'object.png'}]},
                    local_ladder_appearance=enabled)
                views.append(render(out/'baked.obj',room['cameras'][0],return_depth=True))
                if enabled:self.assertIn('primitive_0',report['ladder_tints'])
            self.assertFalse(np.array_equal(np.asarray(views[0][0]),np.asarray(views[1][0])))
            actual=np.asarray(views[1][0]);np.testing.assert_array_equal(actual[part],np.tile([20,30,40],(int(part.sum()),1)))
            np.testing.assert_array_equal(views[0][1],views[1][1]);np.testing.assert_array_equal(views[0][2],views[1][2])
            absent=root/'absent';absent.mkdir()
            report=bake_projected(mesh,room,[image],absent/'baked.obj',segments=[entries],
                group_kinds={'primitive_0':'ladder'},interpreted_groups={'primitive_0'},
                reconstructed_observations={'missing':[{'camera':7,'mask':'object.png'}]},local_ladder_appearance=True)
            self.assertEqual(report['ladder_tints'],{})
            np.testing.assert_array_equal(np.asarray(render(absent/'baked.obj',room['cameras'][0])[0]),np.asarray(views[0][0]))

    def test_retouch_preserves_baked_tint_without_original_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);mesh,room,image,entries,_,_=self.scene(root)
            source=root/'source';source.mkdir()
            recipe={'local_ladder_appearance':True,'reconstructed_object_appearance':True,
                    'primitives':[{'kind':'ladder','interpretation':{'source':'shared_vertical_rail_prior'},
                                   'evidence':{'shared_observations':[{'camera':7,'mask':'object.png'}]}}]}
            bake_projected(mesh,room,[image],source/'assisted.obj',segments=[entries],
                group_kinds={'primitive_0':'ladder'},interpreted_groups={'primitive_0'},
                reconstructed_observations={'primitive_0':[{'camera':7,'mask':'object.png'}]},local_ladder_appearance=True)
            (source/'survey.json').write_text(json.dumps(room))
            (source/'assisted.json').write_text(json.dumps({'fitted_recipe':recipe}))
            image.save(source/'camera7-original.png')
            before=render(source/'assisted.obj',room['cameras'][0],return_depth=True);original=inventory(source)
            previous=source
            for i in range(3):
                out=root/f'retouch{i}';saved=inventory(previous)
                run(previous,{},out,reproject=True)
                report=json.loads((out/'reprojection.json').read_text())
                self.assertEqual(len(report['preserved_ladder_materials']),1)
                after=render(out/'assisted.obj',room['cameras'][0],return_depth=True)
                np.testing.assert_array_equal(np.asarray(before[0]),np.asarray(after[0]))
                np.testing.assert_array_equal(before[1],after[1]);np.testing.assert_array_equal(before[2],after[2])
                self.assertEqual(inventory(previous),saved);previous=out
            self.assertEqual(inventory(source),original)
            with self.assertRaisesRegex(ValueError,'heldout appearance'):
                bake_projected(source/'assisted.obj',room,[image],root/'forbidden.obj',
                               preserve_fixture_appearance=True,exclude_camera=0)
            self.assertFalse((root/'forbidden.obj').exists())
            for name in ('ladder_tint_0','preserved_ladder_12'):self.assertTrue(baked_ladder_material(name))
            for name in ('ladder_tint_0_extra','wall_ladder_tint_0'):self.assertFalse(baked_ladder_material(name))


if __name__=='__main__':unittest.main()
