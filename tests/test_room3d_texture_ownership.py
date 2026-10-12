"""Actual baked texture regression for overlapping object/architecture masks."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from room3d_preview import render
from room3d_reconstruct import bake_projected
from room3d_texture_ownership import (
    completion_segments,
    owned_masks,
    reconstructed_sources,
)


class TextureOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.camera = {'index':7,'origin':[0,0,0],'target':[0,0,1000],'fov':100,'roll':0}
        self.room = {'cameras':[self.camera]}
        self.mask = np.zeros((240,320),bool)
        self.mask[80:160,150:170] = True
        self.item = {'mask':'ladder.png','label':'staircase','score':.9,'predicted_iou':.95}
        self.wall = {'mask':'wall.png','label':'wall','score':.9,'predicted_iou':.95}
        self.segments = [[(self.wall,np.ones((240,320),bool)),(self.item,self.mask)]]
        self.sources = {'primitive_0':[{'camera':7,'mask':'ladder.png'}]}
        self.kinds = {'wall':'wall','primitive_0':'ladder'}

    def test_source_binding_confidence_and_reserved_camera(self):
        original = copy.deepcopy(self.segments)
        mask, evidence = owned_masks(self.room,self.segments,self.sources,self.kinds)
        np.testing.assert_array_equal(mask[0],self.mask)
        self.assertEqual(evidence[0]['score'],.9)
        self.assertEqual(evidence[0]['predicted_iou'],.95)
        for sources, kinds, excluded in [({},self.kinds,None),(self.sources,{},None),
                                        (self.sources,self.kinds,0)]:
            masks, evidence = owned_masks(self.room,self.segments,sources,kinds,excluded)
            self.assertFalse(masks[0].any())
            self.assertEqual(evidence,[])
        for bad in (float('nan'),.2,1.1):
            item = dict(self.item,score=bad)
            masks,_ = owned_masks(self.room,[[(item,self.mask)]],self.sources,self.kinds)
            self.assertFalse(masks[0].any())
        for label in ('staircase','wooden staircase','ladder'):
            item = dict(self.item,label=label,score=.34,predicted_iou=.81)
            masks,evidence = owned_masks(self.room,[[(item,self.mask)]],self.sources,self.kinds)
            np.testing.assert_array_equal(masks[0],self.mask)
            self.assertEqual(evidence[0]['score'],.34)
        item = dict(self.item,score=.9,predicted_iou=.79)
        masks,_ = owned_masks(self.room,[[(item,self.mask)]],self.sources,self.kinds)
        self.assertFalse(masks[0].any())
        for (a,b),(c,d) in zip(self.segments[0],original[0],strict=True):
            self.assertEqual(a,c)
            np.testing.assert_array_equal(b,d)

    def test_opt_in_retained_object_only_and_completion_exclusion(self):
        primitive = {'kind':'ladder','interpretation':{'source':'shared_vertical_rail_prior'},
                     'evidence':{'shared_observations':self.sources['primitive_0']}}
        recipe = {'primitives':[primitive]}
        self.assertIsNone(reconstructed_sources(recipe))
        recipe['reconstructed_object_appearance'] = True
        self.assertEqual(reconstructed_sources(recipe),self.sources)
        primitive['omit_geometry'] = True
        self.assertEqual(reconstructed_sources(recipe),{})
        masks,_ = owned_masks(self.room,self.segments,self.sources,self.kinds)
        filtered = completion_segments(self.segments,masks)
        self.assertFalse(filtered[0][0][1][self.mask].any())
        np.testing.assert_array_equal(filtered[0][1][1],self.mask)
        self.assertTrue(self.segments[0][0][1].all())

    def test_actual_wall_texture_loses_ghost_without_changing_geometry(self):
        self.mask[80:160,120:200] = True
        pixels = np.full((240,320,3),100,dtype='uint8')
        pixels[self.mask] = [255,0,0]
        image = Image.fromarray(pixels)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root/'mesh.obj'
            mesh.write_text('g wall\nv -600 -600 1000\nv 600 -600 1000\n'
                            'v 600 600 1000\nv -600 600 1000\nf 1 2 3 4\n'
                            'g primitive_0\nv -20 -300 900\nv 20 -300 900\n'
                            'v 20 300 900\nv -20 300 900\nf 5 6 7 8\n')
            mesh.with_suffix('.cameras').write_text('')
            views = []
            for toggle in ('off','on','empty','absent'):
                out = root/toggle
                out.mkdir()
                report = bake_projected(mesh,self.room,[image],out/'baked.obj',
                    segments=self.segments,group_kinds={**self.kinds,'missing':'ladder'},
                    reconstructed_observations=(self.sources if toggle=='on' else {} if toggle=='empty' else {'missing':self.sources['primitive_0']} if toggle=='absent' else None))
                views.append(render(out/'baked.obj',self.camera,return_depth=True,only_groups={'wall'}))
                if toggle=='on':
                    box = report['completion_patches']['wall']['source_box']
                    x0,y0,x1,y1 = box
                    self.assertFalse(self.mask[y0:y1,x0:x1].any())
                    filtered = np.asarray(Image.open(out/'baked-camera0-wall.png'))
                    self.assertFalse((filtered[self.mask,0]>200).any())
            colors = [np.asarray(v[0]) for v in views]
            ghost = lambda a: (a[:,:,0]>200)&(a[:,:,1]<30)
            self.assertGreater(ghost(colors[0]).sum(),50)
            self.assertEqual(ghost(colors[1]).sum(),0)
            np.testing.assert_array_equal(colors[0],colors[2])
            np.testing.assert_array_equal(colors[0],colors[3])
            for view in views[1:]:
                np.testing.assert_array_equal(view[1],views[0][1])
                np.testing.assert_array_equal(view[2],views[0][2])
            np.testing.assert_array_equal(np.asarray(image),pixels)

    def test_object_projection_is_unchanged_and_heldout_source_is_not_sampled(self):
        pixels = np.full((240,320,3),90,dtype='uint8')
        pixels[self.mask] = [255,0,0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root/'mesh.obj'
            mesh.write_text('g primitive_0\nv -80 -300 1000\nv 80 -300 1000\n'
                            'v 80 300 1000\nv -80 300 1000\nf 1 2 3 4\n')
            mesh.with_suffix('.cameras').write_text('')
            views = []
            for enabled in (False,True):
                out = root/str(enabled)
                out.mkdir()
                bake_projected(mesh,self.room,[Image.fromarray(pixels)],out/'baked.obj',
                    segments=self.segments,group_kinds=self.kinds,
                    reconstructed_observations=self.sources if enabled else None)
                views.append(render(out/'baked.obj',self.camera))
            np.testing.assert_array_equal(np.asarray(views[0][0]),np.asarray(views[1][0]))
            self.assertGreater((np.asarray(views[1][0])[:,:,0]>200).sum(),50)
            out = root/'heldout'
            out.mkdir()
            report = bake_projected(mesh,self.room,[Image.fromarray(pixels)],out/'baked.obj',
                segments=self.segments,group_kinds=self.kinds,exclude_camera=0,
                reconstructed_observations=self.sources)
            self.assertEqual(report['reconstructed_object_appearance']['sources'],[])
            self.assertEqual(report['completion_patches'],{})


if __name__ == '__main__':
    unittest.main()
