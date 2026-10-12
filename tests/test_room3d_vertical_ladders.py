"""Actual batch-stage geometry, scene vetoes, confidence and idempotence."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from batch_room3d import generate_mesh
from room3d_preview import render
from room3d_segment import texture_support
from room3d_shape_fit import associate, observations
from room3d_shapes import generate
from room3d_vertical_ladders import fit_vertical_ladders


class VerticalLadderTests(unittest.TestCase):
    def scene(self, root):
        room = {'boundaries':[{'index':0,'xmin':-2000,'xmax':2000,'zmin':-100,'zmax':0,
                               'type':1,'flags':256},
                              {'index':1,'xmin':-2000,'xmax':2000,'zmin':6500,'zmax':6600,
                               'type':1,'flags':256}],
                'cameras':[{'index':i,'origin':origin,'target':[0,-1800,180],'roll':0,'fov':230}
                           for i,origin in enumerate(([800,-1800,6000],[-900,-1000,2200],[1500,-1500,5000]))]}
        truth = generate('ladder', {'width':800,'height':3600,'thickness':50,'stand_off':180,
                                    'rungs':9,'rung_phase':-.1})
        recipe = {'primitives':[],'wall_height':6000}
        mesh, baseline = root/'truth.obj', root/'baseline.obj'
        generate_mesh(room, mesh, {**recipe,'primitives':[truth]})
        generate_mesh(room, baseline, recipe)
        segments, depths = [], []
        for i,camera in enumerate(room['cameras']):
            _,mask,_depth = render(mesh,camera,return_depth=True,only_groups={'primitive_0'})
            if i==0:
                y,x = np.nonzero(mask)
                hull = np.zeros(mask.shape,'uint8')
                cv2.fillConvexPoly(hull,cv2.convexHull(np.column_stack((x,y)).astype('int32')),1)
                mask = hull.astype(bool)
            target = render(mesh,camera,return_depth=True)[2]
            depths.append((np.where(np.isfinite(target),target,10000),np.ones(mask.shape,bool)))
            segments.append([({'label':'staircase','mask':f'camera{i}-ladder.png',
                              'score':.9,'predicted_iou':.95},mask)])
        return room,recipe,segments,depths,baseline

    def test_stage_retains_supported_ladder_without_duplicate_and_preserves_confidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            room,recipe,segments,depths,_ = self.scene(root)
            saved = copy.deepcopy(recipe)
            result,report = fit_vertical_ladders(room,recipe,segments,depths,root/'fit',generate_mesh)
            self.assertEqual(report['accepted'],1,report)
            self.assertEqual(recipe,saved)
            ladder = result['primitives'][-1]
            self.assertEqual(ladder['kind'],'ladder')
            self.assertEqual(ladder['shape']['parameters']['rungs'],9)
            self.assertTrue(all(report['objects'][0]['gates'].values()))
            for observation in ladder['evidence']['shared_observations']:
                evidence = observation['confidence_evidence'][0]
                self.assertEqual(evidence['detector_score'],.9)
                self.assertEqual(evidence['sam_prediction_source'],'source_mask')
                self.assertEqual(evidence['predicted_iou'],.95)
            again,repeat = fit_vertical_ladders(room,result,segments,depths,root/'repeat',generate_mesh)
            self.assertEqual(again,result)
            self.assertEqual(repeat['accepted'],0)
            self.assertEqual(repeat['objects'][0]['status'],'already_present')

    def test_reserved_mask_veto_cannot_reselect_training_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            room,recipe,segments,depths,_ = self.scene(root)
            groups = associate(observations(room,segments,depths))
            with patch('room3d_vertical_ladders.associate',return_value=groups):
                _,good = fit_vertical_ladders(room,recipe,segments,depths,root/'good',generate_mesh)
            poisoned = copy.deepcopy(groups)
            poisoned[0][-1]['mask'] = np.roll(poisoned[0][-1]['mask'],80,axis=1)
            with patch('room3d_vertical_ladders.associate',return_value=poisoned):
                result,veto = fit_vertical_ladders(room,recipe,segments,depths,root/'veto',generate_mesh)
            self.assertEqual(result,recipe)
            self.assertEqual(veto['accepted'],0)
            self.assertFalse(veto['objects'][0]['gates']['reserved_masks'])
            self.assertEqual(good['objects'][0]['trials'],veto['objects'][0]['trials'])
            self.assertEqual(good['objects'][0]['selected_training_candidate'],
                             veto['objects'][0]['selected_training_candidate'])

    def test_actual_scene_depth_veto_rolls_back_whole_recipe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            room,recipe,segments,depths,baseline = self.scene(root)
            groups = associate(observations(room,segments,depths))
            wall_depths = []
            for camera,entries in zip(room['cameras'],segments,strict=True):
                depth = render(baseline,camera,return_depth=True)[2]
                wall_depths.append((np.where(np.isfinite(depth),depth,10000),
                                    entries[0][1]&np.isfinite(depth)))
            with patch('room3d_vertical_ladders.associate',return_value=groups):
                result,report = fit_vertical_ladders(room,recipe,segments,wall_depths,root/'fit',generate_mesh)
            self.assertEqual(result,recipe)
            self.assertEqual(report['accepted'],0)
            self.assertFalse(report['objects'][0]['gates']['scene_depth'])

    def test_no_stair_observations_abstains_and_ladder_material_uses_stair_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            room,recipe,_,depths,_ = self.scene(root)
            result,report = fit_vertical_ladders(room,recipe,[[],[],[]],depths,root/'fit',generate_mesh)
            self.assertEqual(result,recipe)
            self.assertEqual(report['objects'],[])
            target = np.zeros((240,320),bool)
            target[50:100,50:100] = True
            entries = [({'label':'staircase','score':.9,'predicted_iou':.95},target),
                       ({'label':'wall','score':.9,'predicted_iou':.95},~target)]
            self.assertEqual(texture_support(entries,'ladder',np.array([[60,60],[120,120]])).tolist(),[True,False])


if __name__=='__main__':
    unittest.main()
