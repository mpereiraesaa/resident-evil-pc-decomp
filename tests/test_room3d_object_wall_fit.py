"""Actual scene fitting, reserved-mask veto, depth veto and support preservation."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from batch_room3d import generate_mesh
from room3d_connections import support_mesh
from room3d_ladder_supports import anchor_brackets
from room3d_object_wall_fit import fit, support_gates
from room3d_preview import render
from room3d_shapes import generate
from room3d_wall_recess import proposals, refit_support


class ObjectWallFitTests(unittest.TestCase):
    def scene(self, root):
        cameras = [{'index':i,'origin':origin,'target':[0,-1800,100],'fov':180,'roll':0}
                   for i,origin in enumerate(([0,-2000,-6000],[1000,-1500,-4000],[1800,-1200,-4200]))]
        room = {'cameras':cameras,'boundaries':[
            {'index':0,'flags':256,'type':1,'xmin':-2000,'xmax':2000,'zmin':0,'zmax':1800},
            {'index':1,'flags':0,'type':1,'xmin':-2000,'xmax':2000,'zmin':-6500,'zmax':-6400}]}
        obj = generate('ladder',{'width':800,'height':3600,'thickness':50,'stand_off':180,'rungs':9},
                       [0,0,100],np.pi+.35)
        recipe = {'shared_wall_boundaries':[0],'skip_boundaries':[1],'primitives':[obj],
                  'structural_surfaces':[{'group':'boundary_0','material':'wall',
                    'points':[[-2000,0,0],[-2000,-4000,0],[2000,-4000,0],[2000,0,0]]}]}
        generate_mesh(room,root/'before.obj',recipe)
        obj,report = anchor_brackets(obj,*support_mesh(root/'before.obj'),{'boundary_0'})
        self.assertEqual(report['status'],'anchored')
        obj['interpretation'] = {'source':'shared_vertical_rail_prior'}
        obj['evidence'] = {'shared_observations':[{'camera':i,'mask':f'object{i}.png'} for i in range(3)]}
        recipe['primitives'][0] = obj
        generate_mesh(room,root/'before.obj',recipe)
        truth,_ = proposals(room,recipe,0,[c['origin'] for c in cameras[:2]],(360,),local=True)[0]
        generate_mesh(room,root/'truth.obj',truth)
        obj,report = refit_support(obj,*support_mesh(root/'truth.obj'),{'boundary_0'})
        self.assertEqual(report['status'],'anchored')
        truth['primitives'][0] = obj
        generate_mesh(room,root/'truth.obj',truth)
        segments,depths = [],[]
        for c in cameras:
            part = render(root/'truth.obj',c,return_depth=True,only_groups={'primitive_0'})[2]
            target = render(root/'truth.obj',c,return_depth=True)[2]
            segments.append([({'label':'staircase','mask':f"object{c['index']}.png",'score':.9,
                               'predicted_iou':.95},np.isfinite(part))])
            valid=np.isfinite(target)
            depths.append((np.where(valid,target,10000),valid))
        return room,recipe,segments,depths,truth

    def test_retention_preserves_confidence_collision_and_rails_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);room,recipe,segments,depths,_=self.scene(root)
            old_room,old_recipe,old_segments=copy.deepcopy((room,recipe,segments))
            selected,report=fit(room,recipe,segments,depths,root/'fit',generate_mesh)
            self.assertEqual(report['accepted'],1,report)
            obj=report['objects'][0]
            self.assertEqual(obj['training_cameras'],[0,1]);self.assertEqual(obj['heldout_cameras'],[2])
            self.assertTrue(all(obj['gates'].values()))
            for source in obj['sources']:
                self.assertEqual(source['masks'][0]['score'],.9)
                self.assertEqual(source['masks'][0]['predicted_iou'],.95)
            np.testing.assert_array_equal(selected['primitives'][0]['parts'][:-4],recipe['primitives'][0]['parts'][:-4])
            self.assertEqual(recipe,old_recipe);self.assertEqual(room,old_room)
            for a,b in zip(segments,old_segments,strict=True):
                self.assertEqual(a[0][0],b[0][0]);np.testing.assert_array_equal(a[0][1],b[0][1])
            repeated,again=fit(room,selected,segments,depths,root/'again',generate_mesh)
            self.assertEqual(repeated,selected);self.assertEqual(again['accepted'],0)
            self.assertEqual(again['objects'][0]['status'],'already_present')

    def test_reserved_actual_baseline_silhouette_can_veto_without_reselection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);room,recipe,segments,depths,_=self.scene(root)
            _,good=fit(room,recipe,segments,depths,root/'good',generate_mesh)
            c=room['cameras'][-1]
            scene=render(root/'before.obj',c,return_depth=True)[2]
            part=render(root/'before.obj',c,return_depth=True,only_groups={'primitive_0'})[2]
            segments[-1][0]=(segments[-1][0][0],np.isfinite(part)&(part<=scene+.001))
            selected,bad=fit(room,recipe,segments,depths,root/'bad',generate_mesh)
            self.assertEqual(selected,recipe)
            a,b=good['objects'][0],bad['objects'][0]
            self.assertEqual(b['status'],'selected_candidate_vetoed')
            self.assertFalse(b['gates']['reserved_masks'])
            self.assertEqual(a['trials'],b['trials'])
            self.assertEqual(a['selected_training_candidate'],b['selected_training_candidate'])

    def test_fixed_actual_scene_depth_can_veto_and_roll_back_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);room,recipe,segments,depths,_=self.scene(root)
            target_depths=[]
            for c,(truth,truth_valid) in zip(room['cameras'],depths,strict=True):
                target=render(root/'before.obj',c,return_depth=True)[2]
                known=np.isfinite(target)&truth_valid
                valid=np.zeros(known.shape,bool)
                valid[known]=abs(target[known]-truth[known])>50
                self.assertTrue(valid.any())
                target_depths.append((np.where(np.isfinite(target),target,10000),valid))
            selected,report=fit(room,recipe,segments,target_depths,root/'fit',generate_mesh)
            self.assertEqual(selected,recipe)
            self.assertEqual(report['objects'][0]['status'],'selected_candidate_vetoed')
            self.assertFalse(report['objects'][0]['gates']['scene_depth'])

    def test_wall_movement_cannot_detach_another_supported_ladder(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);room,recipe,_,_,truth=self.scene(root)
            second=copy.deepcopy(recipe['primitives'][0])
            recipe['primitives'].append(second);truth['primitives'].append(copy.deepcopy(second))
            generate_mesh(room,root/'before2.obj',recipe)
            generate_mesh(room,root/'truth2.obj',truth)
            gates=support_gates(recipe,truth,root/'before2.obj',root/'truth2.obj',0)
            self.assertTrue(gates['finite_wall_contact'])
            self.assertFalse(gates['existing_supported_objects'])

    def test_unreferenced_masks_and_manual_objects_are_not_fitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);room,recipe,segments,depths,_=self.scene(root)
            recipe['primitives'][0]['evidence']['shared_observations']=[]
            selected,report=fit(room,recipe,segments,depths,root/'empty',generate_mesh)
            self.assertEqual(selected,recipe);self.assertEqual(report['accepted'],0)
            self.assertEqual(report['objects'][0]['status'],'insufficient_source_views')
            recipe['primitives'][0]['interpretation']['source']='manual'
            selected,report=fit(room,recipe,segments,depths,root/'manual',generate_mesh)
            self.assertEqual(selected,recipe);self.assertEqual(report['objects'],[])


if __name__=='__main__':
    unittest.main()
