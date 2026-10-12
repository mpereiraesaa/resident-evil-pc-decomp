"""Finite connected visual wall proposals preserve collision and object geometry."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from batch_room3d import generate_mesh
from room3d_connections import support_mesh
from room3d_ladder_supports import anchor_brackets, finite_contact
from room3d_preview import render
from room3d_shapes import generate
from room3d_wall_recess import proposals, refit_support


class WallRecessTests(unittest.TestCase):
    def setUp(self):
        self.room = {'boundaries':[{'index':0,'flags':256,'xmin':-2000,'xmax':2000,
                                   'zmin':0,'zmax':1800}]}
        obj = generate('ladder',{'width':800,'height':3600,'thickness':50,
                                'stand_off':180,'rungs':9},[0,0,100],np.pi)
        obj['wall_attachment'] = {'bracket_groups':['boundary_0']*4}
        self.recipe = {'shared_wall_boundaries':[0],'primitives':[obj],
                       'structural_surfaces':[{'group':'boundary_0','material':'wall',
                         'points':[[-2000,0,0],[-2000,-4000,0],[2000,-4000,0],[2000,0,0]]}]}
        self.cameras = [[0,-2000,-6000],[1000,-1500,-4000]]

    def test_retreat_is_bounded_connected_and_does_not_mutate_collision_or_rails(self):
        before = copy.deepcopy((self.room,self.recipe))
        rows = proposals(self.room,self.recipe,0,self.cameras)
        self.assertEqual(len(rows),7)
        for candidate,evidence in rows:
            self.assertEqual(candidate['primitives'],self.recipe['primitives'])
            surfaces = candidate['structural_surfaces']
            self.assertEqual(len(surfaces),3)
            front = np.asarray(surfaces[0]['points'])
            self.assertTrue((front[:,2]==evidence['distance']).all())
            self.assertTrue((front[:,2]<1800).all())
            original = np.asarray(self.recipe['structural_surfaces'][0]['points'])
            for side,(a,b) in zip(surfaces[1:],((0,1),(2,3)),strict=True):
                np.testing.assert_array_equal(side['points'],[original[a],original[b],front[b],front[a]])
        self.assertEqual((self.room,self.recipe),before)

    def test_protected_doors_unshared_boundaries_and_opposing_cameras_abstain(self):
        self.assertEqual(proposals(self.room,self.recipe,0,[self.cameras[0],[0,-1000,3000]]),[])
        self.recipe['shared_wall_boundaries'] = []
        self.assertEqual(proposals(self.room,self.recipe,0,self.cameras),[])
        self.recipe['shared_wall_boundaries'] = [0]
        self.recipe['primitives'].append({'kind':'door_panel','hide_with':'boundary_0'})
        self.assertEqual(proposals(self.room,self.recipe,0,self.cameras),[])

    def test_invalid_extent_and_distance_are_rejected(self):
        for distances in ((0,),(-1,),(1001,),(float('nan'),)):
            with self.assertRaises(ValueError):
                proposals(self.room,self.recipe,0,self.cameras,distances)
        self.recipe['structural_surfaces'][0]['points'] = [[-100,0,0],[-100,-4000,0],[100,-4000,0],[100,0,0]]
        self.assertEqual(proposals(self.room,self.recipe,0,self.cameras),[])

    def test_rendered_rail_occlusion_improves_with_connected_wall_retreat(self):
        self.room['boundaries'][0]['type'] = 1
        self.room['cameras'] = [{'index':i,'origin':origin,'target':[0,-1800,100],
                                 'fov':160,'roll':0} for i,origin in enumerate(self.cameras)]
        obj = generate('ladder',{'width':800,'height':3600,'thickness':50,
                                'stand_off':180,'rungs':9},[0,0,100],np.pi+.35)
        obj['wall_attachment'] = {'bracket_groups':['boundary_0']*4}
        self.recipe['primitives'] = [obj]
        candidate,_ = proposals(self.room,self.recipe,0,self.cameras,distances=(360,))[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root/'before.obj',root/'after.obj']
            for path,recipe in zip(paths,(self.recipe,candidate),strict=True):
                generate_mesh(self.room,path,recipe)
            for camera in self.room['cameras']:
                part_before = render(paths[0],camera,return_depth=True,only_groups={'primitive_0'})[2]
                part_after = render(paths[1],camera,return_depth=True,only_groups={'primitive_0'})[2]
                np.testing.assert_array_equal(part_before,part_after)
                fractions = []
                for path in paths:
                    scene = render(path,camera,return_depth=True)[2]
                    visible = np.isfinite(part_before)&(part_before<=scene+.001)
                    fractions.append(visible.sum()/np.isfinite(part_before).sum())
                self.assertGreater(fractions[1]-fractions[0],.1)
                self.assertGreater(fractions[1],.95)

    def test_refitted_brackets_keep_rail_endpoints_and_failed_support_rolls_back(self):
        self.room['boundaries'][0]['type'] = 1
        self.room['cameras'] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_mesh(self.room,root/'before.obj',self.recipe)
            attached,report = anchor_brackets(self.recipe['primitives'][0],
                *support_mesh(root/'before.obj'),{'boundary_0'})
            self.assertEqual(report['status'],'anchored')
            self.recipe['primitives'][0] = attached
            candidate,_ = proposals(self.room,self.recipe,0,self.cameras,distances=(360,))[0]
            generate_mesh(self.room,root/'after.obj',candidate)
            triangles,groups = support_mesh(root/'after.obj')
            result,report = refit_support(attached,triangles,groups,{'boundary_0'})
            self.assertEqual(report['status'],'anchored')
            self.assertTrue(finite_contact(result,triangles,groups))
            np.testing.assert_array_equal(result['parts'][:-4],attached['parts'][:-4])
            np.testing.assert_allclose(np.asarray(result['parts'][-4:])[:,4:].mean(1),
                                       np.asarray(attached['parts'][-4:])[:,4:].mean(1),atol=1e-8)
            failed,_ = refit_support(attached,triangles,groups,{'missing'})
            self.assertIs(failed,attached)

    def test_local_recess_keeps_outer_wall_strips_at_original_plane(self):
        rows = proposals(self.room,self.recipe,0,self.cameras,distances=(360,),local=True)
        self.assertEqual(len(rows),1)
        candidate,evidence = rows[0]
        self.assertEqual(evidence['span'],'object')
        surfaces = candidate['structural_surfaces']
        self.assertEqual(len(surfaces),5)
        front = np.asarray(surfaces[0]['points'])
        self.assertGreater(front[:,0].min(),-2000)
        self.assertLess(front[:,0].max(),2000)
        original_vertices = np.asarray(self.recipe['structural_surfaces'][0]['points'])
        retained = np.concatenate([np.asarray(s['points']) for s in surfaces[3:]])
        self.assertTrue((retained[:,2]==0).all())
        for vertex in original_vertices:
            self.assertTrue((retained==vertex).all(axis=1).any())


if __name__ == '__main__':
    unittest.main()
