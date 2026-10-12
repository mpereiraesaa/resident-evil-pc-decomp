"""Geometry invariants independent of a particular mansion room."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from room3d_shape_fit import associate, compare, observations, triangulate_centers
from room3d_shapes import generate


class ShapeTests(unittest.TestCase):
    def test_scene_depth_fit_uses_unobserved_views_but_excludes_reserved_view(self):
        from batch_room3d import generate_mesh
        from room3d_preview import render
        from room3d_refine import depth_error
        from room3d_shape_fit import fit_group

        cameras = [
            {'index': i, 'origin': origin, 'target': [0, -1500, 2500],
             'fov': 240, 'roll': 0}
            for i, origin in enumerate(([0, -2000, -4000], [3000, -2000, -4000],
                                       [-3000, -2000, -4000], [-6000, -1500, 2500]))
        ]
        room = {'cameras': cameras, 'boundaries': [{'index': 0, 'xmin': -4000,
                'xmax': 4000, 'zmin': 0, 'zmax': 10000, 'type': 4, 'flags': 0}]}
        params = {'width': 1600, 'run': 5000, 'rise': 3000, 'steps': 15}
        solid = generate('staircase', params)
        thin = generate('staircase', {**params, 'structure': 'suspended', 'slab_depth': 100})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, candidate = root / 'scene.obj', root / 'solid.obj'
            hollow = root / 'thin.obj'
            generate_mesh(room, scene, {'primitives': []})
            generate_mesh(room, candidate, {'primitives': [solid]})
            generate_mesh(room, hollow, {'primitives': [thin]})
            group, depths, baselines, combined = [], [], [], []
            for camera in cameras:
                floor = render(scene, camera, return_depth=True)[2]
                _, mask, part = render(candidate, camera, return_depth=True,
                                       only_groups={'primitive_0'})
                thin_depth = render(hollow, camera, return_depth=True,
                                    only_groups={'primitive_0'})[2]
                target = np.where(np.isfinite(floor), floor, 10000)
                valid = np.isfinite(floor)
                if camera['index'] in (0, 1):
                    # These views supply silhouettes but no reliable object depth.
                    valid &= ~np.isfinite(part) & ~np.isfinite(thin_depth)
                elif camera['index'] == 3:
                    # Reliable floor visible through the open underside supplies
                    # independent evidence against a solid volume in this view.
                    valid &= np.isfinite(part) & ~np.isfinite(thin_depth)
                self.assertTrue(valid.any())
                depths.append((target, valid))
                baselines.append(depth_error(floor, target, valid))
                combined.append(depth_error(np.minimum(part, floor), target, valid))
                if camera['index'] < 3:
                    group.append({'kind': 'staircase', 'camera': camera['index'], 'mask': mask})
            candidates = [solid, thin]
            _, off = fit_group(room, {'primitives': []}, group, depths,
                               root / 'off', generate_mesh, rounds=0,
                               initial_candidates=candidates)
            recipe = {'primitives': [], 'scene_depth_fit': True}
            fitted, on = fit_group(room, recipe, group, depths, root / 'on',
                                   generate_mesh, rounds=0, initial_candidates=candidates)
            changed = list(depths)
            changed[2] = (depths[2][0] * 100, depths[2][1])
            repeated, altered = fit_group(room, recipe, group, changed, root / 'altered',
                                          generate_mesh, rounds=0, initial_candidates=candidates)
        expected = max(0, *(combined[i] - baselines[i] - .005 for i in (0, 1, 3)))
        self.assertGreater(expected, 0)
        self.assertEqual(on['scene_depth_fit']['training_scene_cameras'], [0, 1, 3])
        self.assertEqual(on['scene_depth_fit']['excluded_heldout_cameras'], [2])
        for original, penalized in zip(off['initializations'][0]['scores'],
                                       on['initializations'][0]['scores'], strict=True):
            self.assertAlmostEqual(penalized['scene_depth_excess'], expected)
            self.assertAlmostEqual(penalized['loss'] - original['loss'], 20 * expected)
        self.assertEqual(fitted['shape'], repeated['shape'])
        self.assertEqual(on['initializations'], altered['initializations'])
        self.assertEqual(on['after'], altered['after'])
        self.assertNotEqual(on['heldout'], altered['heldout'])
        self.assertNotIn('scene_depth_excess', on['heldout'][0])
        self.assertNotIn('scene_depth_fit', off)
        self.assertEqual(off['optimized_initialization'], 0)
        self.assertEqual(fitted['shape']['parameters']['structure'], 'suspended')

    def test_structure_search_keeps_budget_for_a_worse_initial_suspended_seed(self):
        from unittest.mock import patch

        from batch_room3d import generate_mesh
        from room3d_preview import render
        from room3d_shape_fit import fit_group

        camera = {'index': 0, 'origin': [-6000, -1500, 2000],
                  'target': [0, -1500, 2500], 'fov': 240, 'roll': 0}
        room = {'cameras': [camera], 'boundaries': [{'index': 0, 'xmin': -4000,
                'xmax': 4000, 'zmin': 0, 'zmax': 10000, 'type': 4, 'flags': 0}]}
        starts = []
        for steps in range(15, 20):
            obj = generate('staircase', {'width': 1600, 'run': 5000,
                                        'rise': 3000, 'steps': steps})
            obj['initialization'] = 'mask_depth'
            starts.append(obj)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'target.obj'
            generate_mesh(room, target, {'primitives': [starts[0]]})
            _, mask, depth = render(target, camera, return_depth=True,
                                    only_groups={'primitive_0'})
            group = [{'kind': 'staircase', 'camera': 0, 'mask': mask}]
            depths = [(np.where(mask, depth, 10000), np.ones_like(mask))]
            with patch('room3d_shape_fit.initial_shapes', return_value=starts):
                _, report = fit_group(room, {'primitives': [], 'creative_fallback': True,
                                             'suspended_stair_starts': True},
                                      group, depths, root / 'fit', generate_mesh, rounds=1)
            thin = generate('staircase', {**starts[0]['shape']['parameters'],
                                         'structure': 'suspended', 'slab_depth': 100})
            thin['initialization'] = 'mask_depth'
            _, explicit = fit_group(room, {'primitives': [], 'suspended_stair_starts': True},
                                    group, depths, root / 'explicit', generate_mesh, rounds=0,
                                    initial_candidates=[starts[0], thin])
            self.assertEqual(sorted(explicit['structure_search']['optimized_structures']),
                             ['solid', 'suspended'])
        optimized = report['structure_search']['optimized_structures']
        self.assertEqual(optimized.count('solid'), 4)
        self.assertEqual(optimized.count('suspended'), 1)
        self.assertEqual(report['structure_search']['selected_structure'], 'solid')
        # The lower-ranked structure was considered despite four better solid starts.
        thin = report['initializations'][-1]['scores'][0]['loss']
        self.assertTrue(all(item['scores'][0]['loss'] < thin
                            for item in report['initializations'][:5]))

    def test_suspended_structure_is_an_opt_in_rendered_fitting_hypothesis(self):
        from unittest.mock import patch

        from batch_room3d import generate_mesh
        from room3d_preview import render
        from room3d_shape_fit import fit_group

        camera = {'index': 0, 'origin': [0, -2000, -4000],
                  'target': [0, -1000, 2000], 'fov': 240, 'roll': 0}
        room = {'cameras': [camera], 'boundaries': [{'index': 0, 'xmin': -4000,
                'xmax': 4000, 'zmin': 0, 'zmax': 10000, 'type': 4, 'flags': 0}]}
        p = {'width': 1600, 'run': 5000, 'rise': 3000, 'steps': 15}
        solid = generate('staircase', p)
        solid['initialization'] = 'mask_depth'
        truth = generate('staircase', {**p, 'structure': 'suspended', 'slab_depth': 100})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'target.obj'
            generate_mesh(room, target, {'primitives': [truth]})
            _, mask, depth = render(target, camera, return_depth=True,
                                    only_groups={'primitive_0'})
            group = [{'kind': 'staircase', 'camera': 0, 'mask': mask}]
            depths = [(np.where(mask, depth, 10000), np.ones_like(mask))]
            recipe = {'primitives': [], 'creative_fallback': True}
            with patch('room3d_shape_fit.initial_shapes', return_value=[solid]):
                _, off = fit_group(room, recipe, group, depths, root / 'off', generate_mesh, rounds=0)
                fitted, on = fit_group(room, {**recipe, 'suspended_stair_starts': True},
                                       group, depths, root / 'on', generate_mesh, rounds=1)
            self.assertEqual(len(off['initializations']), 1)
            self.assertEqual(len(on['initializations']), 2)
            self.assertEqual(fitted['shape']['parameters']['structure'], 'suspended')
            self.assertEqual(fitted['interpretation']['source'], 'regular_step_prior')
            self.assertGreater(on['after'][0]['silhouette_iou'], .99)


    def test_wall_arm_direction_keeps_plate_anchor(self):
        from room3d_fixtures import anchor_wall_lamp

        params = {'height': 600, 'radius': 300, 'thickness': 40, 'arms': 1,
                  'shade': 200, 'profile': 'wall_sconce'}
        prior = {'point': [100, -1200, 200], 'normal': [0, 0, 1], 'group': 'boundary_0'}
        old = anchor_wall_lamp(generate('lamp', params), prior)
        upward = anchor_wall_lamp(generate('lamp', {**params, 'arm_rise': .5}), prior)
        np.testing.assert_array_equal(old['parts'][1], upward['parts'][1])
        self.assertLess(np.asarray(upward['parts'][2])[:, 1].mean(), prior['point'][1])
        self.assertGreater(np.asarray(old['parts'][2])[:, 1].mean(), prior['point'][1])
        for invalid in (-1.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                generate('lamp', {**params, 'arm_rise': invalid})


    def test_stair_fit_rejects_perfect_silhouette_with_oversized_steps(self):
        from batch_room3d import generate_mesh
        from room3d_assemblies import stair_proportions
        from room3d_preview import render
        from room3d_shape_fit import fit_group

        camera = {'index': 0, 'origin': [0, -2000, -4000],
                  'target': [0, -1000, 2000], 'fov': 240, 'roll': 0}
        room = {'cameras': [camera], 'boundaries': [{'index': 0, 'xmin': -4000,
                'xmax': 4000, 'zmin': 0, 'zmax': 8000, 'type': 4, 'flags': 0}]}
        params = {'width': 1600, 'run': 3600, 'rise': 1800, 'steps': 3}
        bad = generate('staircase', params)
        good = generate('staircase', {**params, 'steps': 10})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mesh = root / 'target.obj'
            generate_mesh(room, mesh, {'primitives': [bad]})
            _, mask, depth = render(mesh, camera, return_depth=True, only_groups={'primitive_0'})
            group = [{'kind': 'staircase', 'camera': 0, 'mask': mask}]
            depths = [(np.where(mask, depth, 10000), np.ones_like(mask))]
            common = {"rounds": 0, "initial_candidates": [bad, good]}
            fitted, _ = fit_group(room, {'primitives': []}, group, depths,
                                  root / 'unconstrained', generate_mesh, **common)
            self.assertFalse(stair_proportions(fitted)['plausible'])
            fitted, report = fit_group(room, {'primitives': [], 'constrain_stair_proportions': True},
                                       group, depths, root / 'constrained', generate_mesh, **common)
            self.assertTrue(stair_proportions(fitted)['plausible'])
            self.assertTrue(report['after'][0]['proportions_valid'])

    def test_single_view_creative_fixture_requires_anchor_and_visible_fit(self):
        from room3d_shape_fit import single_view_fixture_evidence

        obj = {"kind": "lamp", "attachment_prior": {"group": "boundary_2"},
               "interpretation": {"source": "fixture_attachment_prior"}}
        scores = [{"silhouette_iou": .8, "visible_pixels": 100}]
        self.assertTrue(single_view_fixture_evidence(obj, [{}], scores, True))
        self.assertFalse(single_view_fixture_evidence(obj, [{}], scores, False))
        self.assertFalse(single_view_fixture_evidence(obj, [{}, {}], scores, True))
        self.assertFalse(single_view_fixture_evidence({**obj, "attachment_prior": {}}, [{}], scores, True))
        for score in ({"silhouette_iou": .59, "visible_pixels": 100},
                      {"silhouette_iou": .8, "visible_pixels": 8}):
            self.assertFalse(single_view_fixture_evidence(obj, [{}], [score], True))

    def test_balanced_training_loss_penalizes_sacrificing_one_view(self):
        from room3d_shape_fit import view_loss

        uneven = [{"loss": .1}, {"loss": .9}]
        balanced = [{"loss": .55}, {"loss": .55}]
        self.assertLess(view_loss(uneven), view_loss(balanced))
        self.assertGreater(view_loss(uneven, True), view_loss(balanced, True))
        self.assertEqual(view_loss([{"loss": .4}], True), .4)
        self.assertEqual(view_loss(balanced, True), view_loss(list(reversed(balanced)), True))
        with self.assertRaises(ValueError):
            view_loss([], True)

    def test_render_fit_keeps_wall_anchor_while_resizing_lamp(self):
        from batch_room3d import generate_mesh
        from room3d_fixtures import anchor_wall_lamp
        from room3d_preview import render
        from room3d_shape_fit import fit_group

        cameras = [{"index": i, "origin": origin,
                    "target": [0, -800, 0], "fov": 240, "roll": 0}
                   for i, origin in enumerate(([0, -800, 2500],
                                                [1400, -800, 2300],
                                                [-1400, -1000, 2400]))]
        room = {"cameras": cameras, "boundaries": [{
            "index": 0, "xmin": -4000, "xmax": 4000, "zmin": -50,
            "zmax": 0, "type": 1, "flags": 256,
        }]}
        prior = {"point": [0, -1000, 0], "normal": [0, 0, 1], "group": "boundary_0"}
        truth = anchor_wall_lamp(generate("lamp", {
            "height": 600, "radius": 300, "thickness": 40, "arms": 1,
            "shade": 200, "profile": "wall_sconce"}), prior)
        start = anchor_wall_lamp(generate("lamp", {
            **truth["shape"]["parameters"], "radius": 200}), prior)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            path = out / "truth.obj"
            generate_mesh(room, path, {"primitives": [truth]})
            depths, group = [], []
            for camera in cameras:
                _, mask, depth = render(path, camera, return_depth=True,
                                        only_groups={"primitive_0"})
                depths.append((np.where(mask, depth, 10000), np.ones_like(mask)))
                group.append({"camera": camera["index"], "camera_model": camera,
                              "kind": "lamp", "mask": mask, "label": "wall lamp"})
            fitted, report = fit_group(room, {"primitives": []}, group, depths,
                                       out / "fit", generate_mesh,
                                       initial_candidates=[start], rounds=2)
        self.assertTrue(report["trials"])
        self.assertEqual(report["training_cameras"], [0, 1])
        self.assertEqual(report["heldout_cameras"], [2])
        self.assertGreater(report["heldout"][0]["silhouette_iou"], 0.6)
        self.assertTrue(report["heldout"][0]["attachment_valid"])
        for trial in report["trials"]:
            shape = trial["shape"]
            np.testing.assert_allclose(np.array(shape["origin"])
                                       - [0, shape["parameters"]["height"], 0], prior["point"])
            self.assertEqual(shape["yaw"], 0)
        self.assertEqual(fitted["attachment_prior"], prior)
        self.assertTrue(report["after"][0]["attachment_valid"])
        self.assertLessEqual(report["after"][0]["loss"], report["before"][0]["loss"])

    def test_occlusion_proposals_require_training_depth_evidence(self):
        import copy

        from room3d_supports import occlusion_hypotheses

        recipe = {
            "wall_height": 6000,
            "structural_surfaces": [
                {"group": f"boundary_{i}", "points": [[0, -6000, 0]]}
                for i in range(5)
            ],
        }
        original = copy.deepcopy(recipe)
        items = [
            {"group": f"boundary_{i}", "reliable_pixels": 20 + i,
             "observed_depth": 4000, "surface_depth": 1000}
            for i in range(5)
        ]
        report = {"training_cameras": [1], "visibility_diagnostics": [
            {"camera": 1, "occluders": items},
        ]}
        proposals = list(occlusion_hypotheses(recipe, report))
        self.assertEqual(len(proposals), 2)
        self.assertEqual(proposals[0][1]["groups"],
                         ["boundary_4", "boundary_3", "boundary_2"])
        self.assertEqual(recipe, original)
        report["training_cameras"] = [2]
        self.assertEqual(list(occlusion_hypotheses(recipe, report)), [])
        report["training_cameras"] = [1]
        for item in items:
            item["observed_depth"] = 1050
        self.assertEqual(list(occlusion_hypotheses(recipe, report)), [])
        for item in items:
            item.update(observed_depth=4000, reliable_pixels=0)
        self.assertEqual(list(occlusion_hypotheses(recipe, report)), [])
        for item in items:
            item["nearest_pixels"] = 30
        training = [{"camera": 1, "structural_refinement": {
            "structural_evidence": {"tread_edges": 6, "correlation": 0.94},
        }}]
        alternatives = list(occlusion_hypotheses(recipe, report, training))
        self.assertEqual(len(alternatives), 2)
        self.assertEqual(alternatives[0][1]["evidence"]["boundary_0"],
                         ["repeated_tread_structure"])
        training[0]["camera"] = 2
        self.assertEqual(list(occlusion_hypotheses(recipe, report, training)), [])

    def test_isolated_export_preserves_pixels_and_depth(self):
        from batch_room3d import generate_mesh
        from room3d_preview import render
        from room3d_shape_fit import isolated_recipe

        camera = {
            "index": 0,
            "origin": [0, -2000, -3000],
            "target": [0, -1000, 3000],
            "fov": 240,
            "roll": 0,
        }
        room = {
            "cameras": [camera],
            "boundaries": [
                {
                    "index": 7,
                    "xmin": -5000,
                    "xmax": 5000,
                    "zmin": 0,
                    "zmax": 6000,
                    "type": 1,
                    "flags": 0x100,
                }
            ],
        }
        obj = generate(
            "lamp",
            {"height": 1000, "radius": 500, "thickness": 30, "arms": 4, "shade": 250},
            [0, -800, 3000],
        )
        recipe = {
            "primitives": [obj],
            "structural_surfaces": [
                {
                    "points": [
                        [-5000, 0, 6000],
                        [-5000, -6000, 6000],
                        [5000, -6000, 6000],
                        [5000, 0, 6000],
                    ],
                    "group": "boundary_7",
                    "material": "wall",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            before, after = Path(temp) / "before.obj", Path(temp) / "after.obj"
            generate_mesh(room, before, recipe)
            generate_mesh(room, after, isolated_recipe(room, recipe, obj))
            a = render(before, camera, return_depth=True, only_groups={"primitive_0"})
            b = render(after, camera, return_depth=True, only_groups={"primitive_0"})
            self.assertTrue(a[1].any())
            np.testing.assert_array_equal(np.asarray(a[0]), np.asarray(b[0]))
            np.testing.assert_array_equal(a[2], b[2])
            self.assertLess(after.stat().st_size, before.stat().st_size)
            self.assertNotIn("g boundary_7", after.read_text())

    def test_tread_orientation_recovers_diagonal_run_without_depth(self):
        from room3d_preview import camera_basis
        from room3d_shape_fit import initial_shapes, tread_run_direction

        camera = {
            "origin": [0, -3500, -5000],
            "target": [0, -900, 1800],
            "fov": 280,
            "roll": 0,
        }
        yaw = 0.65
        obj = generate(
            "staircase", {"width": 2000, "run": 3600, "rise": 1800, "steps": 9}, yaw=yaw
        )
        noses = np.asarray(obj["parts"])[:, :2]
        view = (noses - camera["origin"]) @ camera_basis(camera).T
        lines = view[..., :2] * camera["fov"] / view[..., 2:] + [160, 120]
        obs = {
            "camera_model": camera,
            "structural_refinement": {
                "structural_evidence": {"line_segments": lines.tolist()}
            },
        }
        direction = tread_run_direction([obs])
        self.assertGreater(abs(direction @ [np.sin(yaw), np.cos(yaw)]), 0.99999)
        self.assertIsNone(tread_run_direction([{"camera_model": camera}]))
        obs.update(
            kind="staircase",
            label="staircase",
            camera=0,
            points=noses.reshape(-1, 3),
            tread_points=noses.reshape(-1, 3),
        )
        starts = list(initial_shapes({"boundaries": []}, [obs]))
        self.assertEqual(
            {s.get("initialization") for s in starts}, {"mask_depth", "tread_region"}
        )

    def test_support_candidates_use_nearby_obstacles_without_mutating_recipe(self):
        from room3d_supports import hypotheses

        boundary = {
            "index": 73,
            "xmin": -2000,
            "xmax": 2000,
            "zmin": 0,
            "zmax": 5000,
            "type": 1,
            "flags": 0,
        }
        room = {"boundaries": [boundary, dict(boundary, index=74, flags=0x100)]}
        recipe = {"primitives": []}
        training = [
            {
                "points": np.column_stack(
                    (
                        np.zeros(40),
                        np.linspace(-4000, -3000, 40),
                        np.linspace(1000, 3000, 40),
                    )
                )
            }
        ]
        candidates = list(hypotheses(room, recipe, training))
        self.assertEqual(recipe, {"primitives": []})
        self.assertEqual(candidates[0][1], {"kind": "floor", "y": 0})
        self.assertTrue(any(s["y"] < -2000 for _, s in candidates))
        for r, s in candidates[1:]:
            self.assertEqual(s["boundary"], 73)
            self.assertEqual(r["boundary_heights"]["73"], -s["y"])
        recipe["skip_boundaries"] = [73]
        self.assertEqual(len(list(hypotheses(room, recipe, training))), 1)

    def test_stair_contact_requires_base_height_and_entire_footprint(self):
        from room3d_supports import contact

        params = {"width": 1000, "run": 3000, "rise": 1200, "steps": 6}
        support = {
            "kind": "collision_box",
            "y": -2500,
            "xmin": -600,
            "xmax": 600,
            "zmin": 0,
            "zmax": 3100,
        }
        stair = generate("staircase", params, [0, -2500, 0])
        self.assertTrue(contact(stair, support))
        self.assertFalse(
            contact(generate("staircase", params, [200, -2500, 0]), support)
        )
        self.assertFalse(contact(generate("staircase", params, [0, -2600, 0]), support))
        self.assertTrue(contact(stair, {"kind": "floor", "y": -2500}))

    def test_visibility_diagnostic_separates_occlusion_and_unknown_depth(self):
        from unittest.mock import patch

        from room3d_shape_fit import visibility_diagnostic

        part = np.full((2, 2), 1000.0)
        scene = np.array([[500.0, 500.0], [np.inf, np.inf]])
        target = np.ones((2, 2), bool)
        depth = np.full((2, 2), 700.0)
        valid = np.array([[True, False], [False, False]])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scene.obj"
            path.write_text("g boundary_2\ng distant\n")

            def renderer(path, camera, return_depth, only_groups):
                surface = (
                    scene if only_groups == {"boundary_2"} else np.full((2, 2), 2000.0)
                )
                return None, None, surface

            with patch("room3d_shape_fit.render", side_effect=renderer):
                report = visibility_diagnostic(
                    path, {}, part, scene, target, depth, valid
                )
        self.assertEqual(report["projected_pixels"], 4)
        self.assertEqual(report["scene_visible_pixels"], 2)
        self.assertEqual(report["depth_visible_pixels"], 3)
        self.assertEqual(report["combined_visible_pixels"], 2)
        self.assertEqual(len(report["occluders"]), 1)
        self.assertEqual(report["occluders"][0]["group"], "boundary_2")
        self.assertEqual(report["occluders"][0]["reliable_pixels"], 1)

    def test_internal_tread_edges_distinguish_counts_and_occlusion(self):
        from room3d_preview import camera_basis
        from room3d_shape_fit import tread_edge_error

        camera = {
            "origin": [0, -3500, -5000],
            "target": [0, -900, 1800],
            "fov": 280,
            "roll": 0,
        }
        params = {"width": 2000, "run": 3600, "rise": 1800, "steps": 9}
        truth = generate("staircase", params)
        noses = np.asarray(truth["parts"])[:, :2]
        view = (noses - camera["origin"]) @ camera_basis(camera).T
        lines = view[..., :2] * camera["fov"] / view[..., 2:] + [160, 120]
        obs = {
            "mask": np.ones((240, 320), bool),
            "structural_refinement": {
                "structural_evidence": {"line_segments": lines.tolist()}
            },
        }
        empty = np.full((240, 320), np.inf)
        exact = tread_edge_error(truth, camera, obs, empty, empty)
        wrong = generate("staircase", dict(params, steps=4))
        self.assertLess(exact, 0.05)
        self.assertGreater(
            tread_edge_error(wrong, camera, obs, empty, empty), exact + 0.1
        )
        self.assertEqual(
            tread_edge_error(truth, camera, obs, empty, np.zeros_like(empty)), 1
        )
        self.assertIsNone(tread_edge_error(truth, camera, {}, empty, empty))
        displaced = []
        for shift in (30, 60):
            obs["structural_refinement"]["structural_evidence"]["line_segments"] = (
                lines + [shift, 0]
            ).tolist()
            displaced.append(tread_edge_error(truth, camera, obs, empty, empty))
        self.assertLess(displaced[0], displaced[1])
        self.assertLess(displaced[1], 1)

    def test_tread_extent_does_not_hide_overwide_steps_outside_mask(self):
        import cv2
        from room3d_preview import camera_basis
        from room3d_shape_fit import tread_edge_error

        camera = {'origin': [0, -3500, -5000], 'target': [0, -900, 1800],
                  'fov': 280, 'roll': 0}
        params = {'width': 1600, 'run': 3600, 'rise': 1800, 'steps': 9}
        truth = generate('staircase', params)
        noses = np.asarray(truth['parts'])[:, :2]
        view = (noses-camera['origin']) @ camera_basis(camera).T
        lines = view[..., :2]*camera['fov']/view[..., 2:] + [160, 120]
        mask = np.zeros((240, 320), 'uint8')
        for a, b in np.rint(lines).astype(int):
            cv2.line(mask, tuple(a), tuple(b), 1, 3)
        mask = cv2.dilate(mask, np.ones((3, 3), 'uint8')).astype(bool)
        obs = {'mask': mask, 'structural_refinement': {
            'structural_evidence': {'line_segments': lines.tolist()}}}
        empty = np.full(mask.shape, np.inf)
        wide = generate('staircase', dict(params, width=4800))
        legacy = tread_edge_error(wide, camera, obs, empty, empty)
        extent = tread_edge_error(wide, camera, obs, empty, empty, constrain_extent=True)
        exact = tread_edge_error(truth, camera, obs, empty, empty, constrain_extent=True)
        self.assertLess(exact, .05)
        self.assertGreater(extent, legacy+.15)
        self.assertGreater(extent, exact+.15)
        # Overshoot hidden by actual scene geometry must not be penalized.
        occluder = empty.copy()
        occluder[~mask] = 0
        hidden = tread_edge_error(wide, camera, obs, empty, occluder, constrain_extent=True)
        self.assertAlmostEqual(hidden, legacy)

    def test_training_tread_direction_is_depth_independent_and_axial(self):
        from room3d_preview import camera_basis
        from room3d_shape_fit import tread_direction_error, tread_run_direction

        yaw = -.8
        truth = generate('staircase', {'width': 1600, 'run': 3600,
                                     'rise': 1800, 'steps': 9}, yaw=yaw)
        camera = {'origin': [0, -3500, -5000], 'target': [0, -900, 1800],
                  'fov': 280, 'roll': 0}
        noses = np.asarray(truth['parts'])[:, :2]
        view = (noses-camera['origin']) @ camera_basis(camera).T
        lines = view[..., :2]*camera['fov']/view[..., 2:] + [160, 120]
        observation = {'camera_model': camera, 'structural_refinement': {
            'structural_evidence': {'line_segments': lines.tolist()}}}
        direction = tread_run_direction([observation])
        self.assertIsNotNone(direction)
        self.assertAlmostEqual(tread_direction_error(truth, direction), 0)
        reverse = generate('staircase', truth['shape']['parameters'], yaw=yaw+np.pi)
        perpendicular = generate('staircase', truth['shape']['parameters'], yaw=yaw+np.pi/2)
        self.assertAlmostEqual(tread_direction_error(reverse, direction), 0)
        self.assertAlmostEqual(tread_direction_error(perpendicular, direction), 1)
        self.assertIsNone(tread_direction_error(truth, None))
        self.assertIsNone(tread_run_direction([{'camera_model': camera}]))
        # Insufficient line evidence must not manufacture a direction prior.
        observation['structural_refinement']['structural_evidence']['line_segments'] = lines[:4].tolist()
        self.assertIsNone(tread_run_direction([observation]))

    def test_reserved_lines_do_not_change_training_orientation_prior(self):
        import copy

        from batch_room3d import generate_mesh
        from room3d_preview import camera_basis, render
        from room3d_shape_fit import fit_group

        truth = generate('staircase', {'width': 1600, 'run': 3600,
                                     'rise': 1800, 'steps': 9}, yaw=-.8)
        cameras = [{'index': i, 'origin': [i*300, -3500, -5000],
                    'target': [0, -900, 1800], 'fov': 280, 'roll': 0} for i in range(3)]
        room = {'cameras': cameras, 'boundaries': [
            {'index': 0, 'xmin': -5000, 'xmax': 5000,
             'zmin': -5000, 'zmax': 5000, 'type': 4, 'flags': 0}]}
        group, depths = [], []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_mesh(room, root/'truth.obj', {'primitives': [truth]})
            for camera in cameras:
                _, mask, depth = render(root/'truth.obj', camera, return_depth=True,
                                        only_groups={'primitive_0'})
                noses = np.asarray(truth['parts'])[:, :2]
                view = (noses-camera['origin']) @ camera_basis(camera).T
                lines = view[..., :2]*camera['fov']/view[..., 2:] + [160, 120]
                group.append({'kind': 'staircase', 'camera': camera['index'],
                              'camera_model': camera, 'mask': mask,
                              'structural_refinement': {'structural_evidence': {
                                  'line_segments': lines.tolist()}}})
                depths.append((np.where(mask, depth, 10000), mask))
            reports = []
            for run in range(2):
                changed = copy.deepcopy(group)
                if run:
                    changed[-1]['structural_refinement']['structural_evidence']['line_segments'] = []
                _, report = fit_group(room, {'constrain_tread_direction': True}, changed,
                                      depths, root/f'run{run}', generate_mesh, rounds=0,
                                      initial_candidates=[truth])
                reports.append(report)
            self.assertEqual(reports[0]['tread_direction_prior'], reports[1]['tread_direction_prior'])
            self.assertEqual(reports[0]['tread_direction_prior']['source_cameras'], [0, 1])
            self.assertEqual(reports[0]['tread_direction_prior']['excluded_cameras'], [2])
            self.assertEqual(reports[0]['after'], reports[1]['after'])





class ObservationTests(unittest.TestCase):
    def observation(self, camera, x, name="a"):
        return {
            "kind": "lamp",
            "camera": camera,
            "mask_name": name,
            "lo": np.array([x, 0, 0]),
            "hi": np.array([x + 100, 100, 100]),
        }

    def test_same_view_lamps_cannot_be_merged(self):
        groups = associate([self.observation(0, 0), self.observation(0, 30, "b")])
        self.assertEqual(len(groups), 2)

    def test_complete_link_prevents_spatial_chains(self):
        groups = associate(
            [self.observation(0, 0), self.observation(1, 250), self.observation(2, 500)]
        )
        self.assertEqual(sorted(map(len, groups)), [1, 2])

    def test_ambiguous_cross_view_match_remains_separate(self):
        groups = associate(
            [
                self.observation(0, 0),
                self.observation(0, 100, "b"),
                self.observation(1, 50),
            ]
        )
        self.assertEqual(len(groups), 3)

    def test_empty_geometry_cannot_win_silhouette_score(self):
        mask = np.zeros((24, 32), bool)
        mask[5:15, 8:18] = True
        part = np.where(mask, 1000.0, np.inf)
        scene = np.full(mask.shape, np.inf)
        depth = np.full(mask.shape, 1000.0)
        valid = np.ones(mask.shape, bool)
        correct = compare(part, scene, mask, depth, valid)
        empty = compare(scene, scene, mask, depth, valid)
        self.assertAlmostEqual(correct["loss"], 0)
        self.assertGreater(empty["loss"], 0.9)

    def test_foreground_occludes_without_becoming_object_surface(self):
        mask = np.zeros((24, 32), bool)
        mask[5:15, 8:18] = True
        full = np.where(mask, 1000.0, np.inf)
        foreground = np.full(mask.shape, 2000.0)
        foreground[:, :13] = 500
        target = mask.copy()
        target[:, :13] = False
        score = compare(
            full,
            foreground,
            target,
            np.full(mask.shape, 1000.0),
            np.ones(mask.shape, bool),
        )
        self.assertEqual(score["silhouette_iou"], 1)


class TriangulationTests(unittest.TestCase):
    def view(self, index, origin, target):
        mask = np.zeros((240, 320), bool)
        mask[115:126, 155:166] = True
        return {
            "camera": index,
            "mask": mask,
            "camera_model": {
                "index": index,
                "origin": origin,
                "target": target,
                "fov": 200,
                "roll": 0,
            },
        }

    def test_known_converging_rays_recover_center(self):
        target = np.array([100.0, -500.0, 4000.0])
        group = [
            self.view(0, [-1000, -500, 0], target),
            self.view(1, [1200, -500, 0], target),
        ]
        np.testing.assert_allclose(triangulate_centers(group), target, atol=1e-8)

    def test_lamp_association_requires_consistent_rays_despite_overlapping_depth_boxes(self):
        target = np.array([100., -500., 4000.])
        views = [self.view(0, [-1000, -500, 0], target),
                 self.view(1, [1200, -500, 0], target)]
        for view in views:
            view.update(kind="lamp", mask_name="lamp.png",
                        lo=np.array([-2000., -2000., 2000.]),
                        hi=np.array([2000., 2000., 6000.]))
        self.assertEqual(len(associate(views)), 1)
        # Same uncertain boxes, but centers now belong to incompatible rays.
        views[1]["camera_model"]["origin"] = [1200, 500, 0]
        views[1]["camera_model"]["target"] = [100, 500, 4000]
        self.assertEqual(len(associate(views)), 2)

    def test_parallel_rays_do_not_invent_depth(self):
        group = [
            self.view(0, [0, 0, 0], [0, 0, 4000]),
            self.view(1, [1000, 0, 0], [1000, 0, 4000]),
        ]
        self.assertIsNone(triangulate_centers(group))


class PipelineTests(unittest.TestCase):
    def test_combined_railing_label_does_not_supervise_solid_stairs(self):
        camera = {
            "index": 0,
            "origin": [0, 0, 0],
            "target": [0, 0, 1000],
            "fov": 100,
            "roll": 0,
        }
        mask = np.zeros((240, 320), bool)
        mask[100:120, 140:160] = True
        det = {
            "label": "staircase railing",
            "score": 0.9,
            "predicted_iou": 0.9,
            "mask": "rail.png",
        }
        obs = observations(
            {"cameras": [camera]},
            [[(det, mask)]],
            [(np.full(mask.shape, 1000.0), np.ones_like(mask))],
        )
        self.assertEqual([x["kind"] for x in obs], ["railing"])

    def test_closed_shapes_export_and_render_with_game_projection(self):
        from batch_room3d import generate_mesh
        from room3d_preview import render

        camera = {
            "index": 0,
            "origin": [0, -600, -4000],
            "target": [0, -600, 1000],
            "fov": 200,
            "roll": 0,
        }
        room = {
            "cameras": [camera],
            "boundaries": [
                {
                    "index": 0,
                    "xmin": -2000,
                    "xmax": 2000,
                    "zmin": -2000,
                    "zmax": 4000,
                    "type": 4,
                    "flags": 0,
                }
            ],
        }
        obj = generate(
            "staircase", {"width": 1200, "run": 3000, "rise": 1800, "steps": 9}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stairs.obj"
            generate_mesh(room, path, {"primitives": [obj]})
            _, visible, depth = render(
                path, camera, return_depth=True, only_groups={"primitive_0"}
            )
            self.assertGreater(visible.sum(), 100)
            self.assertTrue(np.isfinite(depth[visible]).all())
            self.assertTrue((depth[visible] > camera["fov"] * 2).all())


class AppearanceTests(unittest.TestCase):
    def test_foreground_is_not_baked_on_a_farther_wall(self):
        from room3d_reconstruct import observed_visibility

        depth = np.full((10, 10), 500.0)
        valid = np.ones((10, 10), bool)
        visible = observed_visibility(
            np.array([1000.0, 500.0, 300.0]), np.array([[5, 5]] * 3), (depth, valid)
        )
        np.testing.assert_array_equal(visible, [False, True, True])

    def test_unknown_or_discontinuous_depth_does_not_erase_texture(self):
        from room3d_reconstruct import observed_visibility

        depth = np.full((10, 10), 500.0)
        valid = np.ones((10, 10), bool)
        depth[5, 4] = 1000
        self.assertTrue(
            observed_visibility(np.array([2000.0]), np.array([[5, 5]]), (depth, valid))[
                0
            ]
        )
        valid[:] = False
        self.assertTrue(
            observed_visibility(np.array([2000.0]), np.array([[3, 3]]), (depth, valid))[
                0
            ]
        )


class SearchTests(unittest.TestCase):
    def test_coordinate_search_preserves_other_accepted_dimensions(self):
        from room3d_shape_fit import rebase_variant

        params = {"width": 1000, "run": 3000, "rise": 1800, "steps": 9}
        original = generate("staircase", params)
        current = generate("staircase", {**params, "width": 1200}, [100, 0, 0])
        candidate = generate("staircase", {**params, "rise": 2000})
        rebased = rebase_variant(candidate, original, current)["shape"]
        self.assertEqual(rebased["parameters"]["width"], 1200)
        self.assertEqual(rebased["parameters"]["rise"], 2000)
        self.assertEqual(rebased["origin"], [100, 0, 0])

    def test_multiview_recovery_generalizes_to_reserved_synthetic_camera(self):
        from batch_room3d import generate_mesh
        from room3d_preview import render
        from room3d_shape_fit import associate, fit_group, observations

        cameras = [
            {
                "index": i,
                "origin": origin,
                "target": [0, -2200, 4000],
                "fov": 240,
                "roll": 0,
            }
            for i, origin in enumerate(
                [[0, -2000, -2000], [3500, -2000, 1000], [-3000, -3500, 2500]]
            )
        ]
        room = {
            "cameras": cameras,
            "boundaries": [
                {
                    "index": 0,
                    "xmin": -4000,
                    "xmax": 4000,
                    "zmin": 0,
                    "zmax": 8000,
                    "type": 4,
                    "flags": 0,
                }
            ],
        }
        truth = generate(
            "lamp",
            {"height": 1200, "radius": 500, "thickness": 40, "arms": 4, "shade": 250},
            [0, -1500, 4000],
            0.3,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "truth.obj"
            generate_mesh(room, mesh, {"primitives": [truth]})
            segments, depths = [], []
            for c in cameras:
                _, mask, depth = render(
                    mesh, c, return_depth=True, only_groups={"primitive_0"}
                )
                segments.append(
                    [
                        (
                            {
                                "score": 1.0,
                                "predicted_iou": 1.0,
                                "mask": f"m{c['index']}.png",
                                "label": "chandelier",
                            },
                            mask,
                        )
                    ]
                )
                depths.append((np.where(mask, depth, 10000.0), np.ones_like(mask)))
            groups = associate(observations(room, segments, depths))
            self.assertEqual([len(g) for g in groups], [3])
            _, report = fit_group(
                room, {"primitives": []}, groups[0], depths, root / "fit", generate_mesh
            )
            self.assertEqual(report["training_cameras"], [0, 1])
            self.assertEqual(report["heldout_cameras"], [2])
            self.assertGreater(report["heldout"][0]["silhouette_iou"], 0.75)
            self.assertLess(report["heldout"][0]["depth_error"], 0.03)
            fitted, constrained = fit_group(
                room,
                {"primitives": []},
                groups[0],
                depths,
                root / "constrained",
                generate_mesh,
                rounds=1,
                initial_candidates=[truth],
                fixed_parameters=("radius",),
            )
            self.assertTrue(constrained["trials"])
            self.assertEqual(fitted["shape"]["parameters"]["radius"], 500)
            self.assertTrue(
                all(
                    t["shape"]["parameters"]["radius"] == 500
                    for t in constrained["trials"]
                )
            )


class StairEdgeTests(unittest.TestCase):
    def test_original_mask_requires_substantial_reliable_tread_evidence(self):
        from room3d_stair_edges import supported_original

        original = np.ones((20, 20), bool)
        region_mask = np.zeros_like(original)
        region_mask[:5] = True
        detection = {"score": 0.5, "predicted_iou": 0.7}
        region = {"mask": region_mask, "tread_edges": 5, "correlation": 0.94}
        self.assertIs(supported_original(detection, original, [region]), region)
        self.assertEqual(detection["predicted_iou"], 0.7)
        self.assertIsNone(supported_original(detection, original, []))
        for changed in (
            {"tread_edges": 4},
            {"correlation": 0.7},
            {"mask": np.eye(20, dtype=bool)},
        ):
            self.assertIsNone(
                supported_original(detection, original, [{**region, **changed}])
            )
        self.assertIsNone(
            supported_original({**detection, "predicted_iou": 0.5}, original, [region])
        )

    def test_repeated_treads_are_distinguished_from_a_flat_floor(self):
        import cv2
        from PIL import Image
        from room3d_preview import camera_basis
        from room3d_stair_edges import flight_regions

        camera = {
            "index": 0,
            "origin": [0, -1500, -2000],
            "target": [0, -1200, 2500],
            "fov": 240,
            "roll": 0,
        }
        rotation = camera_basis(camera)
        yy, xx = np.mgrid[:240, :320]
        rays = (
            np.stack(((xx - 160) / 240, (yy - 120) / 240, np.ones_like(xx)), axis=-1)
            @ rotation
        )
        origin = np.array(camera["origin"])
        denom = rays[:, :, 1] + 0.5 * rays[:, :, 2]
        depth = np.divide(
            -(origin[1] + 0.5 * origin[2]),
            denom,
            out=np.full_like(denom, 10000.0),
            where=denom > 0.01,
        )
        image = np.zeros((240, 320, 3), "uint8")
        for z in np.arange(1000, 4001, 250):
            endpoints = np.array([[-600, -0.5 * z, z], [600, -0.5 * z, z]])
            view = (endpoints - origin) @ rotation.T
            xy = view[:, :2] / view[:, 2:] * 240 + [160, 120]
            cv2.line(
                image,
                tuple(np.rint(xy[0]).astype(int)),
                tuple(np.rint(xy[1]).astype(int)),
                (255, 255, 255),
                1,
            )
        support = np.ones((240, 320), bool)
        regions = flight_regions(Image.fromarray(image), support, depth, camera)
        self.assertTrue(regions)
        line_sets = [tuple(sorted(r["line_indices"])) for r in regions]
        self.assertEqual(len(line_sets), len(set(line_sets)))
        self.assertTrue(all(r["minimum_edge_spacing"] in (1.5, 3) for r in regions))
        floor = np.divide(
            -origin[1],
            rays[:, :, 1],
            out=np.full_like(denom, 10000.0),
            where=rays[:, :, 1] > 0.01,
        )
        self.assertFalse(flight_regions(Image.fromarray(image), support, floor, camera))

    def test_cached_refinement_rejects_changed_depth(self):
        import hashlib
        import json

        from PIL import Image
        from room3d_stair_edges import load_refinements

        mask = np.ones((4, 4), bool)
        depth = np.ones((4, 4))
        image = Image.new("RGB", (4, 4))
        record = {
            "camera": 0,
            "source_mask": "m.png",
            "accepted_proposal": False,
            "source_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "refinements.json").write_text(json.dumps([record]))
            with self.assertRaisesRegex(ValueError, "depth_sha256"):
                load_refinements(
                    root,
                    {"cameras": [{"index": 0}]},
                    [image],
                    [[({"mask": "m.png"}, mask)]],
                    [(depth * 2, mask)],
                )


class VisualOccluderTests(unittest.TestCase):
    def test_lowering_keeps_both_visual_representations_consistent(self):
        from room3d_supports import lower_visual_occluders

        recipe = {
            "floor_y": 100,
            "wall_height": 6000,
            "structural_surfaces": [
                {
                    "group": "boundary_2",
                    "points": [
                        [0, 100, 0],
                        [0, -5900, 0],
                        [0, -5900, 100],
                        [0, 100, 100],
                    ],
                }
            ],
            "primitives": [{"kind": "door_panel", "box": [0, 0, 100, 100, -1000, 100]}],
        }
        result = lower_visual_occluders(recipe, {"boundary_2"}, 1500)
        self.assertEqual(result["boundary_heights"]["2"], 1500)
        self.assertEqual(
            min(p[1] for p in result["structural_surfaces"][0]["points"]), -1400
        )
        self.assertEqual(result["primitives"], recipe["primitives"])
        self.assertNotIn("boundary_heights", recipe)
        with self.assertRaises(ValueError):
            lower_visual_occluders(recipe, {"boundary_99"}, 1500)


class StairContactTests(unittest.TestCase):
    def test_stair_search_cannot_detach_its_base_from_the_floor(self):
        from room3d_shape_fit import variants

        obj = generate(
            "staircase",
            {"width": 1000, "run": 3000, "rise": 1800, "steps": 9, "landing": 0},
            [200, 100, 400],
        )
        for candidate in variants(obj, 2):
            self.assertEqual(candidate["shape"]["origin"][1], 100)
            self.assertAlmostEqual(np.asarray(candidate["parts"])[:, :, 1].max(), 100)


class FixtureFamilyTests(unittest.TestCase):
    def test_candle_profile_has_narrow_upright_lights(self):
        params = {
            "height": 1000,
            "radius": 500,
            "thickness": 30,
            "arms": 4,
            "shade": 300,
        }
        shade = np.array(generate("lamp", params)["parts"])
        candle = np.array(generate("lamp", {**params, "profile": "candle"})["parts"])
        lights = candle[2::2]
        np.testing.assert_allclose(np.ptp(lights[:, :, 1], axis=1), 300)
        self.assertLess(np.ptp(lights[0, :, 0]), np.ptp(shade[2, :, 0]) * 0.2)
        self.assertEqual(shade.shape, candle.shape)
        with self.assertRaises(ValueError):
            generate("lamp", {**params, "profile": "unknown"})


class CreativeInterpretationTests(unittest.TestCase):
    def test_lamp_proposals_preserve_anchor_and_are_deterministic(self):
        from room3d_interpretation import proposals

        source = generate(
            "lamp",
            {"height": 1000, "radius": 500, "thickness": 30, "arms": 5, "shade": 300},
            [100, -2000, 400],
            0.3,
        )
        first = list(proposals(source))
        self.assertEqual(first, list(proposals(source)))
        for obj in first:
            self.assertEqual(obj["shape"]["origin"], source["shape"]["origin"])
            np.testing.assert_allclose(obj["parts"][0], source["parts"][0])
            self.assertTrue(np.isfinite(obj["parts"]).all())
            self.assertEqual(
                len(obj["parts"]), 1 + 5 * obj["shape"]["parameters"]["arms"]
            )

    def test_placement_rejects_occluded_and_displaced_geometry(self):
        from room3d_interpretation import placement

        target = np.zeros((40, 40), bool)
        target[10:20, 10:20] = True
        part = np.full(target.shape, np.inf)
        part[target] = 100
        scene = np.full(target.shape, 100.0)
        self.assertTrue(placement(part, scene, target)["valid"])
        self.assertFalse(placement(part, scene - 80, target)["valid"])
        self.assertFalse(placement(np.roll(part, 20, axis=1), scene, target)["valid"])

    def test_door_clearance_rejects_intersection(self):
        from room3d_interpretation import door_clear

        recipe = {
            "primitives": [{"kind": "door_panel", "box": [0, 0, 100, 100, -200, 0]}]
        }
        self.assertFalse(door_clear({"box": [50, 50, 150, 150, -100, 0]}, recipe))
        self.assertTrue(door_clear({"box": [100, 0, 200, 100, -200, 0]}, recipe))


if __name__ == "__main__":
    unittest.main()
