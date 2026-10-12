"""Connected procedural assemblies remain attached under arbitrary edge yaw."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from room3d_assemblies import ceiling_hypotheses, stair_landings, zoned_ceiling
from room3d_connections import stair_connections


class AssemblyTests(unittest.TestCase):
    def test_opt_in_fitter_can_start_from_feasible_projection(self):
        import tempfile
        from unittest.mock import patch

        from batch_room3d import generate_mesh
        from room3d_assemblies import feasible_stair_seed
        from room3d_preview import render
        from room3d_shape_fit import fit_group
        from room3d_shapes import generate

        camera = {'index': 0, 'origin': [0, -2000, -4000],
                  'target': [0, -1000, 2000], 'fov': 240, 'roll': 0}
        room = {'cameras': [camera], 'boundaries': [{'index': 0, 'xmin': -4000,
                'xmax': 4000, 'zmin': 0, 'zmax': 18000, 'type': 4, 'flags': 0}]}
        noisy = generate('staircase', {'width': 1600, 'run': 16000,
                                      'rise': 1800, 'steps': 10})
        noisy['initialization'] = 'mask_depth'
        truth = feasible_stair_seed(noisy)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'target.obj'
            generate_mesh(room, target, {'primitives': [truth]})
            _, mask, depth = render(target, camera, return_depth=True,
                                    only_groups={'primitive_0'})
            group = [{'kind': 'staircase', 'camera': 0, 'mask': mask}]
            depths = [(np.where(mask, depth, 10000), np.ones_like(mask))]
            recipe = {'primitives': [], 'creative_fallback': True,
                      'constrain_stair_proportions': True}
            with patch('room3d_shape_fit.initial_shapes', return_value=[noisy]):
                _, off = fit_group(room, recipe, group, depths, root / 'off', generate_mesh, rounds=0)
                fitted, on = fit_group(room, {**recipe, 'feasible_stair_starts': True},
                                       group, depths, root / 'on', generate_mesh, rounds=0)
            self.assertEqual(len(off['initializations']), 1)
            self.assertFalse(off['after'][0]['proportions_valid'])
            self.assertEqual(len(on['initializations']), 2)
            self.assertTrue(on['after'][0]['proportions_valid'])
            self.assertGreater(on['after'][0]['silhouette_iou'], .99)
            self.assertEqual(fitted['shape'], truth['shape'])

    def test_feasible_seed_recovers_shallow_and_steep_initializations_without_mutation(self):
        import copy

        from room3d_assemblies import feasible_stair_seed, stair_proportions
        from room3d_shapes import generate

        for run, steps in ((15000, 16), (2200, 4)):
            stairs = generate("staircase", {"width": 2200, "run": run, "rise": 3552,
                                            "steps": steps, "landing": 500},
                              [123, -76, 456], .7)
            stairs['initialization'] = 'mask_depth'
            original = copy.deepcopy(stairs)
            candidate = feasible_stair_seed(stairs)
            self.assertEqual(stairs, original)
            self.assertIsNotNone(candidate)
            self.assertTrue(stair_proportions(candidate)['plausible'])
            for key in ('width', 'rise', 'landing'):
                self.assertEqual(candidate['shape']['parameters'][key],
                                 stairs['shape']['parameters'][key])
            self.assertEqual(candidate['shape']['origin'], stairs['shape']['origin'])
            self.assertEqual(candidate['shape']['yaw'], stairs['shape']['yaw'])
            self.assertIsNone(feasible_stair_seed(candidate))
            stairs['level_prior'] = {'source': 'independent_level'}
            self.assertIsNone(feasible_stair_seed(stairs))

    def test_feasible_seed_abstains_when_step_height_cannot_fit_generator_limits(self):
        from room3d_assemblies import feasible_stair_seed
        from room3d_shapes import generate

        for rise in (100, 18000):
            stairs = generate('staircase', {'width': 2200, 'run': 30000,
                                            'rise': rise, 'steps': 10})
            self.assertIsNone(feasible_stair_seed(stairs))

    def test_corroborated_assemblies_reserve_cameras_and_bound_duplicate_edges(self):
        from room3d_assemblies import corroborated_stair_assemblies, stair_proportions

        edge = {"endpoints": [[0, -3000, 0], [2000, -3000, 0]],
                "height": -3000, "cameras": [0, 1],
                "corroborating_cameras": [2], "corroborated": True}
        report = {"hypotheses": [edge, edge, {**edge, "corroborated": False}]}
        proposals = list(corroborated_stair_assemblies(report))
        self.assertEqual(len(proposals), 16)
        self.assertEqual(proposals, list(corroborated_stair_assemblies(report)))
        self.assertEqual(list(corroborated_stair_assemblies(report, excluded_cameras=[2])), [])
        self.assertEqual(list(corroborated_stair_assemblies(report, excluded_cameras=[0])), [])
        for proposal in proposals:
            stairs = proposal["stairs"]
            self.assertTrue(stair_proportions(stairs)["plausible"])
            self.assertEqual(proposal["source_cameras"], [0, 1, 2])
            self.assertAlmostEqual(stairs["shape"]["origin"][1] - stairs["shape"]["parameters"]["rise"], -3000)
        json.dumps(proposals)

        reversed_edge = {**edge, "endpoints": edge["endpoints"][::-1]}
        crossing_edge = {**edge, "endpoints": [[1000, -3000, -1000], [1000, -3000, 1000]]}
        proposals = list(corroborated_stair_assemblies(
            {"hypotheses": [edge, reversed_edge, crossing_edge, edge]}
        ))
        self.assertEqual(len(proposals), 32)

    def test_raised_patch_preserves_outer_slab_and_closes_edges(self):
        import copy

        from room3d_assemblies import raised_ceiling_patch

        recipe = {"primitives": [{"kind": "ceiling", "estimated": "inferred",
                                   "box": [0, 0, 10000, 10000, -3050, -3000]}]}
        original = copy.deepcopy(recipe)
        result = raised_ceiling_patch(recipe, 0, [2000, 3000, 6000, 7000], -5000)
        self.assertEqual(recipe, original)
        parts = np.asarray(result["primitives"][0]["parts"])
        self.assertEqual(len(parts), 9)
        np.testing.assert_equal(parts[:4, :, 1].min(axis=1), [-3050] * 4)
        self.assertEqual(parts[4, :, 1].max(), -5000)
        np.testing.assert_equal(parts[5:, :, 1].min(axis=1), [-5000] * 4)
        np.testing.assert_equal(parts[5:, :, 1].max(axis=1), [-3000] * 4)
        with self.assertRaises(ValueError):
            raised_ceiling_patch(recipe, 0, [20000, 0, 21000, 1000], -5000)

    def test_regularization_preserves_evidence_floor_and_labels_inference(self):
        from room3d_assemblies import regularize_stairs
        from room3d_shapes import generate

        obj = generate("staircase", {"width": 2000, "run": 7400, "rise": 6100,
                                    "landing": 2700, "steps": 15}, [0, 0, 0])
        before = [{"loss": 0.4, "silhouette_iou": 0.6}]
        good = lambda candidate: [{"loss": 0.45, "silhouette_iou": 0.53}]
        chosen, _, report = regularize_stairs(obj, before, good)
        self.assertIsNotNone(report["selected"])
        self.assertEqual(chosen["interpretation"]["source"], "regular_step_prior")
        for bad in ({"loss": 0.6, "silhouette_iou": 0.6},
                    {"loss": 0.3, "silhouette_iou": 0.2},
                    {"loss": 0.3, "silhouette_iou": 0.6, "contact_valid": False}):
            rejected, _, report = regularize_stairs(obj, before, lambda candidate, bad=bad: [bad])
            self.assertIs(rejected, obj)
            self.assertIsNone(report["selected"])

    def test_regular_steps_measure_flight_without_landing(self):
        from room3d_assemblies import regular_stair_variants, stair_proportions
        from room3d_shapes import generate

        stairs = generate("staircase", {
            "width": 5600, "run": 7400, "rise": 6100,
            "landing": 2700, "steps": 15,
        }, [100, -200, 300], 0.4)
        self.assertGreater(stair_proportions(stairs)["flight_slope_degrees"], 50)
        self.assertFalse(stair_proportions(stairs)["plausible"])
        candidates = list(regular_stair_variants(stairs))
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertTrue(stair_proportions(candidate)["plausible"])
            self.assertEqual(candidate["shape"]["origin"], stairs["shape"]["origin"])
            self.assertEqual(candidate["shape"]["yaw"], stairs["shape"]["yaw"])
            for key in ("width", "run", "rise"):
                self.assertEqual(candidate["shape"]["parameters"][key],
                                 stairs["shape"]["parameters"][key])
            self.assertEqual(candidate["shape"]["parameters"]["landing"], 0)

    def test_both_sides_and_slopes_connect_to_the_proposed_landing(self):
        edge = {"endpoints": [[100, -3000, 200], [2100, -3000, 1200]], "height": -3000}
        proposals = list(stair_landings(edge))
        self.assertEqual(len(proposals), 8)
        self.assertEqual({p["side"] for p in proposals}, {-1, 1})
        floor = [
            np.array([[-50000, 0, -50000], [50000, 0, -50000], [50000, 0, 50000]]),
            np.array([[-50000, 0, -50000], [50000, 0, 50000], [-50000, 0, 50000]]),
        ]
        for p in proposals:
            vertices = np.array(p["landing"]["parts"][0])
            triangles = floor + [vertices[[0, 1, 5]], vertices[[0, 5, 4]]]
            result = stair_connections(
                p["stairs"],
                triangles,
                groups=["floor", "floor", "landing", "landing"],
                walkable_groups={"floor", "landing"},
            )
            self.assertTrue(result["walkable_valid"])
            json.dumps(p)

    def test_width_fraction_changes_both_connected_parts(self):
        edge = {"endpoints": [[0, -3000, 0], [2000, -3000, 0]], "height": -3000}
        narrow = next(stair_landings(edge, width_fraction=0.5))
        self.assertEqual(narrow["stairs"]["shape"]["parameters"]["width"], 1000)
        self.assertAlmostEqual(
            np.ptp(np.array(narrow["landing"]["parts"][0])[:, 0]), 1000
        )
        with self.assertRaises(ValueError):
            list(stair_landings(edge, width_fraction=0.1))

    def test_ceiling_hypotheses_preserve_thickness_and_authored_geometry(self):
        recipe = {
            "primitives": [
                {
                    "kind": "ceiling",
                    "estimated": "inferred",
                    "box": [0, 0, 1000, 1000, -6050, -6000],
                },
                {"kind": "ceiling", "box": [0, 0, 1000, 1000, -3050, -3000]},
            ]
        }
        candidate, report = next(ceiling_hypotheses(recipe, -6400))
        self.assertEqual(candidate["primitives"][0]["box"][4:], [-8250, -8200])
        self.assertEqual(candidate["primitives"][1], recipe["primitives"][1])
        self.assertEqual(recipe["primitives"][0]["box"][5], -6000)
        self.assertEqual(report["changed_ceilings"], [0])
        self.assertEqual(list(ceiling_hypotheses(recipe, 0)), [])

    def test_zoned_ceiling_retains_slab_thickness_and_closes_the_step(self):
        recipe = {
            "primitives": [
                {
                    "kind": "ceiling",
                    "estimated": "inferred",
                    "box": [0, 0, 4000, 6000, -8050, -8000],
                }
            ]
        }
        result = zoned_ceiling(recipe, 0, 2, 4000, -6000)
        parts = np.array(result["primitives"][0]["parts"])
        self.assertEqual(parts[0, :, 1].max(), -8000)
        self.assertEqual(parts[1, :, 1].max(), -6000)
        self.assertEqual(np.ptp(parts[0, :, 1]), 50)
        self.assertEqual(np.ptp(parts[1, :, 1]), 50)
        self.assertEqual(parts[0, :, 2].max(), parts[1, :, 2].min())
        self.assertEqual(parts[2, :, 1].min(), -8050)
        self.assertEqual(parts[2, :, 1].max(), -6000)
        self.assertNotIn("parts", recipe["primitives"][0])
        with self.assertRaises(ValueError):
            zoned_ceiling(recipe, 0, 2, 9000, -6000)

    def test_ground_level_edge_does_not_invent_a_flight(self):
        self.assertEqual(
            list(stair_landings({"endpoints": [[0, 0, 0], [2000, 0, 0]], "height": 0})),
            [],
        )


if __name__ == "__main__":
    unittest.main()
