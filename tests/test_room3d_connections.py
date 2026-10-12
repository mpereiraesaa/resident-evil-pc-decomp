"""Geometric stair connection checks with known and missing landings."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from room3d_connections import stair_connections
from room3d_shapes import generate


def platform(x0, x1, z0, z1, y):
    a, b, c, d = [[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]]
    return [np.array([a, b, c]), np.array([a, c, d])]


class ConnectionsTests(unittest.TestCase):
    def test_orientable_wall_shade_keeps_support_and_rejects_wall_penetration(self):
        params = {"height": 600, "radius": 240, "thickness": 30,
                  "arms": 1, "shade": 180, "profile": "wall_sconce"}
        base = generate("lamp", params)
        for angle in (-np.pi/2, 0, np.pi/2):
            obj = generate("lamp", {**params, "shade_tilt": angle})
            np.testing.assert_allclose(obj["parts"][:2], base["parts"][:2])
            self.assertGreaterEqual(np.array(obj["parts"][2])[:, 2].min(), 15)
        with self.assertRaises(ValueError):
            generate("lamp", {**params, "radius": 110, "shade_tilt": np.pi/2})
        with self.assertRaises(ValueError):
            generate("lamp", {**params, "shade_tilt": np.pi})

    def test_fixture_anchor_refinement_stays_on_plane_and_bounds_cumulative_motion(self):
        from room3d_fixtures import anchor_fixture, slide_fixture_anchor

        params = {"height": 600, "radius": 200, "thickness": 30,
                  "arms": 1, "shade": 100, "profile": "wall_sconce"}
        prior = {"group": "wall", "point": [0, -1000, 0], "normal": [0, 0, 1]}
        current = anchor_fixture(generate("lamp", params), prior)
        moved = generate("lamp", params, np.array(current["shape"]["origin"])+[200, 50, 80])
        fitted = slide_fixture_anchor(moved, current)
        np.testing.assert_allclose(fitted["attachment_prior"]["point"], [200, -950, 0])
        np.testing.assert_allclose(fitted["attachment_prior"]["seed_point"], prior["point"])
        resized = generate("lamp", {**params, "height": 800}, fitted["shape"]["origin"])
        resized = slide_fixture_anchor(resized, fitted)
        np.testing.assert_allclose(resized["attachment_prior"]["point"], [200, -950, 0])
        far = generate("lamp", params, np.array(fitted["shape"]["origin"])+[900, 0, 0])
        with self.assertRaises(ValueError):
            slide_fixture_anchor(far, fitted)
        table = anchor_fixture(generate("lamp", {**params, "profile": "table_lamp"}),
                               {"support": "horizontal", "group": "desk", "point": [0, -500, 0]})
        moved = generate("lamp", table["shape"]["parameters"], [100, -700, 50])
        fitted = slide_fixture_anchor(moved, table)
        np.testing.assert_allclose(fitted["attachment_prior"]["point"], [100, -500, 50])

    def test_table_fixture_uses_finite_top_and_preserves_base_on_resize(self):
        from room3d_fixtures import (
            anchor_fixture,
            fixture_contact,
            ray_table_lamp_proposals,
        )
        from room3d_interpretation import proposals
        from room3d_preview import camera_basis

        camera = {"index": 2, "origin": [0, -1500, 2500],
                  "target": [0, -700, 0], "fov": 240, "roll": 0}
        mask = np.zeros((240, 320), bool)
        mask[85:132, 150:170] = True
        obs = {"camera": 2, "camera_model": camera, "mask": mask}
        top = platform(-1000, 1000, -1000, 1000, -500)
        bottom = platform(-1000, 1000, -1000, 1000, 0)
        reference = generate("lamp", {"height": 600, "radius": 200, "thickness": 30,
                                      "arms": 3, "shade": 100}, [9000, 0, 9000])
        candidates = list(ray_table_lamp_proposals(reference, [obs], top+bottom,
                                                  ["desk"]*4, {"desk"}))
        self.assertTrue(candidates)
        for obj in candidates:
            prior = obj["attachment_prior"]
            self.assertAlmostEqual(prior["point"][1], -500)
            xyz = (np.array(prior["point"])-camera["origin"]) @ camera_basis(camera).T
            np.testing.assert_allclose(xyz[:2]/xyz[2]*240+[160, 120], prior["pixel"])
            self.assertTrue(fixture_contact(obj, top+bottom, ["desk"]*4))
            self.assertLessEqual(np.asarray(obj["parts"])[:, :, 1].max(), -500 + 1e-6)
            changed = generate("lamp", {**obj["shape"]["parameters"], "height": 800})
            anchored = anchor_fixture(changed, prior)
            np.testing.assert_allclose(anchored["shape"]["origin"], prior["point"])
            for alternative in proposals(obj):
                self.assertEqual(alternative["shape"]["parameters"]["profile"], "table_lamp")
                self.assertEqual(alternative["attachment_prior"], prior)
            tiny = platform(prior["point"][0]-1, prior["point"][0]+1,
                            prior["point"][2]-1, prior["point"][2]+1, -500)
            self.assertFalse(fixture_contact(obj, tiny, ["desk"]*2))
        self.assertEqual(list(ray_table_lamp_proposals(reference, [obs], top,
                                                     ["floor"]*2, {"desk"})), [])
        with self.assertRaises(ValueError):
            generate("lamp", {"height": 120, "radius": 40, "thickness": 30,
                              "arms": 1, "shade": 100, "profile": "table_lamp"})

    def test_lamp_support_height_uses_training_rays_without_restoring_skipped_boxes(self):
        from room3d_fixtures import lamp_support_hypotheses

        camera = {"index": 2, "origin": [0, -1500, 2500],
                  "target": [0, -700, 0], "fov": 240, "roll": 0}
        mask = np.zeros((240, 320), bool)
        mask[85:132, 150:170] = True
        bounds = {"index": 0, "xmin": -1000, "xmax": 1000,
                  "zmin": -1000, "zmax": 1000, "type": 1, "flags": 512}
        room = {"boundaries": [bounds, {**bounds, "index": 1},
                               {**bounds, "index": 2, "flags": 768}]}
        recipe = {"skip_boundaries": [1], "boundary_heights": {"1": 1234}}
        obs = {"camera": 2, "camera_model": camera, "mask": mask}
        candidates = list(lamp_support_hypotheses(room, recipe, [obs]))
        self.assertTrue(candidates)
        self.assertLessEqual(len(candidates), 8)
        for candidate, evidence in candidates:
            self.assertEqual(evidence["boundary"], 0)
            self.assertEqual(evidence["camera"], 2)
            self.assertEqual(candidate["skip_boundaries"], [1])
            self.assertEqual(candidate["boundary_heights"]["1"], 1234)
            self.assertGreaterEqual(evidence["height"], 300)
            self.assertLessEqual(evidence["height"], 3000)
        self.assertEqual(recipe["boundary_heights"], {"1": 1234})
        self.assertEqual(list(lamp_support_hypotheses(room, recipe, [])), [])

    def test_mask_rays_find_finite_wall_without_reference_depth(self):
        from room3d_fixtures import ray_wall_lamp_proposals
        from room3d_preview import camera_basis

        camera = {"index": 7, "origin": [0, -800, 2500],
                  "target": [0, -800, 0], "fov": 240, "roll": 0}
        mask = np.zeros((240, 320), bool)
        mask[90:125, 150:170] = True
        obs = {"camera": 7, "camera_model": camera, "mask": mask}
        reference = generate("lamp", {"height": 600, "radius": 200, "thickness": 30,
                                      "arms": 3, "shade": 100}, [5000, 0, 5000])
        wall = [np.array([[-2000, -2000, 0], [2000, -2000, 0], [2000, 1000, 0]]),
                np.array([[-2000, -2000, 0], [2000, 1000, 0], [-2000, 1000, 0]])]
        behind = [triangle + [0, 0, 3000] for triangle in wall]
        candidates = list(ray_wall_lamp_proposals(reference, [obs], wall + behind,
                                                 ["front"] * 2 + ["behind"] * 2,
                                                 {"front", "behind"}))
        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            prior = candidate["attachment_prior"]
            self.assertEqual(prior["group"], "front")
            point = (np.array(prior["point"]) - camera["origin"]) @ camera_basis(camera).T
            pixel = point[:2] / point[2] * camera["fov"] + [160, 120]
            np.testing.assert_allclose(pixel, prior["pixel"])
            self.assertEqual(prior["camera"], 7)
        distant = [triangle + [20000, 0, 0] for triangle in wall]
        self.assertEqual(list(ray_wall_lamp_proposals(reference, [obs], distant,
                                                    ["front"] * 2, {"front"})), [])

    def test_wall_proposals_attach_and_ignore_ceilings_and_remote_faces(self):
        from room3d_connections import lamp_attachment
        from room3d_fixtures import wall_lamp_proposals, wall_plate_contact

        lamp = generate("lamp", {"height": 600, "radius": 200, "thickness": 30,
                                  "arms": 3, "shade": 120}, [0, 0, 700])
        wall = [np.array([[-2000, -2000, 0], [2000, -2000, 0], [2000, 1000, 0]]),
                np.array([[-2000, -2000, 0], [2000, 1000, 0], [-2000, 1000, 0]])]
        candidates = list(wall_lamp_proposals(lamp, wall, ["wall"] * 2, {"wall"}))
        self.assertTrue(candidates)
        self.assertTrue(wall_plate_contact(candidates[0], wall, ["wall"] * 2))
        from room3d_interpretation import proposals

        for alternative in proposals(candidates[0]):
            self.assertEqual(alternative["attachment_prior"], candidates[0]["attachment_prior"])
            self.assertEqual(alternative["shape"]["parameters"]["profile"], "wall_sconce")
            self.assertTrue(wall_plate_contact(alternative, wall, ["wall"] * 2))
        point = np.asarray(candidates[0]["attachment_prior"]["point"])
        tiny_wall = [point + np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]])]
        self.assertFalse(wall_plate_contact(candidates[0], tiny_wall, ["wall"]))
        for candidate in candidates:
            self.assertEqual(candidate["shape"]["parameters"]["profile"], "wall_sconce")
            self.assertTrue(lamp_attachment(candidate, wall, ["wall"] * 2,
                                            {"wall"})["within_tolerance"])
            self.assertGreater(candidate["attachment_prior"]["normal"][2], 0)
        self.assertEqual(list(wall_lamp_proposals(lamp, wall, ["wall"] * 2,
                                                {"wall"}, maximum_distance=100)), [])
        ceiling = platform(-2000, 2000, -2000, 2000, -600)
        self.assertEqual(list(wall_lamp_proposals(lamp, ceiling, ["ceiling"] * 2,
                                                {"ceiling"})), [])
        reversed_winding = [triangle[::-1] for triangle in wall]
        reverse = list(wall_lamp_proposals(lamp, reversed_winding, ["wall"] * 2, {"wall"}))
        np.testing.assert_allclose(candidates[0]["attachment_prior"]["normal"],
                                   reverse[0]["attachment_prior"]["normal"])

    def test_wall_lamp_attachment_survives_translation_and_yaw(self):
        import math

        from room3d_connections import lamp_attachment

        params = {"height": 600, "radius": 300, "thickness": 30,
                  "arms": 1, "shade": 180, "profile": "wall_sconce"}
        origin = np.array([200, -1000, 500])
        for yaw in (0, 0.7, math.pi / 2):
            lamp = generate("lamp", params, origin, yaw)
            c, s = math.cos(yaw), math.sin(yaw)
            rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
            wall = np.array([[-1000, -2000, 0], [1000, -2000, 0],
                             [1000, 0, 0], [-1000, 0, 0]]) @ rotation + origin
            triangles = [wall[[0, 1, 2]], wall[[0, 2, 3]]]
            report = lamp_attachment(lamp, triangles, ["wall"] * 2, {"wall"})
            self.assertTrue(report["within_tolerance"])
            np.testing.assert_allclose(report["anchor"], origin + [0, -600, 0])
            parts = np.asarray(lamp["parts"])
            local = (parts - origin) @ rotation.T
            self.assertEqual(len(parts), 3)
            self.assertGreater(local[2, :, 2].min(), 0)
        with self.assertRaises(ValueError):
            generate("lamp", {**params, "arms": 3})
        with self.assertRaises(ValueError):
            generate("lamp", {**params, "radius": 80})

    def test_lamp_attachment_uses_finite_surface_and_excludes_its_own_mesh(self):
        from room3d_connections import lamp_attachment, point_triangle_distance

        triangle = np.array([[0, 0, 0], [100, 0, 0], [0, 0, 100]])
        self.assertAlmostEqual(point_triangle_distance([20, 10, 20], triangle), 10)
        self.assertGreater(point_triangle_distance([1000, 0, 1000], triangle), 1000)
        self.assertEqual(point_triangle_distance([0, 3, 0], np.zeros((3, 3))), 3)
        lamp = generate("lamp", {"height": 1000, "radius": 200, "thickness": 30,
                                  "arms": 3, "shade": 80}, [0, 0, 0], 0.8)
        ceiling = platform(-1000, 1000, -1000, 1000, -1000)
        report = lamp_attachment(lamp, ceiling, ["ceiling"] * 2, {"ceiling"})
        self.assertTrue(report["within_tolerance"])
        moved = platform(-1000, 1000, -1000, 1000, -1500)
        self.assertFalse(lamp_attachment(lamp, moved, ["ceiling"] * 2,
                                        {"ceiling"})["within_tolerance"])
        self.assertIsNone(lamp_attachment(lamp, ceiling, ["lamp"] * 2,
                                         {"ceiling"})["distance"])

    def test_headroom_includes_landing_beyond_clear_treads(self):
        from room3d_connections import stair_headroom

        obj = generate("staircase", {"width": 1000, "run": 5000, "rise": 2000,
                                     "steps": 10, "landing": 1000}, [0, 0, 0])
        surfaces = (platform(-1000, 1000, -100, 4000, -4000)
                    + platform(-1000, 1000, 4000, 5100, -2500))
        result = stair_headroom(obj, surfaces, ["ceiling"] * 4, {"ceiling"})
        self.assertFalse(result["clear"])
        blocked = [s for s in result["samples"] if s["obstructed"]]
        self.assertEqual(len(blocked), 9)
        self.assertEqual({s["section"] for s in blocked}, {"landing"})
        self.assertEqual(result["unknown_samples"], 0)

    def test_headroom_distinguishes_low_ceiling_and_missing_evidence(self):
        from room3d_connections import stair_headroom

        obj = generate("staircase", {"width": 1000, "run": 4000, "rise": 2000,
                                     "steps": 10, "landing": 0}, [0, 0, 0])
        groups = ["ceiling"] * 2
        low = stair_headroom(obj, platform(-1000, 1000, -100, 4100, -2500),
                             groups, {"ceiling"})
        self.assertFalse(low["clear"])
        self.assertEqual(low["minimum_clearance"], 500)
        self.assertEqual(low["obstructing_groups"], ["ceiling"])
        high = stair_headroom(obj, platform(-1000, 1000, -100, 4100, -4000),
                              groups, {"ceiling"})
        self.assertTrue(high["clear"])
        unknown = stair_headroom(obj, [], [], {"ceiling"})
        self.assertFalse(unknown["clear"])
        self.assertEqual(unknown["unknown_samples"], 30)
        self.assertEqual(unknown["obstructed_samples"], 0)
        json.dumps(low)

    def setUp(self):
        self.obj = generate(
            "staircase", {"width": 1000, "run": 3000, "rise": 1800, "steps": 9}
        )
        self.floor = platform(-1000, 1000, -1000, 4000, 0)

    def test_floor_does_not_substitute_for_upper_landing(self):
        r = stair_connections(self.obj, self.floor)
        self.assertTrue(r["ends"]["bottom"]["supported"])
        self.assertFalse(r["ends"]["top"]["supported"])
        self.assertFalse(r["valid"])
        json.dumps(r)

    def test_full_width_landing_connects_but_a_corner_does_not(self):
        full = platform(-1000, 1000, 3000, 4000, -1800)
        self.assertTrue(stair_connections(self.obj, self.floor + full)["valid"])
        corner = platform(-500, -300, 3000, 4000, -1800)
        self.assertFalse(stair_connections(self.obj, self.floor + corner)["valid"])

    def test_wall_top_contact_does_not_prove_walkability(self):
        triangles = self.floor + platform(-1000, 1000, 3000, 4000, -1800)
        groups = ["floor", "floor", "wall_cap", "wall_cap"]
        report = stair_connections(
            self.obj, triangles, groups=groups, walkable_groups={"floor"}
        )
        self.assertTrue(report["valid"])
        self.assertFalse(report["walkable_valid"])
        excluded = stair_connections(
            self.obj, triangles, groups=groups, excluded_groups={"wall_cap"}
        )
        self.assertFalse(excluded["valid"])
        self.assertEqual(
            report["ends"]["top"]["samples"][1]["nearest_group"], "wall_cap"
        )
        report = stair_connections(
            self.obj, triangles, groups=groups, walkable_groups={"floor", "wall_cap"}
        )
        self.assertTrue(report["walkable_valid"])

    def test_rotation_and_translation_preserve_contact(self):
        yaw = 0.7
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        shift = np.array([200, -400, 500])
        triangles = [
            t @ rotation + shift
            for t in self.floor + platform(-1000, 1000, 3000, 4000, -1800)
        ]
        obj = generate("staircase", self.obj["shape"]["parameters"], shift, yaw)
        self.assertTrue(stair_connections(obj, triangles)["valid"])


if __name__ == "__main__":
    unittest.main()
