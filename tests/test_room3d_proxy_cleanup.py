"""Footprint safety and rendered training/reserved-view proxy cleanup."""
import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from batch_room3d import generate_mesh
from room3d_assemblies import landing_for_stairs
from room3d_preview import render
from room3d_proxy_cleanup import eligible_boundaries, fit, flight_covers
from room3d_shapes import generate


class ProxyCleanupTests(unittest.TestCase):
    def test_real_footprint_excludes_walls_raised_supports_and_fixture_anchors(self):
        stairs = generate("staircase", {"width": 1000, "run": 3000, "rise": 1800,
                          "steps": 9, "landing": 0})
        recipe = {"primitives": [stairs]}
        boundary = {"index": 7, "type": 1, "flags": 0,
                    "xmin": -300, "xmax": 300, "zmin": 500, "zmax": 2500}
        room = {"boundaries": [boundary]}
        self.assertTrue(flight_covers(boundary, stairs, recipe))
        self.assertEqual(list(eligible_boundaries(room, recipe)), ["boundary_7"])
        for field, value in (("skip_boundaries", [7]), ("shared_wall_boundaries", [7]),
                             ("boundary_heights", {"7": 1400})):
            self.assertEqual(eligible_boundaries(room, {**recipe, field: value}), {})
        self.assertEqual(eligible_boundaries({"boundaries": [{**boundary, "flags": 256}]}, recipe), {})
        anchored = {"primitives": [stairs, {"kind": "lamp", "attachment_prior": {"group": "boundary_7"}}]}
        self.assertEqual(eligible_boundaries(room, anchored), {})
        raised = generate("staircase", stairs["shape"]["parameters"], [0, -1000, 0])
        self.assertFalse(flight_covers(boundary, raised, {"primitives": [raised]}))
        diagonal = generate("staircase", stairs["shape"]["parameters"], yaw=math.pi / 4)
        inside_aabb_only = {**boundary, "xmin": 200, "xmax": 1200, "zmin": 200, "zmax": 1200}
        self.assertFalse(flight_covers(inside_aabb_only, diagonal, {"primitives": [diagonal]}))

    def test_connected_landing_extends_coverage_but_gap_and_wrong_height_do_not(self):
        stairs = generate("staircase", {"width": 1000, "run": 3000, "rise": 1800,
                          "steps": 9, "landing": 0})
        boundary = {"xmin": -300, "xmax": 300, "zmin": 2500, "zmax": 3500}
        landing = landing_for_stairs(stairs, 1000)
        recipe = {"primitives": [stairs, landing]}
        self.assertTrue(flight_covers(boundary, stairs, recipe))
        for shift in ([0, 0, 200], [0, 400, 0]):
            altered = copy.deepcopy(landing)
            altered["parts"] = (np.asarray(altered["parts"]) + shift).tolist()
            self.assertFalse(flight_covers(boundary, stairs, {"primitives": [stairs, altered]}))

    def test_rendered_cleanup_and_reserved_rejection_preserve_training_selection(self):
        room = {"boundaries": [
            {"index": 0, "type": 1, "flags": 0, "xmin": -800, "xmax": 800,
             "zmin": 1000, "zmax": 4500},
            {"index": 1, "type": 4, "flags": 0, "xmin": -5000, "xmax": 5000,
             "zmin": -1000, "zmax": 10000}],
            "cameras": [{"index": i, "origin": origin, "target": [0, -800, 3000],
                         "fov": 240, "roll": 0}
                        for i, origin in enumerate(([4000, -2000, 1500],
                                                    [-4000, -2000, 2000], [0, -1800, 7000]))]}
        stairs = generate("staircase", {"width": 1800, "run": 5000, "rise": 3000,
                          "steps": 17, "structure": "suspended", "slab_depth": 100})
        recipe = {"primitives": [stairs, landing_for_stairs(stairs, 1000)]}
        original = copy.deepcopy((room, recipe))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            truth = root / "truth.obj"
            generate_mesh(room, truth, {**recipe, "skip_boundaries": [0]})
            depths, segments = [], []
            for camera in room["cameras"]:
                depth = render(truth, camera, return_depth=True)[2]
                floor = render(truth, camera, return_depth=True, only_groups={"floor"})[2]
                mask = np.isfinite(floor) & (floor <= depth + 1)
                depths.append((np.where(np.isfinite(depth), depth, 10000), np.isfinite(depth)))
                segments.append([({"label": "wooden floor", "score": .99, "predicted_iou": .99}, mask)])
            fitted, report = fit(room, recipe, segments, depths, root / "fit", generate_mesh,
                                 heldout_cameras=[2])
            self.assertTrue(report["accepted"], report)
            self.assertEqual(fitted["skip_boundaries"], [0])
            self.assertEqual(fitted["primitives"], recipe["primitives"])
            self.assertEqual({d["camera"] for d in report["diagnostics"]}, {0, 1})
            guarded_recipe = {**recipe, "floor_depth_guard": True}
            guarded, guarded_report = fit(room, guarded_recipe, segments, depths,
                                          root / "guarded", generate_mesh, heldout_cameras=[2])
            self.assertTrue(guarded_report["accepted"], guarded_report)
            self.assertEqual(guarded["skip_boundaries"], [0])
            self.assertEqual(set(guarded_report["floor_depth_calibration"]), {0, 1, 2})
            self.assertEqual([v["camera"] for v in guarded_report["trials"][0]["floor_depth_guard"]], [0, 1])
            before = root / "before.obj"
            generate_mesh(room, before, recipe)
            reserved = render(before, room["cameras"][2], return_depth=True)[2]
            changed = [*depths[:2], (np.where(np.isfinite(reserved), reserved, 10000),
                                    np.isfinite(reserved))]
            result, rejected = fit(room, recipe, [*segments[:2], []], changed, root / "rejected", generate_mesh,
                                   heldout_cameras=[2])
            self.assertFalse(rejected["accepted"])
            self.assertEqual(rejected["selected"], report["selected"])
            self.assertEqual(rejected["trials"], report["trials"])
            self.assertEqual(rejected["diagnostics"], report["diagnostics"])
            self.assertEqual(result, recipe)
            _, guarded_rejected = fit(room, guarded_recipe, [*segments[:2], []], changed,
                                      root / "guarded-rejected", generate_mesh, heldout_cameras=[2])
            self.assertFalse(guarded_rejected["accepted"])
            self.assertEqual(guarded_rejected["trials"], guarded_report["trials"])
            self.assertEqual(guarded_rejected["selected"], guarded_report["selected"])
            _, excluded = fit(room, recipe, segments, depths, root / "all-reserved", generate_mesh,
                              heldout_cameras=[0, 1, 2])
            self.assertEqual(excluded["reason"], "no_training_camera")
        self.assertEqual((room, recipe), original)


if __name__ == "__main__":
    unittest.main()
