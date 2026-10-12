"""Rendered landing selection and reserved-view independence."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from batch_room3d import generate_mesh
from room3d_assemblies import landing_for_stairs
from room3d_landing_fit import fit_landing
from room3d_preview import render
from room3d_refine import depth_error
from room3d_shape_fit import fit_objects
from room3d_shapes import generate


class LandingFitTests(unittest.TestCase):
    def test_creative_fallback_can_jointly_connect_a_missing_top_landing(self):
        from room3d_interpretation import interpret

        room, truth = self.scene()
        room["cameras"].append({**room["cameras"][0], "index": 2,
                                "origin": [3500, -4500, 1000]})
        stairs = truth["primitives"][0]
        truth["primitives"].append(landing_for_stairs(stairs, 1200))
        baseline = {"primitives": [copy.deepcopy(truth["primitives"][1])],
                    "creative_stair_landings": True}
        original = copy.deepcopy((room, baseline, stairs))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "truth.obj"
            generate_mesh(room, path, truth)
            depths, group = [], []
            for camera in room["cameras"]:
                depth = render(path, camera, return_depth=True)[2]
                mask = render(path, camera, only_groups={"primitive_0"})[1]
                depths.append((np.where(np.isfinite(depth), depth, 10000), np.isfinite(depth)))
                group.append({"camera": camera["index"], "mask": mask})
            _, _, off = interpret(room, baseline, baseline, group, depths, stairs,
                                  {"kind": "floor", "y": 0}, root / "off", generate_mesh)
            self.assertFalse(off["accepted"])
            self.assertFalse(off["trials"][0]["gates"]["stair_end_connections"])
            self.assertNotIn("landing_assembly", off["trials"][0])
            enabled = {**baseline, "joint_creative_stair_landings": True}
            obj, fitted, on = interpret(room, baseline, enabled, group, depths, stairs,
                                       {"kind": "floor", "y": 0}, root / "on", generate_mesh)
            self.assertTrue(on["accepted"], on)
            assembly = on["landing_assembly"]
            self.assertTrue(assembly["accepted"])
            self.assertEqual(assembly["heldout_cameras"], [2])
            self.assertTrue(on["trials"][0]["connections"]["walkable_valid"])
            self.assertEqual([p["kind"] for p in fitted["primitives"]],
                             ["ceiling", "staircase", "landing"])
            self.assertIs(obj, fitted["primitives"][1])
            self.assertEqual(obj["shape"]["origin"], stairs["shape"]["origin"])
            self.assertEqual(obj["shape"]["yaw"], stairs["shape"]["yaw"])
            self.assertEqual(obj["shape"]["parameters"], {
                **stairs["shape"]["parameters"], "steps": round(3000 / 180)})
            self.assertTrue(all(a <= b + .005 for a, b in zip(
                on["trials"][0]["depth_after"], on["trials"][0]["depth_before"], strict=True)))
            # Change reserved depth only: selected training geometry stays the
            # same, but a newly harmful assembly must not be adopted.
            baseline_path = root / "baseline.obj"
            generate_mesh(room, baseline_path, baseline)
            reserved = render(baseline_path, room["cameras"][2], return_depth=True)[2]
            changed = [*depths[:2], (np.where(np.isfinite(reserved), reserved, 10000),
                                    np.isfinite(reserved))]
            _, _, rejected = interpret(room, baseline, enabled, group, changed, stairs,
                                       {"kind": "floor", "y": 0}, root / "reserved", generate_mesh)
            candidate = rejected["trials"][0]["landing_assembly"]
            self.assertEqual(candidate["selected"], assembly["selected"])
            self.assertEqual(candidate["trials"], assembly["trials"])
            self.assertFalse(candidate["accepted"])
            self.assertFalse(rejected["accepted"])
        self.assertEqual((room, baseline, stairs), original)

    def test_batch_fitter_opt_in_adds_landing_after_accepted_stair(self):
        room, truth = self.scene()
        stairs = truth["primitives"][0]
        truth["primitives"].append(landing_for_stairs(stairs, 1200))
        recipe = {"primitives": [truth["primitives"][1]], "creative_fallback": True,
                  "creative_stair_landings": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "truth.obj"
            generate_mesh(room, path, truth)
            depths, segments = [], []
            for c in room["cameras"]:
                d = render(path, c, return_depth=True)[2]
                mask = render(path, c, return_depth=True, only_groups={"primitive_0"})[1]
                depths.append((np.where(np.isfinite(d), d, 10000), np.isfinite(d)))
                segments.append([({"label": "staircase", "score": .99,
                                   "predicted_iou": .99, "mask": f"camera{c['index']}-stairs.png"}, mask)])
            with patch("room3d_shape_fit.initial_shapes", return_value=[stairs]):
                fitted, result = fit_objects(room, recipe, segments, depths,
                                              root / "fit", generate_mesh)
            generate_mesh(room, root / "final.obj", fitted)
            for i, (c, (target, valid)) in enumerate(zip(room["cameras"], depths, strict=True)):
                final = render(root / "final.obj", c, return_depth=True)[2]
                report = result["objects"][0]
                self.assertAlmostEqual(report["depth_after"][i], depth_error(final, target, valid))
                self.assertLessEqual(report["depth_after"][i], report["depth_before"][i] + .005)
        report = result["objects"][0]
        self.assertTrue(report["accepted"], report)
        self.assertTrue(report["landing_assembly"]["accepted"], report["landing_assembly"])
        self.assertTrue(report["landing_assembly"]["object_fit_preserved"])
        self.assertEqual(fitted["primitives"][-1]["kind"], "landing")
        self.assertEqual(fitted["primitives"][-2]["kind"], "staircase")

    def scene(self):
        room = {"boundaries": [{"index": 0, "xmin": -4000, "zmin": -1000,
                                "xmax": 4000, "zmax": 9000, "type": 4, "flags": 0}],
                "cameras": [{"index": i, "origin": origin, "target": [0, -2000, 4500],
                             "fov": 240, "roll": 0}
                            for i, origin in enumerate(([4000, -4000, 0],
                                                        [-3500, -4500, 3000]))]}
        stairs = generate("staircase", {"width": 1800, "run": 5000, "rise": 3000,
                          "steps": 15, "structure": "suspended", "slab_depth": 100})
        recipe = {"primitives": [stairs, {"kind": "ceiling",
                   "box": [-4000, -1000, 4000, 9000, -6500, -6400]}]}
        return room, recipe

    def test_real_render_selection_keeps_reserved_depth_out_of_training(self):
        room, recipe = self.scene()
        original = copy.deepcopy(recipe)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = copy.deepcopy(recipe)
            truth["primitives"].append(landing_for_stairs(recipe["primitives"][0], 1200))
            path = root / "truth.obj"
            generate_mesh(room, path, truth)
            depths = []
            for c in room["cameras"]:
                d = render(path, c, return_depth=True)[2]
                depths.append((np.where(np.isfinite(d), d, 10000), np.isfinite(d)))
            selected, first = fit_landing(room, recipe, 0, depths, root / "first",
                                          generate_mesh, heldout_cameras=[1])
            changed = [(d.copy(), v.copy()) for d, v in depths]
            # Make only the added landing contradict reliable held-out depth.
            # Restrict reliability to actually changed rendered pixels, rather
            # than hoping a global rescale changes the error in one direction.
            generate_mesh(room, root / "input.obj", recipe)
            generate_mesh(room, root / "selected.obj", selected)
            camera = room["cameras"][1]
            old = render(root / "input.obj", camera, return_depth=True)[2]
            new = render(root / "selected.obj", camera, return_depth=True)[2]
            finite = np.isfinite(old) & np.isfinite(new)
            changed_pixels = np.zeros_like(finite)
            changed_pixels[finite] = np.abs(np.log(new[finite] / old[finite])) > .01
            self.assertTrue(changed_pixels.any())
            changed[1] = (np.where(np.isfinite(old), old, 10000), changed_pixels)
            _, second = fit_landing(room, recipe, 0, changed, root / "changed",
                                    generate_mesh, heldout_cameras=[1])
        self.assertEqual(recipe, original)
        self.assertTrue(first["accepted"], first)
        self.assertEqual(len(selected["primitives"]), len(recipe["primitives"]) + 1)
        self.assertEqual(first["selected"], second["selected"])
        self.assertFalse(second["accepted"])
        self.assertEqual(first["trials"], second["trials"])
        self.assertNotEqual(first["heldout"], second["heldout"])
        self.assertTrue(all(t["headroom"]["unknown_samples"] == 0 for t in first["trials"]))
        self.assertTrue(all(s["camera"] == 0 for t in first["trials"] for s in t["training"]))

    def test_unknown_headroom_does_not_certify_a_landing(self):
        room, recipe = self.scene()
        recipe["primitives"] = recipe["primitives"][:1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "before.obj"
            generate_mesh(room, path, recipe)
            depths = []
            for c in room["cameras"]:
                d = render(path, c, return_depth=True)[2]
                depths.append((np.where(np.isfinite(d), d, 10000), np.isfinite(d)))
            result, report = fit_landing(room, recipe, 0, depths, root / "fit",
                                         generate_mesh, candidate_depths=[400])
        self.assertFalse(report["accepted"])
        self.assertIs(result, recipe)
        self.assertFalse(report["trials"][0]["gates"]["known_headroom"])
        self.assertGreater(report["trials"][0]["headroom"]["unknown_samples"], 0)


if __name__ == "__main__":
    unittest.main()
