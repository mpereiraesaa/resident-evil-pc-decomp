"""Exercise staged geometry fitting through final whole-room acceptance."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from batch_room3d import generate_mesh
from room3d_preview import render
from room3d_refine import depth_error
from room3d_shape_fit import fit_group, fit_objects
from room3d_shapes import generate, prism


class DepthPolishTests(unittest.TestCase):
    def test_creative_fallback_cannot_bypass_known_ceiling_obstruction(self):
        from room3d_interpretation import interpret

        camera = {"index": 0, "origin": [0, -2000, -4000],
                  "target": [0, -1500, 2500], "fov": 240, "roll": 0}
        room = {"cameras": [camera], "boundaries": [{"index": 0, "xmin": -4000,
                "xmax": 4000, "zmin": -1000, "zmax": 10000, "type": 4, "flags": 0}]}
        stairs = generate("staircase", {"width": 1200, "run": 5000, "rise": 3000,
                          "steps": 17, "structure": "suspended", "slab_depth": 100})
        ceiling = {"kind": "ceiling", "box": [-4000, -1000, 4000, 10000, -3600, -3500]}
        landing = {"kind": "landing", "parts": [prism([-1000, -3000, 5000],
                                                                    [1000, -2900, 6500])]}
        stripped = {"primitives": [ceiling, landing]}
        baseline = {"primitives": [ceiling, landing, stairs]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "target.obj"
            generate_mesh(room, path, baseline)
            depth = render(path, camera, return_depth=True)[2]
            mask = render(path, camera, return_depth=True, only_groups={"primitive_2"})[1]
            depths = [(np.where(np.isfinite(depth), depth, 10000), np.isfinite(depth))]
            group = [{"camera": 0, "mask": mask}]
            args = (room, baseline, stripped, group, depths, stairs, {"kind": "floor", "y": 0})
            _, _, off = interpret(*args, root / "off", generate_mesh)
            args = (room, baseline, {**stripped, "scene_depth_polish": True},
                    group, depths, stairs, {"kind": "floor", "y": 0})
            _, _, on = interpret(*args, root / "on", generate_mesh)
        self.assertTrue(off["accepted"])
        self.assertFalse(on["accepted"])
        self.assertFalse(on["trials"][0]["gates"]["headroom_safe"])
        self.assertTrue(all(v for k, v in on["trials"][0]["gates"].items()
                            if k != "headroom_safe"))
        self.assertGreater(on["trials"][0]["headroom"]["obstructed_samples"], 0)

    def test_known_low_ceiling_rejects_wider_better_mask_candidate(self):
        camera = {"index": 0, "origin": [0, -2000, -4000],
                  "target": [0, -1500, 2500], "fov": 240, "roll": 0}
        room = {"cameras": [camera], "boundaries": [{"index": 0, "xmin": -4000,
                "xmax": 4000, "zmin": 0, "zmax": 10000, "type": 4, "flags": 0}]}
        ceiling = {"kind": "ceiling", "parts": [
            prism([-4000, -3600, 0], [-700, -3500, 10000]),
            prism([700, -3600, 0], [4000, -3500, 10000]),
            prism([-700, -6100, 0], [700, -6000, 10000]),
        ]}
        params = {"width": 2400, "run": 5000, "rise": 3000, "steps": 15,
                  "structure": "suspended", "slab_depth": 100}
        wide = generate("staircase", params)
        narrow = generate("staircase", {**params, "width": 1200})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "target.obj"
            generate_mesh(room, path, {"primitives": [wide]})
            _, mask, depth = render(path, camera, return_depth=True, only_groups={"primitive_0"})
            depths = [(np.where(np.isfinite(depth), depth, 10000), np.isfinite(depth))]
            group = [{"kind": "staircase", "camera": 0, "mask": mask}]
            recipe = {"primitives": [ceiling]}
            off, _ = fit_group(room, recipe, group, depths, root / "off", generate_mesh,
                               rounds=0, initial_candidates=[wide, narrow])
            on, report = fit_group(room, {**recipe, "constrain_stair_headroom": True},
                                   group, depths, root / "on", generate_mesh,
                                   rounds=0, initial_candidates=[wide, narrow])
        self.assertEqual(off["shape"]["parameters"]["width"], 2400)
        self.assertEqual(on["shape"]["parameters"]["width"], 1200)
        self.assertGreater(report["initializations"][0]["scores"][0]["headroom_obstructed_samples"], 0)
        self.assertEqual(report["after"][0]["headroom_obstructed_samples"], 0)

    def test_staged_fit_keeps_reserved_view_out_and_runs_final_depth_gates(self):
        cameras = [
            {"index": i, "origin": origin, "target": [0, -1500, 2500],
             "fov": 240, "roll": 0}
            for i, origin in enumerate(([0, -2000, -4000], [3000, -2000, -4000],
                                       [-3000, -2000, -4000], [-6000, -1500, 2500]))
        ]
        room = {"cameras": cameras, "boundaries": [{"index": 0, "xmin": -4000,
                "xmax": 4000, "zmin": 0, "zmax": 10000, "type": 4, "flags": 0}]}
        truth = generate("staircase", {"width": 1600, "run": 5000, "rise": 3000,
                         "steps": 15, "structure": "suspended", "slab_depth": 100})
        recipe = {"primitives": [], "creative_fallback": True,
                  "scene_depth_fit": True, "scene_depth_polish": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "truth.obj"
            generate_mesh(room, target, {"primitives": [truth]})
            segments, depths = [], []
            for camera in cameras:
                depth = render(target, camera, return_depth=True)[2]
                mask = render(target, camera, return_depth=True,
                              only_groups={"primitive_0"})[1]
                depths.append((np.where(np.isfinite(depth), depth, 10000), np.isfinite(depth)))
                entries = [] if camera["index"] == 3 else [(
                    {"label": "staircase", "score": .99, "predicted_iou": .99,
                     "mask": f"camera{camera['index']}-stairs.png"}, mask)]
                segments.append(entries)
            with patch("room3d_shape_fit.initial_shapes", return_value=[truth]):
                fitted, results = fit_objects(room, recipe, segments, depths,
                                              root / "details", generate_mesh)
            self.assertEqual(len(results["objects"]), 1)
            report = results["objects"][0]
            self.assertTrue(report["accepted"], report["gates"])
            self.assertTrue(report["gates"]["headroom_safe"])
            # No ceiling observations imply unknown clearance, not proof of it.
            self.assertGreater(report["headroom"]["unknown_samples"], 0)
            self.assertEqual(report["training_cameras"], [0, 1])
            self.assertEqual(report["heldout_cameras"], [2])
            self.assertEqual(report["scene_depth_fit"]["training_scene_cameras"], [0, 1, 3])
            self.assertNotIn("scene_depth_fit", report["depth_polish"]["initial_report"])
            self.assertNotIn("scene_depth_excess", report["heldout"][0])
            self.assertEqual(report["depth_polish"]["initial_shape"], truth["shape"])
            final = root / "final.obj"
            generate_mesh(room, final, fitted)
            for i, (camera, (depth, valid)) in enumerate(zip(cameras, depths, strict=True)):
                measured = depth_error(render(final, camera, return_depth=True)[2], depth, valid)
                self.assertAlmostEqual(measured, report["depth_after"][i])
                self.assertLessEqual(measured, report["depth_before"][i] + .005)


if __name__ == "__main__":
    unittest.main()
