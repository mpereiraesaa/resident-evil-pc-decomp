"""Scale-bias veto, fixed calibration and deferred reserved-view evidence."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from room3d_depth_guard import FloorDepthGuard, check_selected_scene
from room3d_refine import depth_error


class DepthGuardTests(unittest.TestCase):
    def test_raw_improvement_can_fail_fixed_floor_reference(self):
        floor = np.full((64, 64), 1000.)
        before = floor.copy()
        before[:16] = 500
        after = before.copy()
        after[:16] = 2500
        target = floor * 2
        valid = np.ones(floor.shape, bool)
        mask = valid.copy()
        segments = [[({"label": "wooden floor", "score": .99, "predicted_iou": .99}, mask)]] * 2
        room = {"cameras": [{"index": 0}, {"index": 1}]}
        original = [array.copy() for array in (floor, before, after, target, valid, mask)]
        self.assertLess(depth_error(after, target, valid), depth_error(before, target, valid))
        with patch("room3d_depth_guard.render", return_value=(None, None, floor)) as renderer:
            guard = FloorDepthGuard(room, segments, [(target, valid)] * 2,
                                    Path("baseline.obj"), {0: before, 1: before})
            result = guard.compare(0, after)
            self.assertFalse(result["safe"])
            self.assertTrue(result["calibration_accepted"])
            self.assertEqual(guard.calibration[0]["scale"], .5)
            self.assertEqual(set(guard.calibration), {0})
            self.assertEqual(renderer.call_count, 1)
            # A different candidate does not recompute or tune calibration.
            self.assertTrue(guard.compare(0, floor)["safe"])
            self.assertEqual(renderer.call_count, 1)
            guard.compare(1, after)
            self.assertEqual(set(guard.calibration), {0, 1})
            self.assertEqual(renderer.call_count, 2)
        for array, saved in zip((floor, before, after, target, valid, mask), original, strict=True):
            np.testing.assert_array_equal(array, saved)

    def test_missing_floor_evidence_keeps_raw_reference(self):
        floor = np.full((64, 64), 1000.)
        valid = np.ones(floor.shape, bool)
        target = floor * 2
        with patch("room3d_depth_guard.render", return_value=(None, None, floor)):
            guard = FloorDepthGuard({"cameras": [{"index": 0}]}, [[]], [(target, valid)],
                                    Path("baseline.obj"), {0: floor})
            result = guard.compare(0, target)
        self.assertFalse(result["calibration_accepted"])
        self.assertTrue(result["safe"])
        self.assertEqual(result["depth_after"], 0)
        self.assertEqual(guard.calibration[0]["scale"], 1.)

    def test_selected_assembly_checks_training_before_reserved_without_reselection(self):
        floor = np.full((64, 64), 1000.)
        baseline = floor.copy()
        baseline[:16] = 500
        bad = baseline.copy()
        bad[:16] = 2500
        valid = np.ones(floor.shape, bool)
        room = {"cameras": [{"index": 0}, {"index": 1}]}
        segments = [[({"label": "floor", "score": .99, "predicted_iou": .99}, valid)]] * 2
        depths = [(floor * 2, valid)] * 2
        previous, candidate = {"primitives": []}, {"primitives": [{"kind": "column"}]}
        with tempfile.TemporaryDirectory() as temp:
            def check(name, training, reserved):
                calls = []

                def raster(path, camera, **kwargs):
                    calls.append(camera["index"])
                    if kwargs.get("only_groups"):
                        result = floor
                    elif path.name == "baseline.obj":
                        result = baseline
                    else:
                        result = training if camera["index"] == 0 else reserved
                    return None, None, result

                with patch("room3d_depth_guard.render", side_effect=raster):
                    report = check_selected_scene(room, previous, candidate, segments, depths,
                                                  Path(temp) / name, lambda *args: None,
                                                  heldout_cameras=[1])
                return report, calls

            rejected, calls = check("training-bad", bad, floor)
            self.assertFalse(rejected["accepted"])
            self.assertEqual(rejected["reason"], "training_view_regressed")
            self.assertEqual(set(calls), {0})
            self.assertEqual(rejected["heldout"], [])
            accepted, _ = check("accepted", floor, floor)
            rejected, _ = check("reserved-bad", floor, bad)
            self.assertTrue(accepted["accepted"])
            self.assertFalse(rejected["accepted"])
            self.assertEqual(rejected["reason"], "reserved_view_regressed")
            self.assertEqual(accepted["training"], rejected["training"])
            self.assertEqual(accepted["calibration"][0], rejected["calibration"][0])
        self.assertEqual(previous, {"primitives": []})
        self.assertEqual(candidate, {"primitives": [{"kind": "column"}]})

    def test_all_reserved_skips_generation_and_unknown_camera_fails(self):
        room = {"cameras": [{"index": 0}]}
        with tempfile.TemporaryDirectory() as temp, patch("room3d_depth_guard.render") as renderer:
            generate = unittest.mock.Mock()
            report = check_selected_scene(room, {}, {}, [], [], Path(temp), generate, [0])
            self.assertEqual(report["reason"], "no_training_camera")
            generate.assert_not_called()
            renderer.assert_not_called()
            with self.assertRaises(ValueError):
                check_selected_scene(room, {}, {}, [], [], Path(temp), generate, [99])


if __name__ == "__main__":
    unittest.main()
