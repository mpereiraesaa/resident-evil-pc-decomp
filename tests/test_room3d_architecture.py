"""Column correspondences must reject other pillars and evaluation leakage."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from room3d_architecture import (
    column_axis_hypotheses,
    column_foot_correspondences,
    fit_column_axis,
    projected_object_evidence,
)
from room3d_reconstruct import project


class ColumnCorrespondenceTests(unittest.TestCase):
    def test_foreground_occlusion_is_not_an_object_depth_conflict(self):
        part = np.full((3, 4), 2000.0)
        evidence = projected_object_evidence(part, part, np.full(part.shape, 1000),
                                            np.ones(part.shape, bool))
        self.assertEqual(evidence["counts"]["observed_foreground"], 12)
        self.assertEqual(evidence["counts"]["model_depth_conflict"], 0)
        self.assertEqual(sum(evidence["counts"].values()), evidence["projected_pixels"])

    def test_unknown_depth_cannot_disprove_or_confirm_matching_object(self):
        part = np.full((3, 4), 2000.0)
        evidence = projected_object_evidence(part, part, np.full(part.shape, np.nan),
                                            np.ones(part.shape, bool), np.ones(part.shape, bool))
        self.assertEqual(evidence["counts"]["unknown"], 12)
        self.assertEqual(evidence["counts"]["depth_agreement_with_matching_mask"], 0)

    def test_depth_conflict_and_matching_support_remain_distinct(self):
        part = np.array([[1000, 1000, 1000, np.inf]])
        observed = np.array([[1000, 1000, 2000, 1000]])
        mask = np.array([[True, False, True, True]])
        evidence = projected_object_evidence(part, part, observed,
                                            np.ones(part.shape, bool), mask)
        for label in ("depth_agreement_with_matching_mask", "depth_agreement_without_identity",
                      "model_depth_conflict"):
            self.assertEqual(evidence["counts"][label], 1)
        self.assertEqual(evidence["projected_pixels"], 3)
        with self.assertRaises(ValueError):
            projected_object_evidence(part, part, observed, np.ones((2, 2), bool))

    def test_inferred_scene_occlusion_cannot_overrule_matching_observed_depth(self):
        part = np.full((2, 3), 2000.0)
        scene = np.full(part.shape, 1000.0)
        unknown = projected_object_evidence(part, scene, part, np.zeros(part.shape, bool))
        self.assertEqual(unknown["counts"]["inferred_scene_occluded"], 6)
        known = projected_object_evidence(part, scene, part, np.ones(part.shape, bool),
                                         np.ones(part.shape, bool))
        self.assertEqual(known["counts"]["depth_agreement_with_matching_mask"], 6)
        self.assertEqual(known["counts"]["model_depth_conflict"], 0)

    def test_shared_column_fit_recovers_both_rendered_silhouettes(self):
        from batch_room3d import generate_mesh
        from room3d_preview import render
        from room3d_shapes import generate

        room = {**self.room, "boundaries": [{"index": 0, "xmin": -4000,
                "xmax": 4000, "zmin": -2000, "zmax": 4000, "type": 4, "flags": 0}]}
        truth = generate("column", {"height": 2400, "width": 350})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            generate_mesh(room, root / "truth.obj", {"primitives": [truth]})
            views = [[({"mask": str(c["index"])}, render(
                root / "truth.obj", c, return_depth=True, only_groups={"primitive_0"}
            )[1])] for c in room["cameras"]]
            hypothesis = {"eligible": True, "ambiguous": False, "xz": [0, 0],
                          "observations": [{"camera": i, "mask": str(i)} for i in (0, 1)],
                          "checks": [{"height": 2400}, {"height": 2400}]}
            fitted, report = fit_column_axis(room, hypothesis, views, 0, root / "fit", generate_mesh)
            scores = report["trials"][report["selected"]]["scores"]
            self.assertGreater(min(s["silhouette_iou"] for s in scores), 0.7)
            self.assertEqual(report["training_cameras"], [0, 1])
            self.assertEqual(report["heldout_cameras"], [])
            self.assertFalse(report["accepted_room_edit"])
            np.testing.assert_allclose(fitted["shape"]["origin"], [0, 0, 0])
            with self.assertRaises(ValueError):
                fit_column_axis(room, {**hypothesis, "ambiguous": True}, views, 0,
                                root / "ambiguous", generate_mesh)

    def setUp(self):
        self.room = {"cameras": [
            {"index": i, "origin": [x, -1000, -4000],
             "target": [x, -1000, 0], "fov": 240, "roll": 0}
            for i, x in enumerate((-1000, 1000))
        ]}

    def detection(self, camera, x=0, name="column"):
        uv, _ = project(np.array([[x, 0, 0]]), camera)
        px, py = np.rint(uv[0]).astype(int)
        mask = np.zeros((240, 320), bool)
        mask[py-40:py+1, px-2:px+3] = True
        return {"label": "stone column", "mask": name,
                "predicted_iou": 0.95, "score": 0.3}, mask

    def test_same_column_matches_and_different_column_does_not(self):
        cameras = self.room["cameras"]
        views = [[self.detection(c)] for c in cameras]
        report = column_foot_correspondences(self.room, views)
        self.assertEqual(len(report["pairs"]), 1)
        self.assertFalse(report["pairs"][0]["ambiguous"])
        views[1] = [self.detection(cameras[1], x=800)]
        report = column_foot_correspondences(self.room, views)
        self.assertEqual(report["pairs"], [])
        self.assertEqual(report["rejected_pairs"][0]["reason"], "reprojection_error")
        self.assertGreater(max(report["rejected_pairs"][0]["pixel_errors"]), 3)

    def test_reserved_view_and_truncated_base_cannot_supply_evidence(self):
        views = [[self.detection(c)] for c in self.room["cameras"]]
        report = column_foot_correspondences(self.room, views, excluded_cameras=[1])
        self.assertEqual([a["camera"] for a in report["anchors"]], [0])
        self.assertEqual(report["pairs"], [])
        views[1][0][1][-1, 100] = True
        self.assertEqual(column_foot_correspondences(self.room, views)["pairs"], [])

    def test_multiple_possible_columns_are_marked_ambiguous(self):
        views = [[self.detection(c)] for c in self.room["cameras"]]
        views[1].append(self.detection(self.room["cameras"][1], x=15, name="other"))
        report = column_foot_correspondences(self.room, views)
        self.assertEqual(len(report["pairs"]), 2)
        self.assertTrue(all(p["ambiguous"] for p in report["pairs"]))

    def test_shared_axis_survives_incomplete_visible_foot(self):
        views = [[self.detection(c)] for c in self.room["cameras"]]
        views[1][0][1][175:] = False
        self.assertEqual(column_foot_correspondences(self.room, views)["pairs"], [])
        trials = column_axis_hypotheses(self.room, views)["trials"]
        self.assertEqual(len(trials), 1)
        self.assertTrue(trials[0]["eligible"])
        self.assertFalse(trials[0]["ambiguous"])
        np.testing.assert_allclose(trials[0]["xz"], [0, 0], atol=1e-7)

        views[1].append(self.detection(self.room["cameras"][1], name="duplicate"))
        trials = column_axis_hypotheses(self.room, views)["trials"]
        self.assertTrue(all(t["ambiguous"] for t in trials))

    def test_axis_alone_cannot_accept_other_column_or_inconsistent_height(self):
        views = [[self.detection(c)] for c in self.room["cameras"]]
        views[1] = [self.detection(self.room["cameras"][1], x=800)]
        trials = column_axis_hypotheses(self.room, views)["trials"]
        self.assertTrue(trials[0]["gates"]["axis"])
        self.assertFalse(trials[0]["eligible"])
        views[1] = [self.detection(self.room["cameras"][1])]
        views[1][0][1][:160] = False
        trials = column_axis_hypotheses(self.room, views)["trials"]
        self.assertFalse(trials[0]["gates"]["height"])
        self.assertEqual(column_axis_hypotheses(
            self.room, views, excluded_cameras=[1]
        )["trials"], [])


if __name__ == "__main__":
    unittest.main()
