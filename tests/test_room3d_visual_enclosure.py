"""Training-only proposals for estimated visual enclosure heights."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from room3d_supports import occlusion_hypotheses


class VisualEnclosureTests(unittest.TestCase):
    def setUp(self):
        self.recipe = {
            "creative_stair_replacement": True,
            "floor_y": 100,
            "wall_height": 6000,
            "structural_surfaces": [
                {"group": "boundary_2", "points": [[0, -5900, 0], [0, 100, 0]]}
            ],
            "primitives": [{"kind": "door_panel", "box": [0, 0, 1, 1, -2, 0]}],
        }
        self.training = [
            {"camera": c, "kind": "staircase", "approximate_stair_mask": True}
            for c in (1, 6)
        ]
        self.report = {
            "creative_layout_prior": {"qualification": "estimated"},
            "training_cameras": [1, 6],
            "heldout_cameras": [7],
            "visibility_diagnostics": [{
                "camera": 1, "projected_pixels": 1000, "target_overlap": 900,
                "scene_visible_pixels": 300,
                "occluders": [{"group": "boundary_2", "nearest_pixels": 500,
                               "observed_depth": None, "surface_depth": 1000,
                               "reliable_pixels": 0}],
            }],
        }

    def proposals(self):
        return list(occlusion_hypotheses(self.recipe, self.report, self.training))

    def test_estimated_design_proposes_bounded_visual_heights_without_mutation(self):
        original = copy.deepcopy((self.recipe, self.report, self.training))
        proposals = self.proposals()
        self.assertEqual([p[1]["height"] for p in proposals], [750, 1500, 3000])
        for candidate, evidence in proposals:
            self.assertEqual(candidate["primitives"], self.recipe["primitives"])
            self.assertEqual(candidate["boundary_heights"], {"2": evidence["height"]})
            self.assertEqual(candidate["structural_surfaces"][0]["points"][0][1],
                             100 - evidence["height"])
            self.assertEqual(evidence["evidence"]["boundary_2"],
                             ["estimated_multiview_mask_layout"])
            self.assertIn("requires all-camera acceptance", evidence["qualification"])
        self.assertEqual((self.recipe, self.report, self.training), original)

    def test_default_single_view_observed_and_wrong_projection_do_not_branch(self):
        original = copy.deepcopy((self.recipe, self.report, self.training))
        cases = ("disabled", "no_prior", "single", "observed", "miss", "visible", "tiny")
        for case in cases:
            with self.subTest(case=case):
                self.recipe, self.report, self.training = copy.deepcopy(original)
                if case == "disabled":
                    self.recipe["creative_stair_replacement"] = False
                elif case == "no_prior":
                    self.report.pop("creative_layout_prior")
                elif case == "single":
                    self.training.pop()
                elif case == "observed":
                    self.training[1].pop("approximate_stair_mask")
                elif case == "miss":
                    self.report["visibility_diagnostics"][0]["target_overlap"] = 599
                elif case == "visible":
                    self.report["visibility_diagnostics"][0]["scene_visible_pixels"] = 801
                else:
                    self.report["visibility_diagnostics"][0]["occluders"][0]["nearest_pixels"] = 49
                self.assertEqual(self.proposals(), [])

    def test_reserved_view_cannot_supply_qualification_or_occlusion_votes(self):
        self.report["visibility_diagnostics"][0]["camera"] = 7
        self.training.append({"camera": 7, "kind": "staircase",
                              "approximate_stair_mask": True})
        self.assertEqual(self.proposals(), [])
        self.report["visibility_diagnostics"][0]["camera"] = 1
        expected = self.proposals()
        heldout = copy.deepcopy(self.report["visibility_diagnostics"][0])
        heldout.update(camera=7)
        heldout["occluders"][0].update(group="boundary_99", nearest_pixels=100000)
        self.report["visibility_diagnostics"].append(heldout)
        self.assertEqual(self.proposals(), expected)


if __name__ == "__main__":
    unittest.main()
