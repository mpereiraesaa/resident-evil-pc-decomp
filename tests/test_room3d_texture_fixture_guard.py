"""Prevent confident foreground masks from becoming architecture textures."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from batch_room3d import generate_mesh
from room3d_preview import render
from room3d_reconstruct import bake_projected
from room3d_segment import texture_support


class ForegroundFixtureGuardTests(unittest.TestCase):
    def test_veto_includes_unknown_surfaces_and_preserves_own_category(self):
        mask = np.array([[True, False]])
        item = {"label": "wall lamp chandelier", "score": .35, "predicted_iou": .85}
        xy = np.array([[0, 0], [1, 0]])
        for kind in (None, "collision_block", "floor", "wall", "staircase"):
            self.assertTrue(texture_support([(item, mask)], kind, xy).all())
            np.testing.assert_array_equal(
                texture_support([(item, mask)], kind, xy, foreground_fixture_guard=True),
                [False, True])
        self.assertTrue(texture_support([(item, mask)], "lamp", xy,
                                        foreground_fixture_guard=True).all())
        wall = {"label": "wall", "score": .95, "predicted_iou": .99}
        entries = [(item, mask), (wall, np.ones_like(mask))]
        self.assertFalse(texture_support(entries, "wall", xy, foreground_fixture_guard=True)[0])

    def test_weak_or_ambiguous_labels_do_not_gain_veto_authority(self):
        mask = np.ones((1, 1), dtype=bool)
        xy = np.array([[0, 0]])
        for label, score, iou in (("chandelier", .29, .9), ("chandelier", .35, .79),
                                  ("lamp table", .35, .9), ("shelf railing", .35, .9)):
            item = {"label": label, "score": score, "predicted_iou": iou}
            self.assertTrue(texture_support([(item, mask)], None, xy,
                                            foreground_fixture_guard=True).all())

    def test_rendered_bake_removes_foreground_bulb_from_floor(self):
        camera = {"index": 0, "origin": [0, -2000, -4000], "target": [0, 0, 3000],
                  "fov": 240, "roll": 0}
        room = {"cameras": [camera], "boundaries": [{"index": 0, "xmin": -4000,
                "xmax": 4000, "zmin": 0, "zmax": 10000, "type": 4, "flags": 0}]}
        rgb = np.full((240, 320, 3), 80, dtype=np.uint8)
        mask = np.zeros((240, 320), dtype=bool)
        mask[130:180, 130:180] = True
        rgb[mask] = 255
        segments = [[({"label": "wall lamp chandelier", "score": .35,
                       "predicted_iou": .85}, mask)]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.obj"
            generate_mesh(room, source, {"primitives": []})
            for name, enabled in (("off", False), ("on", True)):
                out = root / name
                out.mkdir()
                report = bake_projected(source, room, [Image.fromarray(rgb)], out / "baked.obj",
                                        segments=segments, group_kinds={"floor": "floor"},
                                        foreground_fixture_guard=enabled)
                im = np.asarray(render(out / "baked.obj", camera)[0])
                if enabled:
                    after = im
                    self.assertIn("foreground_fixture_guard", report)
                else:
                    before = im
                    self.assertNotIn("foreground_fixture_guard", report)
            self.assertGreater(int((before.min(-1) > 220).sum()), 100)
            self.assertEqual(int((after.min(-1) > 220).sum()), 0)
            # Shape and visibility are identical: only the appearance changed.
            old_depth = render(root / "off/baked.obj", camera, return_depth=True)[2]
            new_depth = render(root / "on/baked.obj", camera, return_depth=True)[2]
            np.testing.assert_array_equal(old_depth, new_depth)


if __name__ == "__main__":
    unittest.main()
