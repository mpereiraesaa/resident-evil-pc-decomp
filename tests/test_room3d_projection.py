"""Background projection: face splitting, camera choice and the .proj sidecar."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_projection import (View, assign_polygon, clip_polygon, polygon_area,  # noqa: E402
                               write_proj)

SETTINGS = dict(min_density=0.000015, depth_slack=0.03, min_facing=0.2, max_depth=6,
                min_piece=150.0 ** 2)


def camera(index=0, origin=(0, -1500, -6000), target=(0, -1500, 0), fov=300):
    return {'index': index, 'origin': list(origin), 'target': list(target), 'roll': 0, 'fov': fov}


def view(cam, depth=None, hides=()):
    if depth is None:
        depth = np.full((240, 320), np.inf)
    return View(cam, depth, set(hides), 1.0)


def wall(x0, x1, y0=0.0, y1=-3000.0, z=0.0):
    """A quad in the plane z (x, y, z, r, g, b, u, v per corner)."""
    return np.array([[x0, y0, z, 1, 1, 1, 0, 0], [x1, y0, z, 1, 1, 1, 1, 0],
                     [x1, y1, z, 1, 1, 1, 1, 1], [x0, y1, z, 1, 1, 1, 0, 1]], dtype=float)


def covered_area(pieces, camera_index):
    return sum(polygon_area(p[:, :3]) for p, c in pieces if c == camera_index)


class ClipPolygonTests(unittest.TestCase):
    def test_half_plane_keeps_area_and_interpolates_attributes(self):
        square = wall(0, 1000, 0, 1000)
        half = clip_polygon(square, np.array([1.0, 0, 0]), -500.0)
        self.assertAlmostEqual(polygon_area(half[:, :3]), 500_000.0)
        self.assertTrue(np.all(half[:, 0] >= 500.0 - 1e-9))
        # u runs with x across the square, so the cut edge sits at u = 0.5.
        on_cut = half[np.isclose(half[:, 0], 500.0)]
        np.testing.assert_allclose(on_cut[:, 6], 0.5)

    def test_fully_outside_is_empty_and_fully_inside_unchanged(self):
        square = wall(0, 1000, 0, 1000)
        self.assertEqual(len(clip_polygon(square, np.array([1.0, 0, 0]), -2000.0)), 0)
        np.testing.assert_array_equal(clip_polygon(square, np.array([1.0, 0, 0]), 10.0), square)


class AssignTests(unittest.TestCase):
    def test_face_inside_the_frame_takes_that_camera_whole(self):
        pieces = assign_polygon(wall(-1000, 1000), frozenset(['wall_0']), [view(camera())], SETTINGS)
        self.assertEqual([c for _, c in pieces], [0])

    def test_face_crossing_the_frame_edge_is_split_there(self):
        # At z = 6000 the 300-unit focal length sees x up to 159/300*6000 = 3180.
        face = wall(-1000, 8000)
        pieces = assign_polygon(face, frozenset(['wall_0']), [view(camera())], SETTINGS)
        inside = covered_area(pieces, 0)
        self.assertGreater(inside, 0.0)
        self.assertAlmostEqual(inside + covered_area(pieces, -1), polygon_area(face[:, :3]), delta=1.0)
        for piece, cam in pieces:
            if cam == 0:
                self.assertTrue(np.all(piece[:, 0] <= 3180.0 + 1.0))
        self.assertGreater(covered_area(pieces, -1), 0.0)

    def test_edge_on_face_keeps_the_procedural_material(self):
        side = np.array([[500, 0, 1000, 1, 1, 1, 0, 0], [500, 0, 3000, 1, 1, 1, 1, 0],
                         [500, -3000, 3000, 1, 1, 1, 1, 1], [500, -3000, 1000, 1, 1, 1, 0, 1]],
                        dtype=float)
        straight_down_the_wall = camera(origin=(500, -1500, -6000), target=(500, -1500, 0))
        pieces = assign_polygon(side, frozenset(['wall_1']), [view(straight_down_the_wall)], SETTINGS)
        self.assertTrue(all(c == -1 for _, c in pieces))

    def test_occluded_face_is_not_painted_and_hidden_groups_are_skipped(self):
        blocked = np.full((240, 320), 3000.0)  # something 3000 units out covers the frame
        pieces = assign_polygon(wall(-1000, 1000), frozenset(['wall_0']), [view(camera(), blocked)],
                                SETTINGS)
        self.assertEqual(covered_area(pieces, 0), 0.0)
        pieces = assign_polygon(wall(-1000, 1000), frozenset(['wall_0']),
                                [view(camera(), hides=['wall_0'])], SETTINGS)
        self.assertEqual([c for _, c in pieces], [-1])

    def test_closer_camera_wins(self):
        # Both see the whole panel; the nearer one gives it more texels.
        near = camera(index=1, origin=(0, -1500, -3000))
        panel = wall(-500, 500, -1000, -2000)
        pieces = assign_polygon(panel, frozenset(['wall_0']), [view(camera()), view(near)], SETTINGS)
        self.assertEqual([c for _, c in pieces], [1])


class SidecarTests(unittest.TestCase):
    def test_proj_lists_one_character_per_face_then_tints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'room1000.proj'
            write_proj(path, np.array([0, -1, 10, 31]), {'wallpaper': np.array([0.5, 0.25, 1.0])})
            self.assertEqual(path.read_text().splitlines(),
                             ['proj1 4', '0-av', 'tint wallpaper 0.500 0.250 1.000'])


if __name__ == '__main__':
    unittest.main()
