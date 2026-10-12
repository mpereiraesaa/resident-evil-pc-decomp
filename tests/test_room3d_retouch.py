"""Retouch must isolate shared vertices and preserve appearance coordinates."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from room3d_retouch import catalog, local_file, read_mesh, transforms, write_mesh


class RetouchTests(unittest.TestCase):
    def test_shared_vertices_do_not_move_other_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.obj"
            target = Path(folder) / "target.obj"
            contents = ("v 0 0 0 1 .5 .3\nv 2 0 0\nv 0 0 2\n"
                        "vt .1 .2\nvt .3 .4\nvt .5 .6\n"
                        "g moved\nusemtl texture\nf 1/1 2/2 3/3\n"
                        "g fixed\nf -3/-3 -2/-2 -1/-1\n")
            source.write_text(contents)
            original = read_mesh(source)
            write_mesh(original, {"moved": {"translate": [5, -10, 0]}}, target)
            edited = read_mesh(target)
            for face_index, offset in ((0, [5, -10, 0]), (1, [0, 0, 0])):
                for old_ref, new_ref in zip(original[2][face_index][2], edited[2][face_index][2], strict=True):
                    np.testing.assert_allclose(edited[0][new_ref[0]][:3],
                                               np.array(original[0][old_ref[0]][:3]) + offset)
                    self.assertEqual(original[1][old_ref[1]], edited[1][new_ref[1]])
            self.assertEqual(source.read_text(), contents)
            self.assertEqual(edited[0][0][3:], [1, .5, .3])
            write_mesh(original, {"moved": {"hide": True}}, target)
            self.assertEqual([face[0] for face in read_mesh(target)[2]], ["fixed"])

    def test_rotation_scale_and_invalid_edits(self):
        mesh = ([[0, 0, 0], [2, 0, 0], [0, 0, 2]], [],
                [("lamp", "", [(0, None), (1, None), (2, None)])], [])
        groups = catalog(mesh)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "edit.obj"
            write_mesh(mesh, {"lamp": {"pivot": [0, 0, 0], "scale": [2, 1, 1], "yaw": 90}}, target)
            np.testing.assert_allclose(read_mesh(target)[0][1], [0, 0, 4], atol=1e-10)
        for edit in ({"missing": {}}, {"lamp": {"scale": [-1, 1, 1]}},
                     {"lamp": {"translate": [float("nan"), 0, 0]}},
                     {"lamp": {"yaw": float("inf")}}, {"lamp": {"typo": 1}}):
            with self.assertRaises(ValueError):
                transforms(edit, groups)

    def test_material_dependencies_cannot_escape_room(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            room = root / "room"
            room.mkdir()
            (root / "outside.png").write_bytes(b"outside")
            with self.assertRaises(ValueError):
                local_file(room, "../outside.png")


if __name__ == "__main__":
    unittest.main()
