"""Multi-room packaging must preserve textures despite identical input filenames."""

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from package_room3d import room_files
from room3d_preview import render


class PackageTests(unittest.TestCase):
    def test_same_names_different_textures_remain_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            camera = {'index': 0, 'origin': [0,0,0], 'target': [0,0,1000], 'roll': 0, 'fov': 100}
            for name, color in [('room1000', 'red'), ('room1001', 'blue')]:
                source = root / name
                source.mkdir()
                (source / 'assisted.obj').write_text(
                    'mtllib assisted.mtl\ng panel\nusemtl texture\n'
                    'v -100 -100 500\nv 100 -100 500\nv 0 100 500\n'
                    'vt 0 0\nvt 1 0\nvt .5 1\nf 1/1 2/2 3/3\n')
                (source / 'assisted.mtl').write_text('newmtl texture\nKd 1 1 1\nmap_Kd source-camera0.png\n')
                (source / 'assisted.cameras').write_text('hide 1 panel\n')
                Image.new('RGB', (2,2), color).save(source / 'source-camera0.png')
                files, _ = room_files(source, name)
                for path, payload in files.items():
                    target = root / 'package' / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                before, _ = render(source / 'assisted.obj', camera)
                after, _ = render(root / f'package/Stage1/{name}.obj', camera)
                self.assertEqual(before.tobytes(), after.tobytes())
                self.assertEqual((root / f'package/Stage1/{name}.cameras').read_text(), 'hide 1 panel\n')
            red = Image.open(root / 'package/Stage1/room1000/texture000.png').getpixel((0,0))
            blue = Image.open(root / 'package/Stage1/room1001/texture000.png').getpixel((0,0))
            self.assertNotEqual(red, blue)

    def test_missing_dependencies_and_invalid_room_name(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            (source / 'assisted.obj').write_text('mtllib missing.mtl\n')
            with self.assertRaises(ValueError):
                room_files(source, 'room1000')
            with self.assertRaises(ValueError):
                room_files(source, '../room1000')


if __name__ == '__main__':
    unittest.main()
