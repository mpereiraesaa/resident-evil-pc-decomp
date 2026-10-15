"""Room readings: inheritance, evidence helpers, variant install and new kinds."""

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from room3d_evidence import floor_points, project  # noqa: E402
from room3d_reading import Builder, load_reading  # noqa: E402
from room3d_readings_batch import install  # noqa: E402

ASSETS = next((p / 'assets' / 'USA' for p in (ROOT, *ROOT.parents) if (p / 'assets' / 'USA' / 'Stage1').is_dir()), None)

CAMERA = {'index': 0, 'origin': [4000, -4000, 2000], 'target': [6000, 0, 9000], 'roll': 0, 'fov': 230}


class LoadReadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        (self.dir / 'ROOMBASE.toml').write_text(
            'room = "ROOMBASE"\nseed = 1\nbackdrop = [2]\n'
            '[shell]\nfloor = 0\nceiling = -9000\ndado = 1200\n'
            '[materials]\nplaster = [0.5, 0.5, 0.5]\nwood = [0.2, 0.1, 0.1]\n'
            '[[zone]]\nbox = [0, 0, 1000, 1000]\nceiling = -8000\n'
            '[[door]]\nwall = "west"\nspan = [100, 900]\ntop = -3300\n'
            '[[sconce]]\nwall = "north"\nat = [-3000, 500]\n'
            '[[light]]\nat = [500, -4000, 500]\n'
            '[[painting]]\nwall = "east"\nspan = [100, 400]\ny = [-3000, -2000]\n'
            '[[object]]\nkind = "rail"\nfrom = [0, 0]\nto = [100, 0]\ny = -6000\n'
            '[[object]]\nkind = "sofa"\nbox = [0, 0, 10, 10]\nback = "west"\n'
            '[[hide]]\ncamera = 0\ngroups = ["walls"]\n'
        )

    def test_base_is_shifted_and_extended(self):
        path = self.dir / 'ROOMUP.toml'
        path.write_text(
            'room = "ROOMUP"\nbase = { reading = "ROOMBASE", y = 6000 }\n'
            '[materials]\nplaster = [0.9, 0.9, 0.8]\n'
            '[[door]]\nwall = "east"\nspan = [100, 900]\nbottom = 0\ntop = -3300\n'
        )
        r = load_reading(path)
        self.assertEqual(r['room'], 'ROOMUP')
        self.assertEqual(r['shell']['floor'], 6000)
        self.assertEqual(r['shell']['ceiling'], -3000)
        self.assertEqual(r['shell']['dado'], 1200)            # a height, not moved
        self.assertEqual(r['zone'][0]['ceiling'], -2000)
        self.assertEqual(r['door'][0]['top'], 2700)
        self.assertEqual(r['door'][1], {'wall': 'east', 'span': [100, 900], 'bottom': 0, 'top': -3300})
        self.assertEqual(r['sconce'][0]['at'], [3000, 500])  # [y, along]
        self.assertEqual(r['light'][0]['at'], [500, 2000, 500])  # [x, y, z]
        self.assertEqual(r['painting'][0]['y'], [3000, 4000])
        self.assertEqual(r['object'][0]['y'], 0)
        self.assertEqual(r['object'][1]['back'], 'west')     # a side name stays
        self.assertEqual(r['materials']['plaster'], [0.9, 0.9, 0.8])
        self.assertEqual(r['materials']['wood'], [0.2, 0.1, 0.1])
        self.assertNotIn('hide', r)
        self.assertNotIn('backdrop', r)

    def test_plain_reading_is_unchanged(self):
        r = load_reading(self.dir / 'ROOMBASE.toml')
        self.assertEqual(r['shell']['floor'], 0)
        self.assertEqual(r['backdrop'], [2])
        self.assertEqual(len(r['hide']), 1)


class EvidenceTests(unittest.TestCase):
    def test_floor_point_projects_back_to_its_pixel(self):
        for point in ((5000, 0, 6000), (7000, 0, 8000), (4500, 0, 4000)):
            pixel = project(CAMERA, point)
            self.assertIsNotNone(pixel)
            (hit,) = floor_points(CAMERA, [pixel])
            self.assertAlmostEqual(hit[0], point[0], delta=1)
            self.assertAlmostEqual(hit[1], point[2], delta=1)

    def test_points_on_a_raised_plane_and_behind_the_camera(self):
        pixel = project(CAMERA, (5200, -900, 7000))
        (hit,) = floor_points(CAMERA, [pixel], y=-900)
        self.assertAlmostEqual(hit[0], 5200, delta=1)
        self.assertAlmostEqual(hit[1], 7000, delta=1)
        self.assertIsNone(project(CAMERA, (4000, 0, -5000)))
        (above,) = floor_points(CAMERA, [(160, -400)])          # far above the horizon
        self.assertIsNone(above)


class InstallTests(unittest.TestCase):
    def test_jill_variant_gets_the_same_mesh_and_cameras(self):
        with tempfile.TemporaryDirectory() as tmp:
            built, mods = Path(tmp) / 'built', Path(tmp) / 'mods'
            built.mkdir()
            (built / 'room1040.obj').write_text('mtllib room1040.mtl\nv 0 0 0\n')
            (built / 'room1040.mtl').write_text('newmtl a\nmap_Kd a.png\n')
            (built / 'a.png').write_bytes(b'png')
            (built / 'room1040.cameras').write_text('backdrop 3\n')
            install(built, 'room1040', mods, 1)
            stage = mods / 'Stage1'
            self.assertEqual((stage / 'room1041.obj').read_text(), (stage / 'room1040.obj').read_text())
            self.assertEqual((stage / 'room1041.cameras').read_text(), 'backdrop 3\n')
            self.assertEqual((stage / 'room1040.mtl').read_text(), 'newmtl a\nmap_Kd room1040-a.png\n')
            self.assertTrue((stage / 'room1040-a.png').exists())
            self.assertFalse((stage / 'room1041.mtl').exists())    # shared library


@unittest.skipUnless(ASSETS, 'game assets not present')
class BuilderKindTests(unittest.TestCase):
    def build(self, extra):
        reading = load_reading_text(
            'room = "ROOM4020"\nseed = 1\nbackdrop = [1]\n'
            '[shell]\nfloor = 0\nceiling = -4200\n'
            '[[zone]]\nbox = [1764, 2250, 7074, 8136]\n' + extra)
        builder = Builder(reading, ASSETS / 'Stage4' / 'ROOM4020.RDT')
        mesh = builder.build()
        return builder, mesh

    def test_backdrop_cameras_are_listed(self):
        builder, _ = self.build('')
        self.assertIn('backdrop 1', builder.hides_text().splitlines())

    def test_new_kinds_add_geometry(self):
        _, empty = self.build('')
        for kind in (
            '[[object]]\nkind = "tub"\nbox = [1800, 2300, 3400, 5000]\n',
            '[[object]]\nkind = "toilet"\nat = [2600, 7700]\nback = "north"\n',
            '[[object]]\nkind = "sink"\nwall = "north"\nat = 6300\n',
            '[[object]]\nkind = "bookcase"\nbox = [4000, 7600, 6000, 8136]\nfront = ["south", "west"]\n',
            '[[object]]\nkind = "ladder"\nbox = [6900, 4000, 7074, 5000]\ntop = -4200\n',
            '[[object]]\nkind = "trophy"\nwall = "west"\nat = [-3000, 5000]\nscale = 1.4\n',
            '[[window]]\nkind = "rect"\nwall = "north"\nat = 4000\nsill = -1200\nheight = 2000\npanes = 3\ntransom = -2600\n',
            '[[object]]\nkind = "table_round"\nat = [4000, 5000]\nradius = 500\ntop = -800\ncloth = "linen"\n',
            '[[object]]\nkind = "plant"\nat = [4000, 5000]\nheight = 1400\n',
            '[[object]]\nkind = "pipe"\nfrom = [2000, -3000, 3000]\nto = [6000, -3000, 3000]\nvia = [[4000, -3000, 6000]]\nradius = 150\n',
            '[[object]]\nkind = "ring"\nat = [4000, 5000]\ninner = 900\nouter = 1200\ntop = -500\nfill = "water"\n',
            '[[object]]\nkind = "spiral_stair"\nat = [4000, 5000]\nradius = 1400\ntop = -3000\nbase = 3000\nsteps = 20\n',
        ):
            with self.subTest(kind=kind.split('\n')[1]):
                _, mesh = self.build(kind)
                self.assertGreater(mesh.triangles, empty.triangles)

    def test_zone_can_turn_the_dado_off(self):
        reading = 'room = "ROOM4020"\nseed = 1\n[shell]\nfloor = 0\nceiling = -4200\ndado = 1200\n'
        a = Builder(load_reading_text(reading + '[[zone]]\nbox = [1764, 2250, 7074, 8136]\n'), ASSETS / 'Stage4' / 'ROOM4020.RDT')
        b = Builder(load_reading_text(reading + '[[zone]]\nbox = [1764, 2250, 7074, 8136]\ndado = 0\n'), ASSETS / 'Stage4' / 'ROOM4020.RDT')
        self.assertIn('usemtl panel', a.build().lines)
        self.assertNotIn('usemtl panel', b.build().lines)

    def test_open_sides_drop_a_zone_wall(self):
        reading = 'room = "ROOM4020"\nseed = 1\n[shell]\nfloor = 0\nceiling = -4200\nskirting = 0\ncornice = 0\n'
        walls = Builder(load_reading_text(reading + '[[zone]]\nbox = [1764, 2250, 7074, 8136]\n'), ASSETS / 'Stage4' / 'ROOM4020.RDT').build()
        open_ = Builder(load_reading_text(reading + '[[zone]]\nbox = [1764, 2250, 7074, 8136]\nopen_sides = ["west", "north"]\n'), ASSETS / 'Stage4' / 'ROOM4020.RDT').build()
        self.assertLess(open_.triangles, walls.triangles)

    def test_solid_variants_take_their_own_tint(self):
        builder, _ = self.build('[materials]\n"iron.tread" = [0.5, 0.4, 0.3]\n'
                                '[[object]]\nkind = "box"\nbox = [2000, 3000, 2500, 3500]\ntop = -300\nmaterial = "iron.tread"\n')
        self.assertEqual(builder.palette.spec['iron.tread']['kd'], (0.5, 0.4, 0.3))
        self.assertNotEqual(builder.palette.spec['iron']['kd'], (0.5, 0.4, 0.3))

    def test_door_can_start_above_the_floor(self):
        _, mesh = self.build('[[door]]\nwall = "north"\nspan = [3000, 4000]\nbottom = -1000\ntop = -3000\n')
        verts, material, door_y = [], None, []
        for line in mesh.lines:
            if line.startswith('v '):
                verts.append(float(line.split()[2]))
            elif line.startswith('usemtl '):
                material = line.split()[1]
            elif line.startswith('f ') and material == 'door':
                door_y += [verts[int(ref.split('/')[0]) - 1] for ref in line.split()[1:]]
        self.assertEqual((min(door_y), max(door_y)), (-3000, -1000))


def load_reading_text(text):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'ROOMTEST.toml'
        path.write_text(text)
        return load_reading(path)


class TubeTests(unittest.TestCase):
    def test_cylinder_and_tube_sides_face_outward(self):
        from room3d_handmade import Mesh
        for build, centre, axis in (
            (lambda m: m.cylinder(0, 0, 100, -200, 0, 'steel', segments=8, caps=False), (0, -100, 0), 1),
            (lambda m: m.tube((0, -500, 0), (1000, -500, 0), 100, 'steel', segments=8, caps=False), (500, -500, 0), 0),
        ):
            seen = []
            mesh = Mesh('t', {'steel': {'kd': (0.5, 0.5, 0.5)}}, lambda p, n: seen.append((p, n)) or (1, 1, 1))
            build(mesh)
            for p, n in seen:
                radial = [p[k] - centre[k] for k in range(3)]
                radial[axis] = 0
                self.assertGreater(sum(radial[k] * n[k] for k in range(3)), 0)

    def test_tube_vertices_lie_on_the_cylinder(self):
        from room3d_handmade import Mesh
        mesh = Mesh('t', {'steel': {'kd': (0.5, 0.5, 0.5)}}, lambda p, n: (1, 1, 1))
        mesh.tube((0, 0, 0), (1000, 0, 0), 100, 'steel', segments=8)
        verts = [tuple(map(float, line.split()[1:4])) for line in mesh.lines if line.startswith('v ')]
        self.assertTrue(verts)
        for x, y, z in verts:
            self.assertIn(round(x), (0, 1000))
            self.assertAlmostEqual((y * y + z * z) ** 0.5, 100, delta=1)


if __name__ == '__main__':
    unittest.main()
