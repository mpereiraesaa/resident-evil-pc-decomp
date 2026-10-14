"""Scenario variants are grouped by measured layout; status states are validated."""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_rooms import build, group_variants, inventory, load_status, propagate, set_status


def rdt_bytes(cameras, boundaries, salt=b''):
    header = bytearray(0x94)
    header[1] = len(cameras)
    block = b''
    for camera in cameras:
        record = bytearray(44)
        struct.pack_into('<9i', record, 8, *camera['origin'], *camera['target'], camera['roll'], 0, camera['fov'])
        block += record
    struct.pack_into('<I', header, 0x4C, 0x94 + len(block) + len(salt))
    collision = struct.pack('<hh5I', 0, 0, len(boundaries), 0, 0, 0, 0)
    for xmin, zmin, xmax, zmax in boundaries:
        collision += struct.pack('<6H', xmax, zmax, xmin, zmin, 1, 768)
    return bytes(header) + block + salt + collision


CAMERAS = [{'origin': [0, -500, 0], 'target': [0, -500, 2000], 'roll': 0, 'fov': 230}]


class RoomsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        stage = self.root / 'assets' / 'Stage1'
        stage.mkdir(parents=True)
        (stage / 'ROOM1000.RDT').write_bytes(rdt_bytes(CAMERAS, [(0, 0, 100, 100)]))
        (stage / 'ROOM1001.RDT').write_bytes(rdt_bytes(CAMERAS, [(0, 0, 100, 100)], salt=b'script differs'))
        (stage / 'ROOM1010.RDT').write_bytes(rdt_bytes(CAMERAS, [(0, 0, 100, 200)]))
        (stage / 'ROOM1011.RDT').write_bytes(rdt_bytes(CAMERAS, [(0, 0, 100, 300)]))
        (stage / 'ROOM1020.RDT').write_bytes(bytes(4))
        revisit = self.root / 'assets' / 'Stage6'
        revisit.mkdir()
        (revisit / 'ROOM6000.RDT').write_bytes(rdt_bytes(CAMERAS, [(0, 0, 100, 100)], salt=b'revisit'))
        (stage / 'RC1000.pak').write_bytes(b'image')
        (stage / 'RC1010.pak').write_bytes(b'image')

    def test_variants_group_by_layout_and_images(self):
        records = {r['rdt']: r for r in group_variants(inventory(self.root / 'assets'))}
        self.assertEqual(records['ROOM1000']['same_geometry'], ['ROOM1001'])
        self.assertEqual(records['ROOM1001']['same_backgrounds'], ['ROOM1000'])
        self.assertEqual(records['ROOM1001']['same_rdt'], [])
        self.assertEqual(records['ROOM1010']['same_geometry'], [])
        self.assertEqual(records['ROOM1010']['variant_group'], '1-01')
        self.assertTrue(records['ROOM1020']['empty_slot'])
        self.assertEqual(records['ROOM6000']['same_geometry'], [])
        self.assertEqual(records['ROOM6000']['shared_layout_with'], ['ROOM1000', 'ROOM1001'])
        self.assertEqual(records['ROOM1000']['shared_layout_with'], ['ROOM1001', 'ROOM6000'])
        self.assertEqual(records['ROOM1000']['backgrounds'], {'0': records['ROOM1010']['backgrounds']['0']})

    def test_status_is_validated_and_catalog_reports_counts(self):
        status = self.root / 'status.json'
        entry = set_status(status, 'room1000', review='needs_work', note='door too wide')
        self.assertEqual(entry['review'], 'needs_work')
        self.assertEqual(entry['gameplay'], 'not_validated')
        with self.assertRaises(ValueError):
            set_status(status, 'ROOM1000', review='approved')
        with self.assertRaises(ValueError):
            set_status(status, 'ROOM9000', gameplay='validated')
        set_status(status, 'ROOM1010', review='visually_approved', gameplay='validated')
        broken = self.root / 'broken.json'
        broken.write_text(json.dumps({'schema': 1, 'rooms': {'ROOM1000': {'review': 'done'}}}))
        with self.assertRaises(ValueError):
            load_status(broken)
        overrides = self.root / 'overrides'
        overrides.mkdir()
        (overrides / 'ROOM1000.json').write_text(json.dumps({'history': [{}, {}], 'created': 'now',
                                                             'appearance': {'mode': 'reprojected'}}))
        state = build(self.root / 'assets', self.root / 'catalog', overrides, status)
        self.assertEqual(state['counts']['rdt_files'], 6)
        self.assertEqual(state['counts']['empty_slots'], 1)
        self.assertEqual(state['counts']['unique_geometry'], 3)
        self.assertEqual(state['counts']['variants_with_same_geometry'], 2)
        self.assertEqual(state['counts']['variants_with_same_geometry_and_backgrounds'], 2)
        self.assertEqual(state['counts']['overrides'], 3)
        self.assertEqual(state['counts']['unique_layout_and_backgrounds'], 3)
        self.assertEqual(state['counts']['review'], {'unreviewed': 3, 'needs_work': 1, 'visually_approved': 1})
        self.assertEqual(state['counts']['gameplay'], {'not_validated': 4, 'validated': 1, 'broken': 0})
        rooms = {r['rdt']: r for r in state['rooms']}
        self.assertEqual(rooms['ROOM1000']['overrides']['steps'], 2)
        self.assertEqual(rooms['ROOM1000']['note'], 'door too wide')
        self.assertEqual(rooms['ROOM6000']['overrides_shared_from'], 'ROOM1000')
        page = (self.root / 'catalog/index.html').read_text()
        self.assertIn('Misma geometría e imágenes que ROOM1001', page)
        self.assertIn('Necesita trabajo', page)
        with self.assertRaises(ValueError):
            build(self.root / 'assets', self.root / 'assets' / 'catalog', None, None)

    def test_review_propagates_only_to_identical_layouts(self):
        status = self.root / 'status.json'
        set_status(status, 'ROOM1000', review='visually_approved', gameplay='validated', note='fine')
        changed = propagate(self.root / 'assets', status)
        self.assertEqual(sorted(changed), ['ROOM1001', 'ROOM6000'])
        self.assertEqual(changed['ROOM6000']['review'], 'visually_approved')
        self.assertEqual(changed['ROOM6000']['gameplay'], 'not_validated')
        self.assertEqual(changed['ROOM6000']['inherited_from'], 'ROOM1000')
        self.assertIn('not reviewed separately', changed['ROOM6000']['note'])
        self.assertEqual(propagate(self.root / 'assets', status), {})
        loaded = load_status(status)['rooms']
        self.assertEqual(loaded.get('ROOM1010', {}).get('review', 'unreviewed'), 'unreviewed')
        set_status(status, 'ROOM1000', review='needs_work')
        self.assertEqual(propagate(self.root / 'assets', status), {})
        self.assertEqual(load_status(status)['rooms']['ROOM1001']['review'], 'visually_approved')
        state = build(self.root / 'assets', self.root / 'catalog', None, status)
        self.assertEqual(state['counts']['inherited_review'], 2)
        self.assertIn('Estado heredado de ROOM1000', (self.root / 'catalog/index.html').read_text())


if __name__ == '__main__':
    unittest.main()
