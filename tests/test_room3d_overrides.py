"""Portable overrides replay from a fresh baseline, resolve identity and report conflicts."""

import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from room3d_overrides import (
    Conflict,
    add_step,
    compare,
    initialize,
    object_identities,
    replay,
    replay_for_batch,
    resolve,
    validate,
)
from room3d_retouch import catalog, read_mesh, run

CAMERA = {'index': 0, 'origin': [0, -500, 0], 'target': [0, -500, 2000], 'roll': 0, 'fov': 230}
BOUNDARIES = [{'index': 0, 'xmin': -2000, 'zmin': 3990, 'xmax': 2000, 'zmax': 4010, 'type': 1, 'flags': 768,
               'quadrants': [0], 'record_indices': [0]}]
DOOR = ('door_panel', [500, -2000, 3990], [1500, 0, 4010], 618)
BARREL_A = ('barrel', [-800, -600, 2500], [-300, 0, 3000], None)
BARREL_B = ('barrel', [1500, -600, 2500], [2000, 0, 3000], None)
OBJECTS = [DOOR, BARREL_A, BARREL_B]
LAMP = {'name': 'sconce', 'kind': 'lamp', 'origin': [0, -1500, 3990],
        'parameters': {'profile': 'wall_sconce', 'height': 300, 'radius': 120, 'thickness': 20, 'shade': 140,
                       'arms': 1}}


def box_lines(group, material, lo, hi, textured):
    corners = [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    lines = []
    for quad in quads:
        lines.extend(['g ' + group, 'usemtl ' + material])
        refs = []
        for corner in quad:
            lines.append('v ' + ' '.join(map(str, corners[corner])))
            if textured:
                lines.append('vt .25 .75')
            refs.append('-1/-1' if textured else '-1')
        lines.append('f ' + ' '.join(refs))
    return lines


def make_room(root, objects=OBJECTS, rdt=b'room input', boundaries=BOUNDARIES, name='ROOM1180',
              background=b'background'):
    """A synthetic automatic export with the files the batcher writes."""
    room = Path(root)
    room.mkdir(parents=True, exist_ok=True)
    lines = ['mtllib assisted.mtl']
    lines.extend(box_lines('floor', 'camera0', [-2000, 0, 0], [2000, 1, 4000], True))
    lines.extend(box_lines('boundary_0', 'camera0', [-2000, -3000, 3990], [2000, 0, 4010], True))
    primitives = []
    for i, (kind, lo, hi, offset) in enumerate(objects):
        lines.extend(box_lines(f'primitive_{i}', 'camera0', lo, hi, True))
        item = {'kind': kind, 'box': [lo[0], lo[2], hi[0], hi[2], lo[1], hi[1]]}
        if offset is not None:
            item['source_offset'] = offset
        primitives.append(item)
    (room / 'assisted.obj').write_text('\n'.join(lines) + '\n')
    (room / 'assisted.mtl').write_text('newmtl camera0\nKd 1 1 1\nmap_Kd source-camera0.png\n')
    (room / 'assisted.cameras').write_text('# No camera cutaways\n')
    Image.new('RGB', (320, 240), (120, 80, 60)).save(room / 'camera0-original.png')
    Image.new('RGB', (8, 8), (80, 90, 100)).save(room / 'source-camera0.png')
    survey = {'source': name + '.RDT', 'sha256': hashlib.sha256(rdt).hexdigest(), 'cameras': [CAMERA],
              'boundaries': boundaries, 'quadrant_split': [0, 0], 'coordinate_system': 'X/Z room units; negative Y is up'}
    (room / 'survey.json').write_text(json.dumps(survey))
    (room / 'result.json').write_text(json.dumps({'room': name, 'status': 'image_assisted', 'source_sha256': survey['sha256'],
                                                  'cameras': [{'index': 0, 'background_sha256': hashlib.sha256(background).hexdigest()}]}))
    (room / 'assisted.json').write_text(json.dumps({'fitted_recipe': {'primitives': primitives, 'parametric_details': True}}))
    return room


def fingerprints(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(folder).iterdir() if p.is_file()}


class OverrideTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.room = make_room(self.root / 'auto' / 'room1180')
        self.edits = {'primitive_1': {'translate': [100, 0, 0]}, 'primitive_0': {'scale': [1, 1.2, 1], 'yaw': 5},
                      'primitive_2': {'hide': True}}

    def record(self, additions=None, edits=None):
        overrides = initialize(self.room, note='test room')
        add_step(overrides, self.room, self.edits if edits is None else edits, additions, note='first pass')
        return overrides

    def test_override_file_is_portable_and_identifies_the_room(self):
        overrides = self.record([LAMP])
        text = json.dumps(overrides)
        self.assertNotIn(str(self.root), text)
        self.assertIsNone(re.search(r'/home/|/tmp/', text))
        validate(overrides)
        self.assertEqual(overrides['room'], {'rdt': 'ROOM1180', 'stage': 1, 'room': 0x18, 'player': 0})
        self.assertEqual(overrides['source']['rdt_sha256'], hashlib.sha256(b'room input').hexdigest())
        self.assertEqual(overrides['baseline']['settings'], {'parametric_details': True})
        step = overrides['history'][0]
        self.assertEqual(step['objects']['primitive_0']['source_offset'], 618)
        self.assertEqual(step['objects']['primitive_1']['kind'], 'barrel')
        self.assertEqual(step['additions'], [LAMP])
        with self.assertRaises(ValueError):
            validate({**overrides, 'note': '/home/someone/room'})
        with self.assertRaises(ValueError):
            validate({**overrides, 'schema': 2})

    def test_replay_matches_direct_retouch_and_is_repeatable(self):
        overrides = self.record([LAMP])
        before = fingerprints(self.room)
        direct = run(self.room, self.edits, self.root / 'direct', additions=[LAMP])
        first = replay(overrides, self.room, self.root / 'replay-1')
        second = replay(overrides, self.room, self.root / 'replay-2')
        self.assertEqual(before, fingerprints(self.room))
        self.assertEqual((self.root / 'direct/assisted.obj').read_bytes(), (self.root / 'replay-1/assisted.obj').read_bytes())
        self.assertEqual((self.root / 'replay-1/assisted.obj').read_bytes(), (self.root / 'replay-2/assisted.obj').read_bytes())
        self.assertEqual(first['status'], 'applied')
        self.assertEqual(first['assisted_obj_sha256'], second['assisted_obj_sha256'])
        self.assertEqual(first['source_check']['status'], 'exact')
        self.assertEqual({r['group']: r['method'] for r in first['resolutions']},
                         {'primitive_0': 'script_anchor', 'primitive_1': 'kind_and_bounds', 'primitive_2': 'kind_and_bounds'})
        bounds = catalog(read_mesh(self.root / 'replay-1/assisted.obj'))
        self.assertNotIn('primitive_2', bounds)
        self.assertEqual(bounds['primitive_1']['min'][0], -700)
        self.assertIn('added_sconce', bounds)
        self.assertEqual(direct['additions'][0]['shape']['parameters'], LAMP['parameters'])
        agreement = compare(self.root / 'direct', self.root / 'replay-1')
        self.assertTrue(agreement['identical_geometry'] and agreement['identical_bytes'])
        self.assertEqual(agreement['views'][0]['mean_abs_difference'], 0)
        self.assertTrue((self.root / 'replay-1/index.html').is_file())
        self.assertTrue((self.root / 'replay-1/overrides.json').is_file())

    def test_uvs_are_preserved_and_reprojection_is_deterministic(self):
        overrides = self.record()
        replay(overrides, self.room, self.root / 'uv')
        source, edited = read_mesh(self.room / 'assisted.obj'), read_mesh(self.root / 'uv/assisted.obj')
        self.assertEqual(source[1], edited[1])
        for (group, _, old_refs), (_, _, new_refs) in zip(
                [f for f in source[2] if f[0] != 'primitive_2'], edited[2], strict=True):
            self.assertEqual([source[1][t] for _, t in old_refs], [edited[1][t] for _, t in new_refs])
        overrides['appearance']['mode'] = 'reprojected'
        first = replay(overrides, self.room, self.root / 'bake-1')
        second = replay(overrides, self.room, self.root / 'bake-2')
        self.assertEqual(first['appearance'], 'reprojected')
        self.assertEqual((self.root / 'bake-1/assisted.obj').read_bytes(), (self.root / 'bake-2/assisted.obj').read_bytes())
        self.assertTrue((self.root / 'bake-1/reprojection.json').is_file())
        self.assertEqual(first['assisted_obj_sha256'], second['assisted_obj_sha256'])

    def test_reindexed_and_relocated_objects_resolve_by_identity(self):
        overrides = self.record()
        # A regenerated export orders the objects differently and nudges one barrel.
        moved = ('barrel', [-790, -600, 2510], [-290, 0, 3010], None)
        fresh = make_room(self.root / 'fresh' / 'room1180', [BARREL_B, moved, DOOR])
        report = replay(overrides, fresh, self.root / 'fresh-replay')
        resolved = {r['group']: r for r in report['resolutions']}
        self.assertEqual(resolved['primitive_0']['resolved'], 'primitive_2')
        self.assertEqual(resolved['primitive_1']['resolved'], 'primitive_1')
        self.assertEqual(resolved['primitive_2']['resolved'], 'primitive_0')
        self.assertAlmostEqual(resolved['primitive_1']['bounds_delta'], 10)
        self.assertFalse(report['source_check']['baseline_mesh_match'])
        bounds = catalog(read_mesh(self.root / 'fresh-replay/assisted.obj'))
        self.assertEqual(bounds['primitive_1']['min'][0], -690)
        self.assertNotIn('primitive_0', bounds)
        self.assertAlmostEqual(bounds['primitive_2']['max'][1], 200, places=6)

    def test_conflicts_are_reported_never_guessed(self):
        overrides = self.record()
        missing = make_room(self.root / 'missing' / 'room1180', [DOOR, BARREL_B])
        with self.assertRaises(Conflict) as caught:
            replay(overrides, missing, self.root / 'missing-replay')
        self.assertEqual(caught.exception.record['report']['conflicts'][0]['conflict'], 'missing_object')
        self.assertFalse((self.root / 'missing-replay').exists())
        twin = ('barrel', [-780, -600, 2500], [-280, 0, 3000], None)
        ambiguous = make_room(self.root / 'ambiguous' / 'room1180', [DOOR, BARREL_A, BARREL_B, twin])
        with self.assertRaises(Conflict) as caught:
            replay(overrides, ambiguous, self.root / 'ambiguous-replay')
        self.assertEqual(caught.exception.record['report']['conflicts'][0]['conflict'], 'ambiguous_object')
        other = make_room(self.root / 'other' / 'room1190', [DOOR, BARREL_A, BARREL_B], rdt=b'other room',
                          boundaries=[{**BOUNDARIES[0], 'xmax': 2500}], name='ROOM1190')
        with self.assertRaises(Conflict) as caught:
            replay(overrides, other, self.root / 'other-replay')
        self.assertEqual(caught.exception.record['conflict'], 'source_mismatch')
        partial = replay(overrides, missing, self.root / 'partial', allow_partial=True)
        self.assertEqual(partial['status'], 'partial')
        self.assertEqual([c['group'] for c in partial['conflicts']], ['primitive_1'])
        bounds = catalog(read_mesh(self.root / 'partial/assisted.obj'))
        self.assertNotIn('primitive_1', bounds)
        self.assertAlmostEqual(bounds['primitive_0']['min'][1], -2200, places=6)
        current = object_identities(self.room)
        with self.assertRaises(Conflict):
            resolve('primitive_0', {**current['primitive_0'], 'source_offset': 999}, current)
        with self.assertRaises(Conflict):
            resolve('boundary_0', {**current['boundary_0'], 'max': [2000, 0, 4500]}, current)
        self.assertEqual(resolve('boundary_0', current['boundary_0'], current)[1], 'stable_name')

    def test_scenario_variant_with_same_layout_is_accepted(self):
        overrides = self.record()
        variant = make_room(self.root / 'variant' / 'room1181', rdt=b'room input for the other player',
                            name='ROOM1181', background=b'different lighting')
        report = replay(overrides, variant, self.root / 'variant-replay')
        self.assertEqual(report['source_check']['status'], 'scenario_variant')
        self.assertFalse(report['source_check']['backgrounds_match'])
        self.assertEqual(report['source']['rdt'], 'ROOM1181')
        self.assertEqual(json.loads((self.root / 'variant-replay/replay.json').read_text())['source']['rdt_sha256'],
                         hashlib.sha256(b'room input for the other player').hexdigest())

    def test_steps_compose_from_the_replayed_state_only(self):
        overrides = self.record(edits={'primitive_1': {'translate': [100, 0, 0]}})
        replay(overrides, self.room, self.root / 'state-1')
        with self.assertRaises(ValueError):
            add_step(overrides, self.room, {'primitive_1': {'translate': [50, 0, 0]}})
        add_step(overrides, self.root / 'state-1', {'primitive_1': {'translate': [50, 0, 0]}}, [LAMP], note='second pass')
        self.assertEqual(overrides['history'][1]['objects']['primitive_1']['min'][0], -700)
        self.assertEqual(overrides['history'][1]['base'], 'replayed')
        with self.assertRaises(ValueError):
            add_step(overrides, self.root / 'state-1', {'primitive_1': {'translate': [1, 0, 0]}})
        replay(overrides, self.room, self.root / 'state-2')
        add_step(overrides, self.root / 'state-2', {'added_sconce': {'translate': [0, -10, 0]}})
        final = replay(overrides, self.room, self.root / 'state-3')
        bounds = catalog(read_mesh(self.root / 'state-3/assisted.obj'))
        self.assertEqual(bounds['primitive_1']['min'][0], -650)
        self.assertEqual(final['steps'], 3)
        self.assertEqual([r['method'] for r in final['resolutions'] if r['group'] == 'added_sconce'], ['stable_name'])
        self.assertAlmostEqual(catalog(read_mesh(self.root / 'state-2/assisted.obj'))['added_sconce']['min'][1] - 10,
                               bounds['added_sconce']['min'][1])

    def test_recording_refuses_identities_that_could_not_replay(self):
        twin = ('barrel', [-780, -600, 2500], [-280, 0, 3000], None)
        room = make_room(self.root / 'twins' / 'room1180', [DOOR, BARREL_A, twin])
        overrides = initialize(room)
        with self.assertRaises(Conflict):
            add_step(overrides, room, {'primitive_1': {'translate': [1, 0, 0]}})
        add_step(overrides, room, {'primitive_0': {'translate': [1, 0, 0]}})
        with self.assertRaises(ValueError):
            add_step(overrides, room, {})

    def test_batch_hook_keeps_automatic_and_corrected_outputs_distinct(self):
        batch = self.root / 'auto'
        overrides_dir = self.root / 'overrides'
        overrides_dir.mkdir()
        self.assertEqual(replay_for_batch(overrides_dir, batch, 'room1180')['status'], 'no_overrides')
        (overrides_dir / 'ROOM1180.json').write_text(json.dumps(self.record([LAMP])))
        before = fingerprints(self.room)
        outcome = replay_for_batch(overrides_dir, batch, 'room1180')
        self.assertEqual(outcome['status'], 'applied')
        self.assertEqual(outcome['page'], 'corrected/room1180/index.html')
        self.assertEqual(before, fingerprints(self.room))
        self.assertTrue((batch / 'corrected/room1180/replay.json').is_file())
        self.assertEqual(replay_for_batch(overrides_dir, batch, 'room1180')['status'], 'applied')
        missing = make_room(batch / 'room1181', [DOOR, BARREL_B], name='ROOM1181', rdt=b'variant')
        (overrides_dir / 'ROOM1181.json').write_text((overrides_dir / 'ROOM1180.json').read_text())
        outcome = replay_for_batch(overrides_dir, batch, 'room1181')
        self.assertEqual(outcome['status'], 'conflict')
        self.assertEqual(outcome['conflicts'][0]['conflict'], 'missing_object')
        self.assertFalse((batch / 'corrected/room1181/assisted.obj').exists())
        self.assertEqual(json.loads((batch / 'corrected/room1181/replay.json').read_text())['status'], 'conflict')
        self.assertIsNotNone(missing)


if __name__ == '__main__':
    unittest.main()
