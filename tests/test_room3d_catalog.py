"""An exported room is distinct from a baseline, pending room or changed artifact."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from batch_room3d import render_visibility
from room3d_catalog import classify, detail_failures
from room3d_checkpoint import save


class CatalogTests(unittest.TestCase):
    def test_rejections_keep_category_and_multiple_causes(self):
        summary = detail_failures([
            {'kind': 'lamp', 'accepted': True, 'gates': {'supported': False}},
            {'kind': 'lamp', 'accepted': False, 'gates': {'supported': False, 'consistent': True}},
            {'kind': 'staircase', 'accepted': False,
             'gates': {'supported': False, 'plausible_proportions': False}},
            {'kind': 'lamp', 'accepted': False, 'gates': {'supported': None}},
        ])
        self.assertEqual(summary['lamp'], {'rejected': 2, 'failed_gates': {'supported': 1},
                                          'without_failed_gate': 1})
        self.assertEqual(summary['staircase']['rejected'], 1)
        self.assertEqual(summary['staircase']['failed_gates'],
                         {'supported': 1, 'plausible_proportions': 1})

    def test_export_evidence_and_changed_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rdt = root / 'ROOM1000.RDT'
            rdt.write_bytes(b'room input')
            batch = root / 'batch'
            room = batch / 'room1000'
            room.mkdir(parents=True)
            for name in ('assisted.obj', 'survey.json', 'index.html', 'result.json'):
                (room / name).write_text(name)
            result = {'status': 'image_assisted', 'source_sha256': hashlib.sha256(rdt.read_bytes()).hexdigest(),
                      'cameras': [{'index': 0}], 'visual_review': {'status': 'pending'}}
            (batch / 'checkpoints').mkdir()
            checkpoint = batch / 'checkpoints/room1000.json'
            (room / 'details').mkdir()
            (room / 'details/results.json').write_text(json.dumps({'objects': [
                {'accepted': True, 'path': 'procedural_interpretation', 'observations': [{'camera': 0}]},
                {'accepted': False}]}))
            save(checkpoint, 'test', result, room)
            state = classify(rdt, batch)
            self.assertEqual(state['status'], 'exported')
            self.assertEqual(state['creative_details'], 1)
            self.assertEqual(state['single_view_details'], 1)
            self.assertEqual(state['rejected_details'], 1)
            self.assertEqual(state['review_status'], 'pending')
            self.assertEqual(state['render_visibility']['status'], 'not_measured')
            # A valid export can contain an entirely invisible mesh. Keep its
            # artifacts reviewable, but never conflate it with visible geometry.
            result['render_visibility'] = render_visibility([
                {'index': 0, 'assisted_screen_coverage': 0.0}])
            result['depth_scale'] = {'scale': 'estimated_from_camera_target_distance',
                                     'evidence': {'metric_scale_verified': False}}
            save(checkpoint, 'test', result, room)
            state = classify(rdt, batch)
            self.assertEqual(state['status'], 'exported_empty_geometry')
            self.assertFalse(state['depth_scale']['evidence']['metric_scale_verified'])
            # Neither a changed failure report nor a missing texture may leave
            # the export marked complete just because its four main files match.
            details = room / 'details/results.json'
            old_details = details.read_bytes()
            details.write_text('{"objects": []}')
            self.assertEqual(classify(rdt, batch)['status'], 'incomplete_or_changed')
            details.write_bytes(old_details)
            texture = room / 'material.png'
            texture.write_bytes(b'generated texture')
            save(checkpoint, 'test', result, room)
            texture.unlink()
            self.assertEqual(classify(rdt, batch)['status'], 'incomplete_or_changed')
            save(checkpoint, 'test', result, room)
            extra = room / 'unrecorded.obj'
            extra.write_text('unrecorded geometry')
            self.assertEqual(classify(rdt, batch)['status'], 'incomplete_or_changed')
            extra.unlink()
            self.assertEqual(classify(rdt, batch)['status'], 'exported_empty_geometry')
            (room / 'assisted.obj').write_text('modified')
            self.assertEqual(classify(rdt, batch)['status'], 'incomplete_or_changed')

    def test_visibility_distinguishes_missing_partial_and_measured(self):
        self.assertEqual(render_visibility([{'index': 0}])['status'], 'not_measured')
        report = render_visibility([{'index': 0, 'assisted_screen_coverage': .000001},
                                    {'index': 1, 'assisted_screen_coverage': 0.}, {'index': 2}])
        self.assertEqual(report['status'], 'partially_empty')
        self.assertEqual(report['empty_cameras'], [1])
        self.assertEqual(report['missing_cameras'], [2])
        self.assertEqual(render_visibility([{'index': 0, 'assisted_screen_coverage': 0.},
                                            {'index': 1}])['status'], 'partially_empty')

    def test_missing_checkpoint_is_not_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rdt = root / 'ROOM1000.RDT'
            rdt.write_bytes(bytes(4))
            self.assertEqual(classify(rdt, root / 'batch')['status'], 'empty_slot')
            rdt.write_bytes(b'room data')
            self.assertEqual(classify(rdt, root / 'batch')['status'], 'pending')
            (root / 'batch/room1000').mkdir(parents=True)
            self.assertEqual(classify(rdt, root / 'batch')['status'], 'in_progress')


if __name__ == '__main__':
    unittest.main()
