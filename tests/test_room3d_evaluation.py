"""Evaluation must not turn absent evidence or replaced-fit scores into success."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from evaluate_room3d_details import (
    final_geometry_metrics,
    object_evidence,
    summarize,
    write_report,
)


class EvaluationTests(unittest.TestCase):
    def test_refined_creative_shape_replaces_earlier_interpretation_geometry(self):
        from evaluate_room3d_details import creative_evidence

        refined = {'accepted': True, 'selected_shape': {'parameters': {'shade': 120}},
                   'placement': [{'valid': True}], 'gates': {'raw_depth': True, 'floor_reference': True},
                   'raw_depth': [{'before': .1, 'after': .099}]}
        obj = {'accepted': True, 'creative_part_refinement': refined,
               'interpretation': {'accepted': True, 'trials': [{'shape': {'parameters': {'shade': 300}}}]}}
        self.assertEqual(creative_evidence(obj)['shape'], refined['selected_shape'])
        obj.update(kind='lamp', candidate=[{'silhouette_iou': .99}], interpretation={})
        self.assertIsNone(object_evidence(obj)['minimum_mask_iou'])
        refined['gates']['raw_depth'] = False
        self.assertIsNone(creative_evidence(obj))

    def test_final_measurements_require_current_mesh_and_visibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertIsNone(final_geometry_metrics(root))
            mesh = root/'fitted.obj'
            mesh.write_text('final geometry')
            record = {'schema': 1, 'mesh_sha256': hashlib.sha256(mesh.read_bytes()).hexdigest(),
                      'visibility_sha256': None, 'objects': []}
            (root/'final-details.json').write_text(json.dumps(record))
            self.assertEqual(final_geometry_metrics(root), record)
            mesh.with_suffix('.cameras').write_text('hide 7 primitive_0\n')
            with self.assertRaises(ValueError):
                final_geometry_metrics(root)
            record['visibility_sha256'] = hashlib.sha256(mesh.with_suffix('.cameras').read_bytes()).hexdigest()
            (root/'final-details.json').write_text(json.dumps(record))
            self.assertEqual(final_geometry_metrics(root), record)
            mesh.write_text('modified geometry')
            with self.assertRaises(ValueError):
                final_geometry_metrics(root)

    def test_depth_veto_is_reported_without_reusing_precleanup_measurements(self):
        obj = {'kind': 'lamp', 'accepted': False, 'observations': [{'camera': 1}],
               'depth_reference': {'kind': 'background_without_matched_inferred_fixture_proxies',
                                   'removed_proxy_indices': [0, 1]},
               'floor_depth_guard': {'accepted': False, 'reason': 'reserved_view_regressed',
                                    'training': [{'camera': 1, 'safe': True}],
                                    'heldout': [{'camera': 3, 'safe': False}]}}
        original = copy.deepcopy(obj)
        evidence = object_evidence(obj)
        self.assertEqual(evidence['depth_reference'], obj['depth_reference'])
        self.assertEqual(evidence['floor_depth_guard']['depth_reference'], obj['depth_reference'])
        self.assertEqual(evidence['floor_depth_guard']['heldout'], [{'camera': 3, 'safe': False}])
        changed = object_evidence({**obj, 'visibility_evidence_scope': 'before_visual_proxy_cleanup'})
        self.assertFalse(changed['floor_depth_guard']['accepted'])
        self.assertEqual(changed['floor_depth_guard']['reason'], 'reserved_view_regressed')
        self.assertEqual(changed['floor_depth_guard']['training'], [])
        self.assertEqual(changed['floor_depth_guard']['heldout'], [])
        self.assertEqual(obj, original)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root / 'batch'
            details = batch / 'room10e0/details'
            details.mkdir(parents=True)
            (batch / 'manifest.json').write_text(json.dumps({'results': [
                {'room': 'room10e0', 'status': 'image_assisted'}]}))
            (details / 'results.json').write_text(json.dumps({'accepted': 0, 'objects': [obj]}))
            write_report([batch], root / 'evaluation')
            page = (root / 'evaluation/index.html').read_text()
            self.assertIn('Control de profundidad de la geometría seleccionada', page)
            self.assertIn('reserved_view_regressed', page)

    def test_visibility_before_proxy_cleanup_is_not_a_final_measurement(self):
        obj = {'kind': 'staircase', 'accepted': True,
               'observations': [{'camera': 1}, {'camera': 6}],
               'candidate': [{'silhouette_iou': .9}], 'heldout': [{'silhouette_iou': .8}],
               'visibility_evidence_scope': 'before_visual_proxy_cleanup',
               'interpretation': {'accepted': True, 'selected': 0, 'trials': [{
                   'gates': {'placement': True}, 'shape': {'parameters': {'width': 1000}},
                   'placement': [{'valid': True}], 'headroom': {'clear': True},
                   'depth_before': [.2], 'depth_after': [.1]}]}}
        original = copy.deepcopy(obj)
        result = object_evidence(obj)
        self.assertIsNone(result['minimum_mask_iou'])
        self.assertEqual(result['heldout_scores'], [])
        self.assertEqual(result['creative_geometry']['placement'], [])
        self.assertEqual(result['creative_geometry']['depth_before'], [])
        self.assertEqual(result['creative_geometry']['depth_after'], [])
        self.assertEqual(result['creative_geometry']['shape']['parameters']['width'], 1000)
        self.assertEqual(result['creative_geometry']['visibility_evidence_scope'],
                         'before_visual_proxy_cleanup')
        self.assertEqual(obj, original)

    def test_selected_joint_geometry_reports_its_own_headroom_not_rejected_metrics(self):
        obj = {'kind': 'staircase', 'accepted': True,
               'observations': [{'camera': 1}, {'camera': 6}],
               'candidate': [{'silhouette_iou': .1}],
               'headroom': {'clear': False, 'obstructed_samples': 99},
               'interpretation': {'accepted': True, 'selected': 0, 'trials': [{
                   'shape': {'parameters': {'width': 1800}},
                   'gates': {'placement': True, 'stair_end_connections': True},
                   'placement': [{'valid': True, 'centroid_error': 3}],
                   'connections': {'walkable_valid': True},
                   'headroom': {'clear': False, 'obstructed_samples': 11},
                   'depth_before': [.2, .3], 'depth_after': [.19, .29],
                   'landing_assembly': {'accepted': True, 'selected': 0,
                       'training_cameras': [1, 6], 'heldout_cameras': [7],
                       'heldout': [{'camera': 7, 'valid': True}],
                       'trials': [{'depth': 1200,
                           'gates': {'known_headroom': True, 'walkable_contacts': True},
                           'headroom': {'clear': True, 'obstructed_samples': 0},
                           'contacts': {'walkable_valid': True}}]},
               }]}}
        original = copy.deepcopy(obj)
        result = object_evidence(obj)
        self.assertTrue(result['headroom_clear'])
        self.assertEqual(result['obstructed_samples'], 0)
        self.assertIsNone(result['minimum_mask_iou'])
        self.assertEqual(result['heldout_scores'], [])
        creative = result['creative_geometry']
        self.assertEqual(creative['shape'], {'parameters': {'width': 1800}})
        self.assertEqual(creative['landing']['depth'], 1200)
        self.assertEqual(creative['landing']['heldout_checks'], [{'camera': 7, 'valid': True}])
        self.assertEqual(creative['connections'], {'walkable_valid': True})
        self.assertEqual(obj, original)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root / 'batch'
            details = batch / 'room1060/details'
            details.mkdir(parents=True)
            (batch / 'manifest.json').write_text(json.dumps({'results': [
                {'room': 'room1060', 'status': 'image_assisted'}]}))
            (details / 'results.json').write_text(json.dumps({'accepted': 1, 'objects': [obj]}))
            output = root / 'evaluation'
            reports = write_report([batch], output)
            self.assertEqual(reports[0]['rooms'][0]['accepted_with_unknown_or_failed_headroom'], 0)
            self.assertIn('Geometría creativa seleccionada', (output / 'index.html').read_text())
            self.assertIn('1200', (output / 'index.html').read_text())
            self.assertEqual(reports[0]['rooms'][0]['visual_audit']['status'], 'not_inspected')
        for mutate in ('index', 'boolean_index', 'failed_gate', 'bad_gates', 'landing_rejected', 'object_rejected'):
            with self.subTest(mutate=mutate):
                obj = copy.deepcopy(original)
                selected = obj['interpretation']
                if mutate == 'index':
                    selected['selected'] = 99
                elif mutate == 'boolean_index':
                    selected['selected'] = False
                elif mutate == 'failed_gate':
                    selected['trials'][0]['gates']['placement'] = False
                elif mutate == 'bad_gates':
                    selected['trials'][0]['gates'] = [True]
                elif mutate == 'landing_rejected':
                    selected['trials'][0]['landing_assembly']['accepted'] = False
                else:
                    obj['accepted'] = False
                self.assertIsNone(object_evidence(obj)['creative_geometry'])

    def test_frozen_module_filename_manifest_checks_actual_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'sources/tools/probe_source.py'
            source.parent.mkdir(parents=True)
            source.write_text('original snapshot')
            (root/'manifest.json').write_text(json.dumps({'results': []}))
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            for name in ('probe_source.py', 'tools/probe_source.py'):
                (root/'frozen-config.json').write_text(json.dumps({'source_sha256': {name: expected}}))
                self.assertEqual(summarize(root)['source_execution_snapshot'], 'frozen_sources_verified')
                source.write_text('changed snapshot')
                with self.assertRaisesRegex(ValueError, 'Frozen source differs'):
                    summarize(root)
                source.write_text('original snapshot')

    def test_committed_source_inventory_checks_nonpython_missing_and_extra_files(self):
        from room3d_checkpoint import inventory

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = root/'sources'
            source = sources/'tools/probe.py'
            source.parent.mkdir(parents=True)
            source.write_text('original code')
            support = sources/'tools/template.html'
            support.write_text('original template')
            (root/'manifest.json').write_text(json.dumps({'results': []}))
            (root/'frozen-config.json').write_text(json.dumps({
                'source_inventory': inventory(sources), 'source_git_commit': 'recorded_separately'}))
            self.assertEqual(summarize(root)['source_execution_snapshot'], 'frozen_sources_verified')
            support.write_text('changed template')
            with self.assertRaisesRegex(ValueError, 'Frozen source inventory differs'):
                summarize(root)
            support.write_text('original template')
            source.unlink()
            with self.assertRaisesRegex(ValueError, 'Frozen source inventory differs'):
                summarize(root)
            source.write_text('original code')
            extra = sources/'unrecorded.py'
            extra.write_text('unrecorded')
            with self.assertRaisesRegex(ValueError, 'Frozen source inventory differs'):
                summarize(root)
            extra.unlink()
            self.assertEqual(summarize(root)['source_execution_snapshot'], 'frozen_sources_verified')

    def test_live_source_file_cannot_substitute_for_a_missing_frozen_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'manifest.json').write_text(json.dumps({'results': []}))
            live = root/'live.py'
            live.write_text('current checkout, not an execution snapshot')
            digest = hashlib.sha256(live.read_bytes()).hexdigest()
            for config in ({'source_sha256': {str(live): digest}},
                           {'source_sha256': {'missing.py': digest}},
                           {'source_inventory': {}}, {'command': ['unknown']}):
                (root/'frozen-config.json').write_text(json.dumps(config))
                with self.subTest(config=config), self.assertRaises(ValueError):
                    summarize(root)

    def test_external_visual_review_is_bound_to_complete_export(self):
        from room3d_checkpoint import inventory, save
        from room3d_visual_audit import load_audit, record_audit

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batch = root/'batch'
            room = batch/'room1060'
            (room/'details').mkdir(parents=True)
            (room/'details/results.json').write_text(json.dumps({'objects': [], 'accepted': 0}))
            (room/'survey.json').write_text(json.dumps({'cameras': [{'index': 7}]}))
            (room/'camera7-original.png').write_bytes(b'original')
            (room/'camera7-assisted.png').write_bytes(b'rendered')
            (room/'assisted.mtl').write_text('newmtl fixture\nKd .2 .3 .4\n')
            checkpoint = batch/'checkpoints/room1060.json'
            save(checkpoint, 'key', {'status': 'image_assisted'}, room)
            before = inventory(batch)
            audits = root/'audits'
            record_audit(room, audits, 'needs_further_geometry_work', [7],
                         ['Flight intersects the room.'], 'assistant_visual_review')
            self.assertEqual(inventory(batch), before)
            audit = load_audit(audits, room)
            self.assertEqual(audit['binding'], 'complete_export_verified')
            summary = summarize(batch, completed_only=True, audits=audits)
            self.assertEqual(summary['rooms'][0]['visual_audit']['findings'],
                             ['Flight intersects the room.'])
            output = root/'report'
            write_report([batch], output, completed_only=True, audits=audits)
            self.assertIn('needs_further_geometry_work', (output/'index.html').read_text())
            # A material change invalidates the review even if its rendered PNG is unchanged.
            (room/'assisted.mtl').write_text('newmtl fixture\nKd 1 0 0\n')
            self.assertEqual(load_audit(audits, room)['status'], 'stale_or_invalid_audit')
            save(checkpoint, 'new-key', {'status': 'image_assisted'}, room)
            self.assertEqual(load_audit(audits, room)['status'], 'stale_or_invalid_audit')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                record_audit(room, audits, 'approximate_visual_pass', [7],
                             ['Reviewed only this view.'], 'assistant_visual_review')

    def test_audit_registration_rejects_bad_scope_or_unfinished_export(self):
        from room3d_checkpoint import save
        from room3d_visual_audit import record_audit

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            room = root/'batch/room1060'
            room.mkdir(parents=True)
            (room/'survey.json').write_text(json.dumps({'cameras': [{'index': 1}]}))
            (room/'camera1-original.png').write_bytes(b'original')
            (room/'camera1-assisted.png').write_bytes(b'rendered')
            checkpoint = room.parent/'checkpoints/room1060.json'
            save(checkpoint, 'key', {'status': 'image_assisted'}, room)
            with self.assertRaisesRegex(ValueError, 'outside'):
                record_audit(room, room/'audits', 'approximate_visual_pass', [1], ['Note'], 'reviewer')
            for cameras in ([], [1, 1], [2], [True]):
                with self.subTest(cameras=cameras), self.assertRaises(ValueError):
                    record_audit(room, root/'unused', 'approximate_visual_pass', cameras, ['Note'], 'reviewer')
            self.assertFalse((root/'unused').exists())
            (room/'new-partial.obj').write_text('partial output')
            with self.assertRaisesRegex(ValueError, 'unchanged completed'):
                record_audit(room, root/'unused', 'approximate_visual_pass', [1], ['Note'], 'reviewer')

    def test_malformed_external_audit_cannot_approve_a_room(self):
        from room3d_visual_audit import audit_path, load_audit

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            room = root/'batch/room1060'
            audits = root/'audits'
            audits.mkdir()
            for malformed in ([], {'status': 'approximate_visual_pass'}, 'approved'):
                audit_path(audits, room).write_text(json.dumps(malformed))
                self.assertEqual(load_audit(audits, room)['status'], 'stale_or_invalid_audit')

    def test_live_evaluation_excludes_partial_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            for name in ("room1030", "room1031", "room1040"):
                details = root / name / "details/results.json"
                details.parent.mkdir(parents=True)
                details.write_text(json.dumps({"accepted": 0, "objects": [{
                    "kind": "lamp", "accepted": False, "observations": [{"camera": 0}],
                    "gates": {"supported": False}, "heldout_cameras": []}]}))
                record = {"schema": 1, "key": name, "result": {"status": "image_assisted"},
                          "artifacts": {"details/results.json": hashlib.sha256(details.read_bytes()).hexdigest()}}
                (checkpoints / (name + ".json")).write_text(json.dumps(record))
            (root / "room1040/details/results.json").write_text('{"objects": [], "accepted": 0}')
            (checkpoints / "room1050.json").write_text(json.dumps({
                "schema": 1, "result": {"status": "mesh"}, "artifacts": {}}))
            # No manifest exists while the real batch is still running.
            result = summarize(root, completed_only=True)
            with self.assertRaises(ValueError):
                write_report([root], root / "inside", completed_only=True)
            self.assertFalse((root / "inside").exists())
        self.assertEqual([r["room"] for r in result["rooms"]], ["room1030", "room1031"])
        self.assertEqual(result["variant_groups"], {"room103": ["room1030", "room1031"]})
        self.assertEqual(result["category_totals"]["lamp"]["rejected"], 2)
        self.assertEqual(result["category_totals"]["lamp"]["failed_gates"], {"supported": 2})
        self.assertEqual(len(result["excluded_checkpoints"]), 2)
        self.assertEqual(result["source_execution_snapshot"], "not_recorded")

    def test_accepted_creative_fit_does_not_count_old_failed_gates_as_rejections(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text(json.dumps({"results": [{
                "room": "room1030", "status": "image_assisted"}]}))
            path = root / "room1030/details/results.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"accepted": 1, "objects": [{
                "kind": "lamp", "accepted": True, "interpretation": {"accepted": True},
                "observations": [{"camera": 0}], "gates": {"improving": False}}]}))
            result = summarize(root)
        self.assertEqual(result["rooms"][0]["failed_gates"], {})
        self.assertEqual(result["category_totals"]["lamp"], {
            "accepted": 1, "rejected": 0, "failed_gates": {}})

    def test_empty_room_reports_missing_categories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'manifest.json').write_text(json.dumps({'results': [
                {'room': 'ROOMTEST', 'status': 'image_assisted'}]}))
            details = root / 'roomtest/details'
            details.mkdir(parents=True)
            (details / 'results.json').write_text(json.dumps({'objects': [], 'accepted': 0}))
            self.assertEqual(summarize(root)['rooms'][0]['visual_audit']['status'], 'not_inspected')
            audit_dir = root / 'roomtest/visual-audit'
            audit_dir.mkdir()
            (audit_dir / 'findings.json').write_text(json.dumps({
                'status': 'visual_failure', 'findings': ['Door is missing.']}))
            self.assertIsNone(summarize(root)['rooms'][0]['appearance']['sampled_vertex_fraction'])
            (root / 'roomtest/assisted.json').write_text(json.dumps({'bake': {
                'vertices': 100, 'sampled_vertices': 12, 'inferred_vertices': 0}}))
            room = summarize(root)['rooms'][0]
        self.assertEqual(room['appearance']['sampled_vertex_fraction'], .12)
        self.assertEqual(room['visual_audit']['status'], 'visual_failure')
        self.assertEqual(room['categories_without_fit_observations'],
                         ['lamp', 'railing', 'staircase'])
        self.assertEqual(room['accepted_shared_objects'], 0)

    def test_creative_replacement_does_not_inherit_rejected_fit_metrics(self):
        obj = {'kind': 'staircase', 'accepted': True,
               'observations': [{'camera': 1}, {'camera': 1}],
               'candidate': [{'silhouette_iou': 0.1}],
               'headroom': {'clear': True, 'obstructed_samples': 0},
               'interpretation': {'accepted': True}}
        result = object_evidence(obj)
        self.assertIsNone(result['minimum_mask_iou'])
        self.assertIsNone(result['headroom_clear'])
        self.assertFalse(result['shared_between_cameras'])
        obj['interpretation']['accepted'] = False
        obj['observations'].append({'camera': 2})
        result = object_evidence(obj)
        self.assertTrue(result['shared_between_cameras'])
        self.assertEqual(result['minimum_mask_iou'], 0.1)
        self.assertTrue(result['headroom_clear'])


if __name__ == '__main__':
    unittest.main()
