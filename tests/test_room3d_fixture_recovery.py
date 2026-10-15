"""Automatic focused fixture recovery preserves inputs and requires shared evidence."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from batch_room3d import generate_mesh
from room3d_checkpoint import inventory
from room3d_fixture_recovery import (
    appearance_segments,
    fixture_trigger,
    run,
    shared_fan_proposals,
    superseded_masks,
)
from room3d_preview import render
from room3d_segment import load_masks
from room3d_shape_fit import observations
from room3d_shapes import generate


class FixtureRecoveryTests(unittest.TestCase):
    def test_trigger_needs_wide_upper_fixture_and_finite_confidence(self):
        mask = np.zeros((240, 320), bool)
        mask[20:35, 60:240] = True
        item = {'label': 'chandelier', 'score': .7, 'predicted_iou': .9, 'mask': 'old.png'}
        self.assertEqual(len(fixture_trigger([[(item, mask)]])), 1)
        self.assertEqual(fixture_trigger([[]]), [])
        for change in ({'label': 'railing'}, {'score': float('nan')}, {'predicted_iou': .7}):
            self.assertEqual(fixture_trigger([[({**item, **change}, mask)]]), [])
        self.assertEqual(fixture_trigger([[(item, np.roll(mask, 150, axis=0))]]), [])

    def test_supersession_is_bounded_unambiguous_and_fixture_only(self):
        old_mask = np.zeros((240, 320), bool)
        old_mask[30:50, 30:50] = True
        item = {'label': 'wall lamp', 'score': .6, 'predicted_iou': .9, 'mask': 'old.png'}
        new = {**item, 'label': 'ceiling fan', 'mask': 'new.png'}
        mapping = superseded_masks([(item, old_mask)], [(new, old_mask)])
        self.assertEqual(mapping, {'new.png': ['old.png']})
        ambiguous = [(new, old_mask), ({**new, 'mask': 'other.png'}, old_mask)]
        self.assertTrue(all(not names for names in superseded_masks([(item, old_mask)], ambiguous).values()))
        for change in ({'label': 'door'}, {'score': float('nan')}, {'predicted_iou': .7}):
            self.assertEqual(superseded_masks([({**item, **change}, old_mask)], [(new, old_mask)]), {'new.png': []})
        self.assertEqual(superseded_masks([(item, old_mask)], [(new, np.ones_like(old_mask))]), {'new.png': []})

    def example(self, root):
        for name in ('images', 'segmentation', 'depth', 'proposals', 'input'):
            (root/name).mkdir()
        room = {'boundaries': [{'index': 0, 'xmin': -2500, 'xmax': 2500,
                               'zmin': -2500, 'zmax': 2500, 'flags': 0, 'type': 4}],
                'cameras': [{'index': i, 'origin': origin, 'target': [0, -800, 0], 'roll': 0, 'fov': 220}
                            for i, origin in zip((4, 7, 9), ([0, -600, -3000], [3000, -800, 0], [-2500, -700, -2000]), strict=True)]}
        survey = root/'input/survey.json'
        survey.write_text(json.dumps(room))
        fan = generate('lamp', {'profile': 'ceiling_fan', 'height': 600, 'radius': 800,
                               'thickness': 15, 'blade_width': 140, 'arms': 4, 'shade': 100}, [0, -1000, 0])
        path = root/'fan.obj'
        generate_mesh(room, path, {'primitives': [fan]})
        image = Image.new('RGB', (320, 240), 'blue')
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        proposals, depths, views = [], [], []
        for position, camera in enumerate(room['cameras']):
            index = camera['index']
            image.save(root/f'images/camera{index}-original.png')
            _, mask, depth = render(path, camera, return_depth=True, only_groups={'primitive_0'})
            observed = np.where(np.isfinite(depth), depth, 10000)
            np.save(root/f'depth/camera{index}-depth.npy', observed)
            np.save(root/f'depth/camera{index}-confidence.npy', np.ones_like(observed))
            depths.append((observed, np.ones_like(mask)))
            item = {'label': 'ceiling fan', 'score': .7, 'predicted_iou': .9, 'mask': f'fan-{index}.png'}
            if position == 2:
                mask = np.ones_like(mask)  # High confidence frame-filling false proposal.
            proposals.append([(item, mask)])
            Image.fromarray(mask.astype('uint8')*255).save(root/'proposals'/item['mask'])
            (root/f'proposals/camera{index}-original.json').write_text(json.dumps({'rgb_sha256': digest, 'segments': [item]}))
            old = {**item, 'label': 'wall lamp chandelier', 'mask': f'old-{index}.png'}
            Image.fromarray(mask.astype('uint8')*255).save(root/'segmentation'/old['mask'])
            original = [old]
            if position == 0:
                trigger = np.zeros_like(mask)
                trigger[20:35, 60:240] = True
                Image.fromarray(trigger.astype('uint8')*255).save(root/'segmentation/trigger.png')
                original.append({**old, 'mask': 'trigger.png'})
            (root/f'segmentation/camera{index}-original.json').write_text(json.dumps({'rgb_sha256': digest, 'segments': original}))
            views.append({'camera': index, 'rgb_sha256': digest})
        (root/'depth/depth.json').write_text(json.dumps({'survey_sha256': hashlib.sha256(survey.read_bytes()).hexdigest(), 'views': views}))
        return room, survey, proposals, depths

    def test_cached_recovery_groups_views_preserves_scores_aliases_and_original_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            room, survey, proposals, depths = self.example(root)
            selected, groups, rejected = shared_fan_proposals(room, proposals, depths)
            self.assertEqual({camera for camera, _ in selected}, {4, 7})
            self.assertEqual(len(groups), 1)
            self.assertEqual(rejected[0]['reason'], 'small_or_frame_filling_mask')
            inputs = {name: inventory(root/name) for name in ('segmentation', 'depth', 'proposals')}
            report = run(survey, root/'images', root/'segmentation', root/'depth', root/'recovered', root/'proposals')
            self.assertTrue(report['source_inputs_unchanged'])
            self.assertEqual(inputs, {name: inventory(root/name) for name in inputs})
            images = [Image.open(root/f"images/camera{c['index']}-original.png").convert('RGB') for c in room['cameras']]
            segments = load_masks(root/'recovered/segmentation', room, images)
            original = load_masks(root/'segmentation', room, images)
            for old_entries, recovered_entries in zip(original, segments, strict=True):
                for item, mask in old_entries:
                    retained = next((d, m) for d, m in recovered_entries if d['mask'] == item['mask'])
                    self.assertEqual(retained[0], item)
                    np.testing.assert_array_equal(retained[1], mask)
            recovered = [item for entries in segments for item, _ in entries if 'fixture_recovery' in item]
            self.assertEqual(len(recovered), 2)
            self.assertTrue(all(item['score'] == .7 and item['predicted_iou'] == .9 for item in recovered))
            self.assertEqual(recovered[0]['fixture_recovery']['superseded_masks'], ['old-4.png'])
            obs = observations(room, segments, depths)
            self.assertTrue(any('old-4.png' in o['mask_names'] and 'recovery' in o['mask_name'] for o in obs))
            self.assertFalse(any(o['mask_name'] == 'old-4.png' for o in obs))
            self.assertTrue((root/'recovered/segmentation/old-4.png').exists())
            self.assertEqual(report['triggers'][0]['camera'], 4)
            # A failed fit must leave original texture semantics intact, even
            # though its proposals formed a plausible shared detector group.
            appearance, decision = appearance_segments(room, original, segments, {'primitives': []})
            self.assertEqual(decision['selected'], [])
            self.assertEqual(len(decision['discarded']), 2)
            for before, after in zip(original, appearance, strict=True):
                self.assertEqual([d for d, _ in before], [d for d, _ in after])
                for (_, a), (_, b) in zip(before, after, strict=True):
                    np.testing.assert_array_equal(a, b)
            sources = [{'camera': camera, 'mask': item['mask']}
                       for camera, entries in zip((4, 7, 9), segments, strict=True)
                       for item, _ in entries if 'fixture_recovery' in item]
            appearance, decision = appearance_segments(room, original, segments, {
                'primitives': [{'kind': 'lamp', 'evidence': {'shared_observations': sources}}]})
            self.assertEqual(len(decision['selected']), 2)
            self.assertEqual(decision['discarded'], [])
            self.assertFalse(any(d['mask'] == 'old-4.png' for d, _ in appearance[0]))
            # Wrong image binding is rejected before inference or cache adoption.
            p = root/'proposals/camera4-original.json'
            record = json.loads(p.read_text()); record['rgb_sha256'] = 'wrong'; p.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                run(survey, root/'images', root/'segmentation', root/'depth', root/'wrong', root/'proposals')

    def test_negative_room_keeps_identical_cache_and_needs_no_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, survey, _, _ = self.example(root)
            for p in (root/'segmentation').glob('*.json'):
                record = json.loads(p.read_text())
                for item in record['segments']:
                    item['label'] = 'wall'
                p.write_text(json.dumps(record))
            before = inventory(root/'segmentation')
            report = run(survey, root/'images', root/'segmentation', root/'depth', root/'negative')
            self.assertEqual(report['triggers'], [])
            self.assertEqual(report['shared_groups'], [])
            self.assertEqual(inventory(root/'negative/segmentation'), before)


if __name__ == '__main__':
    unittest.main()
