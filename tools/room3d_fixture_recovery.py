"""Offline focused fan proposals; multiview grouping is not geometry acceptance."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from room3d_checkpoint import inventory
from room3d_refine import load_depth
from room3d_segment import load_masks
from room3d_shape_fit import associate, observations

LABELS = ['ceiling fan']


def fixture_trigger(segments):
    """A wide upper-image lamp hypothesis triggers a room-wide category query."""
    triggers = []
    for camera_position, entries in enumerate(segments):
        for item, mask in entries:
            if (not any(word in item['label'] for word in ('lamp', 'chandelier', 'ceiling fan'))
                    or not np.isfinite([item['score'], item['predicted_iou']]).all()
                    or item['score'] < .3 or item['predicted_iou'] < .8 or mask.sum() < 150):
                continue
            yy, xx = np.nonzero(mask)
            width, height = np.ptp(xx)+1, np.ptp(yy)+1
            if width >= 80 and width >= 3*height and np.median(yy) <= mask.shape[0]*.45:
                triggers.append({'camera_position': camera_position, 'mask': item['mask'],
                                 'width': int(width), 'height': int(height)})
    return triggers


def shared_fan_proposals(room, proposals, depths):
    """Keep only bounded exact-category masks that associate in >=2 cameras."""
    filtered, rejected = [], []
    for camera, entries in zip(room['cameras'], proposals, strict=True):
        eligible = []
        for item, mask in entries:
            reason = None
            if item['label'].strip().casefold() != 'ceiling fan':
                reason = 'different_category'
            elif (not np.isfinite([item['score'], item['predicted_iou']]).all()
                  or item['score'] < .3 or item['predicted_iou'] < .8):
                reason = 'weak_or_nonfinite_confidence'
            elif not 150 <= mask.sum() <= mask.size*.3:
                reason = 'small_or_frame_filling_mask'
            if reason:
                rejected.append({'camera': camera['index'], 'mask': item['mask'], 'reason': reason})
            else:
                eligible.append((item, mask))
        filtered.append(eligible)
    groups = associate(observations(room, filtered, depths))
    selected = set()
    shared = []
    for group in groups:
        members = [{'camera': o['camera'], 'mask': o['mask_name']} for o in group]
        if len({o['camera'] for o in group}) >= 2:
            selected.update((o['camera'], name) for o in group for name in o['mask_names'])
            shared.append(members)
        else:
            rejected.extend({**member, 'reason': 'no_shared_multiview_group'} for member in members)
    return selected, shared, rejected


def superseded_masks(existing, recovered):
    """Map contained lamp-only hypotheses to exactly one focused fan proposal.

    Coverage links observations, not script-owned geometry. Ambiguous overlaps
    remain untouched. The growth bound avoids treating a huge mask as identity.
    """
    result = {item['mask']: [] for item, _ in recovered}
    for old, old_mask in existing:
        if (not any(word in old['label'] for word in ('lamp', 'chandelier', 'ceiling fan'))
                or not np.isfinite([old['score'], old['predicted_iou']]).all()
                or old['score'] < .3 or old['predicted_iou'] < .8 or old_mask.sum() < 24):
            continue
        matches = []
        for new, new_mask in recovered:
            area = int(old_mask.sum())
            if (new_mask.sum() <= 10*area
                    and np.count_nonzero(old_mask & new_mask) / area >= .85):
                matches.append(new['mask'])
        if len(matches) == 1:
            result[matches[0]].append(old['mask'])
    return result


def run(survey, images_root, segmentation, depth_root, out, proposals_root=None):
    if out.exists():
        raise FileExistsError(out)
    roots = [survey.parent, images_root, segmentation, depth_root]
    if proposals_root is not None:
        roots.append(proposals_root)
    if any(out.resolve().is_relative_to(p.resolve()) for p in roots):
        raise ValueError('Keep recovered proposals outside input folders')
    room = json.loads(survey.read_text())
    images = [Image.open(images_root/f"camera{c['index']}-original.png").convert('RGB')
              for c in room['cameras']]
    segments = load_masks(segmentation, room, images)
    depths = load_depth(depth_root, survey, room, images)
    inputs = {'segmentation': inventory(segmentation), 'depth': inventory(depth_root)}
    if proposals_root is not None:
        inputs['proposals'] = inventory(proposals_root)
    triggers = fixture_trigger(segments)
    for trigger in triggers:
        trigger['camera'] = room['cameras'][trigger['camera_position']]['index']
    out.mkdir(parents=True)
    merged = out/'segmentation'
    shutil.copytree(segmentation, merged)
    shared, rejected, reports = [], [], []
    if triggers:
        if proposals_root is None:
            from room3d_segment import run as segment
            from room3d_vision import run as detect

            image_paths = [images_root/f"camera{c['index']}-original.png" for c in room['cameras']]
            detect(image_paths, out/'detections', labels=LABELS)
            segment(image_paths, out/'detections', out/'proposals')
            proposals_root = out/'proposals'
        proposals = load_masks(proposals_root, room, images)
        selected, shared, rejected = shared_fan_proposals(room, proposals, depths)
        for camera, existing, candidates in zip(room['cameras'], segments, proposals, strict=True):
            index = camera['index']
            kept = [(item, mask) for item, mask in candidates if (index, item['mask']) in selected]
            aliases = superseded_masks(existing, kept)
            removed = {name for names in aliases.values() for name in names}
            path = merged/f'camera{index}-original.json'
            record = json.loads(path.read_text())
            prior = [item for item in record['segments'] if item['mask'] in removed]
            # Keep original observations for coarse geometry and rollback.
            # Only the detail fitter shadows their aliases while testing a fan.
            additions = []
            for number, (item, mask) in enumerate(kept):
                name = f'camera{index}-original-ceiling-fan-recovery{number:02d}.png'
                if (merged/name).exists():
                    raise ValueError('Recovery mask already exists')
                Image.fromarray(mask.astype('uint8')*255).save(merged/name)
                entry = {**item, 'mask': name, 'mask_pixels': int(mask.sum()),
                         'status': 'unverified_mask_proposal', 'fixture_recovery': {
                             'source_mask': item['mask'], 'source_root': str(proposals_root.resolve()),
                             'superseded_masks': aliases[item['mask']], 'prompt_labels': LABELS,
                             'qualification': 'Shared mask/depth hypothesis; category and geometry unverified',
                         }}
                record['segments'].append(entry)
                additions.append({'mask': name, 'source_mask': item['mask'],
                                  'superseded_masks': aliases[item['mask']],
                                  'score': item['score'], 'predicted_iou': item['predicted_iou']})
            path.write_text(json.dumps(record, indent=2))
            reports.append({'camera': index, 'added': additions, 'superseded_entries': prior})
    load_masks(merged, room, images)
    if (inputs['segmentation'] != inventory(segmentation)
            or inputs['depth'] != inventory(depth_root)
            or ('proposals' in inputs and inputs['proposals'] != inventory(proposals_root))):
        raise ValueError('Input cache changed during fixture recovery')
    report = {'triggers': triggers, 'shared_groups': shared, 'rejected': rejected,
              'views': reports, 'source_inputs_unchanged': True, 'input_sha256': inputs,
              'prompt_labels': LABELS, 'qualification': 'Automatic proposals only; original caches retained. '
              'No confidence inflation, fitted geometry, reserved-view evaluation or room completion.'}
    (out/'recovery.json').write_text(json.dumps(report, indent=2))
    return report


def appearance_segments(room, original, recovered, recipe):
    """Only fitted geometry may promote recovered semantics into texture baking."""
    used = set()
    for obj in recipe.get('primitives', []):
        evidence = obj.get('evidence', {})
        if 'camera' in evidence and 'mask' in evidence:
            used.add((evidence['camera'], evidence['mask']))
        used.update((o['camera'], o['mask']) for o in evidence.get('shared_observations', []))
    result, selected, discarded = [], [], []
    for camera, old, new in zip(room['cameras'], original, recovered, strict=True):
        additions, aliases = [], set()
        for item, mask in new:
            recovery = item.get('fixture_recovery')
            if not recovery:
                continue
            record = {'camera': camera['index'], 'mask': item['mask']}
            if (camera['index'], item['mask']) in used:
                additions.append((item, mask))
                aliases.update(recovery['superseded_masks'])
                selected.append(record)
            else:
                discarded.append(record)
        result.append([(item, mask) for item, mask in old if item['mask'] not in aliases] + additions)
    return result, {'selected': selected, 'discarded': discarded,
                    'qualification': 'Recovered semantics used only for geometry retained in the final recipe; original observations otherwise preserved'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('survey', 'images', 'segmentation', 'depth', 'out'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--proposals', type=Path, help='Optional image-hash-verified focused mask cache')
    args = parser.parse_args()
    run(args.survey, args.images, args.segmentation, args.depth, args.out, args.proposals)
