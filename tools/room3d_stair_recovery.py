"""Local focused stair proposals for missing views; geometry gates remain required."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from room3d_checkpoint import inventory
from room3d_refine import load_depth
from room3d_segment import load_masks, verified_boxes
from room3d_stair_edges import STAIR_LABELS, flight_regions

LABELS = ['staircase', 'carpeted staircase', 'railing', 'wooden floor']


def retry_cameras(room, segments):
    """An existing stair hypothesis triggers retries only in missing/weak views."""
    def has_stairs(entries):
        return any(item['label'].strip().casefold() in STAIR_LABELS
                   and item.get('score', 0) >= .3 and item.get('predicted_iou', 0) >= .65
                   and mask.sum() >= 150 for item, mask in entries)
    present = [has_stairs(entries) for entries in segments]
    if not any(present):
        return []
    return [camera['index'] for camera, found in zip(room['cameras'], present, strict=True) if not found]


def select_candidates(image, camera, depth, proposals, existing, region_finder=flight_regions):
    """Keep at most two distinct stair hypotheses with repeated tread/depth evidence."""
    selected, seen = [], [mask for item, mask in existing
                          if item['label'].strip().casefold() in STAIR_LABELS
                          and item.get('score', 0) >= .3 and item.get('predicted_iou', 0) >= .65]
    for item, mask in sorted(proposals, key=lambda p: p[0]['score'], reverse=True):
        if (item['label'].strip().casefold() not in STAIR_LABELS
                or not np.isfinite([item['score'], item['predicted_iou']]).all()
                or item['score'] < .3 or item['predicted_iou'] < .65 or mask.sum() < 150):
            continue
        if any(np.count_nonzero(mask & old) / max(1, np.count_nonzero(mask | old)) >= .8 for old in seen):
            continue
        regions = region_finder(image, mask, depth, camera)
        if not regions:
            continue
        selected.append(({**item, 'stair_recovery': {
            'status': 'unverified_focused_proposal', 'prompt_labels': list(LABELS),
            'tread_regions': [{k: v for k, v in r.items() if k != 'mask'} for r in regions[:2]],
            'qualification': 'Repeated edges and predicted depth, not object identity or fitted geometry',
        }}, mask))
        seen.append(mask)
        if len(selected) == 2:
            break
    return selected


def run(survey, images_root, segmentation, depth_root, out):
    if out.exists():
        raise FileExistsError(out)
    if any(out.resolve().is_relative_to(p.resolve())
           for p in (survey.parent, images_root, segmentation, depth_root)):
        raise ValueError('Keep recovered proposals outside all input folders')
    room = json.loads(survey.read_text())
    images = [Image.open(images_root / f"camera{c['index']}-original.png").convert('RGB') for c in room['cameras']]
    segments = load_masks(segmentation, room, images)
    depths = load_depth(depth_root, survey, room, images)
    retry = retry_cameras(room, segments)
    original_hashes = inventory(segmentation)
    out.mkdir(parents=True)
    merged = out / 'segmentation'
    shutil.copytree(segmentation, merged)
    reports = []
    detector = segmenter = None
    for camera, image, entries, (depth, _) in zip(room['cameras'], images, segments, depths, strict=True):
        if camera['index'] not in retry:
            continue
        if detector is None:
            from room3d_segment import MODEL as SAM_MODEL
            from room3d_segment import REVISION as SAM_REVISION
            from room3d_segment import Segmenter
            from room3d_vision import MODEL as DINO_MODEL
            from room3d_vision import REVISION as DINO_REVISION
            from room3d_vision import Detector

            detector, segmenter = Detector(LABELS), Segmenter()
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        detections = {'rgb_sha256': digest, 'detections': detector.detect(image)}
        masks, scores = segmenter.predict(image, verified_boxes(image, detections))
        proposals = [({**item, 'predicted_iou': float(score)}, mask)
                     for item, mask, score in zip(detections['detections'], masks, scores, strict=True)]
        selected = select_candidates(image, camera, depth, proposals, entries)
        path = merged / f"camera{camera['index']}-original.json"
        record = json.loads(path.read_text())
        for index, (item, mask) in enumerate(selected):
            name = f"camera{camera['index']}-original-recovery{index:02d}.png"
            if (merged / name).exists():
                raise ValueError('Recovery mask name already exists')
            Image.fromarray(mask.astype('uint8') * 255).save(merged / name)
            record['segments'].append({**item, 'mask': name, 'mask_pixels': int(mask.sum()),
                                       'status': 'unverified_mask_proposal'})
        path.write_text(json.dumps(record, indent=2))
        reports.append({'camera': camera['index'], 'rgb_sha256': digest,
                        'depth_sha256': hashlib.sha256(depth.tobytes()).hexdigest(),
                        'detections': detections['detections'], 'added_proposals': len(selected),
                        'detector': [DINO_MODEL, DINO_REVISION], 'segmenter': [SAM_MODEL, SAM_REVISION]})
        print(f"camera{camera['index']}: {len(selected)} focused stair proposals", flush=True)
    if original_hashes != inventory(segmentation):
        raise ValueError('Original segmentation changed during recovery')
    # Round-trip through the same hash/shape verifier consumed by the batcher.
    load_masks(merged, room, images)
    report = {'retry_cameras': retry, 'views': reports, 'source_segmentation_unchanged': True,
              'source_segmentation_sha256': original_hashes, 'prompt_labels': LABELS,
              'qualification': 'Proposals only; original masks retained, no geometry acceptance or room completion'}
    (out / 'recovery.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('survey', 'images', 'segmentation', 'depth', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    run(args.survey, args.images, args.segmentation, args.depth, args.out)
