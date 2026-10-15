"""Conservative cross-category conflicts, not semantic ground truth."""
import numpy as np


def filter_stair_conflicts(room, segments):
    """Abstain on a staircase mask contained by a stronger, reliable door mask.

    Horizontal door panels can mimic treads. Keep uncertain conflicts and every
    other category for the existing geometry/depth checks. Never mutate inputs.
    """
    filtered, rejected = [], []
    for camera, entries in zip(room['cameras'], segments, strict=True):
        kept = []
        for item, mask in entries:
            if 'staircase' not in item['label'] or 'railing' in item['label']:
                kept.append((item, mask))
                continue
            count = np.count_nonzero(mask)
            conflicts = []
            for other, door in entries:
                if (other['label'].strip() != 'door'
                        or not np.isfinite([item['score'], other['score'],
                                            other.get('predicted_iou', 0)]).all()
                        or other['score'] < .5
                        or other.get('predicted_iou', 0) < .9
                        or other['score'] - item['score'] < .15):
                    continue
                if door.shape != mask.shape:
                    raise ValueError('Semantic masks must share image dimensions')
                overlap = np.count_nonzero(mask & door) / max(1, count)
                if count >= 16 and overlap >= .9:
                    conflicts.append((overlap, other))
            if not conflicts:
                kept.append((item, mask))
                continue
            overlap, other = max(conflicts, key=lambda c: (c[0], c[1]['score']))
            rejected.append({
                'camera': camera['index'], 'mask': item['mask'],
                'label': item['label'], 'detector_score': item['score'],
                'conflicting_mask': other['mask'], 'conflicting_label': 'door',
                'conflicting_detector_score': other['score'],
                'contained_fraction': float(overlap), 'reason': 'stronger_door_mask',
                'qualification': 'Ambiguous category abstention; not proof of door identity',
            })
        filtered.append(kept)
    return filtered, rejected
