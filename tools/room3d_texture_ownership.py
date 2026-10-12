"""Keep observed reconstructed objects out of architectural texture sources.

Mask ownership is estimated from retained geometry evidence. It does not recover
the wall behind an object or change geometry, source images or model confidence.
"""
import numpy as np
from room3d_segment import matches_material

ARCHITECTURE = {'wall', 'floor', 'ceiling', 'door_panel'}


def reconstructed_sources(recipe):
    if not recipe.get('reconstructed_object_appearance', False):
        return None
    return {f'primitive_{i}': p.get('evidence', {}).get('shared_observations', [])
            for i, p in enumerate(recipe.get('primitives', []))
            if p.get('kind') == 'ladder' and not p.get('omit_geometry')
            and p.get('interpretation', {}).get('source') == 'shared_vertical_rail_prior'}


def owned_masks(room, segments, observations, group_kinds, exclude_camera=None):
    masks, evidence = [], []
    wanted = {(e['camera'], e['mask'])
              for group, sources in (observations or {}).items()
              if (group_kinds or {}).get(group) == 'ladder' for e in sources}
    for index, camera in enumerate(room['cameras']):
        owned = np.zeros((240, 320), dtype=bool)
        if segments is not None and index != exclude_camera:
            for item, mask in segments[index]:
                scores = [item.get('score', 0), item.get('predicted_iou', 0)]
                if ((camera['index'], item['mask']) not in wanted
                        or not matches_material(item['label'].strip().casefold(), ('staircase', 'ladder'))
                        or not np.isfinite(scores).all() or not all(0 <= s <= 1 for s in scores)
                        or scores[0] < .3 or scores[1] < .8):
                    continue
                if mask.shape != owned.shape or mask.dtype != bool:
                    raise ValueError('Expected native boolean ownership mask')
                owned |= mask
                evidence.append({'camera': camera['index'], 'mask': item['mask'],
                                 'score': scores[0], 'predicted_iou': scores[1],
                                 'pixels': int(mask.sum())})
        masks.append(owned)
    return masks, evidence


def completion_segments(segments, masks):
    """Exclude owned pixels before choosing an inferred architecture patch."""
    if segments is None:
        return None
    return [[(item, mask & ~owned if item['label'].strip().casefold()
              in ('wall', 'floor', 'ceiling', 'door') else mask)
             for item, mask in entries]
            for entries, owned in zip(segments, masks, strict=True)]
