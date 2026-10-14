"""Observed baked detail on a retained ceiling panel's visible underside.

Only referenced high-confidence own masks and actual visible underside pixels
may supply detail. Unknown texels retain estimated tint, not recovered artwork.
"""
import hashlib

import numpy as np
from PIL import Image
from room3d_preview import camera_basis, render
from room3d_segment import matches_material

LABELS = ('lamp', 'chandelier', 'ceiling fan')
MATERIALS = {'fixture_shade', 'retouch_added_shade'}


def panel_groups(recipe):
    if not recipe.get('observed_panel_appearance', False):
        return None
    return {f'primitive_{i}' for i, p in enumerate(recipe.get('primitives', []))
            if p.get('kind') == 'lamp' and not p.get('omit_geometry')
            and p.get('shape', {}).get('parameters', {}).get('profile') == 'ceiling_panel'}


def qualified(item):
    scores = [item.get('score', 0), item.get('predicted_iou', 0)]
    return (np.isfinite(scores).all() and all(0 <= s <= 1 for s in scores)
            and scores[0] >= .3 and scores[1] >= .8
            and matches_material(item['label'].strip().casefold(), LABELS))


def textures(mesh, triangles, groups, charts, room, images, segments,
             observations, palettes, wanted_groups, scene_depths,
             exclude_camera=None, minimum_cosine=.25, size=64):
    """Geometry stays fixed; excluded cameras are never sampled or opened."""
    if segments is None:
        return {}
    result = {}
    for group in sorted(wanted_groups & set(palettes)):
        shade = [(tri[:, :3], chart) for (tri, material), g, chart in
                 zip(triangles, groups, charts, strict=True)
                 if g == group and material in MATERIALS]
        if not shade:
            continue
        bottom = max(float(xyz[:, 1].max()) for xyz, _ in shade)
        chart = next((chart for xyz, chart in shade
                      if chart is not None and np.allclose(xyz[:, 1], bottom, atol=1e-6, rtol=0)), None)
        if chart is None:
            continue
        origin, axes, low, scale = chart
        yy, xx = np.indices((size, size))
        local = (np.column_stack((xx.ravel(), yy.ravel()))+.5)/size*scale+low
        world = local @ axes + origin
        rgb = np.broadcast_to(palettes[group]['roles']['shade'], (size*size, 3)).copy()
        best = np.zeros(size*size)
        evidence = []
        wanted = {(s['camera'], name) for s in observations.get(group, [])
                  for name in [s['mask'], *[e['mask'] for e in s.get('confidence_evidence', [])]]}
        for index, (camera, image, entries, scene) in enumerate(
                zip(room['cameras'], images, segments, scene_depths, strict=True)):
            if index == exclude_camera:
                continue
            owned, foreign = np.zeros((240, 320), bool), np.zeros((240, 320), bool)
            confidence = []
            for item, mask in entries:
                if not qualified(item):
                    continue
                if mask.shape != owned.shape or mask.dtype != bool:
                    raise ValueError('Expected native boolean fixture masks')
                if (camera['index'], item['mask']) in wanted:
                    owned |= mask
                    confidence.append({'mask':item['mask'], 'score':item['score'],
                                       'predicted_iou':item['predicted_iou'],
                                       'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest()})
                else:
                    foreign |= mask
            owned &= ~foreign
            if not owned.any():
                continue
            basis = camera_basis(camera)
            view = (world-camera['origin']) @ basis.T
            z = view[:, 2]
            uv = view[:, :2]/np.maximum(z[:, None], 1)*camera['fov']+[160, 120]
            inside = ((z > 2*camera['fov']) & (uv[:, 0] >= 0) & (uv[:, 0] < 320)
                      & (uv[:, 1] >= 0) & (uv[:, 1] < 240))
            xy = np.floor(np.clip(uv, [0, 0], [319, 239])).astype(int)
            part = render(mesh, camera, return_depth=True, only_groups={group},
                          only_materials=MATERIALS)[2]
            pixel_rays = np.column_stack(((xy[:, 0]+.5-160)/camera['fov'],
                                          (xy[:, 1]+.5-120)/camera['fov'], np.ones(len(xy)))) @ basis
            expected = np.divide(bottom-camera['origin'][1], pixel_rays[:, 1],
                                 out=np.zeros(len(xy)), where=abs(pixel_rays[:, 1]) > 1e-8)
            actual = part[xy[:, 1], xy[:, 0]]
            visible = (inside & owned[xy[:, 1], xy[:, 0]] & np.isfinite(actual)
                       & (abs(actual-expected) <= .001)
                       & (actual <= scene[xy[:, 1], xy[:, 0]]+.001))
            rays = np.asarray(camera['origin'])-world
            cosine = abs(rays[:, 1])/np.maximum(np.linalg.norm(rays, axis=1), 1)
            visible &= cosine >= minimum_cosine
            unique = np.unique(xy[visible], axis=0)
            if len(unique) < 16:
                continue
            pixels = np.asarray(image.convert('RGB'))
            if pixels.shape != (240, 320, 3):
                raise ValueError('Expected native source RGB')
            score = cosine/np.maximum(z, 1)**2
            update = visible & (score > best)
            rgb[update] = pixels[xy[update, 1], xy[update, 0]]/255
            best[update] = score[update]
            evidence.append({'camera':camera['index'], 'source_masks':confidence,
                             'unique_visible_source_pixels':len(unique),
                             'sample_coordinates_sha256':hashlib.sha256(unique.tobytes()).hexdigest(),
                             'eligible_texels':int(visible.sum()), 'contributed_texels':int(update.sum())})
        observed = int(np.count_nonzero(best))
        if observed < .05*size*size:
            continue
        texture = Image.fromarray(np.rint(np.clip(rgb.reshape(size, size, 3), 0, 1)*255).astype('uint8'))
        result[group] = (texture, {'face_y':bottom, 'observed_texels':observed,
            'estimated_texels':size*size-observed, 'sources':evidence,
            'qualification':'Own-mask baked detail on visible underside; unknown texels use estimated tint. Hidden faces, original material, lighting and fidelity unverified'})
    return result
