"""Observed local ladder tint, distinct from geometry and wall appearance.

The uniform material is estimated from source pixels; it is not recovered
albedo, texture detail, metallic reflectance or lighting.
"""
import hashlib
import re

import numpy as np
from room3d_preview import render
from room3d_segment import texture_support
from room3d_texture_ownership import owned_masks


def baked_ladder_material(name):
    return re.fullmatch(r'(?:ladder_tint|preserved_ladder)_\d+', name) is not None


def palettes(mesh, room, images, segments, observations, group_kinds,
             scene_depths, exclude_camera=None):
    result = {}
    yy, xx = np.indices((240, 320))
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    for group, sources in (observations or {}).items():
        if (group_kinds or {}).get(group) != 'ladder':
            continue
        masks, _ = owned_masks(room, segments, {group:sources}, group_kinds, exclude_camera)
        medians, evidence = [], []
        for index, (camera, image, owned, scene) in enumerate(
                zip(room['cameras'], images, masks, scene_depths, strict=True)):
            if index == exclude_camera or not owned.any():
                continue
            part = render(mesh, camera, return_depth=True, only_groups={group})[2]
            selected = owned & np.isfinite(part) & (part <= scene+.001)
            # A filled staircase mask can overlap a lamp. Only use the existing
            # high-confidence foreground fixture veto, not wall-mask ownership.
            safe = texture_support(segments[index], None, xy,
                                   foreground_fixture_guard=True).reshape(240, 320)
            selected &= safe
            if selected.sum() < 16:
                continue
            pixels = np.asarray(image.convert('RGB'))
            if pixels.shape != (240, 320, 3):
                raise ValueError('Expected native RGB source image')
            rgb = np.median(pixels[selected].astype(float)/255, axis=0)
            medians.append(rgb)
            _, confidence = owned_masks({'cameras':[camera]}, [segments[index]],
                {group:sources}, group_kinds)
            evidence.append({'camera':camera['index'],'matched_visible_pixels':int(selected.sum()),
                             'sample_mask_sha256':hashlib.sha256(selected.tobytes()).hexdigest(),
                             'source_confidence':confidence,'rgb_median':rgb.tolist()})
        if medians:
            result[group] = {'rgb':np.median(medians, axis=0).tolist(), 'sources':evidence,
                             'aggregation':'equal-view median of own visible source-pixel medians',
                             'qualification':'Estimated uniform baked tint; hidden appearance, albedo, metallic reflectance and lighting unverified'}
    return result
