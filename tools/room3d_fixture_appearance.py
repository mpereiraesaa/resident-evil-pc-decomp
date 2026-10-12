"""Estimated local bright cores for creative shades, independent of geometry."""

import re

import numpy as np
from PIL import Image

SHADE_MATERIALS = {'fixture_shade', 'retouch_added_shade'}


def baked_fixture_role(material):
    """Recognize only the role names emitted by our fixture appearance bakers."""
    match = re.fullmatch(r'(?:fixture_tint|preserved_fixture)_\d+_(shade|support)', material)
    return None if match is None else match[1]


def core_patch(palette, room, images, segments, projected_areas, exclude_camera=None):
    """Own-mask brightness sets a bounded pattern area; no reserved pixels used."""
    wanted = {(source['camera'], source['mask']) for source in palette['sources']}
    bright_pixels = total_area = 0
    evidence = []
    for index, (camera, image, entries) in enumerate(zip(room['cameras'], images, segments, strict=True)):
        if index == exclude_camera:
            continue
        area = projected_areas.get(camera['index'], 0)
        if not np.isfinite(area) or area < 16:
            continue
        for item, mask in entries:
            if (camera['index'], item['mask']) not in wanted:
                continue
            samples = np.asarray(image.convert('RGB'), float)[mask] / 255
            if len(samples) < 16:
                continue
            luminance = samples @ np.array([.2126, .7152, .0722])
            low, high = np.quantile(luminance, [.25, .85])
            if high - low < .15:
                continue
            count = int(np.count_nonzero(luminance >= high))
            bright_pixels += count
            total_area += area
            evidence.append({'camera': camera['index'], 'mask': item['mask'],
                             'bright_pixels': count, 'projected_shade_pixels': area})
    if not evidence:
        return None
    fraction = float(np.clip(bright_pixels / total_area, .04, .65))
    yy, xx = np.mgrid[:64, :64]
    distance = np.hypot((xx+.5)/64-.5, (yy+.5)/64-.5)
    radius = np.sqrt(fraction / np.pi)
    blend = np.clip((radius - distance) / .04 + .5, 0, 1)[..., None]
    rgb = np.asarray(palette['roles']['support']) * (1-blend) + np.asarray(palette['roles']['shade']) * blend
    texture = Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype('uint8'))
    return texture, {'fraction': fraction, 'sources': evidence,
                     'qualification': 'Estimated centered bright pattern on shade faces; baked own-object palette, not recovered light shape, albedo or emission'}
