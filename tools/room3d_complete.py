"""Approximate material patches for unseen surfaces, derived locally from masks.

These are inferred continuations of observed materials, never recovered hidden
art. No original source images are modified.
"""

import cv2
import numpy as np
from PIL import Image
from room3d_segment import matches_material

LABELS = {
    "staircase": ("staircase",),
    "railing": ("railing",),
    "barrel": ("barrel",),
    "lamp": ("lamp", "chandelier", "ceiling fan"),
    "floor": ("floor",),
    "wall": ("wall",),
    "ceiling": ("ceiling",),
    "door_panel": ("door",),
    "chest_body": ("chest", "crate"),
    "typewriter_table": ("table", "typewriter"),
}


def fixture_tints(room, images, segments, observations, exclude_camera=None):
    """Observed median tint for accepted fixtures, without borrowing other masks."""
    result = {}
    if segments is None:
        return result
    for group, sources in (observations or {}).items():
        pixels, evidence, selected_labels = [], [], []
        wanted = {(o['camera'], o['mask']) for o in sources}
        for index, (camera, image, entries) in enumerate(
            zip(room['cameras'], images, segments, strict=True)
        ):
            if index == exclude_camera:
                continue
            for item, mask in entries:
                if ((camera['index'], item['mask']) not in wanted
                        or item.get('predicted_iou', 0) < .7
                        or not matches_material(item['label'], LABELS['lamp'])
                        or np.count_nonzero(mask) < 16):
                    continue
                pixels.append(np.asarray(image.convert('RGB'))[mask])
                selected_labels.append(item['label'].strip().casefold())
                evidence.append({'camera': camera['index'], 'mask': item['mask'],
                                 'detector_score': item['score'],
                                 'predicted_iou': item['predicted_iou']})
        if pixels:
            samples = np.concatenate(pixels).astype(float) / 255
            luminance = samples @ np.array([.2126, .7152, .0722])
            # Fan blades occupy most of the mask; a small central light can
            # account for less than 15% of its pixels. Keep that category's
            # own bright tail rather than assigning black blade tint to it.
            fan_only = all(label == 'ceiling fan' for label in selected_labels)
            quantiles = [.25, .98 if fan_only else .85]
            low, high = np.quantile(luminance, quantiles)
            median = np.median(samples, axis=0).tolist()
            # A procedural shade is an interpretation, not a pixel correspondence.
            # Preserve observed contrast without fabricating emission or borrowing
            # another object's bright pixels. Flat masks retain their original tint.
            roles = {'support': median, 'shade': median}
            if high - low >= .15:
                roles = {
                    'support': np.median(samples[luminance <= low], axis=0).tolist(),
                    'shade': np.median(samples[luminance >= high], axis=0).tolist(),
                }
            result[group] = {'rgb': median, 'roles': roles,
                             'palette_quantiles': quantiles,
                             'sources': evidence,
                             'qualification': 'Observed baked palette; procedural part assignment, not albedo or light emission'}
    return result


def material_patches(images, segments, out, exclude_camera=None):
    result = {}
    if segments is None:
        return result
    for kind, labels in LABELS.items():
        choices = []
        for camera, (image, entries) in enumerate(zip(images, segments, strict=True)):
            if camera == exclude_camera:
                continue
            for item, mask in entries:
                if (
                    item["score"] < 0.4
                    or item["predicted_iou"] < 0.7
                    or not matches_material(item["label"], labels)
                ):
                    continue
                # A square strictly inside the mask prevents borrowing neighboring
                # objects. Distance to the image border is also constrained.
                padded = np.pad(mask.astype(np.uint8), 1)
                distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
                y, x = np.unravel_index(distance.argmax(), distance.shape)
                radius = min(24, int(distance[y, x] / np.sqrt(2)) - 1)
                if radius < 3:
                    continue
                choices.append((radius * item["score"], camera, x, y, radius, image))
        if not choices:
            continue
        _, camera, x, y, radius, image = max(choices, key=lambda c: c[0])
        patch = image.crop((x - radius, y - radius, x + radius, y + radius)).resize(
            (64, 64), Image.Resampling.BILINEAR
        )
        name = f"completion-{kind}-exclude{exclude_camera}.png"
        patch.save(out / name)
        result[kind] = {
            "file": name,
            "camera": camera,
            "source_box": [
                int(x - radius),
                int(y - radius),
                int(x + radius),
                int(y + radius),
            ],
            "mean": (np.asarray(patch, float).mean(axis=(0, 1)) / 255).tolist(),
            "status": "inferred_material_continuation",
        }
    return result
