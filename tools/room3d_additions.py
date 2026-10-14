"""Insert editable parametric objects into an exported mesh without inference."""

import re

import numpy as np
from room3d_shapes import generate

LIBRARY = 'retouch-additions.mtl'
MATERIAL = 'retouch_added_neutral'
ROLE_MATERIALS = {'support': 'retouch_added_support', 'shade': 'retouch_added_shade'}


def material_library_text():
    """Canonical neutral materials can be shared by successive insertion steps."""
    return ''.join(f'newmtl {name}\nKd .55 .55 .55\n'
                   for name in (MATERIAL, *ROLE_MATERIALS.values()))


def append_objects(mesh, additions):
    """Return a new mesh plus explicit manual/estimated evidence; keep source intact."""
    if not isinstance(additions, list) or len(additions) > 32:
        raise ValueError('Additions must be a list of at most 32 objects')
    vertices, uvs, faces, libraries = mesh
    vertices, faces, libraries = list(vertices), list(faces), list(libraries)
    groups = {face[0] for face in faces}
    generated = []
    for spec in additions:
        if (not isinstance(spec, dict)
                or set(spec) - {'name', 'kind', 'parameters', 'origin', 'yaw'}
                or not {'name', 'kind', 'parameters', 'origin'} <= spec.keys()):
            raise ValueError('Object requires name, kind, parameters and origin; yaw is optional')
        name = spec['name']
        if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]{0,39}', name):
            raise ValueError('Object name must be a short alphanumeric identifier')
        group = 'added_' + name
        if group in groups:
            raise ValueError('Duplicate object group: ' + group)
        if spec['kind'] not in ('staircase', 'railing', 'lamp', 'column', 'block'):
            raise ValueError('Unsupported parametric object family')
        if not isinstance(spec['parameters'], dict):
            raise TypeError('Object parameters must be a mapping')
        try:
            item = generate(spec['kind'], spec['parameters'], spec['origin'], spec.get('yaw', 0))
        except KeyError as exc:
            raise ValueError(f'Missing shape parameter: {exc.args[0]}') from exc
        roles = item.get('part_roles')
        if spec['kind'] == 'lamp' and spec['parameters'].get('profile') in ('wall_sconce', 'table_lamp'):
            roles = ['shade' if i == 2 else 'support' for i in range(len(item['parts']))]
        for part_index, part in enumerate(item['parts']):
            material = ROLE_MATERIALS[roles[part_index]] if roles else MATERIAL
            start = len(vertices)
            vertices.extend(part)
            for quad in ((0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1),
                         (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)):
                faces.append((group, material, [(start + i, None) for i in quad]))
        groups.add(group)
        generated.append({**item, 'group': group, 'origin_of_geometry': 'manual_parametric_addition',
                          'fidelity': 'unverified', 'navigation': 'unchanged_not_validated'})
    if additions and LIBRARY not in libraries:
        libraries.append(LIBRARY)
    return (vertices, list(uvs), faces, libraries), generated


def associate_fixture_masks(mesh, room, segments, groups, minimum_iou=.35, margin=.1):
    """Associate confident masks to visible added fixtures; ambiguous matches abstain.

    Rendered depth includes the entire room, so hidden object pixels cannot select
    a mask. A source mask must also have an unambiguous owner among the objects.
    These are approximate geometric associations, not recognition ground truth.
    """
    from room3d_preview import render
    from room3d_segment import matches_material

    result = {group: [] for group in groups}
    if segments is None or not groups:
        return result
    if not 0 <= minimum_iou <= 1 or not 0 <= margin <= 1:
        raise ValueError('Invalid association thresholds')
    for camera, entries in zip(room['cameras'], segments, strict=True):
        _, _, scene_depth = render(mesh, camera, return_depth=True)
        eligible = [(item, mask) for item, mask in entries
                    if item.get('score', 0) >= .3 and item.get('predicted_iou', 0) >= .7
                    and matches_material(item['label'], ('lamp', 'chandelier'))
                    and np.count_nonzero(mask) >= 16]
        if not eligible:
            continue
        scores = np.zeros((len(groups), len(eligible)))
        for row, group in enumerate(groups):
            _, visible, depth = render(mesh, camera, return_depth=True, only_groups={group})
            visible &= depth <= scene_depth + .001
            if visible.sum() < 16:
                continue
            for col, (_, mask) in enumerate(eligible):
                union = np.count_nonzero(visible | mask)
                scores[row, col] = np.count_nonzero(visible & mask) / max(1, union)
        for row, group in enumerate(groups):
            col = int(scores[row].argmax())
            best = scores[row, col]
            competing_masks = np.delete(scores[row], col)
            competing_objects = np.delete(scores[:, col], row)
            if (best < minimum_iou or any(values.size and best - values.max() < margin
                                          for values in (competing_masks, competing_objects))):
                continue
            result[group].append({'camera': camera['index'], 'mask': eligible[col][0]['mask'],
                                  'association_iou': float(best),
                                  'qualification': 'Unique visible silhouette match; approximate association'})
    return result
