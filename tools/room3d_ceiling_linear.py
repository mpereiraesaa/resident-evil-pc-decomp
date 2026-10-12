"""Finite ceiling attachment for estimated elongated light assemblies.

This supplies geometry/contact proposals, not image fitting or acceptance.
Hidden suspension and roof height remain estimates; no ceiling is fabricated.
"""
import copy

import numpy as np
from room3d_connections import point_triangle_distance
from room3d_shapes import generate


def anchor(candidate, prior):
    """Preserve housing/tubes while extending suspension to a ceiling point."""
    shape = candidate['shape']
    if candidate.get('kind') != 'lamp' or shape['parameters'].get('profile') != 'ceiling_linear':
        raise ValueError('Expected a ceiling linear lamp')
    point = np.asarray(prior['point'], float)
    if (prior.get('support') != 'ceiling' or not prior.get('group')
            or point.shape != (3,) or not np.isfinite(point).all()):
        raise ValueError('Expected a finite named ceiling anchor')
    origin = np.asarray(shape['origin'], float)
    if not np.allclose(point[[0, 2]], origin[[0, 2]], atol=1e-8, rtol=0):
        raise ValueError('Ceiling attachment cannot translate the visible body')
    params = {**shape['parameters'], 'height': float(origin[1] - point[1])}
    result = generate('lamp', params, origin, shape['yaw'])
    for key in ('evidence', 'interpretation'):
        if key in candidate:
            result[key] = copy.deepcopy(candidate[key])
    result['attachment_prior'] = copy.deepcopy(prior)
    return result


def contact(obj, triangles, groups, tolerance=10):
    """All four mounting corners and their center must touch finite ceiling."""
    prior = obj.get('attachment_prior', {})
    if (obj.get('kind') != 'lamp' or prior.get('support') != 'ceiling'
            or obj.get('shape', {}).get('parameters', {}).get('profile') != 'ceiling_linear'
            or not np.isfinite(tolerance) or tolerance < 0):
        return False
    surfaces = [np.asarray(t, float)[:, :3] for t, g in zip(triangles, groups, strict=True)
                if g == prior.get('group') and np.ptp(np.asarray(t)[:, 1]) <= 1e-6]
    if not surfaces:
        return False
    # Prism order: [0,1,4,5] is the upper Y face, including after yaw rotation.
    corners = np.asarray(obj['parts'][0], float)[[0, 1, 4, 5]]
    return all(min(point_triangle_distance(p, t) for t in surfaces) <= tolerance
               for p in [*corners, corners.mean(axis=0)])


def proposals(reference, triangles, groups, ceiling_groups, maximum_extension=2000):
    """Try the nearest finite ceiling per group; retain visible geometry exactly."""
    if reference.get('shape', {}).get('parameters', {}).get('profile') != 'ceiling_linear':
        return
    if not np.isfinite(maximum_extension) or maximum_extension <= 0:
        raise ValueError('Expected a positive finite extension bound')
    origin = np.asarray(reference['shape']['origin'], float)
    old_y = origin[1] - reference['shape']['parameters']['height']
    surfaces = []
    for triangle, group in zip(triangles, groups, strict=True):
        xyz = np.asarray(triangle, float)[:, :3]
        if group not in ceiling_groups or np.ptp(xyz[:, 1]) > 1e-6:
            continue
        point = origin.copy()
        point[1] = xyz[0, 1]
        if (0 <= old_y - point[1] <= maximum_extension
                and point_triangle_distance(point, xyz) <= 1e-6):
            surfaces.append((point[1], group, point))
    emitted = set()
    for _, group, point in sorted(surfaces, key=lambda item: -item[0]):
        if group in emitted:
            continue
        prior = {'support': 'ceiling', 'group': group, 'point': point.tolist(),
                 'qualification': 'Estimated finite ceiling attachment; mechanical support and hidden suspension unverified'}
        candidate = anchor(reference, prior)
        if contact(candidate, triangles, groups):
            emitted.add(group)
            yield candidate
