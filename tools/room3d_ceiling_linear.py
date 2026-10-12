"""Finite ceiling attachment for estimated elongated light assemblies.

This supplies geometry/contact proposals, not image fitting or acceptance.
Hidden suspension and roof height remain estimates; no ceiling is fabricated.
"""
import copy

import numpy as np
from room3d_connections import point_triangle_distance
from room3d_preview import camera_basis
from room3d_shapes import generate


def anchor(candidate, prior):
    """Preserve housing/tubes while extending suspension to a ceiling point."""
    shape = candidate['shape']
    if candidate.get('kind') != 'lamp' or shape['parameters'].get('profile') not in ('ceiling_linear', 'ceiling_panel'):
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
            or obj.get('shape', {}).get('parameters', {}).get('profile') not in ('ceiling_linear', 'ceiling_panel')
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
    if reference.get('shape', {}).get('parameters', {}).get('profile') not in ('ceiling_linear', 'ceiling_panel'):
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


def mask_starts(training, triangles, groups, ceiling_groups):
    """Bounded elongated-light hypotheses from training masks on finite ceilings.

    Tube counts and drop are design alternatives, not observed component counts.
    Camera-clipped extents remain estimated. Callers must reserve observations
    before invoking this function and apply their final scene validation.
    """
    if len(training) < 2:
        return
    ceilings = {}
    for triangle, group in zip(triangles, groups, strict=True):
        xyz = np.asarray(triangle, float)[:, :3]
        if group in ceiling_groups and np.ptp(xyz[:, 1]) <= 1e-6:
            ceilings[group] = max(ceilings.get(group, -np.inf), xyz[0, 1])
    for group, level in sorted(ceilings.items()):
        for drop in (50., 100., 200., 350.):
            clouds = []
            for observation in training:
                camera = observation['camera_model']
                mask = observation['mask']
                if mask.shape != (240, 320) or mask.dtype != bool:
                    raise ValueError('Expected a native boolean fixture mask')
                y, x = np.nonzero(mask)
                if len(x) < 24:
                    break
                take = np.linspace(0, len(x)-1, min(512, len(x))).astype(int)
                rays = np.column_stack(((x[take]+.5-160)/camera['fov'],
                                        (y[take]+.5-120)/camera['fov'], np.ones(len(take)))) @ camera_basis(camera)
                origin = np.asarray(camera['origin'], float)
                z = np.divide(level+drop-origin[1], rays[:, 1],
                              out=np.zeros(len(rays)), where=abs(rays[:, 1]) > 1e-8)
                good = np.isfinite(z) & (z > 2*camera['fov'])
                if good.sum() < .8*len(z):
                    break
                clouds.append((origin+rays[good]*z[good, None])[:, [0, 2]])
            if len(clouds) != len(training):
                continue
            # Equal observation weight, even when masks have different areas.
            center = np.mean([c.mean(0) for c in clouds], axis=0)
            covariance = np.mean([(c-center).T @ (c-center)/len(c) for c in clouds], axis=0)
            _, axes = np.linalg.eigh(covariance)
            along, across = axes[:, 1], axes[:, 0]
            basis = np.column_stack((along, across))
            coordinates = np.concatenate([(c-center) @ basis for c in clouds])
            low, high = np.quantile(coordinates, [.01, .99], axis=0)
            length, width = high-low
            if not 300 <= length <= 5000 or not 50 <= width <= 1000:
                continue
            midpoint = center + ((low+high)/2) @ basis.T
            yaw = float(np.arctan2(-along[1], along[0]))
            designs = [('ceiling_panel', 1)]
            if length >= 3*width:
                designs += [('ceiling_linear', 1), ('ceiling_linear', 2)]
            for profile, tubes in designs:
                shade = (min(10., width*.05) if profile == 'ceiling_panel'
                         else float(width/(1.25*(tubes-1)+.7)*.65))
                thickness = shade if profile == 'ceiling_panel' else min(20., shade*.25)
                params = {'profile': profile, 'height': drop, 'radius': float(length/2),
                          'housing_width': float(width), 'thickness': thickness, 'shade': shade, 'arms': tubes}
                try:
                    obj = generate('lamp', params, [midpoint[0], level+drop, midpoint[1]], yaw)
                except ValueError:
                    continue
                prior = {'support': 'ceiling', 'group': group,
                         'point': [float(midpoint[0]), float(level), float(midpoint[1])],
                         'normal': [0, 1, 0], 'source_cameras': [o['camera'] for o in training],
                         'qualification': 'Estimated training-mask extent on finite ceiling; tube count, drop and clipped/hidden geometry are design hypotheses'}
                obj = anchor(obj, prior)
                if contact(obj, triangles, groups):
                    obj['initialization'] = f'{profile}_{group}_{drop:g}_{tubes}'
                    yield obj
