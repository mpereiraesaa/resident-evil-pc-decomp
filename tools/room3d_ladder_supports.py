"""Finite wall attachment hypotheses for shared vertical ladders.

Rails may be reoriented to a nearby wall; brackets move along the rails within
a bounded allowance. Existing walls and collision data are never extended.
"""
import copy
import math

import numpy as np
from room3d_shapes import beam, generate


def closest_points(point, triangles):
    """Vectorized closest points on finite triangles, including degenerate edges."""
    xyz = np.asarray(triangles, float)[..., :3]
    if xyz.ndim != 3 or xyz.shape[1:] != (3, 3) or not np.isfinite(xyz).all():
        raise ValueError('Expected finite triangles')
    point = np.asarray(point, float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Expected a finite point')
    if not len(xyz):
        return np.empty((0, 3))
    a, b, c = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    u, v = b-a, c-a
    normal = np.cross(u, v)
    area = np.sum(normal*normal, axis=1)
    projected = point-normal*np.divide(np.sum((point-a)*normal, axis=1), area,
                                       out=np.zeros(len(xyz)), where=area>1e-12)[:, None]
    uu, uv, vv = np.sum(u*u, axis=1), np.sum(u*v, axis=1), np.sum(v*v, axis=1)
    pu, pv = np.sum((projected-a)*u, axis=1), np.sum((projected-a)*v, axis=1)
    denominator = uu*vv-uv*uv
    alpha = np.divide(pu*vv-pv*uv, denominator, out=np.zeros(len(xyz)), where=denominator>1e-12)
    beta = np.divide(pv*uu-pu*uv, denominator, out=np.zeros(len(xyz)), where=denominator>1e-12)
    inside = (denominator>1e-12) & (alpha>=-1e-10) & (beta>=-1e-10) & (alpha+beta<=1+1e-10)
    candidates = [projected]
    distances = [np.where(inside, np.sum((projected-point)**2, axis=1), np.inf)]
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end-start
        length = np.sum(edge*edge, axis=1)
        t = np.divide(np.sum((point-start)*edge, axis=1), length,
                       out=np.zeros(len(xyz)), where=length>1e-12)
        closest = start+np.clip(t, 0, 1)[:, None]*edge
        candidates.append(closest)
        distances.append(np.sum((closest-point)**2, axis=1))
    candidates = np.stack(candidates, axis=1)
    return candidates[np.arange(len(xyz)), np.argmin(np.stack(distances, axis=1), axis=1)]


def finite_contact(obj, triangles, groups, tolerance=10):
    prior = obj.get('wall_attachment', {})
    if obj.get('kind') != 'ladder' or not prior or not np.isfinite(tolerance) or tolerance < 0:
        return False
    assigned = prior.get('bracket_groups', [prior.get('group')]*4)
    if len(assigned) != 4:
        return False
    for part, assignment in zip(np.asarray(obj['parts'][-4:]), assigned, strict=True):
        surfaces = [t for t, group in zip(triangles, groups, strict=True) if group==assignment]
        if not surfaces or any(np.linalg.norm(closest_points(point, surfaces)-point, axis=1).min()>tolerance
                               for point in [*part[:4], part[:4].mean(0)]):
            return False
    return True


def anchor_brackets(obj, triangles, groups, wall_groups, maximum_distance=1000):
    """Adapt only wall caps/links; preserve the observed rails and rungs exactly.

    Finite contact alone does not prove mechanical support or absence of
    intersections with provisional wall volumes. Scene gates remain external.
    """
    if obj.get('kind') != 'ladder' or not np.isfinite(maximum_distance) or maximum_distance <= 0:
        raise ValueError('Expected a ladder and positive finite attachment distance')
    allowed = set(wall_groups)
    planes = {}
    for triangle, group in zip(triangles, groups, strict=True):
        if group not in allowed:
            continue
        xyz = np.asarray(triangle, float)[:, :3]
        normal = np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0])
        length = np.linalg.norm(normal)
        if length < 1e-8 or abs(normal[1])/length > 1e-8:
            continue
        normal /= length
        if tuple(normal)<tuple(-normal):
            normal = -normal
        key = (group, *np.round(normal, 6), round(float(xyz[0]@normal), 4))
        planes.setdefault(key, []).append(xyz)
    report = {'status': 'no_finite_bracket_support', 'brackets': [],
              'qualification': 'Estimated finite bracket-cap contact only; provisional wall intersections, original construction, scene depth and navigation unverified.'}
    brackets = []
    for old in np.asarray(obj['parts'][-4:], float):
        start, end = old[:4].mean(0), old[4:].mean(0)
        axis = end-start
        axis /= np.linalg.norm(axis)
        choices = []
        for key, raw_faces in planes.items():
            normal, faces = np.array(key[1:4]), np.asarray(raw_faces)
            if abs(normal@axis)<.5:
                continue
            points = closest_points(start, faces)
            nearest = points[np.linalg.norm(points-start, axis=1).argmin()]
            if np.linalg.norm(nearest-start)>maximum_distance:
                continue
            toward = faces.mean(axis=(0, 1))-nearest
            distance = np.linalg.norm(toward)
            toward = toward/max(distance, 1e-12)
            for inset in (0, obj['shape']['parameters']['thickness'], 2*obj['shape']['parameters']['thickness']):
                anchor = nearest+toward*min(inset, distance)
                length = np.linalg.norm(end-anchor)
                if not 1 <= length <= maximum_distance or np.linalg.norm(anchor-start)>maximum_distance:
                    continue
                if abs((end-anchor)@normal)/length<.15:
                    continue
                part = np.asarray(beam(anchor, end, obj['shape']['parameters']['thickness']/2))
                part[:4] -= (((part[:4]-faces[0, 0])@normal)[:, None]*normal)
                if any(np.linalg.norm(closest_points(point, faces)-point, axis=1).min()>10
                       for point in part[:4]):
                    continue
                volume = abs(np.linalg.det([part[1]-part[0], part[3]-part[0], part[4]-part[0]]))
                if volume <= 1e-6:
                    continue
                choices.append((float(np.linalg.norm(anchor-start)), part, key[0], length))
        if not choices:
            report['failed_bracket'] = len(brackets)
            return obj, report
        displacement, part, group, length = min(choices, key=lambda value:value[0])
        brackets.append(part)
        report['brackets'].append({'group':group,'wall_anchor_displacement':displacement,
                                   'link_length':float(length)})
    shape = obj['shape']
    c, s = math.cos(shape['yaw']), math.sin(shape['yaw'])
    rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    local = (np.asarray(brackets)-shape['origin']) @ rotation.T
    candidate = generate('ladder', {**shape['parameters'], 'bracket_parts':local.tolist()},
                         shape['origin'], shape['yaw'])
    for key in ('initialization', 'ladder_evidence', 'spacing_fit'):
        if key in obj:
            candidate[key] = copy.deepcopy(obj[key])
    candidate['wall_attachment'] = {'bracket_groups':[r['group'] for r in report['brackets']],
                                    'qualification':report['qualification']}
    if not finite_contact(candidate, triangles, groups):
        report['status'] = 'finite_contact_failed'
        return obj, report
    report['status'] = 'anchored'
    return candidate, report


def wall_proposals(obj, triangles, groups, wall_groups, camera_origins,
                   maximum_distance=1000, maximum_planes=4):
    """Generate finite, room-facing alternatives; render/depth gates stay external."""
    if obj.get('kind') != 'ladder':
        raise ValueError('Expected a generated ladder')
    if (not np.isfinite(maximum_distance) or maximum_distance <= 0
            or type(maximum_planes) is not int or maximum_planes < 1):
        raise ValueError('Expected positive attachment distance and plane count')
    origins = np.asarray(camera_origins, float)
    if origins.ndim != 2 or origins.shape[1] != 3 or not len(origins) or not np.isfinite(origins).all():
        raise ValueError('Observed camera origins establish the room-facing side')
    allowed = set(wall_groups)
    planes = {}
    for triangle, group in zip(triangles, groups, strict=True):
        if group not in allowed:
            continue
        xyz = np.asarray(triangle, float)[:, :3]
        normal = np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0])
        length = np.linalg.norm(normal)
        if length < 1e-8 or abs(normal[1])/length > 1e-8:
            continue
        normal /= length
        if tuple(normal) < tuple(-normal):
            normal = -normal
        key = (group, *np.round(normal, 6), round(float(xyz[0]@normal), 4))
        planes.setdefault(key, []).append(xyz)
    old_shape, p = obj['shape'], obj['shape']['parameters']
    center = np.asarray(obj['parts'][:2], float)[:, :4].mean(axis=(0, 1))
    nearest = []
    for key, faces in planes.items():
        points = closest_points(center, faces)
        distances = np.linalg.norm(points-center, axis=1)
        index = int(distances.argmin())
        if distances[index] <= maximum_distance:
            nearest.append((float(distances[index]), key, np.asarray(faces), points[index]))
    for _, key, faces, anchor in sorted(nearest, key=lambda item: item[0])[:maximum_planes]:
        normal = np.array(key[1:4])
        facing = (origins.mean(0)-anchor)@normal
        if abs(facing) < 1e-8:
            continue
        if facing < 0:
            normal = -normal
        tangent = np.array([normal[2], 0, -normal[0]])
        floor_y = old_shape['origin'][1]
        lower, upper = faces[:, :, 1].min()+p['thickness']/2, faces[:, :, 1].max()-p['thickness']/2
        desired = floor_y-p['height']*np.asarray(p.get('bracket_levels', [.1, .9]))
        levels_y = np.clip(desired, lower, upper)
        if (upper <= lower or np.max(abs(levels_y-desired)) > p['height']*.15
                or levels_y[0] <= levels_y[1]):
            continue
        levels = ((floor_y-levels_y)/p['height']).tolist()
        for width_scale in (.9, 1, 1.1):
            for horizontal in (-.15, 0, .15):
                for stand_off_scale in (1, 1.5, 2):
                    point = anchor+tangent*(horizontal*p['width'])
                    point[1] = floor_y
                    parameters = {**p, 'width': p['width']*width_scale,
                                  'stand_off': p['stand_off']*stand_off_scale,
                                  'bracket_levels': levels}
                    parameters.pop('bracket_parts', None)
                    candidate = generate('ladder', parameters, point,
                                         math.atan2(normal[0], normal[2]))
                    candidate['wall_attachment'] = {
                        'group': key[0], 'point': point.tolist(), 'normal': normal.tolist(),
                        'bracket_level_displacement': abs(levels_y-desired).tolist(),
                        'qualification': 'Estimated wall-aligned ladder with bounded bracket shifts on finite existing surfaces; original support design and navigation unverified'}
                    if finite_contact(candidate, faces, [key[0]]*len(faces)):
                        yield candidate
