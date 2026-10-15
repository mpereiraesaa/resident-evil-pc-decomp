"""Experimental continuous wall/attachment visual recipes; collision data unchanged."""
import copy

import numpy as np


def attachment_recipe(room, recipe, plane, candidate, height):
    """Emit a candidate in either footprint axis and protrusion direction.

    The entire attachment must fall inside the observed lateral interval. Vertical
    continuation is explicitly estimated. This generator does not select or accept
    its own result, establish object identity, or modify original collision records.
    """
    if candidate.get('normal_axis') not in ('x', 'z') or candidate.get('protrusion_side') not in (-1, 1):
        raise ValueError('Unsupported attachment orientation')
    if not np.isfinite(height) or height <= 0:
        raise ValueError('Expected positive finite attachment height')
    axis = 0 if candidate['normal_axis'] == 'x' else 2
    tangent = 2 if axis == 0 else 0
    n_axis, n_tangent = axis // 2, tangent // 2
    normal = np.asarray(plane['normal'], float)
    if normal.shape != (2,) or not np.isfinite(normal).all() or abs(normal[n_axis]) < .98:
        raise ValueError('Plane and footprint axis disagree')
    offset = float(plane['offset'])
    patch = np.asarray(plane['points'], float)
    if patch.shape != (4, 3) or not np.isfinite(patch).all() or not np.isfinite(offset):
        raise ValueError('Invalid plane patch')
    low, high = float(patch[:, tangent].min()), float(patch[:, tangent].max())
    if high <= low:
        raise ValueError('Empty lateral interval')
    side = candidate['protrusion_side']
    backing_id, attachment_id = candidate['backing'], candidate['attachment']
    by_id = {boundary['index']: boundary for boundary in room['boundaries']}
    if backing_id == attachment_id or backing_id not in by_id or attachment_id not in by_id:
        raise ValueError('Invalid boundary pair')
    obstacle = by_id[attachment_id]
    t_prefix, a_prefix = ('z', 'x') if axis == 0 else ('x', 'z')
    t_low, t_high = obstacle[t_prefix + 'min'], obstacle[t_prefix + 'max']
    if t_low < low or t_high > high or t_high <= t_low:
        raise ValueError('Attachment extends outside observed lateral interval')
    floor = float(recipe.get('floor_y', 0))
    group = f'boundary_{backing_id}'
    backing_surfaces = [s for s in recipe.get('structural_surfaces', []) if s['group'] == group]
    if not backing_surfaces:
        raise ValueError('No inferred backing surfaces')
    ceiling = min(float(p[1]) for surface in backing_surfaces for p in surface['points'])
    if ceiling >= floor or height > floor - ceiling:
        raise ValueError('Attachment exceeds visual vertical extent')

    def position(t):
        return float((offset - normal[n_tangent] * t) / normal[n_axis])

    front = float(obstacle[a_prefix + ('max' if side > 0 else 'min')])
    if any(side * (front - position(t)) <= 0 for t in (t_low, t_high)):
        raise ValueError('Attachment does not protrude from the plane')

    result = copy.deepcopy(recipe)
    surfaces = []
    changed = 0
    for surface in recipe.get('structural_surfaces', []):
        if surface['group'] == f'boundary_{attachment_id}':
            continue
        points = np.asarray(surface['points'], float)
        lo, hi = points[:, tangent].min(), points[:, tangent].max()
        eligible = (surface['group'] == group and np.ptp(points[:, axis]) < 1e-6
                    and side * (points[:, axis].mean() - position((low + high) / 2)) >= 0
                    and min(hi, high) > max(lo, low))
        if not eligible:
            surfaces.append(copy.deepcopy(surface))
            continue
        changed += 1
        for left, right in [(lo, min(hi, low)), (max(lo, high), hi)]:
            if right <= left:
                continue
            revised = points.copy()
            revised[:, tangent] = np.where(np.abs(points[:, tangent] - lo) < 1e-6, left, right)
            surfaces.append({**copy.deepcopy(surface), 'points': revised.tolist()})
    if not changed:
        raise ValueError('No overlapping front backing surface')

    def point(a, y, t):
        p = [0., float(y), 0.]
        p[axis], p[tangent] = float(a), float(t)
        return p

    surfaces.append({'points': [point(position(low), floor, low), point(position(low), ceiling, low),
                                point(position(high), ceiling, high), point(position(high), floor, high)],
                     'material': 'wall', 'group': f'attachment_backing_{backing_id}_{attachment_id}'})
    # Back corners follow the fitted plane exactly, including its small tilt.
    depth = front - position((t_low + t_high) / 2)
    bottom = [point(position(t_low), floor, t_low), point(position(t_low) + depth, floor, t_low),
              point(position(t_high) + depth, floor, t_high), point(position(t_high), floor, t_high)]
    top = [[p[0], floor - height, p[2]] for p in bottom]
    faces = [bottom, top[::-1]] + [[bottom[i], top[i], top[(i + 1) % 4], bottom[(i + 1) % 4]] for i in range(4)]
    for face in faces:
        surfaces.append({'points': face, 'material': 'wall', 'group': f'boundary_{attachment_id}'})
    result['structural_surfaces'] = surfaces
    result['skip_boundaries'] = sorted(set(recipe.get('skip_boundaries', [])) | {attachment_id})
    result['attachment_proposal'] = {'backing': backing_id, 'attachment': attachment_id,
                                      'height': float(height), 'changed_backing_patches': changed,
                                      'qualification': 'estimated_vertical_extent_and_attached_volume'}
    return result


def propose_attachments(room, recipe, cloud):
    """Generate audited alternatives; never replace the caller's active recipe."""
    from room3d_enclosure import extract_wall_planes, identify_wall_attachment
    from room3d_reconstruct import unique_correspondences

    matches = unique_correspondences(cloud['points'])
    report = {'unique_matches': len(matches), 'planes': [], 'status': 'insufficient_matches',
              'qualification': 'Unselected experimental alternatives; not validated geometry'}
    if len(matches) < 6:
        return [], report
    points = np.array([match['xyz'] for match in matches])
    distances = [np.linalg.norm(np.array(match['xyz']) - room['cameras'][view]['origin'])
                 for match in matches for view in match['views']]
    planes, _ = extract_wall_planes(points, float(np.median(distances)), min_samples=6)
    variants = []
    for plane_index, plane in enumerate(planes):
        records = []
        for index in plane['indices']:
            match = matches[index]
            target = np.array(match['xyz'])
            for view in match['views']:
                origin = np.array(room['cameras'][view]['origin'])
                ray = target - origin
                hits = []
                for surface in recipe.get('structural_surfaces', []):
                    a, b, _c, d = np.asarray(surface['points'], float)
                    u, v = b - a, d - a
                    normal = np.cross(u, v)
                    denominator = normal @ ray
                    if abs(denominator) < 1e-8:
                        continue
                    fraction = float(normal @ (a - origin) / denominator)
                    if not 0 < fraction < .98:
                        continue
                    delta = origin + ray * fraction - a
                    q, r = delta @ u / (u @ u), delta @ v / (v @ v)
                    if 0 <= q <= 1 and 0 <= r <= 1:
                        hits.append({'group': surface['group']})
                records.append({'camera': view, 'target': target.tolist(), 'occluders': hits})
        selection = identify_wall_attachment(room['boundaries'], plane, records)
        entry = {'plane': plane, 'selection': selection, 'trials': [], 'source_records': records}
        report['planes'].append(entry)
        if selection['status'] != 'unique_hypothesis':
            continue
        candidate = selection['candidates'][0]
        base_height = float(recipe.get('obstacle_height', 1500))
        for factor in (2 / 3, 1, 4 / 3):
            height = base_height * factor
            name = f'plane{plane_index}_height{height:g}'
            try:
                proposal = attachment_recipe(room, recipe, plane, candidate, height)
            except ValueError as error:
                entry['trials'].append({'name': name, 'status': 'rejected', 'reason': str(error)})
                continue
            variants.append({'name': name, 'recipe': proposal})
            entry['trials'].append({'name': name, 'status': 'unselected_proposal'})
    report['status'] = 'proposals_ready_for_evaluation' if variants else 'no_eligible_proposal'
    return variants, report
