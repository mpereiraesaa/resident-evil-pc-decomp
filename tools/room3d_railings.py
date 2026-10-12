"""Observation-derived handrail alternatives, without room-specific geometry."""

import math

import numpy as np
from room3d_shapes import generate


def handrail_seed(group):
    """Fit one continuous rail to equally sampled training-view point clouds.

    Callers must exclude reserved views. This estimates a visible beam, not
    wall attachment, hidden brackets, collision or semantic object identity.
    """
    if not group or any(o['kind'] != 'railing' for o in group):
        return None
    clouds = []
    for obs in group:
        points = np.asarray(obs['points'], float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('Expected N x 3 railing points')
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 24:
            return None
        clouds.append(points)
    count = min(2048, min(len(p) for p in clouds))
    points = np.concatenate([p[np.linspace(0, len(p)-1, count).astype(int)] for p in clouds])
    center = np.median(points[:, [0, 2]], axis=0)
    xz = points[:, [0, 2]]-center
    values, vectors = np.linalg.eigh(np.cov(xz.T))
    if values[-1] <= 0 or values[0] > .25*values[-1]:
        return None
    direction = vectors[:, -1]
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    along = xz @ direction
    slope, intercept = np.polyfit(along, points[:, 1], 1)
    if abs(slope) < 1e-10:
        slope = 0.
    if slope > 0:
        direction = -direction
        along = -along
        slope = -slope
    lo, hi = np.percentile(along, [5, 95])
    run = float(hi-lo)
    rise = float(-slope*run)
    if run < 500 or rise > 7000:
        return None
    cross = xz @ np.array([-direction[1], direction[0]])
    vertical = points[:, 1]-(intercept+slope*along)
    thickness = float(np.clip(max(np.ptp(np.percentile(cross, [5, 95])),
                                  np.ptp(np.percentile(vertical, [5, 95]))), 25, 120))
    base = center + direction*lo
    height = 900.
    obj = generate('railing', {'profile': 'handrail', 'run': run, 'rise': rise,
                   'height': height, 'thickness': thickness},
                   [float(base[0]), float(intercept+slope*lo+height), float(base[1])],
                   math.atan2(direction[0], direction[1]))
    obj['initialization'] = 'continuous_handrail'
    obj['estimated'] = 'Visible continuous rail hypothesis; hidden supports and wall attachment unverified'
    return obj


def surface_handrail_seed(room, group, mesh):
    """Use visible native wall faces as an alternative to biased model depth.

    This is an existing-geometry design prior, not evidence that native collision
    extrusions faithfully represent the wall. Reserved views must be excluded.
    """
    from room3d_connections import support_mesh
    from room3d_enclosure import extract_wall_planes, fit_wall
    from room3d_preview import camera_basis, render

    _, groups = support_mesh(mesh)
    walls = {g for g in groups if g.startswith('boundary_')}
    if not walls:
        return None
    cameras = {c['index']: c for c in room['cameras']}
    mapped = []
    for obs in group:
        camera = cameras[obs['camera']]
        scene = render(mesh, camera, return_depth=True)[2]
        wall = render(mesh, camera, return_depth=True, only_groups=walls)[2]
        visible = obs['mask'] & np.isfinite(wall) & (wall <= scene+50)
        if visible.sum() < max(24, .8*obs['mask'].sum()):
            return None
        y, x = np.nonzero(visible)
        z = wall[y, x]
        points = np.column_stack(((x-160)*z/camera['fov'], (y-120)*z/camera['fov'], z))
        mapped.append({**obs, 'points': points @ camera_basis(camera)+camera['origin']})
    if not mapped:
        return None
    # Multiple native faces can explain one broad mask; do not bridge them
    # with a rail that floats across the room.
    cloud = np.concatenate([o['points'] for o in mapped])
    try:
        plane = fit_wall(cloud)
    except ValueError:
        return None
    if plane['p90_residual'] > 25:
        distances = [np.linalg.norm(o['points']-cameras[o['camera']]['origin'], axis=1)
                     for o in mapped]
        planes, _ = extract_wall_planes(cloud, float(np.median(np.concatenate(distances))),
                                       min_samples=max(24, int(.1*len(cloud))))
        segments, widths = [], []
        offsets = np.cumsum([0, *[len(o['points']) for o in mapped]])
        for surface in planes:
            indices = np.asarray(surface['indices'])
            members = []
            for i, obs in enumerate(mapped):
                local = indices[(indices >= offsets[i]) & (indices < offsets[i+1])]-offsets[i]
                if len(local) >= 24:
                    members.append({**obs, 'points': obs['points'][local]})
            segment = handrail_seed(members)
            if segment is None:
                continue
            normal = np.asarray(surface['normal'])
            camera_center = np.mean([cameras[o['camera']]['origin'] for o in members], axis=0)[[0, 2]]
            if camera_center @ normal < surface['offset']:
                normal = -normal
            vertices = np.asarray(segment['parts'][0])
            ends = np.stack([vertices[:4].mean(0), vertices[4:].mean(0)])
            width = segment['shape']['parameters']['thickness']
            ends[:, [0, 2]] += normal*width/2
            segments.append(ends.tolist())
            widths.append(width)
        if len(segments) < 2:
            return None
        segments = np.asarray(segments)
        center = segments.reshape(-1, 3).mean(axis=0)
        obj = generate('railing', {'profile': 'handrail_segments',
                                   'segments': (segments-center).tolist(),
                                   'thickness': float(np.median(widths))}, center.tolist())
        obj['initialization'] = 'visible_wall_handrail_segments'
        obj['estimated'] = 'Visible native-wall segment design prior; corner joins, brackets and wall fidelity unverified'
        return obj
    obj = handrail_seed(mapped)
    if obj is None:
        return None
    shape = obj['shape']
    normal = np.array(plane['normal'])
    camera_center = np.mean([cameras[o['camera']]['origin'] for o in group], axis=0)[[0, 2]]
    if camera_center @ normal < plane['offset']:
        normal = -normal
    origin = np.array(shape['origin'])
    origin[[0, 2]] += normal*(shape['parameters']['thickness']/2)
    obj = generate('railing', shape['parameters'], origin.tolist(), shape['yaw'])
    obj['initialization'] = 'visible_wall_handrail'
    obj['estimated'] = 'Existing visible wall design prior; hidden brackets, native wall fidelity and navigation unverified'
    return obj
