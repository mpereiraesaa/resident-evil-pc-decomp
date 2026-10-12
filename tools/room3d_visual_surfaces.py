"""Opt-in shared visual surfaces when native collision extrusion is invisible.

Observed depth and semantic masks are estimated evidence. Extended floor area
and wall-to-floor contact are creative completion, never navigation geometry.
"""

import copy

import numpy as np
from room3d_enclosure import extract_wall_planes, reconcile_walls
from room3d_preview import camera_basis, render
from room3d_refine import depth_error
from room3d_structure import semantic_masks


def propose(room, recipe, depths, segments):
    """Use training views only; reserve the final camera when there are >=3."""
    views = list(zip(room['cameras'], depths, segments, strict=True))
    if len({c['index'] for c, _, _ in views}) != len(views):
        raise ValueError('Expected distinct camera indices')
    training = views[:-1] if len(views) >= 3 else views
    report = {'status': 'insufficient_shared_evidence',
              'training_cameras': [c['index'] for c, _, _ in training],
              'heldout_cameras': [c['index'] for c, _, _ in views[len(training):]],
              'qualification': 'Estimated shared visual wall and floor; rectangular floor extent and wall contact are creative completion. Hidden parts, original fidelity and navigation unverified.'}
    if len(training) < 2:
        return recipe, report
    walls, floors, floor_depths, floor_cameras, extents = [], [], [], [], []
    for camera, (depth, valid), entries in training:
        if depth.shape != (240, 320) or valid.shape != depth.shape:
            raise ValueError('Expected native viewport depth and confidence')
        masks = semantic_masks(entries)
        y, x = np.mgrid[0:240:2, 0:320:2]
        z = depth[y, x]
        points = np.stack(((x-160)*z/camera['fov'], (y-120)*z/camera['fov'], z), -1)
        points = points @ camera_basis(camera) + camera['origin']
        for kind in ('wall', 'floor'):
            keep = valid[y, x] & masks[kind][y, x] & np.isfinite(z) & (z > 0)
            cloud, distances = points[keep], z[keep]
            if len(cloud) < 300:
                continue
            if kind == 'floor':
                floors.append(float(np.median(cloud[:, 1])))
                floor_cameras.append(camera['index'])
                floor_depths.append(float(np.median(distances)))
                extents.extend(np.percentile(cloud, [2, 98], axis=0).tolist())
            else:
                planes, _ = extract_wall_planes(cloud, float(np.median(distances)))
                for plane in planes:
                    walls.append({'camera': camera['index'], 'points': cloud[plane['indices']],
                                  'median_depth': float(np.median(distances[plane['indices']]))})
    if len(floors) < 2 or not walls:
        return recipe, report
    # This is model consistency, not a measured world-height guarantee.
    floor_limit = max(300., .05 * float(np.median(floor_depths)))
    report['floor'] = {'view_medians': floors, 'cameras': floor_cameras,
                       'maximum_spread': floor_limit}
    floor = float(np.mean(floors))
    report['floor']['policy'] = 'training_view_mean'
    if np.ptp(floors) > floor_limit:
        # Keep the existing editing plane only if every training-view estimate
        # admits it within the same uncertainty allowance. Never move toward
        # a reserved observation or silently widen the allowance after failure.
        prior = float(recipe.get('floor_y', 0))
        report['floor']['existing_prior'] = prior
        report['floor']['prior_residuals'] = [abs(y-prior) for y in floors]
        if not np.isfinite(prior) or any(abs(y-prior) > floor_limit for y in floors):
            report['status'] = 'inconsistent_training_floor'
            return recipe, report
        floor = prior
        report['floor']['policy'] = 'estimated_existing_floor_prior'
    planes, decisions = reconcile_walls(walls)
    shared = [p for p in planes if len(set(p['cameras'])) >= 2]
    report['wall_reconciliation'] = decisions
    if not shared:
        return recipe, report
    surfaces = []
    for index, plane in enumerate(shared):
        points = np.array(plane['points'], copy=True)
        points[[0, 3], 1] = floor
        extents.extend(points.tolist())
        surfaces.append({'points': points.tolist(), 'material': 'wall',
                         'group': f'estimated_visual_wall_{index}',
                         'estimated': True, 'observed_cameras': plane['cameras']})
    extents.extend(c['origin'] for c, _, _ in training)
    bounds = np.asarray(extents)[:, [0, 2]]
    x0, z0 = bounds.min(0)
    x1, z1 = bounds.max(0)
    surfaces.append({'points': [[x0, floor, z0], [x0, floor, z1],
                               [x1, floor, z1], [x1, floor, z0]],
                     'material': 'floor', 'group': 'estimated_visual_floor',
                     'estimated': True, 'observed_cameras': floor_cameras})
    candidate = copy.deepcopy(recipe)
    existing = candidate.setdefault('structural_surfaces', [])
    if {s['group'] for s in existing} & {s['group'] for s in surfaces}:
        report['status'] = 'surface_groups_already_present'
        return recipe, report
    existing.extend(surfaces)
    report.update(status='proposed', floor_y=floor, shared_walls=len(shared))
    return candidate, report


def recover(room, recipe, depths, segments, out, generate_mesh):
    """Select once on training evidence, then allow reserved depth only to veto."""
    out.mkdir(parents=True, exist_ok=True)
    before = out/'before.obj'
    generate_mesh(room, before, recipe)
    baseline = [render(before, c, return_depth=True)[2] for c in room['cameras']]
    if any(np.isfinite(d).any() for d in baseline):
        return recipe, {'accepted': False, 'reason': 'existing_visible_geometry'}
    candidate, report = propose(room, recipe, depths, segments)
    report['accepted'] = False
    if report['status'] != 'proposed':
        report['reason'] = report['status']
        return recipe, report
    selected = out/'selected.obj'
    generate_mesh(room, selected, candidate)
    heldout = set(report['heldout_cameras'])
    report['views'] = []
    for reserved in (False, True):
        for camera, old, (target, valid) in zip(room['cameras'], baseline, depths, strict=True):
            if (camera['index'] in heldout) != reserved:
                continue
            new = render(selected, camera, return_depth=True)[2]
            a, b = depth_error(old, target, valid), depth_error(new, target, valid)
            report['views'].append({'camera': camera['index'], 'reserved': reserved,
                'before': a, 'after': b, 'geometry_pixels': int(np.isfinite(new).sum()),
                'safe': bool(b <= a+.005 and np.isfinite(new).sum() >= 150)})
        if not all(v['safe'] for v in report['views']):
            report['reason'] = 'reserved_depth_or_visibility_failed' if reserved else 'training_depth_or_visibility_failed'
            return recipe, report
    if not any(v['after'] < v['before']-.0005 for v in report['views'] if not v['reserved']):
        report['reason'] = 'no_training_depth_improvement'
        return recipe, report
    report.update(accepted=True, reason='estimated_visual_surfaces_safe')
    return candidate, report
