"""Bounded visual wall retreat proposals behind a retained upright ladder.

These are estimated alternatives, never accepted replacements. Original room
collision data and rail geometry remain untouched. Render, reserved-view,
scene-depth, finite support and independent-room checks remain necessary.
"""
import copy

import numpy as np
from room3d_ladder_supports import anchor_brackets
from room3d_shapes import generate


def refit_support(obj, triangles, groups, wall_groups):
    """Reinitialize only bracket caps before attaching to a changed wall.

    Previously fitted links can be oblique: their old shaft direction is not
    a new wall normal. The procedural seed retains each rail-end attachment.
    """
    shape = obj['shape']
    parameters = {k: v for k, v in shape['parameters'].items() if k != 'bracket_parts'}
    seed = generate('ladder', parameters, shape['origin'], shape['yaw'])
    if (not np.array_equal(seed['parts'][:-4], obj['parts'][:-4])
            or not np.allclose(np.asarray(seed['parts'][-4:])[:, 4:].mean(1),
                               np.asarray(obj['parts'][-4:])[:, 4:].mean(1), atol=1e-8, rtol=0)):
        return obj, {'status':'nonregenerable_attachment', 'qualification':'Rails and rail-end attachments must remain unchanged'}
    attached, report = anchor_brackets(seed, triangles, groups, wall_groups)
    if report['status'] != 'anchored':
        return obj, report
    return {**copy.deepcopy(obj), **attached}, report


def proposals(room, recipe, primitive_index, camera_origins,
              distances=(90, 180, 270, 360, 540, 720, 900), *, local=False):
    distances = np.asarray(distances, float)
    cameras = np.asarray(camera_origins, float)
    if (distances.ndim != 1 or not len(distances) or not np.isfinite(distances).all()
            or (distances <= 0).any() or (distances > 1000).any()
            or cameras.ndim != 2 or cameras.shape[1:] != (3,)
            or not len(cameras) or not np.isfinite(cameras).all()):
        raise ValueError('Expected bounded positive retreats and finite training cameras')
    obj = recipe['primitives'][primitive_index]
    if obj.get('kind') != 'ladder' or obj.get('omit_geometry'):
        raise ValueError('Expected retained ladder geometry')
    assigned = set(obj.get('wall_attachment', {}).get('bracket_groups', []))
    bounds = {f"boundary_{b['index']}": b for b in room['boundaries']
              if b['flags'] & 256 and b['index'] in recipe.get('shared_wall_boundaries', [])}
    protected = {p.get('hide_with') for p in recipe['primitives'] if p.get('kind') == 'door_panel'}
    rails = np.asarray(obj['parts'][:-4], float).reshape(-1, 3)
    if not len(rails) or not np.isfinite(rails).all():
        raise ValueError('Expected finite rail and rung geometry')
    result = []
    for index, surface in enumerate(recipe.get('structural_surfaces', [])):
        group = surface['group']
        if (surface['material'] != 'wall' or group not in assigned
                or group not in bounds or group in protected):
            continue
        points = np.asarray(surface['points'], float)
        if points.shape != (4, 3) or not np.isfinite(points).all() or np.ptp(points[:, 1]) <= 0:
            continue
        axes = [axis for axis in (0, 2) if np.ptp(points[:, axis]) < 1e-6]
        if len(axes) != 1:
            continue
        axis = axes[0]
        cross = 2-axis
        position = points[0, axis]
        sides = cameras[:, axis]-position
        if not ((sides > 1).all() or (sides < -1).all()):
            continue
        direction = -float(np.sign(sides[0]))
        if (abs(np.median(rails[:, axis])-position) > 1000
                or rails[:, cross].min() <= points[:, cross].min()
                or rails[:, cross].max() >= points[:, cross].max()):
            continue
        b = bounds[group]
        lo, hi = (b['xmin'], b['xmax']) if axis == 0 else (b['zmin'], b['zmax'])
        patch = points.copy()
        outer = []
        if local:
            margin = 2*obj['shape']['parameters']['thickness']
            low = max(points[:, cross].min(), rails[:, cross].min()-margin)
            high = min(points[:, cross].max(), rails[:, cross].max()+margin)
            # Current shared walls use two vertical edge pairs. Preserve their
            # order and winding, and retain the two outer strips unchanged.
            if not (points[0, cross] == points[1, cross]
                    and points[2, cross] == points[3, cross]):
                continue
            for ids in ((0, 1), (2, 3)):
                edge = points[ids[0], cross]
                patch[list(ids), cross] = low if edge == points[:, cross].min() else high
                strip = np.asarray([points[ids[0]], points[ids[1]],
                                    patch[ids[1]], patch[ids[0]]])
                if np.linalg.norm(np.cross(strip[1]-strip[0], strip[3]-strip[0])) > 1e-8:
                    outer.append({**surface, 'points':strip.tolist()})
        for distance in sorted(set(distances)):
            moved = patch.copy()
            moved[:, axis] += direction*distance
            if not lo < moved[0, axis] < hi:
                continue
            candidate = copy.deepcopy(recipe)
            evidence = {'source':'bounded_object_wall_retreat', 'primitive':primitive_index,
                        'surface':index,'distance':float(distance),
                        'span':'object' if local else 'wall_run',
                        'qualification':'Estimated visual wall retreat inside original collision footprint; support, depth, heldout views and fidelity unverified'}
            candidate['structural_surfaces'][index] = {**surface, 'points':moved.tolist(),
                                                       'object_wall_fit':evidence}
            # Connect the moved run to its original ends. Existing neighboring
            # walls keep their vertices; no gap is opened at either junction.
            for a, b in ((0, 1), (2, 3)):
                side = np.asarray([patch[a], patch[b], moved[b], moved[a]])
                if np.linalg.norm(np.cross(side[1]-side[0], side[3]-side[0])) > 1e-8:
                    candidate['structural_surfaces'].append({**surface,'points':side.tolist(),
                                                            'object_wall_fit':evidence})
            candidate['structural_surfaces'].extend(copy.deepcopy(outer))
            result.append((candidate, evidence))
    return result
