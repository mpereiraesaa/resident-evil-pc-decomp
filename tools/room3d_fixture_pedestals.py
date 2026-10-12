"""Local creative supports for raised table fixtures; no furniture extrusion."""

import copy

import numpy as np
from room3d_shapes import prism


def propose(room, recipe, fixture):
    """Return a supported fixture and recipe, or no admissible local pedestal."""
    prior = fixture.get('attachment_prior', {})
    params = fixture.get('shape', {}).get('parameters', {})
    group = prior.get('group', '')
    if (fixture.get('kind') != 'lamp' or params.get('profile') != 'table_lamp'
            or prior.get('support') != 'horizontal'
            or not group.startswith('boundary_') or not group[9:].isdigit()):
        return None
    index = int(group[9:])
    boundary = next((b for b in room['boundaries'] if b['index'] == index), None)
    if (boundary is None or boundary['flags'] & 256 or boundary['type'] & 255 not in (1, 5)
            or index in recipe.get('skip_boundaries', [])
            or index in recipe.get('shared_wall_boundaries', [])):
        return None
    height = float(recipe.get('boundary_heights', {}).get(str(index), recipe.get('obstacle_height', 1500)))
    bottom = float(recipe.get('floor_y', 0)) - height
    feet = np.asarray(fixture['parts'][1], float)[[2, 3, 6, 7]]
    top, margin = float(feet[:, 1].mean()), float(params['thickness'])
    if (not np.isfinite([bottom, top, margin, *feet.ravel()]).all()
            or margin <= 0 or not 0 < bottom - top <= 1000
            or np.ptp(feet[:, 1]) > 1e-5):
        return None
    lo, hi = feet.min(0), feet.max(0)
    lo[[0, 2]] -= margin
    hi[[0, 2]] += margin
    lo[1], hi[1] = top, bottom
    if not (boundary['xmin'] <= lo[0] < hi[0] <= boundary['xmax']
            and boundary['zmin'] <= lo[2] < hi[2] <= boundary['zmax']):
        return None
    result, lamp = copy.deepcopy(recipe), copy.deepcopy(fixture)
    pedestal_index = len(result.setdefault('primitives', []))
    result['primitives'].append({
        'kind': 'fixture_pedestal', 'parts': [prism(lo, hi)],
        'parent_boundary': index,
        'estimated': 'Creative local fixture support; original furniture height/footprint and native collision unchanged',
    })
    lamp['attachment_prior']['group'] = f'primitive_{pedestal_index}'
    lamp['attachment_prior']['qualification'] = 'Estimated local pedestal anchor, not recovered furniture'
    return result, lamp


def contact(recipe, triangles, groups):
    """Require each generated pedestal to rest on its original parent surface."""
    from room3d_connections import point_triangle_distance

    for primitive in recipe.get('primitives', []):
        if primitive.get('kind') != 'fixture_pedestal':
            continue
        parent = f"boundary_{primitive['parent_boundary']}"
        surfaces = [t for t, g in zip(triangles, groups, strict=True) if g == parent]
        if not surfaces:
            return False
        bottom = np.asarray(primitive['parts'][0], float)[[2, 3, 6, 7]]
        if any(min(point_triangle_distance(p, t) for t in surfaces) > 1e-5 for p in bottom):
            return False
    return True
