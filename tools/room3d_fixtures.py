"""Wall-attached fixture proposals; placement still requires image validation."""

import math

import numpy as np
from room3d_connections import closest_triangle_point, point_triangle_distance
from room3d_shapes import generate


def anchor_wall_lamp(candidate, prior):
    """Resize an anchored fixture without translating or rotating it off its wall."""
    params = candidate['shape']['parameters']
    wall_params = (params if params.get('profile') in ('wall_candelabra', 'wall_linear')
                   else {**params, 'profile': 'wall_sconce', 'arms': 1})
    normal = np.asarray(prior['normal'], float)
    point = np.asarray(prior['point'], float)
    obj = generate('lamp', wall_params,
                   point + [0, params['height'], 0], math.atan2(normal[0], normal[2]))
    obj['attachment_prior'] = dict(prior)
    return obj


def wall_plate_contact(obj, triangles, groups, tolerance=10):
    """Check plate corners and center against its designated finite wall group."""
    prior = obj.get('attachment_prior')
    if not prior or obj['shape']['parameters'].get('profile') not in ('wall_sconce', 'wall_candelabra', 'wall_linear'):
        return False
    surfaces = [triangle for triangle, group in zip(triangles, groups, strict=True)
                if group == prior['group']]
    if not surfaces:
        return False
    corners = np.asarray(obj['parts'][1], float)[4:8]
    samples = [*corners, corners.mean(axis=0)]
    return all(min(point_triangle_distance(point, triangle) for triangle in surfaces)
               <= tolerance for point in samples)


def inset_wall_lamp(reference, triangles, groups):
    """Move a clipped plate a bounded distance within its existing wall plane.

    Try only a plate half-extent in each tangent direction. Keep the nearest
    valid contact; do not extend walls or infer a new support from image depth.
    """
    prior = reference.get('attachment_prior')
    if not prior or reference['shape']['parameters'].get('profile') not in ('wall_sconce', 'wall_candelabra'):
        return None
    if wall_plate_contact(reference, triangles, groups):
        return None
    normal = np.array(prior['normal'], dtype=float, copy=True)
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or np.linalg.norm(normal) == 0 or abs(normal[1]) > 1e-8):
        raise ValueError('Expected a finite horizontal wall normal')
    normal /= np.linalg.norm(normal)
    tangent = np.array([normal[2], 0, -normal[0]])
    params = reference['shape']['parameters']
    vertical = params['height'] * .1 + 10
    horizontal = params['thickness'] * 1.5 + 10
    offsets = [np.array([0, y, 0]) + tangent * x
               for y in (0, vertical, -vertical)
               for x in (0, horizontal, -horizontal) if y or x]
    for offset in sorted(offsets, key=lambda d: float(d @ d)):
        moved = dict(prior)
        moved.update(point=(np.asarray(prior['point']) + offset).tolist(),
                     seed_point=prior.get('seed_point', prior['point']),
                     inset_displacement=float(np.linalg.norm(offset)),
                     qualification='Estimated finite-wall plate inset; image placement unverified')
        candidate = anchor_wall_lamp(reference, moved)
        if wall_plate_contact(candidate, triangles, groups):
            return candidate
    return None


def multilight_wall_proposals(reference, triangles, groups):
    """Bounded two/three-light alternatives on an already valid finite wall."""
    if not wall_plate_contact(reference, triangles, groups):
        return
    params = reference['shape']['parameters']
    if params.get('profile') == 'wall_linear':
        return
    for arms in (2, 3):
        for shade_scale in (.35, .55):
            shade = params['shade'] * shade_scale
            for offset in (.35, .65):
                prior = dict(reference['attachment_prior'])
                prior['point'] = (np.asarray(prior['point']) + [0, params['height'] * offset, 0]).tolist()
                prior['seed_point'] = reference['attachment_prior'].get('seed_point', reference['attachment_prior']['point'])
                p = {**params, 'profile': 'wall_candelabra', 'arms': arms,
                     'shade': shade, 'arm_rise': .35,
                     'span': max(params['shade'] * .8, (arms - 1) * (shade + params['thickness'] * 2))}
                candidate = anchor_wall_lamp(generate('lamp', p), prior)
                if wall_plate_contact(candidate, triangles, groups):
                    yield candidate


def wall_lamp_proposals(reference, triangles, groups, wall_groups,
                        camera_origins=(), maximum_distance=2000, maximum_candidates=4):
    """Snap the stem to nearby finite wall faces, facing observed room space."""
    if reference.get('kind') != 'lamp':
        raise ValueError('Expected a lamp reference')
    if not np.isfinite(maximum_distance) or maximum_distance <= 0:
        raise ValueError('Attachment search distance must be finite and positive')
    if not isinstance(maximum_candidates, int) or maximum_candidates < 1:
        raise ValueError('Expected a positive candidate count')
    anchor = np.asarray(reference['parts'][0], float)[:4].mean(axis=0)
    allowed = set(wall_groups)
    candidates = []
    for triangle, group in zip(triangles, groups, strict=True):
        if group not in allowed:
            continue
        xyz = np.asarray(triangle, float)[:, :3]
        normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
        length = np.linalg.norm(normal)
        if length < 1e-6 or abs(normal[1]) / length > 0.05:
            continue
        point = closest_triangle_point(anchor, xyz)
        distance = float(np.linalg.norm(point - anchor))
        if distance > maximum_distance:
            continue
        normal[1] = 0
        normal /= np.linalg.norm(normal)
        direction = anchor - point
        if abs(direction @ normal) < 1e-6 and len(camera_origins):
            direction = np.mean(camera_origins, axis=0) - point
        if abs(direction @ normal) < 1e-6:
            continue  # Surface winding alone cannot establish the room-facing side.
        if direction @ normal < 0:
            normal *= -1
        candidates.append((distance, group, point, normal))
    emitted = []
    for distance, group, point, normal in sorted(candidates, key=lambda item: (item[0], item[1])):
        if any(np.linalg.norm(point - p) < 1 and normal @ n > 0.99 for p, n in emitted):
            continue
        params = {**reference['shape']['parameters'], 'profile': 'wall_sconce', 'arms': 1}
        params['radius'] = max(params['radius'], (params['shade'] + params['thickness']) / 2)
        obj = generate('lamp', params, point + [0, params['height'], 0],
                       math.atan2(normal[0], normal[2]))
        obj['attachment_prior'] = {
            'group': group, 'point': point.tolist(), 'normal': normal.tolist(),
            'displacement': distance,
            'qualification': 'Inferred wall anchor; image agreement and plate contact unverified',
        }
        yield obj
        emitted.append((point, normal))
        if len(emitted) == maximum_candidates:
            break


def ray_wall_lamp_proposals(reference, training, triangles, groups, wall_groups):
    """Intersect training-mask rays with finite walls, independent of DA3 scale."""
    from room3d_preview import camera_basis

    allowed = set(wall_groups)
    surfaces = []
    for triangle, group in zip(triangles, groups, strict=True):
        if group not in allowed:
            continue
        xyz = np.asarray(triangle, float)[:, :3]
        normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
        length = np.linalg.norm(normal)
        if length < 1e-6 or abs(normal[1]) / length > 0.05:
            continue
        normal[1] = 0
        surfaces.append((xyz, group, normal / np.linalg.norm(normal)))
    emitted = []
    for obs in training:
        camera = obs['camera_model']
        y, x = np.nonzero(obs['mask'])
        if len(x) < 8:
            continue
        origin = np.asarray(camera['origin'], float)
        for fraction in (0.1, 0.35):
            pixel = np.array([np.median(x), np.quantile(y, fraction)])
            ray = np.array([*( (pixel - [160, 120]) / camera['fov']), 1]) @ camera_basis(camera)
            hits = []
            for xyz, group, normal in surfaces:
                matrix = np.column_stack((xyz[1] - xyz[0], xyz[2] - xyz[0], -ray))
                if abs(np.linalg.det(matrix)) < 1e-8:
                    continue
                u, v, depth = np.linalg.solve(matrix, origin - xyz[0])
                if u < -1e-6 or v < -1e-6 or u + v > 1 + 1e-6 or depth <= 2 * camera['fov']:
                    continue
                hits.append((float(depth), group, origin + depth * ray, normal))
            unique = []
            for depth, group, point, normal in sorted(hits, key=lambda h: (h[0], h[1])):
                normal = normal if normal @ (origin - point) > 0 else -normal
                if any(np.linalg.norm(point - p) < 1 for p in unique):
                    continue
                unique.append(point)
                if len(unique) > 2:
                    break
                if any(np.linalg.norm(point - p) < 1 and normal @ n > .99 for p, n in emitted):
                    continue
                width, height = np.ptp(x) * depth / camera['fov'], np.ptp(y) * depth / camera['fov']
                params = {**reference['shape']['parameters'], 'profile': 'wall_sconce', 'arms': 1,
                          'height': float(np.clip(height / .65, 150, 3000)),
                          'radius': float(np.clip(width * .5, 40, 600)),
                          'shade': float(np.clip(width * .8, 60, 400))}
                params['radius'] = max(params['radius'], (params['shade'] + params['thickness']) / 2)
                prior = {'group': group, 'point': point.tolist(), 'normal': normal.tolist(),
                         'camera': obs['camera'], 'pixel': pixel.tolist(),
                         'qualification': 'Mask-ray wall intersection; attachment position inferred'}
                candidate = generate('lamp', params)
                yield anchor_wall_lamp(candidate, prior)
                emitted.append((point, normal))
                if len(emitted) >= 8:
                    return


def linear_highlights(image, mask):
    """Return dominant elongated bright components, not a replacement object mask.

    Border-clipped tubes can initialize a lower-bound housing extent. Flat
    regions, round glints and fragmented highlights abstain. Brightness does
    not establish emission or tube identity.
    """
    import cv2

    rgb = np.asarray(image.convert('RGB'), float)/255
    if mask.dtype != bool or mask.shape != rgb.shape[:2]:
        raise ValueError('Expected a boolean mask matching the source image')
    if mask.sum() < 40:
        return None
    luminance = rgb @ np.array([.2126, .7152, .0722])
    low, high = np.quantile(luminance[mask], [.25, .85])
    if high-low < .15:
        return None
    region = mask & (luminance >= high)
    if region.sum() > mask.sum()*.35:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(region.astype('uint8'), 8)
    selected, axes = [], []
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        if area < max(8, .15*region.sum()):
            continue
        yy, xx = np.nonzero(labels == index)
        eigenvalues, vectors = np.linalg.eigh(np.cov(np.column_stack((xx, yy)).T))
        if eigenvalues[-1] < 25 or eigenvalues[-1] < 25*max(eigenvalues[0], 1e-8):
            continue
        selected.append(index)
        axes.append(vectors[:, -1])
    if not 1 <= len(selected) <= 3 or any(abs(a @ b) < .95 for a in axes for b in axes):
        return None
    core = np.isin(labels, selected)
    if core.sum() < max(40, .8*region.sum()):
        return None
    clipped = bool(core[0].any() or core[-1].any() or core[:, 0].any() or core[:, -1].any())
    return core, {'components': len(selected), 'pixels': int(core.sum()),
                  'clipped': clipped, 'luminance_contrast': float(high-low),
                  'qualification': 'Elongated training highlights only; clipped extent is partial, identity and emission unverified'}


def linear_wall_proposals(reference, training, triangles, groups, images=None, tube_counts=(1, 2)):
    """Infer a long housing in a finite wall plane using training masks only.

    Cropped observations supply partial extents, combined across cameras.
    Tube count and unseen housing thickness remain explicit design hypotheses.
    Existing render, attachment and reserved-view gates decide adoption.
    """
    from room3d_preview import camera_basis

    prior = reference.get('attachment_prior', {})
    if prior.get('support') in ('horizontal', 'ceiling') or not wall_plate_contact(reference, triangles, groups):
        return
    if images is not None:
        highlighted, evidence = [], []
        for obs in training:
            if obs['camera'] not in images:
                continue
            core = linear_highlights(images[obs['camera']], obs['mask'])
            if core is None:
                continue
            mask, record = core
            highlighted.append({**obs, 'mask': mask})
            evidence.append({'camera': obs['camera'], **record})
        if highlighted:
            counts = {record['components'] for record in evidence}
            agreed = len({record['camera'] for record in evidence}) >= 2 and len(counts) == 1
            hypotheses = tuple(sorted(counts)) if agreed else tube_counts
            for proposal in linear_wall_proposals(reference, highlighted, triangles, groups,
                                                  tube_counts=hypotheses):
                proposal['attachment_prior'].update(region='elongated_training_highlights',
                                                    highlight_evidence=evidence,
                                                    tube_count_consensus=agreed)
                yield proposal
    normal = np.array(prior['normal'], dtype=float, copy=True)
    normal /= np.linalg.norm(normal)
    tangent = np.array([normal[2], 0, -normal[0]])
    point = np.asarray(prior['point'], float)
    samples = []
    cameras = []
    for obs in training:
        y, x = np.nonzero(obs['mask'])
        if len(x) < 40:
            continue
        # All pixels, evenly subsampled in deterministic raster order.
        indices = np.linspace(0, len(x)-1, min(256, len(x))).astype(int)
        pixels = np.column_stack((x[indices]+.5, y[indices]+.5))
        camera = obs['camera_model']
        origin = np.asarray(camera['origin'], float)
        rays = np.column_stack(((pixels-[160, 120])/camera['fov'], np.ones(len(pixels)))) @ camera_basis(camera)
        denominator = rays @ normal
        good = np.abs(denominator) > 1e-8
        depth = np.divide((point-origin) @ normal, denominator,
                          out=np.zeros(len(rays)), where=good)
        good &= np.isfinite(depth) & (depth > camera['fov']*2)
        if good.sum() < 40:
            continue
        xyz = origin + rays[good]*depth[good, None]
        samples.append(np.column_stack((xyz @ tangent, xyz[:, 1])))
        cameras.append(obs['camera'])
    if not samples:
        return
    points = np.concatenate(samples)
    lo, hi = np.quantile(points, [.02, .98], axis=0)
    width, vertical = hi-lo
    if not (200 <= width <= 10000 and 50 <= vertical <= 1200 and width >= vertical*3):
        return
    center = point + tangent*((lo[0]+hi[0])/2-point @ tangent)
    center[1] = (lo[1]+hi[1])/2
    for arms in tube_counts:
        shade = vertical/(arms+.75)
        thickness = min(40., shade*.2)
        params = {'profile': 'wall_linear', 'height': float(vertical),
                  'radius': float(width/2), 'thickness': float(thickness),
                  'arms': arms, 'shade': float(shade)}
        anchored = dict(prior)
        anchored.update(point=center.tolist(), source_cameras=cameras,
                        qualification='Training-mask wall-plane housing extent; tube count and hidden housing inferred')
        proposal = anchor_wall_lamp(generate('lamp', params), anchored)
        if wall_plate_contact(proposal, triangles, groups):
            yield proposal


def anchor_fixture(candidate, prior):
    """Keep the selected support family and contact point during shape fitting."""
    if prior.get('support') == 'ceiling':
        return anchor_ceiling_fan(candidate, prior)
    if prior.get('support') != 'horizontal':
        return anchor_wall_lamp(candidate, prior)
    params = {**candidate['shape']['parameters'], 'profile': 'table_lamp', 'arms': 1}
    obj = generate('lamp', params, prior['point'], candidate['shape']['yaw'])
    obj['attachment_prior'] = dict(prior)
    return obj


def fixture_contact(obj, triangles, groups, tolerance=10):
    prior = obj.get('attachment_prior', {})
    if prior.get('support') == 'ceiling':
        if obj['shape']['parameters'].get('profile') != 'ceiling_fan':
            return False
        surfaces = [t for t, g in zip(triangles, groups, strict=True) if g == prior['group']]
        if not surfaces:
            return False
        corners = np.asarray(obj['parts'][0], float)[:4]
        return all(min(point_triangle_distance(p, t) for t in surfaces) <= tolerance
                   for p in [*corners, corners.mean(axis=0)])
    if prior.get('support') != 'horizontal':
        return wall_plate_contact(obj, triangles, groups, tolerance)
    if obj['shape']['parameters'].get('profile') != 'table_lamp':
        return False
    surfaces = [t for t, g in zip(triangles, groups, strict=True) if g == prior['group']]
    if not surfaces:
        return False
    corners = np.asarray(obj['parts'][1], float)[[2, 3, 6, 7]]
    return all(min(point_triangle_distance(p, t) for t in surfaces) <= tolerance
               for p in [*corners, corners.mean(axis=0)])


def anchor_ceiling_fan(candidate, prior):
    """Change the inferred suspension stem, preserving the fitted light/blade level."""
    shape = candidate['shape']
    params = shape['parameters']
    if candidate.get('kind') != 'lamp' or params.get('profile') != 'ceiling_fan':
        raise ValueError('Ceiling support requires a ceiling fan')
    point = np.asarray(prior['point'], float)
    origin = np.asarray(shape['origin'], float).copy()
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Expected a finite ceiling anchor')
    origin[[0, 2]] = point[[0, 2]]
    anchored = {**params, 'light_drop': params.get('light_drop', params['height']*.35),
                'height': float(origin[1] - point[1])}
    obj = generate('lamp', anchored, origin, shape['yaw'])
    obj['attachment_prior'] = dict(prior)
    return obj


def ceiling_fan_proposals(reference, triangles, groups, ceiling_groups, maximum_extension=2000):
    """Suspend a fan from a finite horizontal ceiling without moving its visible body.

    Roof height remains an estimate. Only the nearest contacting underside per
    group is proposed; side faces and distant roofs cannot become supports.
    """
    if reference.get('shape', {}).get('parameters', {}).get('profile') != 'ceiling_fan':
        return
    if not np.isfinite(maximum_extension) or maximum_extension <= 0:
        raise ValueError('Expected a positive finite ceiling extension limit')
    origin = np.asarray(reference['shape']['origin'], float)
    old_anchor = np.asarray(reference['parts'][0], float)[:4].mean(0)
    allowed = set(ceiling_groups)
    surfaces = []
    for triangle, group in zip(triangles, groups, strict=True):
        xyz = np.asarray(triangle, float)[:, :3]
        if group not in allowed or np.ptp(xyz[:, 1]) > 1e-6:
            continue
        point = origin.copy()
        point[1] = xyz[0, 1]
        if not 0 <= old_anchor[1]-point[1] <= maximum_extension:
            continue
        if point_triangle_distance(point, xyz) <= 1e-6:
            surfaces.append((point[1], group, point))
    emitted = set()
    for _, group, point in sorted(surfaces, key=lambda item: -item[0]):
        if group in emitted:
            continue
        prior = {'support': 'ceiling', 'group': group, 'point': point.tolist(),
                 'normal': [0, 1, 0],
                 'qualification': 'Estimated ceiling/stem connection; visible body unchanged, mechanical support unverified'}
        candidate = anchor_ceiling_fan(reference, prior)
        if fixture_contact(candidate, triangles, groups):
            candidate['initialization'] = 'ceiling_fan_attachment'
            yield candidate
            emitted.add(group)


def ray_table_lamp_proposals(reference, training, triangles, groups, support_groups):
    """Propose bases on finite horizontal furniture tops using lower-mask rays.

    Surface geometry is inferred. These are hypotheses, not confirmed furniture
    identity. Only explicitly supplied furniture groups may supply support.
    """
    from room3d_preview import camera_basis

    surfaces = []
    for triangle, group in zip(triangles, groups, strict=True):
        xyz = np.asarray(triangle, float)[:, :3]
        if group not in support_groups or np.ptp(xyz[:, 1]) > 1e-5:
            continue
        surfaces.append((xyz, group))
    tops = {group: min(t[0, 1] for t, g in surfaces if g == group)
            for _, group in surfaces}
    surfaces = [(t, g) for t, g in surfaces if abs(t[0, 1]-tops[g]) < 1e-5]
    emitted = []
    for obs in training:
        camera = obs['camera_model']
        y, x = np.nonzero(obs['mask'])
        if len(x) < 8:
            continue
        origin = np.asarray(camera['origin'], float)
        for fraction in (.9, 1.):
            pixel = np.array([np.median(x), np.quantile(y, fraction)])
            ray = np.array([*((pixel - [160, 120]) / camera['fov']), 1]) @ camera_basis(camera)
            hits = []
            for xyz, group in surfaces:
                # Game Y points down; camera must be above the support surface.
                if origin[1] >= xyz[0, 1]:
                    continue
                matrix = np.column_stack((xyz[1]-xyz[0], xyz[2]-xyz[0], -ray))
                if abs(np.linalg.det(matrix)) < 1e-8:
                    continue
                u, v, depth = np.linalg.solve(matrix, origin-xyz[0])
                if u < -1e-6 or v < -1e-6 or u+v > 1+1e-6 or depth <= 2*camera['fov']:
                    continue
                hits.append((float(depth), group, origin+depth*ray))
            for depth, group, point in sorted(hits, key=lambda h: (h[0], h[1])):
                if any(np.linalg.norm(point-p) < 1 for p in emitted):
                    continue
                width = np.ptp(x)*depth/camera['fov']
                height = np.ptp(y)*depth/camera['fov']
                params = {**reference['shape']['parameters'], 'profile': 'table_lamp',
                          'arms': 1, 'height': float(np.clip(height, 120, 2000)),
                          'shade': float(np.clip(min(width, height*.5), 40, 500)),
                          'radius': float(np.clip(width*.18, 15, 150)),
                          'thickness': float(np.clip(height*.04, 5, 40))}
                prior = {'support': 'horizontal', 'group': group, 'point': point.tolist(),
                         'camera': obs['camera'], 'pixel': pixel.tolist(),
                         'qualification': 'Mask-ray furniture support hypothesis; identity unverified'}
                try:
                    obj = anchor_fixture(generate('lamp', params), prior)
                except ValueError:
                    continue
                if not fixture_contact(obj, triangles, groups):
                    continue
                yield obj
                emitted.append(point)
                if len(emitted) >= 8:
                    return


def lamp_support_hypotheses(room, recipe, training):
    """Infer bounded top-height proposals where lower-mask rays cross obstacles.

    Existing footprints are preserved. Skipped geometry stays skipped; restoring
    a rejected obstacle requires separate evidence. These proposals must be
    evaluated jointly with the fixture and the surrounding room.
    """
    import copy

    from room3d_preview import camera_basis

    floor = float(recipe.get('floor_y', 0))
    emitted = set()
    for obs in training:
        y, x = np.nonzero(obs['mask'])
        if len(x) < 8:
            continue
        camera = obs['camera_model']
        origin = np.asarray(camera['origin'], float)
        for fraction in (.9, 1.):
            pixel = np.array([np.median(x), np.quantile(y, fraction)])
            ray = np.array([*((pixel-[160, 120])/camera['fov']), 1]) @ camera_basis(camera)
            for b in room['boundaries']:
                index = b['index']
                if (b['flags'] & 0x100 or b['type'] & 255 not in (1, 5)
                        or index in recipe.get('skip_boundaries', [])
                        or index in recipe.get('shared_wall_boundaries', [])):
                    continue
                near, far = 2*camera['fov'], float('inf')
                for axis, lower, upper in ((0, b['xmin'], b['xmax']),
                                           (2, b['zmin'], b['zmax'])):
                    if abs(ray[axis]) < 1e-9:
                        if not lower <= origin[axis] <= upper:
                            far = -1
                    else:
                        a, c = sorted(((lower-origin[axis])/ray[axis],
                                       (upper-origin[axis])/ray[axis]))
                        near, far = max(near, a), min(far, c)
                if near >= far or not np.isfinite(far):
                    continue
                for blend in (.25, .75):
                    point = origin + (near+(far-near)*blend)*ray
                    height = float(floor-point[1])
                    key = index, round(height/25)
                    if not 300 <= height <= 3000 or key in emitted:
                        continue
                    candidate = copy.deepcopy(recipe)
                    candidate.setdefault('boundary_heights', {})[str(index)] = height
                    evidence = {'boundary': index, 'height': height,
                                'camera': obs['camera'], 'pixel': pixel.tolist(),
                                'qualification': 'Ray-footprint support hypothesis, not recovered furniture'}
                    yield candidate, evidence
                    emitted.add(key)
                    if len(emitted) >= 8:
                        return


def fit_fixture_supports(room, recipe, group, depths, out, generate_mesh, images=None):
    """Compare existing supports with joint top-height/fixture hypotheses.

    Selection uses training views only. The caller must retain its final
    all-camera checks before adopting the modified scene recipe.
    """
    import copy
    import json

    from room3d_connections import support_mesh
    from room3d_preview import render
    from room3d_refine import depth_error
    from room3d_shape_fit import compare, fit_group, initial_shapes

    obj, report = fit_group(room, recipe, group, depths, out, generate_mesh, images=images)
    training = group[:-1] if len(group) >= 3 else group
    reference = next(initial_shapes(room, training), None)
    if reference is None:
        return obj, report, recipe
    by_camera = {c['index']: (c, d) for c, d in zip(room['cameras'], depths, strict=True)}
    baseline_path = out/'support-baseline.obj'
    generate_mesh(room, baseline_path, recipe)
    before = [depth_error(render(baseline_path, by_camera[o['camera']][0],
                                 return_depth=True)[2], *by_camera[o['camera']][1])
              for o in training]
    best = float(np.mean([s['loss'] for s in report.get('after', [])[:len(training)]])) if obj else float('inf')
    selected_recipe = recipe
    trials = []
    for index, (candidate, evidence) in enumerate(lamp_support_hypotheses(room, recipe, training)):
        dest = out/f'fixture-support{index}'
        dest.mkdir(exist_ok=True)
        scene = dest/'scene.obj'
        generate_mesh(room, scene, candidate)
        triangles, names = support_mesh(scene)
        starts = list(ray_table_lamp_proposals(reference, training, triangles, names,
                                               {f"boundary_{evidence['boundary']}"}))
        trial = {'support': evidence, 'starts': len(starts), 'selected': False}
        if starts:
            fitted, fitted_report = fit_group(room, candidate, group, depths, dest,
                                              generate_mesh, initial_candidates=starts)
            if fitted is None:
                trial['status'] = fitted_report.get('status')
                trials.append(trial)
                continue
            (dest/'fit-result.json').write_text(json.dumps(fitted_report, indent=2))
            variants = [('whole_support_height', candidate, fitted, fitted_report)]
            if recipe.get('local_fixture_pedestals', False):
                from room3d_fixture_pedestals import propose

                localized = propose(room, recipe, fitted)
                if localized is not None:
                    local_recipe, local_lamp = localized
                    # Equal training loss prefers the local change. Reserved
                    # views cannot choose between these alternatives.
                    variants.insert(0, ('local_pedestal', local_recipe, local_lamp,
                                        copy.deepcopy(fitted_report)))
            for variant, variant_recipe, variant_obj, variant_report in variants:
                combined = copy.deepcopy(variant_recipe)
                combined.setdefault('primitives', []).append(variant_obj)
                mesh = dest/f'{variant}.obj'
                generate_mesh(room, mesh, combined)
                errors, scores = [], []
                attached = True
                if variant == 'local_pedestal':
                    from room3d_fixture_pedestals import contact

                    triangles, names = support_mesh(mesh)
                    attached = fixture_contact(variant_obj, triangles, names) and contact(variant_recipe, triangles, names)
                for observation in training:
                    camera, (target, valid) = by_camera[observation['camera']]
                    scene_depth = render(mesh, camera, return_depth=True)[2]
                    errors.append(depth_error(scene_depth, target, valid))
                    if variant == 'local_pedestal':
                        part = render(mesh, camera, return_depth=True,
                                      only_groups={f"primitive_{len(combined['primitives'])-1}"})[2]
                        score = compare(part, scene_depth, observation['mask'], target, valid)
                        score['attachment_valid'] = bool(attached)
                        scores.append(score)
                if variant == 'local_pedestal':
                    variant_report['after'] = scores
                    variant_report['localized_support'] = {
                        'pedestal': variant_recipe['primitives'][-1],
                        'qualification': 'Pose fitted on a height proposal; training scores recomputed on the unchanged furniture plus local pedestal. Earlier search diagnostics describe the height proposal.',
                    }
                loss = float(np.mean([s['loss'] for s in variant_report['after'][:len(training)]]))
                depth_safe = all(a <= b+.005 for a, b in zip(errors, before, strict=True))
                attached &= all(s.get('attachment_valid', False) for s in variant_report['after'][:len(training)])
                keep = depth_safe and attached and loss < best
                variant_trial = {'support': evidence, 'starts': len(starts), 'variant': variant,
                                 'loss': loss, 'depth_before': before, 'depth_after': errors,
                                 'depth_safe': depth_safe, 'attached': bool(attached), 'selected': bool(keep)}
                (dest/f'{variant}-fit-result.json').write_text(json.dumps(variant_report, indent=2))
                if keep:
                    best, obj, report, selected_recipe = loss, variant_obj, variant_report, variant_recipe
                    for previous in trials:
                        previous['selected'] = False
                trials.append(variant_trial)
            continue
        trials.append(trial)
    if report.get('localized_support'):
        selected_path = out/'selected-local-support.obj'
        combined = copy.deepcopy(selected_recipe)
        combined.setdefault('primitives', []).append(obj)
        generate_mesh(room, selected_path, combined)
        report['heldout'] = []
        for observation in group[len(training):]:
            camera, (target, valid) = by_camera[observation['camera']]
            scene_depth = render(selected_path, camera, return_depth=True)[2]
            part = render(selected_path, camera, return_depth=True,
                          only_groups={f"primitive_{len(combined['primitives'])-1}"})[2]
            score = compare(part, scene_depth, observation['mask'], target, valid)
            triangles, names = support_mesh(selected_path)
            score['attachment_valid'] = fixture_contact(obj, triangles, names)
            report['heldout'].append(score)
    report['fixture_support_trials'] = trials
    report['fixture_support_training_cameras'] = [o['camera'] for o in training]
    return obj, report, selected_recipe


def slide_fixture_anchor(candidate, current, maximum_displacement=1000):
    """Allow inferred support points to move tangentially during image fitting.

    Finite support contact must still be evaluated by the caller. The original
    ray point is retained as provenance and bounds cumulative anchor motion.
    """
    prior = dict(current['attachment_prior'])
    point = np.asarray(prior['point'], float)
    seed = np.asarray(prior.get('seed_point', point), float)
    normal = np.asarray(prior.get('normal', [0, -1, 0]), float)
    normal /= np.linalg.norm(normal)
    delta = (np.asarray(candidate['shape']['origin'], float)
             - np.asarray(current['shape']['origin'], float))
    moved = point + delta - (delta @ normal)*normal
    if np.linalg.norm(moved-seed) > maximum_displacement:
        raise ValueError('Fixture anchor exceeds its search neighborhood')
    prior.update(point=moved.tolist(), seed_point=seed.tolist(),
                 qualification='Support-plane anchor refined from image agreement; identity unverified')
    return anchor_fixture(candidate, prior)
