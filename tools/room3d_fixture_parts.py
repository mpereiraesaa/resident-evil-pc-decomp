"""Optional wall-fixture part fitting from local baked brightness hypotheses.

Brightness is evidence of appearance, not proof of bulb identity or emission.
Only training observations influence selection; final scene gates remain required.
"""
import copy
from itertools import permutations

import cv2
import numpy as np
from room3d_fixtures import anchor_wall_lamp, fixture_contact
from room3d_shapes import generate


def _bright_components(image, mask):
    rgb = np.asarray(image.convert('RGB'), float) / 255
    if mask.shape != rgb.shape[:2] or mask.dtype != bool:
        raise ValueError('Expected a boolean mask matching the image')
    if mask.sum() < 16:
        return None
    luminance = rgb @ np.array([.2126, .7152, .0722])
    low, high = np.quantile(luminance[mask], [.25, .85])
    if high - low < .15:
        return None
    region = mask & (luminance >= high)
    if region.sum() < 4 or region.sum() > .35 * mask.sum():
        return None
    # Clipped or fragmented highlights do not reliably locate a whole shade.
    if region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any():
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(region.astype('uint8'), 8)
    if count <= 1:
        return None
    return labels, stats[1:, cv2.CC_STAT_AREA], int(region.sum())


def bright_region(image, mask):
    """Legacy single-shade evidence: one component must contain 60% of highlights."""
    components = _bright_components(image, mask)
    if components is None:
        return None
    labels, areas, total = components
    largest = int(np.argmax(areas))
    return labels == largest + 1 if areas[largest] >= .6 * total else None


def bright_regions(image, mask):
    """Two to four contained light hypotheses covering >=80% of highlights.

    Bright components describe appearance; they do not establish bulb identity.
    Small isolated glints cannot become additional modeled lights.
    """
    components = _bright_components(image, mask)
    if components is None:
        return None
    labels, areas, total = components
    selected = [i for i, area in enumerate(areas) if area >= max(4, .15 * total)]
    if not 2 <= len(selected) <= 4 or sum(areas[i] for i in selected) < .8 * total:
        return None
    return [labels == i + 1 for i in selected]


def light_assignment_score(matrix):
    """Match distinct shades to observed components, weighting the weakest fit."""
    matrix = np.asarray(matrix, float)
    if (matrix.ndim != 2 or not 1 <= matrix.shape[0] <= matrix.shape[1] <= 4
            or not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any()):
        raise ValueError('Expected a finite component/shade IoU matrix with up to four lights')
    matches = [[matrix[row, column] for row, column in enumerate(order)]
               for order in permutations(range(matrix.shape[1]), len(matrix))]
    return float(max(.5 * min(m) + .5 * np.mean(m) for m in matches))


def size_part_variants(obj):
    """Vary shade size/spacing while preserving wall anchor and light-center Y."""
    params = obj['shape']['parameters']
    profile = params.get('profile')
    if profile not in ('wall_sconce', 'wall_candelabra'):
        return
    for shade_scale in (.4, .6, .8, 1.):
        for span_scale in ((.7, .85, 1., 1.15) if profile == 'wall_candelabra' else (1.,)):
            if shade_scale == span_scale == 1:
                continue
            p = {**params, 'shade': params['shade'] * shade_scale}
            # Negative Y is upwards. Move the smaller shade's lower edge up
            # half the size difference to retain its original vertical centre.
            p['arm_rise'] = params.get('arm_rise', -.65) + (params['shade'] - p['shade']) / (2 * params['height'])
            if profile == 'wall_candelabra':
                p['span'] = params['span'] * span_scale
            try:
                candidate = anchor_wall_lamp(generate('lamp', p), copy.deepcopy(obj['attachment_prior']))
            except ValueError:
                continue
            yield candidate, {'stage': 'shade_size_spacing', 'shade_scale': shade_scale, 'span_scale': span_scale}


def joint_size_part_variants(obj):
    """Bound joint search to three shade sizes and 56 anchor/arm alternatives each."""
    for sized, metadata in size_part_variants(obj):
        if metadata['span_scale'] != 1:
            continue
        params = sized['shape']['parameters']
        for offset in (0, .3, .65, 1):
            for rise in np.linspace(-.65, .65, 14):
                prior = copy.deepcopy(sized['attachment_prior'])
                prior['point'][1] += offset*params['height']
                candidate = anchor_wall_lamp(generate('lamp', {**params, 'arm_rise': float(rise)}), prior)
                yield candidate, {**metadata, 'stage': 'joint_shade_size_anchor',
                                  'offset': offset, 'arm_rise': float(rise)}


def refine_wall_parts(room, recipe, obj, group, depths, images, out, generate_mesh, *, creative=False):
    """Bounded arm/anchor search, preserving whole silhouette support and contact."""
    from room3d_connections import support_mesh
    from room3d_preview import render
    from room3d_shape_fit import compare, isolated_recipe

    profile = obj.get('shape', {}).get('parameters', {}).get('profile')
    if creative and not obj.get('interpretation'):
        raise ValueError('Creative part refinement requires an already interpreted fixture')
    if profile not in ('wall_sconce', 'wall_candelabra'):
        return obj, {'status': 'unsupported_profile'}
    training = group[:-1] if len(group) >= 3 else group
    cameras = {c['index']: (c, d, im) for c, d, im in
               zip(room['cameras'], depths, images, strict=True)}
    targets = []
    for obs in training:
        if profile == 'wall_candelabra':
            regions = bright_regions(cameras[obs['camera']][2], obs['mask'])
        else:
            region = bright_region(cameras[obs['camera']][2], obs['mask'])
            regions = [region] if region is not None else None
        if regions is not None and len(regions) <= obj['shape']['parameters'].get('arms', 1):
            targets.append((obs, regions))
    if not targets:
        return obj, {'status': 'no_reliable_bright_region'}
    out.mkdir(parents=True, exist_ok=True)
    scene = out / 'scene.obj'
    generate_mesh(room, scene, recipe)
    triangles, groups = support_mesh(scene)
    scene_depths = {o['camera']: render(scene, cameras[o['camera']][0], return_depth=True)[2]
                    for o in group}
    path = out / 'candidate.obj'

    def score(candidate, observations):
        generate_mesh(room, path, isolated_recipe(room, recipe, candidate))
        results = []
        for obs in observations:
            camera, (depth, valid), _ = cameras[obs['camera']]
            part = render(path, camera, return_depth=True, only_groups={'primitive_0'})[2]
            result = compare(part, scene_depths[obs['camera']], obs['mask'], depth, valid)
            if creative:
                from room3d_interpretation import placement

                result['creative_placement'] = placement(part, scene_depths[obs['camera']], obs['mask'])
            results.append(result)
        return results

    def part_scores(candidate):
        count = int(candidate['shape']['parameters']['arms']) if profile == 'wall_candelabra' else 1
        matrices = [np.zeros((len(regions), count)) for _, regions in targets]
        for index in range(count):
            shade = copy.deepcopy(candidate)
            shade['parts'] = [shade['parts'][2 + index]]
            generate_mesh(room, path, isolated_recipe(room, recipe, shade))
            for matrix, (obs, regions) in zip(matrices, targets, strict=True):
                camera, (depth, valid), _ = cameras[obs['camera']]
                part = render(path, camera, return_depth=True, only_groups={'primitive_0'})[2]
                for row, region in enumerate(regions):
                    matrix[row, index] = compare(part, scene_depths[obs['camera']], region, depth, valid)['silhouette_iou']
        result = []
        for matrix in matrices:
            # One shade per component: a single large shade cannot explain two
            # lights. Worst matched component contributes to the objective.
            result.append(light_assignment_score(matrix))
        return result

    baseline = score(obj, training)
    base_parts = part_scores(obj)
    best, best_scores, best_parts = obj, baseline, base_parts
    def loss(whole, parts):
        return .35 * np.mean([1-s['silhouette_iou'] for s in whole]) + .65 * np.mean(1-np.array(parts))
    best_loss = loss(baseline, base_parts)
    trials = []
    params = obj['shape']['parameters']
    # Single-view wall acceptance requires .6 downstream. Do not select a
    # prettier part that turns an otherwise supported object into a rejection.
    minimum_whole_iou = (.6 if len(group) == 1
                         and recipe.get('single_view_wall_fixtures', False) else .35)

    def supported(candidate, whole):
        if creative:
            moved = np.linalg.norm(np.asarray(candidate['attachment_prior']['point'])
                                   - obj['attachment_prior']['point'])
            return moved <= min(1000, params['height']) and all(s['creative_placement']['valid'] for s in whole)
        return all(s['silhouette_iou'] >= max(minimum_whole_iou, b['silhouette_iou']-.15)
                   for s, b in zip(whole, baseline, strict=True))

    for offset in (0, .3, .65, 1):
        for rise in np.linspace(-.65, .65, 14):
            prior = copy.deepcopy(obj['attachment_prior'])
            prior['point'][1] += offset * params['height']
            candidate = anchor_wall_lamp(generate('lamp', {**params, 'arm_rise': float(rise)}), prior)
            attached = fixture_contact(candidate, triangles, groups)
            if not attached:
                continue
            whole = score(candidate, training)
            if not supported(candidate, whole):
                continue
            parts = part_scores(candidate)
            value = loss(whole, parts)
            improved = bool(min(p-b for p, b in zip(parts, base_parts, strict=True)) >= .05)
            keep = improved and value < best_loss
            trials.append({'offset': offset, 'arm_rise': float(rise), 'whole': whole,
                           'part_iou': parts, 'loss': float(value), 'improved': improved})
            if keep:
                best, best_scores, best_parts, best_loss = candidate, whole, parts, value
    if recipe.get('fixture_size_part_fit', False) or creative:
        variants = joint_size_part_variants(best) if creative else size_part_variants(best)
        for candidate, parameters in variants:
            if not fixture_contact(candidate, triangles, groups):
                trials.append({**parameters, 'status': 'rejected', 'reason': 'fixture_attachment'})
                continue
            whole = score(candidate, training)
            if not supported(candidate, whole):
                trials.append({**parameters, 'status': 'rejected',
                               'reason': 'creative_placement_or_anchor_bound' if creative else 'whole_silhouette_regressed',
                               'whole': whole})
                continue
            parts = part_scores(candidate)
            value = loss(whole, parts)
            improved = bool(min(p-b for p, b in zip(parts, base_parts, strict=True)) >= .05)
            trials.append({**parameters, 'whole': whole, 'part_iou': parts,
                           'loss': float(value), 'improved': improved})
            if improved and value < best_loss:
                best, best_scores, best_parts, best_loss = candidate, whole, parts, value
    selected = best is not obj
    if selected:
        for key in ('evidence', 'interpretation', 'estimated'):
            if key in obj:
                best[key] = copy.deepcopy(obj[key])
    after = score(best, group)
    attached = fixture_contact(best, triangles, groups)
    for s in after:
        s['attachment_valid'] = attached
    return best, {'status': 'selected' if selected else 'unchanged',
                  'training_cameras': [o['camera'] for o in training],
                  'part_cameras': [o['camera'] for o, _ in targets],
                  'light_components': [len(regions) for _, regions in targets],
                  'heldout_cameras': [o['camera'] for o in group[len(training):]],
                  'baseline': baseline, 'baseline_part_iou': base_parts,
                  'minimum_whole_iou': None if creative else minimum_whole_iou,
                  'support_criterion': 'creative_placement_and_bounded_anchor' if creative else 'whole_mask_iou',
                  'selected_training': best_scores, 'selected_part_iou': best_parts,
                  'after': after, 'trials': trials,
                  **({'size_search_enabled': True} if recipe.get('fixture_size_part_fit', False) else {}),
                  **({'creative_size_search_enabled': True} if creative else {}),
                  'qualification': 'Bright-region hypothesis, not bulb identity; final all-camera scene gates still required'}
