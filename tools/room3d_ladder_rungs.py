"""Fit ladder rung counts with actual rasterized edges, preserving shared pose."""
import copy
import math

import cv2
import numpy as np
from room3d_preview import camera_basis, render
from room3d_shapes import generate


def target_edges(obj, observation, image):
    """Exclude rail sides and retain image edges parallel to projected rungs."""
    rgb = np.asarray(image.convert('RGB'))
    mask = np.asarray(observation['mask'])
    if rgb.shape != (240, 320, 3) or mask.shape != (240, 320) or mask.dtype != bool:
        raise ValueError('Expected native original image and boolean object mask')
    shape = obj['shape']
    p = shape['parameters']
    c, s = math.cos(shape['yaw']), math.sin(shape['yaw'])
    rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    local = np.array([[x, y, p['stand_off']] for x, y in
                      [(-p['width']*.3, -p['height']*.98),
                       (p['width']*.3, -p['height']*.98),
                       (p['width']*.3, -p['height']*.02),
                       (-p['width']*.3, -p['height']*.02)]])
    camera = observation['camera_model']
    projected = (local @ rotation + shape['origin'] - camera['origin']) @ camera_basis(camera).T
    if np.any(projected[:, 2] <= camera['fov']*2):
        return None
    pixels = projected[:, :2]*camera['fov']/projected[:, 2, None]+[160, 120]
    region = np.zeros(mask.shape, 'uint8')
    cv2.fillConvexPoly(region, np.rint(pixels).astype('int32'), 1)
    region = region.astype(bool) & (cv2.dilate(mask.astype('uint8'), np.ones((7, 7), 'uint8')) > 0)
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (3, 3), .8)
    gx = cv2.Sobel(gray.astype('float32'), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray.astype('float32'), cv2.CV_32F, 0, 1)
    magnitude = np.hypot(gx, gy)
    normal = np.array([s, 0, c])
    plane_point = np.asarray(shape['origin'])+normal*p['stand_off']
    # Projected horizontal direction varies with image height. A single
    # averaged direction can reject real rungs in an oblique close-up.
    yy, xx = np.nonzero(region)
    rays = np.column_stack(((xx-160)/camera['fov'], (yy-120)/camera['fov'],
                            np.ones(len(xx)))) @ camera_basis(camera)
    denominator = rays @ normal
    depths = np.divide((plane_point-camera['origin'])@normal, denominator,
                       out=np.zeros(len(xx)), where=abs(denominator)>1e-8)
    valid = depths > 2*camera['fov']
    yy, xx, rays, depths = yy[valid], xx[valid], rays[valid], depths[valid]
    world_y = (np.asarray(camera['origin'])+rays*depths[:, None])[:, 1]
    ends = np.zeros((len(xx), 2, 3))
    ends[:, :, 0] = [-p['width']*.3, p['width']*.3]
    ends[:, :, 1] = world_y[:, None]-shape['origin'][1]
    ends[:, :, 2] = p['stand_off']
    ends = (ends@rotation+shape['origin']-camera['origin']) @ camera_basis(camera).T
    valid = (ends[:, :, 2] > 2*camera['fov']).all(axis=1)
    yy, xx, ends = yy[valid], xx[valid], ends[valid]
    endpoints = ends[:, :, :2]*camera['fov']/ends[:, :, 2, None]
    tangents = endpoints[:, 1]-endpoints[:, 0]
    lengths = np.linalg.norm(tangents, axis=1)
    tangents /= np.maximum(lengths[:, None], 1e-8)
    parallel = np.zeros(mask.shape, bool)
    parallel[yy, xx] = (abs(gx[yy, xx]*tangents[:, 0]+gy[yy, xx]*tangents[:, 1])
                        <= .5*magnitude[yy, xx]) & (lengths >= 1)
    edges = (cv2.Canny(gray, 10, 25) > 0) & parallel & region & (magnitude >= 8)
    # Pixel noise and masonry grain can make an increasingly dense ladder
    # appear better. Keep only line segments spanning a substantial part of
    # the local projected rung width, using the shared plane to handle scale.
    lines = cv2.HoughLinesP(edges.astype('uint8')*255, 1, np.pi/360, 5,
                            minLineLength=5, maxLineGap=2)
    coherent = np.zeros(mask.shape, 'uint8')
    if lines is not None:
        for x0, y0, x1, y1 in lines.reshape(-1, 4):
            midpoint = np.array([(x0+x1)/2, (y0+y1)/2])
            ray = np.r_[(midpoint-[160, 120])/camera['fov'], 1] @ camera_basis(camera)
            den = ray@normal
            if abs(den) < 1e-8:
                continue
            depth = (plane_point-camera['origin'])@normal/den
            if depth <= 2*camera['fov']:
                continue
            world_y = (np.asarray(camera['origin'])+ray*depth)[1]
            ends = np.array([[-p['width']*.3, world_y-shape['origin'][1], p['stand_off']],
                             [p['width']*.3, world_y-shape['origin'][1], p['stand_off']]])
            projected_ends = (ends@rotation+shape['origin']-camera['origin']) @ camera_basis(camera).T
            if np.any(projected_ends[:, 2] <= 2*camera['fov']):
                continue
            projected_ends = projected_ends[:, :2]*camera['fov']/projected_ends[:, 2, None]
            tangent = projected_ends[1]-projected_ends[0]
            segment = np.array([x1-x0, y1-y0], float)
            length = np.linalg.norm(segment)
            span = np.linalg.norm(tangent)
            if (length >= max(5, .4*span)
                    and abs(segment@tangent) >= .95*length*span):
                cv2.line(coherent, (int(x0), int(y0)), (int(x1), int(y1)), 1, 1)
    edges &= coherent.astype(bool)
    if edges.sum() < 30:
        return None
    return region, edges


def edge_score(predicted, target, region):
    edge = (cv2.Canny(predicted.astype('uint8')*255, 50, 150) > 0) & region
    if edge.sum() < 10:
        return 0.0
    to_target = cv2.distanceTransform((~target).astype('uint8'), cv2.DIST_L2, 5)
    to_edge = cv2.distanceTransform((~edge).astype('uint8'), cv2.DIST_L2, 5)
    precision = float(np.exp(-to_target[edge]/.75).mean())
    recall = float(np.exp(-to_edge[target]/.75).mean())
    return 2*precision*recall/max(precision+recall, 1e-12)


def refine_count(room, obj, observations, images, out, generate_mesh,
                 counts=range(2, 33)):
    """Select on training views; veto on reserved evidence without reselection.

    Does not accept a room replacement: full scene-depth/contact gates remain
    the caller's responsibility. Pose, rail dimensions and brackets stay fixed.
    """
    if obj.get('kind') != 'ladder':
        raise ValueError('Expected a generated ladder')
    ids = [o['camera'] for o in observations]
    if len(ids) != len(set(ids)):
        raise ValueError('Expected distinct camera observations')
    training = observations[:-1] if len(observations) >= 3 else observations
    reserved = observations[len(training):]
    report = {'status': 'insufficient_training_edges',
              'training_cameras': [o['camera'] for o in training],
              'heldout_cameras': [o['camera'] for o in reserved], 'trials': [],
              'qualification': 'Rung count from original training image edges and rendered rung geometry. Full scene depth, finite support, textures, independent-room fidelity and navigation unverified.'}
    if len(training) < 2:
        return obj, report
    targets = [target_edges(obj, o, images[o['camera']]) for o in training]
    report['training_edge_evidence'] = [
        {'camera': observation['camera'], 'usable': target is not None,
         'coherent_edge_pixels': int(target[1].sum()) if target is not None else None}
        for observation, target in zip(training, targets, strict=True)]
    if any(t is None for t in targets):
        return obj, report
    parameters = obj['shape']['parameters']
    counts = list(counts)
    if any(isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= 64 for n in counts):
        raise ValueError('Expected integer rung counts in 2..64')
    counts = sorted({parameters['rungs'], *counts})
    out.mkdir(parents=True, exist_ok=True)

    def evaluate(candidate, views, evidence, path):
        # Separate part groups expose rung silhouettes without rail/bracket
        # edges dominating the image agreement score.
        parts = [{'kind': 'ladder', 'parts': [part]} for part in candidate['parts']]
        generate_mesh(room, path, {'primitives': parts,
                                  'skip_boundaries': [b['index'] for b in room['boundaries']]})
        groups = {f'primitive_{i}' for i in range(2, 2+candidate['shape']['parameters']['rungs'])}
        scores = []
        for observation, target in zip(views, evidence, strict=True):
            mask = render(path, observation['camera_model'], only_groups=groups)[1]
            scores.append(edge_score(mask, target[1], target[0]))
        return scores

    candidates = {}
    for count in counts:
        if parameters['height'] <= 2*parameters['thickness']*(count+1):
            continue
        candidate = generate('ladder', {**parameters, 'rungs': count},
                             obj['shape']['origin'], obj['shape']['yaw'])
        for key in ('initialization', 'ladder_evidence'):
            if key in obj:
                candidate[key] = copy.deepcopy(obj[key])
        scores = evaluate(candidate, training, targets, out/f'rungs{count}.obj')
        candidates[count] = candidate, scores
        report['trials'].append({'rungs': count, 'training_edge_scores': scores})
    old_count = parameters['rungs']
    if old_count not in candidates:
        raise ValueError('Invalid baseline ladder rung spacing')
    baseline = candidates[old_count][1]
    best = max(candidates, key=lambda n: (float(np.mean(candidates[n][1])), n == old_count, -n))
    candidate, scores = candidates[best]
    report.update(baseline_rungs=old_count, selected_training_rungs=best,
                  baseline_scores=baseline, selected_training_scores=scores)
    if (np.mean(scores) <= np.mean(baseline)+.02
            or any(a < b-.03 for a, b in zip(scores, baseline, strict=True))):
        report['status'] = 'no_consistent_training_improvement'
        return obj, report
    if reserved:
        heldout_targets = [target_edges(obj, o, images[o['camera']]) for o in reserved]
        if any(t is None for t in heldout_targets):
            report['status'] = 'reserved_edges_unavailable'
            return obj, report
        old = evaluate(obj, reserved, heldout_targets, out/'heldout-baseline.obj')
        new = evaluate(candidate, reserved, heldout_targets, out/'heldout-selected.obj')
        report['heldout'] = {'baseline': old, 'selected': new}
        if any(a < b-.03 for a, b in zip(new, old, strict=True)):
            report['status'] = 'reserved_edges_failed'
            return obj, report
    report['status'] = 'selected'
    candidate['rung_fit'] = copy.deepcopy(report)
    return candidate, report
