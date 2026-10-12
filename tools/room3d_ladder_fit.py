"""Shared upright rail hypotheses; masks may fill holes or clip ladder ends.

Only training observations initialize geometry. These are estimated proposals,
not accepted replacements or proof of physical support/navigation.
"""
import itertools
import math

import cv2
import numpy as np
from room3d_preview import camera_basis
from room3d_shapes import generate


def rail_lines(observation):
    """Find two long mask edges compatible with projected game gravity."""
    mask = np.asarray(observation['mask'])
    if mask.shape != (240, 320) or mask.dtype != bool:
        raise ValueError('Expected a native boolean object mask')
    y, x = np.nonzero(mask)
    if len(x) < 100 or np.ptp(y) < 35:
        return None
    camera = observation['camera_model']
    edges = cv2.Canny(mask.astype('uint8') * 255, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, 15,
                            minLineLength=max(25, int(np.ptp(y) * .25)), maxLineGap=6)
    if lines is None:
        return None
    reference_y = float(np.median(y))
    basis = camera_basis(camera)
    candidates = []
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        length = float(np.hypot(x1-x0, y1-y0))
        line = np.cross([x0+.5, y0+.5, 1], [x1+.5, y1+.5, 1])
        line /= np.linalg.norm(line[:2])
        a, b, c = line
        if abs(a) < .2:
            continue
        normal = np.array([a*camera['fov'], b*camera['fov'], c+160*a+120*b]) @ basis
        normal /= np.linalg.norm(normal)
        if abs(normal[1]) > .035:
            continue
        position = -(b*reference_y+c)/a
        # A clipped rail can extrapolate outside the viewport at the common
        # reference row; its observed segment still constrains the shared axis.
        # Constrain a noisy mask edge to game gravity while retaining its
        # intersection with the observed middle row. Merely zeroing world Y
        # would move the line and bias the triangulated rail position.
        ray = np.array([(position-160)/camera['fov'],
                        (reference_y-120)/camera['fov'], 1]) @ basis
        normal = np.cross(ray, [0, 1, 0])
        normal /= np.linalg.norm(normal)
        local = normal @ basis.T
        line = np.array([local[0]/camera['fov'], local[1]/camera['fov'],
                         local[2]-(160*local[0]+120*local[1])/camera['fov']])
        line /= np.linalg.norm(line[:2])
        if line[0] < 0:
            line, normal = -line, -normal
        candidates.append((position, length, line, normal))
    clusters = []
    for item in sorted(candidates, key=lambda value: -value[1]):
        if not any(abs(item[0]-old[0]) < 5 for old in clusters):
            clusters.append(item)
    pairs = [(min(a[1], b[1]), abs(a[0]-b[0]), a, b)
             for a, b in itertools.combinations(clusters, 2)
             if abs(a[0]-b[0]) >= 8]
    if not pairs:
        return None
    _, _, a, b = max(pairs, key=lambda value: (value[0], value[1]))
    return [{'image_line': item[2].tolist(), 'world_normal': item[3].tolist(),
             'length_pixels': item[1]} for item in sorted((a, b), key=lambda item: item[0])]


def proposals(observations, floor_y=0, rung_spacing=400):
    """Triangulate rail axes; infer height from union of partial training masks.

    The last observation is reserved whenever there are at least three. Both
    two-view correspondence permutations remain candidates for later render
    fitting. No wall is extended and no attachment is asserted here.
    """
    if not np.isfinite([floor_y, rung_spacing]).all() or rung_spacing <= 0:
        raise ValueError('Expected finite floor and positive rung spacing')
    camera_ids = [o['camera'] for o in observations]
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError('Expected distinct camera observations')
    training = observations[:-1] if len(observations) >= 3 else observations
    report = {'status': 'insufficient_shared_rails',
              'training_cameras': [o['camera'] for o in training],
              'heldout_cameras': camera_ids[len(training):], 'rail_evidence': [], 'trials': [],
              'qualification': 'Training-mask vertical rail axes; partial height, stand-off and rung count estimated. Finite wall support, appearance, acceptance and navigation unverified.'}
    if len(training) < 2:
        return [], report
    rails = [rail_lines(o) for o in training]
    report['rail_evidence'] = [{'camera': o['camera'], 'lines': r}
                               for o, r in zip(training, rails, strict=True)]
    if any(r is None for r in rails):
        return [], report
    candidates = []
    for permutation in ((0, 1), (1, 0)):
        trial = {'permutation': list(permutation), 'status': 'rejected'}
        report['trials'].append(trial)
        positions = []
        for side in range(2):
            normals = np.array([rails[0][side]['world_normal'], rails[1][permutation[side]]['world_normal']])
            origins = np.array([o['camera_model']['origin'] for o in training[:2]])
            matrix = normals[:, [0, 2]]
            if abs(np.linalg.det(matrix)) < .01:
                break
            positions.append(np.linalg.solve(matrix, np.sum(normals*origins, axis=1)-normals[:, 1]*floor_y))
        if len(positions) != 2:
            continue
        positions = np.asarray(positions)
        width = float(np.linalg.norm(positions[1]-positions[0]))
        trial['width'] = width
        if not 200 <= width <= 2500:
            continue
        horizontal = (positions[1]-positions[0])/width
        normal = np.array([-horizontal[1], 0, horizontal[0]])
        center = np.array([positions[:, 0].mean(), floor_y, positions[:, 1].mean()])
        origins = np.array([o['camera_model']['origin'] for o in training])
        if (origins.mean(axis=0)-center)@normal < 0:
            normal = -normal
            horizontal = -horizontal
        heights, extents, residuals = [], [], []
        for obs, lines in zip(training, rails, strict=True):
            camera = obs['camera_model']
            # Additional training views validate shared rail axes without using
            # the reserved image or requiring matching clipped end extents.
            projected = []
            for position in positions:
                local = (np.array([position[0], floor_y, position[1]])-camera['origin']) @ camera_basis(camera).T
                if local[2] <= 2*camera['fov']:
                    break
                projected.append(np.r_[local[:2]*camera['fov']/local[2]+[160, 120], 1])
            if len(projected) != 2:
                break
            distances = np.abs(np.array([line['image_line'] for line in lines]) @ np.array(projected).T)
            residuals.append(float(min(distances[0, 0]+distances[1, 1], distances[0, 1]+distances[1, 0])/2))
            y, x = np.nonzero(obs['mask'])
            select = np.linspace(0, len(x)-1, min(512, len(x))).astype(int)
            rays = np.column_stack(((x[select]+.5-160)/camera['fov'],
                                    (y[select]+.5-120)/camera['fov'], np.ones(len(select)))) @ camera_basis(camera)
            origin = np.asarray(camera['origin'])
            den = rays @ normal
            depth = np.divide((center-origin)@normal, den, out=np.zeros(len(den)), where=abs(den)>1e-8)
            good = np.isfinite(depth) & (depth>2*camera['fov']) & (abs(den)>1e-8)
            if good.sum() < 40:
                break
            world_y = (origin+rays[good]*depth[good, None])[:, 1]
            low, high = np.quantile(world_y, [.01, .99])
            clipped = bool(obs['mask'][[0, -1], :].any() or obs['mask'][:, [0, -1]].any())
            extents.append({'camera': obs['camera'], 'y_min': float(low), 'y_max': float(high), 'border_clipped': clipped})
            heights.append(floor_y-low)
        if len(heights) != len(training) or max(residuals) > 3:
            continue
        height = max(heights)
        trial.update(height=float(height), observed_extents=extents,
                     rail_residual_pixels=residuals)
        if not 1000 <= height <= 15000 or height < 2*width:
            continue
        if any(abs(e['y_max']-floor_y)>max(300, .08*height) for e in extents if not e['border_clipped']):
            continue
        thickness = float(np.clip(width*.06, 20, 90))
        stand_off = thickness*3
        maximum_rungs = min(64, math.ceil(height/(2*thickness))-2)
        # Triangulated axes are the rails, not a measured wall: shift the local
        # plane behind them to keep the placement hypothesis unchanged.
        origin = center-normal*stand_off
        obj = generate('ladder', {'width': width+thickness, 'height': height,
                                 'thickness': thickness, 'stand_off': stand_off,
                                 'rungs': int(np.clip(round(height/rung_spacing)-1, 2, maximum_rungs))},
                       origin, math.atan2(normal[0], normal[2]))
        obj['initialization'] = 'shared_vertical_mask_rails'
        obj['ladder_evidence'] = {'training_cameras': report['training_cameras'],
                                  'rail_residual_pixels': residuals, 'observed_extents': extents,
                                  'qualification': report['qualification']}
        candidates.append(obj)
        trial['status'] = 'proposed'
    report['status'] = 'proposed' if candidates else 'shared_rail_geometry_rejected'
    report['candidates'] = len(candidates)
    return candidates, report
