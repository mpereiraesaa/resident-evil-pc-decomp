"""Infer repeated rung spacing from open masks and refine actual raster geometry.

Filled masks supply no count evidence. Partial masks may omit one rung; the
result is a regular structural hypothesis, not recovered hidden geometry.
"""
import copy
import math

import cv2
import numpy as np
from room3d_preview import camera_basis, render
from room3d_shapes import generate


def sampling_grid(obj, camera):
    """A narrow central strip avoids rails and tolerates modest plane error."""
    shape, p = obj['shape'], obj['shape']['parameters']
    c, s = math.cos(shape['yaw']), math.sin(shape['yaw'])
    rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    x, y = np.meshgrid(np.linspace(-p['width']*.08, p['width']*.08, 64),
                       -np.linspace(0, p['height'], 512))
    points = np.stack((x, y, np.full(x.shape, p['stand_off'])), axis=-1)
    local = (points@rotation+shape['origin']-camera['origin']) @ camera_basis(camera).T
    depth = local[..., 2]
    uv = np.divide(local[..., :2]*camera['fov'], depth[..., None],
                   out=np.zeros_like(local[..., :2]), where=abs(depth[..., None])>1e-8)+[160, 120]
    valid = ((depth > 2*camera['fov']) & (uv[..., 0] >= 0) & (uv[..., 0] <= 319)
             & (uv[..., 1] >= 0) & (uv[..., 1] <= 239))
    return uv.astype('float32'), valid


def rectify(mask, grid):
    uv, valid = grid
    return cv2.remap(mask.astype('float32'), uv[..., 0], uv[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), valid


def spacing_evidence(obj, observation):
    mask = np.asarray(observation['mask'])
    if mask.shape != (240, 320) or mask.dtype != bool:
        raise ValueError('Expected a native boolean mask')
    grid = sampling_grid(obj, observation['camera_model'])
    strip, valid = rectify(mask, grid)
    count = valid.sum(1)
    profile = np.divide((strip*valid).sum(1), count, out=np.zeros(512), where=count>0)
    observed = count >= 32
    high = observed & (profile > .65)
    bounds = np.flatnonzero(np.diff(np.r_[False, high, False])).reshape(-1, 2)
    centers = [(a+b-1)/2/511 for a, b in bounds
               if 2 <= b-a <= 40 and a > 0 and b < 512]
    report = {'camera': observation['camera'], 'status': 'insufficient_open_rungs',
              'centers': centers, 'observed_rows': int(observed.sum()),
              'qualification': 'Repeated central-strip mask occupancy; filled envelopes, missing rungs and hidden extent are not measured identities.'}
    if len(centers) < 3 or np.count_nonzero(observed & (profile < .2)) < 20:
        return None, report
    gaps = np.diff(centers)
    periods = []
    for period in sorted({float(gap/k) for gap in gaps for k in (1, 2)}, reverse=True):
        multiples = np.rint(gaps/period).astype(int)
        if (multiples.min() < 1 or multiples.max() > 2 or not np.any(multiples == 1)
                or np.max(abs(gaps/period-multiples)) > .15):
            continue
        spacing = float(np.mean(gaps/multiples))
        rungs = round(1/spacing)-1
        if not 2 <= rungs <= 64:
            continue
        phase = float(np.median(np.array(centers)*(rungs+1)
                                - np.rint(np.array(centers)*(rungs+1))))
        if not -.45 <= phase <= .45:
            continue
        periods.append((spacing, rungs, phase, multiples.tolist()))
    if not periods:
        report['status'] = 'irregular_spacing'
        return None, report
    spacing, rungs, phase, multiples = max(periods)
    report.update(status='estimated_spacing', normalized_spacing=spacing,
                  inferred_rungs=rungs, inferred_phase=phase, gap_multiples=multiples)
    return {'grid': grid, 'target': strip, 'valid': valid,
            'rungs': rungs, 'phase': phase}, report


def refine_spacing(room, obj, observations, out, generate_mesh):
    """Select count/phase on open training masks; reserve the final view.

    This is initialization refinement only. Scene depth and finite support
    still need validation before an automatic room replacement is accepted.
    """
    if obj.get('kind') != 'ladder':
        raise ValueError('Expected a generated ladder')
    ids = [o['camera'] for o in observations]
    if len(ids) != len(set(ids)):
        raise ValueError('Expected distinct observations')
    training = observations[:-1] if len(observations) >= 3 else observations
    reserved = observations[len(training):]
    report = {'status': 'insufficient_open_training_masks',
              'training_cameras': [o['camera'] for o in training],
              'heldout_cameras': [o['camera'] for o in reserved], 'evidence': [], 'trials': [],
              'qualification': 'Regular rung count/phase hypothesis from open training masks. Missing intervals, hidden extent, support, scene depth and independent-room fidelity unverified.'}
    if len(training) < 2:
        return obj, report
    active = []
    for observation in training:
        evidence, record = spacing_evidence(obj, observation)
        report['evidence'].append(record)
        if evidence is not None:
            active.append((observation, evidence))
    if not active:
        return obj, report
    counts = [e['rungs'] for _, e in active]
    if max(counts)-min(counts) > 1:
        report['status'] = 'inconsistent_training_spacing'
        return obj, report
    out.mkdir(parents=True, exist_ok=True)

    def evaluate(candidate, views, path):
        parts = [{'kind': 'ladder', 'parts': [part]} for part in candidate['parts']]
        generate_mesh(room, path, {'primitives': parts,
                                  'skip_boundaries': [b['index'] for b in room['boundaries']]})
        groups = {f'primitive_{i}' for i in range(2, 2+candidate['shape']['parameters']['rungs'])}
        scores = []
        for observation, evidence in views:
            raster = render(path, observation['camera_model'], only_groups=groups)[1]
            predicted, valid = rectify(raster, evidence['grid'])
            target = evidence['target']
            a, b = predicted[valid], target[valid]
            scores.append(float(2*np.minimum(a, b).sum()/max(a.sum()+b.sum(), 1e-8)))
        return scores

    baseline = evaluate(obj, active, out/'baseline.obj')
    inferred = round(float(np.mean(counts)))
    eligible = []
    for count in range(max(2, inferred-1), min(64, inferred+1)+1):
        for phase in np.linspace(-.4, .4, 9):
            p = {**obj['shape']['parameters'], 'rungs': count, 'rung_phase': float(phase)}
            if p['height'] <= 2*p['thickness']*(count+1):
                continue
            candidate = generate('ladder', p, obj['shape']['origin'], obj['shape']['yaw'])
            scores = evaluate(candidate, active, out/f'count{count}-phase{phase:+.1f}.obj')
            keep = bool(np.mean(scores) > np.mean(baseline)+.02
                        and all(a >= b-.03 for a, b in zip(scores, baseline, strict=True)))
            report['trials'].append({'rungs': count, 'phase': float(phase),
                                     'training_scores': scores, 'eligible': keep})
            if keep:
                eligible.append((float(np.mean(scores)), candidate, scores))
    report['baseline_scores'] = baseline
    if not eligible:
        report['status'] = 'no_consistent_training_improvement'
        return obj, report
    _, selected, scores = max(eligible, key=lambda value: value[0])
    report.update(selected_parameters=selected['shape']['parameters'], selected_training_scores=scores)
    if reserved:
        heldout = []
        for observation in reserved:
            evidence, record = spacing_evidence(obj, observation)
            report.setdefault('reserved_evidence', []).append(record)
            if evidence is None:
                report['status'] = 'reserved_spacing_unavailable'
                return obj, report
            heldout.append((observation, evidence))
        old = evaluate(obj, heldout, out/'heldout-baseline.obj')
        new = evaluate(selected, heldout, out/'heldout-selected.obj')
        report['heldout_scores'] = {'baseline': old, 'selected': new}
        if any(a < b-.03 for a, b in zip(new, old, strict=True)):
            report['status'] = 'reserved_spacing_failed'
            return obj, report
    for key in ('initialization', 'ladder_evidence'):
        if key in obj:
            selected[key] = copy.deepcopy(obj[key])
    report['status'] = 'selected'
    selected['spacing_fit'] = copy.deepcopy(report)
    return selected, report
