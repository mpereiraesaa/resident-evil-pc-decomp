"""Bounded visual occlusion correction for an already fitted raised staircase.

Selection uses training views only. The caller must apply final scene gates.
Original collision records are not changed.
"""
import copy

import numpy as np


def refine_raised_occlusion(room, recipe, obj, report, group, depths, out, generate_mesh):
    from room3d_preview import render
    from room3d_refine import depth_error
    from room3d_shape_fit import compare, tread_edge_error
    from room3d_supports import occlusion_hypotheses

    training = group[:-1] if len(group) >= 3 else group
    cameras = {c['index']: (c, d) for c, d in zip(room['cameras'], depths, strict=True)}
    out.mkdir(parents=True, exist_ok=True)
    def evaluate(candidate, name):
        combined = copy.deepcopy(candidate)
        combined.setdefault('primitives', []).append(obj)
        path = out / f'{name}.obj'
        generate_mesh(room, path, combined)
        scores, errors = [], []
        for obs in group:
            camera, (target, valid) = cameras[obs['camera']]
            scene = render(path, camera, return_depth=True)[2]
            part = render(path, camera, return_depth=True,
                          only_groups={f"primitive_{len(combined['primitives'])-1}"})[2]
            score = compare(part, scene, obs['mask'], target, valid)
            detail = tread_edge_error(obj, camera, obs, part, scene)
            if detail is not None:
                score['tread_edge_error'] = detail
                score['loss'] = .8 * score['loss'] + .2 * detail
            scores.append(score)
            errors.append(depth_error(scene, target, valid))
        return scores, errors
    before, old_errors = evaluate(recipe, 'baseline')
    n = len(training)
    best, best_scores = recipe, before
    best_loss = float(np.mean([s['loss'] for s in before[:n]]) + .2*np.mean(old_errors[:n]))
    trials = []
    for i, (candidate, evidence) in enumerate(occlusion_hypotheses(recipe, report, training)):
        scores, errors = evaluate(candidate, f'alternative{i}')
        # Do not trade a view's agreement/depth for another training view.
        safe = all(a <= b+.005 for a, b in zip(errors[:n], old_errors[:n], strict=True))
        consistent = all(a['loss'] <= b['loss']+.03
                         for a, b in zip(scores[:n], before[:n], strict=True))
        improved = np.mean([s['loss'] for s in scores[:n]]) < np.mean([s['loss'] for s in before[:n]])-.02
        loss = float(np.mean([s['loss'] for s in scores[:n]]) + .2*np.mean(errors[:n]))
        keep = bool(safe and consistent and improved and loss < best_loss)
        trials.append({'proposal': evidence, 'scores': scores, 'depth': errors,
                       'training_depth_safe': safe, 'training_consistent': consistent,
                       'eligible': bool(safe and consistent and improved)})
        if keep:
            best, best_scores, best_loss = candidate, scores, loss
    return best, best_scores, {
        'selected': best is not recipe, 'before': before, 'before_depth': old_errors,
        'selection_loss': best_loss,
        'training_cameras': [o['camera'] for o in training],
        'heldout_cameras': [o['camera'] for o in group[n:]], 'trials': trials,
        'qualification': 'Training-view visual correction; final all-camera scene gates required',
    }


def refine_support_candidates(room, solutions, group, depths, out, generate_mesh):
    """Correct up to two raised alternatives before choosing floor versus raised."""
    expanded = list(solutions)
    searches = []
    ranked = sorted((s for s in solutions if s[4]['kind'] != 'floor'), key=lambda s: s[0])[:2]
    for index, (_, obj, report, recipe, support) in enumerate(ranked):
        candidate, _, correction = refine_raised_occlusion(
            room, recipe, obj, report, group, depths, out / f'candidate{index}', generate_mesh,
        )
        searches.append({'support': support, **correction})
        if correction['selected']:
            corrected_report = copy.deepcopy(report)
            corrected_report['raised_support_occlusion'] = correction
            expanded.append((correction['selection_loss'], obj, corrected_report, candidate, support))
    return expanded, searches
