"""Opt-in shared ladder reconstruction with transactional scene validation."""
import copy
import json

import cv2
import numpy as np
from room3d_connections import support_mesh
from room3d_interpretation import door_clear
from room3d_ladder_fit import proposals
from room3d_ladder_spacing import refine_spacing
from room3d_ladder_supports import anchor_brackets, closest_points, finite_contact
from room3d_preview import render
from room3d_refine import depth_error
from room3d_shape_fit import associate, observations


def envelope(mask):
    result = np.zeros(mask.shape, 'uint8')
    y, x = np.nonzero(mask)
    if len(x):
        cv2.fillConvexPoly(result, cv2.convexHull(np.column_stack((x, y)).astype('int32')), 1)
    return result.astype(bool)


def mask_score(predicted, target):
    """Filled source envelopes constrain placement, not rung hole topology."""
    hull = envelope(target)
    filled = target.sum()/max(1, hull.sum()) > .75
    if filled:
        predicted = envelope(predicted)
    return {'iou':float((predicted & target).sum()/max(1, (predicted | target).sum())),
            'mode':'filled_envelope' if filled else 'open_geometry'}


def fit_vertical_ladders(room, recipe, segments, depths, out, generate_mesh):
    """Preserve rejected recipes exactly and never reselect after reserved veto."""
    out.mkdir(parents=True, exist_ok=True)
    current = copy.deepcopy(recipe)
    reports = []
    groups = [g for g in associate(observations(room, segments, depths))
              if g[0]['kind']=='staircase' and len(g)>=2]
    for number, group in enumerate(groups):
        dest = out/f'object{number}'
        dest.mkdir()
        training = group[:-1] if len(group)>=3 else group
        reserved = group[len(training):]
        references = [{'camera':o['camera'],'mask':o['mask_name'],'label':o['label'],
                       'confidence_evidence':copy.deepcopy(o.get('confidence_evidence', []))}
                      for o in group]
        report = {'status':'no_shared_rail_proposal','accepted':False,
                  'training_cameras':[o['camera'] for o in training],
                  'heldout_cameras':[o['camera'] for o in reserved],
                  'observations':references,'trials':[],
                  'qualification':'Estimated vertical ladder; source-mask representation is explicit. Hidden geometry, provisional wall intersections, mechanical support, original fidelity and navigation unverified.'}
        reports.append(report)
        hypotheses, initialization = proposals(group, floor_y=current.get('floor_y', 0))
        report['initialization'] = initialization
        if not hypotheses:
            continue
        keys = {(o['camera'], name) for o in group for name in o.get('mask_names', [o['mask_name']])}
        if any(p.get('kind')=='ladder' and any((e['camera'], e['mask']) in keys
               for e in p.get('evidence', {}).get('shared_observations', []))
               for p in current.get('primitives', [])):
            report['status']='already_present'
            continue
        matched = [i for i, p in enumerate(current.get('primitives', []))
                   if p.get('kind')=='staircase' and 'camera' in p.get('evidence', {})
                   and ((p['evidence']['camera'], p['evidence'].get('mask')) in keys
                        or any((e['camera'], e['mask']) in keys for e in p['evidence'].get('shared_observations', [])))]
        before = dest/'before.obj'
        generate_mesh(room, before, current)
        stripped = copy.deepcopy(current)
        kept = [i for i in range(len(current.get('primitives', []))) if i not in matched]
        remap = {f'primitive_{old}':f'primitive_{new}' for new, old in enumerate(kept)}
        stripped['primitives'] = [stripped['primitives'][i] for i in kept]
        stripped['hide'] = [[c, remap.get(g, g)] for c, g in stripped.get('hide', [])
                            if not g.startswith('primitive_') or g in remap]
        support_path = dest/'supports.obj'
        generate_mesh(room, support_path, stripped)
        triangles, surface_groups = support_mesh(support_path)
        walls = {s['group'] for s in stripped.get('structural_surfaces', []) if s['material']=='wall'}
        walls |= {f"boundary_{b['index']}" for b in room['boundaries'] if b['flags'] & 256}
        floors = [t for t, g in zip(triangles, surface_groups, strict=True) if g=='floor']
        eligible = []
        for index, hypothesis in enumerate(hypotheses):
            candidate, spacing = refine_spacing(room, hypothesis, training, dest/f'spacing{index}',
                                                 generate_mesh, reserve_last=False)
            trial = {'index':index,'spacing':spacing,'eligible':False}
            report['trials'].append(trial)
            if (not any(e['status']=='estimated_spacing' for e in spacing['evidence'])
                    or spacing['status']=='inconsistent_training_spacing'):
                continue
            candidate, support = anchor_brackets(candidate, triangles, surface_groups, walls)
            trial['support'] = support
            if support['status']!='anchored':
                continue
            isolated = dest/f'candidate{index}.obj'
            generate_mesh(room, isolated, {'primitives':[candidate],
                'floor_y':stripped.get('floor_y', 0),
                'skip_boundaries':[b['index'] for b in room['boundaries']]})
            scores, old_scores = [], []
            for observation in training:
                predicted = render(isolated, observation['camera_model'], only_groups={'primitive_0'})[1]
                scores.append(mask_score(predicted, observation['mask']))
                old_mask = render(before, observation['camera_model'], only_groups={f'primitive_{i}' for i in matched})[1]
                old_scores.append(mask_score(old_mask, observation['mask']))
            keep = (min(s['iou'] for s in scores)>=.3
                    and all(a['iou']>=b['iou']-.03 for a, b in zip(scores, old_scores, strict=True)))
            trial.update(training_scores=scores, previous_scores=old_scores, eligible=bool(keep))
            if keep:
                eligible.append((float(np.mean([s['iou'] for s in scores])), index, candidate, isolated))
        if not eligible:
            report['status']='no_supported_training_candidate'
            continue
        _, index, candidate, isolated = max(eligible, key=lambda value:value[0])
        report['selected_training_candidate']=index
        # The choice is frozen here. Reserved masks and all-camera scene gates
        # can veto it, never select another initialization.
        reserved_scores = [mask_score(render(isolated, o['camera_model'], only_groups={'primitive_0'})[1],
                                       o['mask']) for o in reserved]
        candidate['evidence'] = {'camera':group[0]['camera'],'mask':group[0]['mask_name'],
                                 'shared_observations':references}
        candidate['interpretation'] = {'source':'shared_vertical_rail_prior',
                                       'qualification':report['qualification']}
        selected = copy.deepcopy(stripped)
        selected['primitives'].append(candidate)
        after = dest/'selected.obj'
        generate_mesh(room, after, selected)
        depth_views = []
        scenes = {}
        for camera, (target, valid) in zip(room['cameras'], depths, strict=True):
            a = render(before, camera, return_depth=True)[2]
            b = render(after, camera, return_depth=True)[2]
            old_error, new_error = depth_error(a, target, valid), depth_error(b, target, valid)
            depth_views.append({'camera':camera['index'],'before':old_error,'after':new_error,
                                'safe':bool(new_error<=old_error+.005)})
            scenes[camera['index']] = b
        visibility = []
        group_name = f"primitive_{len(selected['primitives'])-1}"
        for observation in group:
            part = render(after, observation['camera_model'], return_depth=True, only_groups={group_name})[2]
            scene = scenes[observation['camera']]
            visible = np.isfinite(part) & (part<=scene+.001)
            visibility.append(float(visible.sum()/max(1, np.isfinite(part).sum())))
        foot_contact = bool(floors) and all(np.linalg.norm(closest_points(p, floors)-p, axis=1).min()<=10
                                            for part in np.asarray(candidate['parts'][:2]) for p in part[:4])
        gates = {'reserved_masks':all(s['iou']>=.3 for s in reserved_scores),
                 'scene_depth':all(v['safe'] for v in depth_views),
                 'finite_wall_contact':finite_contact(candidate, triangles, surface_groups),
                 'finite_floor_contact':bool(foot_contact),'door_clearance':bool(door_clear(candidate, stripped)),
                 'observed_visibility':min(visibility)>=.5}
        report.update(reserved_scores=reserved_scores, depth_views=depth_views,
                      observed_visible_fractions=visibility,gates=gates,matched_inferred_stairs=matched)
        if all(gates.values()):
            report.update(status='estimated_ladder_retained',accepted=True)
            current = selected
        else:
            report['status']='selected_candidate_vetoed'
    result = {'objects':reports,'accepted':sum(r['accepted'] for r in reports),
              'qualification':'Opt-in estimated ladders; accepted scene gates do not constitute a whole-room visual pass or navigation validation.'}
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    (out/'index.html').write_text('<meta charset="utf-8"><title>Vertical ladder trials</title>'
        '<h1>Vertical ladder trials</h1><p>Estimated geometry; inspect final room views.</p>'
        '<p><a href="results.json">Training selection, reserved vetoes and scene gates</a></p>')
    return current, result
