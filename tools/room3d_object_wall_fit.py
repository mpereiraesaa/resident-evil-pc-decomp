"""Transactional training-render selection of local wall retreats.

Retained geometry remains estimated. Reserved masks and fixed-support depth can
only veto the selected candidate; they never choose a different wall location.
"""
import copy
import hashlib
import json

import numpy as np
from room3d_connections import lamp_attachment, support_mesh
from room3d_ladder_supports import closest_points, finite_contact
from room3d_preview import render
from room3d_refine import depth_error
from room3d_segment import matches_material
from room3d_vertical_ladders import mask_score
from room3d_wall_recess import proposals, refit_support


def source_observations(room, obj, segments):
    wanted = {(e['camera'], name)
              for e in obj.get('evidence', {}).get('shared_observations', [])
              for name in {e['mask'], *(c['mask'] for c in e.get('confidence_evidence', []))}}
    result = []
    for camera, entries in zip(room['cameras'], segments, strict=True):
        masks, evidence = [], []
        for item, mask in entries:
            scores = [item['score'], item['predicted_iou']]
            if ((camera['index'], item['mask']) not in wanted
                    or not matches_material(item['label'].strip().casefold(), ('staircase', 'ladder'))
                    or not np.isfinite(scores).all() or not all(0 <= s <= 1 for s in scores)
                    or scores[0] < .3 or scores[1] < .8):
                continue
            if mask.shape != (240, 320) or mask.dtype != bool:
                raise ValueError('Expected native boolean source mask')
            masks.append(mask)
            evidence.append({'mask':item['mask'],'score':scores[0], 'predicted_iou':scores[1],
                             'pixels_sha256':hashlib.sha256(mask.tobytes()).hexdigest()})
        if masks:
            result.append({'camera':camera,'mask':np.logical_or.reduce(masks),'evidence':evidence})
    return result


def measure(mesh, primitive_index, observations):
    result = []
    for observation in observations:
        camera = observation['camera']
        scene = render(mesh, camera, return_depth=True)[2]
        part = render(mesh, camera, return_depth=True,
                      only_groups={f'primitive_{primitive_index}'})[2]
        visible = np.isfinite(part) & (part <= scene+.001)
        result.append({'camera':camera['index'],
                       'visible_fraction':float(visible.sum()/max(1, np.isfinite(part).sum())),
                       'mask_score':mask_score(visible, observation['mask'])})
    return result


def training_score(rows):
    return float(np.mean([r['mask_score']['iou']+.25*r['visible_fraction'] for r in rows]))


def no_view_regression(after, before):
    return all(a['mask_score']['iou'] >= b['mask_score']['iou']-.03
               and a['visible_fraction'] >= b['visible_fraction']-.02
               for a, b in zip(after, before, strict=True))


def support_gates(before_recipe, after_recipe, before_mesh, after_mesh, target_index):
    old_triangles, old_groups = support_mesh(before_mesh)
    triangles, groups = support_mesh(after_mesh)
    walls = {s['group'] for s in before_recipe.get('structural_surfaces', []) if s['material']=='wall'}
    walls |= {g for g in groups if g.startswith('boundary_')}
    contacts = []
    for old, new in zip(before_recipe['primitives'], after_recipe['primitives'], strict=True):
        if old.get('omit_geometry'):
            continue
        if old.get('kind')=='ladder' and finite_contact(old, old_triangles, old_groups):
            contacts.append(finite_contact(new, triangles, groups))
        if old.get('kind')=='lamp' and old.get('shape') and old.get('parts'):
            a = lamp_attachment(old, old_triangles, old_groups, walls)
            b = lamp_attachment(new, triangles, groups, walls)
            if a['within_tolerance']:
                contacts.append(b['within_tolerance'])
    target = after_recipe['primitives'][target_index]
    floors = [t for t, g in zip(triangles, groups, strict=True) if g=='floor']
    floor_contact = bool(floors) and all(
        np.linalg.norm(closest_points(p, floors)-p, axis=1).min() <= 10
        for part in np.asarray(target['parts'][:2]) for p in part[:4])
    return {'finite_wall_contact':finite_contact(target, triangles, groups),
            'finite_floor_contact':bool(floor_contact),
            'existing_supported_objects':all(contacts)}


def fit(room, recipe, segments, depths, out, generate_mesh):
    out.mkdir(parents=True, exist_ok=True)
    current = copy.deepcopy(recipe)
    reports = []
    for index in range(len(current.get('primitives', []))):
        obj = current['primitives'][index]
        if (obj.get('kind') != 'ladder' or obj.get('omit_geometry')
                or obj.get('interpretation', {}).get('source') != 'shared_vertical_rail_prior'):
            continue
        report = {'primitive':index,'status':'insufficient_source_views','accepted':False,'trials':[]}
        reports.append(report)
        if obj.get('wall_recess_fit'):
            report['status'] = 'already_present'
            continue
        observations = source_observations(room, obj, segments)
        report['sources'] = [{'camera':o['camera']['index'],'masks':o['evidence']} for o in observations]
        if len(observations) < 2:
            continue
        training = observations[:-1] if len(observations)>=3 else observations
        reserved = observations[len(training):]
        report.update(training_cameras=[o['camera']['index'] for o in training],
                      heldout_cameras=[o['camera']['index'] for o in reserved])
        dest = out/f'primitive{index}'
        dest.mkdir()
        before_mesh = dest/'before.obj'
        generate_mesh(room, before_mesh, current)
        before = measure(before_mesh, index, training)
        report['before_training'] = before
        eligible = []
        candidates = proposals(room, current, index,
                               [o['camera']['origin'] for o in training], local=True)
        for number, (candidate, evidence) in enumerate(candidates):
            mesh = dest/f'trial{number}.obj'
            generate_mesh(room, mesh, candidate)
            triangles, groups = support_mesh(mesh)
            support = [(t, g) for t, g in zip(triangles, groups, strict=True)
                       if g != f'primitive_{index}']
            attached, attachment = refit_support(obj, [t for t, _ in support],
                [g for _, g in support], set(obj['wall_attachment']['bracket_groups']))
            trial = {'index':number,'proposal':evidence,'attachment':attachment,'eligible':False}
            report['trials'].append(trial)
            if attachment['status'] != 'anchored':
                continue
            candidate['primitives'][index] = attached
            generate_mesh(room, mesh, candidate)
            rows = measure(mesh, index, training)
            keep = (no_view_regression(rows, before)
                    and any(a['visible_fraction'] >= b['visible_fraction']+.1
                            for a, b in zip(rows, before, strict=True))
                    and training_score(rows) >= training_score(before)+.02)
            trial.update(training=rows, eligible=bool(keep))
            if keep:
                eligible.append((training_score(rows), number, candidate, mesh))
        if not eligible:
            report['status'] = 'no_training_improvement'
            continue
        _, number, selected, mesh = max(eligible, key=lambda candidate:candidate[0])
        # Freeze the choice before reserved evidence or scene-level vetoes.
        report['selected_training_candidate'] = number
        reserved_before = measure(before_mesh, index, reserved)
        reserved_after = measure(mesh, index, reserved)
        views = []
        for camera, (target, valid) in zip(room['cameras'], depths, strict=True):
            a = depth_error(render(before_mesh, camera, return_depth=True)[2], target, valid)
            b = depth_error(render(mesh, camera, return_depth=True)[2], target, valid)
            views.append({'camera':camera['index'],'before':a,'after':b,'safe':bool(b<=a+.005)})
        gates = support_gates(current, selected, before_mesh, mesh, index)
        gates.update(reserved_masks=no_view_regression(reserved_after, reserved_before),
                     scene_depth=all(v['safe'] for v in views))
        report.update(heldout_before=reserved_before,heldout_after=reserved_after,
                      depth_views=views,gates=gates)
        if all(gates.values()):
            report.update(status='estimated_wall_retreat_retained',accepted=True)
            selected['primitives'][index]['wall_recess_fit'] = {
                'source':'bounded_object_wall_retreat','selected_training_candidate':number,
                'qualification':'Estimated local wall retreat; hidden geometry, fidelity and navigation unverified'}
            current = selected
        else:
            report['status'] = 'selected_candidate_vetoed'
    result = {'objects':reports,'accepted':sum(r['accepted'] for r in reports),
              'qualification':'Opt-in estimated visual wall fitting; source collision unchanged. Numerical acceptance is not a whole-room visual pass or independent validation.'}
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    (out/'index.html').write_text('<meta charset="utf-8"><title>Object wall fitting</title>'
        '<h1>Estimated object wall fitting</h1><p>Inspect final room renders.</p>'
        '<a href="results.json">Training choice, reserved vetoes and scene gates</a>')
    return current, result
