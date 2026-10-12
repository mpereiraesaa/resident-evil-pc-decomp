"""Selected-only safety veto for optional part fitting of creative wall fixtures."""

import copy

from room3d_depth_guard import check_selected_scene
from room3d_fixture_parts import refine_wall_parts
from room3d_interpretation import door_clear
from room3d_preview import render
from room3d_refine import depth_error


def refine(room, recipe, index, group, depths, images, segments, out, generate_mesh):
    """Optimize training highlights, then either keep one selected assembly or roll back."""
    out.mkdir(parents=True, exist_ok=True)
    original = recipe['primitives'][index]
    stripped = copy.deepcopy(recipe)
    del stripped['primitives'][index]
    fitted, parts = refine_wall_parts(room, stripped, original, group, depths, images,
                                     out/'parts', generate_mesh, creative=True)
    report = {'accepted': False, 'parts': parts, 'raw_depth': [],
        'qualification': 'Post-acceptance creative part fitting. Training brightness selects once; reserved images never optimize it. Final placement, wall contact, door-panel clearance, complete-object silhouette, raw depth and fixed floor-reference checks can veto it without candidate reselection. Not bulb identity, fidelity or navigation proof.'}
    if parts['status'] != 'selected':
        report['reason'] = parts['status']
        return recipe, original, report
    candidate = copy.deepcopy(recipe)
    candidate['primitives'][index] = fitted
    placement_safe = all(s['creative_placement']['valid'] and s['attachment_valid'] for s in parts['after'])
    gates = {'placement_and_attachment': placement_safe, 'door_clearance': door_clear(fitted, recipe)}
    report['gates'] = gates
    if not all(gates.values()):
        report['reason'] = 'placement_attachment_or_door_failed'
        return recipe, original, report
    before, after = out/'before.obj', out/'selected.obj'
    generate_mesh(room, before, recipe)
    generate_mesh(room, after, candidate)
    # Highlights can improve while the support and complete fixture disappear.
    # Veto the selected assembly using the same observed whole-object masks;
    # reserved masks validate once and never initiate another candidate search.
    from room3d_shape_fit import compare

    heldout = set(parts['heldout_cameras'])
    observations = {o['camera']: o for o in group}
    report['whole_silhouette'] = []
    for reserved in (False, True):
        for camera, (target, valid) in zip(room['cameras'], depths, strict=True):
            if (camera['index'] in heldout) != reserved:
                continue
            old_scene = render(before, camera, return_depth=True)[2]
            new_scene = render(after, camera, return_depth=True)[2]
            if camera['index'] in observations:
                mask = observations[camera['index']]['mask']
                scores = []
                for mesh, scene in ((before, old_scene), (after, new_scene)):
                    part = render(mesh, camera, return_depth=True,
                                  only_groups={f'primitive_{index}'})[2]
                    scores.append(compare(part, scene, mask, target, valid)['silhouette_iou'])
                report['whole_silhouette'].append({'camera': camera['index'], 'reserved': reserved,
                    'before': scores[0], 'after': scores[1], 'maximum_drop': .05,
                    'safe': scores[1] + .05 + 1e-12 >= scores[0]})
            old = depth_error(old_scene, target, valid)
            new = depth_error(new_scene, target, valid)
            report['raw_depth'].append({'camera': camera['index'], 'reserved': reserved,
                                        'before': old, 'after': new, 'safe': new <= old+.005})
        if not all(c['safe'] for c in report['whole_silhouette']):
            gates['whole_silhouette'] = False
            report['reason'] = 'reserved_whole_silhouette_regressed' if reserved else 'training_whole_silhouette_regressed'
            return recipe, original, report
        if not all(c['safe'] for c in report['raw_depth']):
            gates['raw_depth'] = False
            report['reason'] = 'reserved_raw_depth_regressed' if reserved else 'training_raw_depth_regressed'
            return recipe, original, report
    gates['raw_depth'] = True
    gates['whole_silhouette'] = True
    floor = check_selected_scene(room, recipe, candidate, segments, depths,
                                 out/'floor-reference', generate_mesh, heldout_cameras=heldout)
    report['floor_reference'] = floor
    gates['floor_reference'] = floor['accepted']
    if not floor['accepted']:
        report['reason'] = floor['reason']
        return recipe, original, report
    report.update(accepted=True, reason='selected_creative_parts_safe', selected_shape=fitted['shape'],
                  placement=[s['creative_placement'] for s in parts['after']])
    return candidate, fitted, report
