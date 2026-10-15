"""Common-region render comparison for experimental attachment alternatives."""
import html
import json

import cv2
import numpy as np
from room3d_preview import render
from room3d_reconstruct import bake_projected


def rank_alternatives(scores, rgb_allowance=1 / 255, edge_allowance=.1):
    """Recommend only bounded per-view regressions; no independent fidelity claim."""
    for views in scores.values():
        for view in views:
            if not all(np.isfinite(view[key]) for key in ('rgb_mae', 'edge_distance', 'pixels', 'uncovered')):
                raise ValueError('Nonfinite render metrics')
    baseline = scores['baseline']
    decisions = []
    for name, views in scores.items():
        if name == 'baseline':
            continue
        reasons, improvements = [], []
        for old, new in zip(baseline, views, strict=True):
            if old['pixels'] < 64:
                continue
            if new['pixels'] != old['pixels'] or new['camera'] != old['camera']:
                raise ValueError('Candidates must use the same camera regions')
            delta = new['rgb_mae'] - old['rgb_mae']
            edge_delta = new['edge_distance'] - old['edge_distance']
            improvements.append(-delta)
            if delta > rgb_allowance or edge_delta > edge_allowance or new['uncovered'] > old['uncovered']:
                reasons.append(f"camera{old['camera']}_regression")
        if not improvements:
            reasons.append('insufficient_affected_pixels')
        elif max(improvements) <= rgb_allowance:
            reasons.append('no_material_image_improvement')
        decisions.append({'name': name, 'eligible': not reasons, 'reasons': reasons,
                          'mean_rgb_improvement': float(np.mean(improvements)) if improvements else 0.})
    eligible = [d for d in decisions if d['eligible']]
    best = max(eligible, key=lambda d: d['mean_rgb_improvement'])['name'] if eligible else None
    return {'recommended_candidate': best, 'decisions': decisions,
            'qualification': 'Source-view diagnostic recommendation; not held-out or 3D validation',
            'thresholds': {'rgb_allowance': rgb_allowance, 'edge_allowance_pixels': edge_allowance}}


def evaluate_attachments(room, baseline, alternatives, images, segments, depths, out, generate_mesh):
    """Bake/render all variants and score one fixed region per original camera."""
    out.mkdir(parents=True, exist_ok=True)
    renders = {}
    for name, recipe in [('baseline', baseline)] + [(a['name'], a['recipe']) for a in alternatives]:
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('Invalid candidate name')
        folder = out / name
        folder.mkdir(exist_ok=True)
        mesh = folder / 'geometry.obj'
        generate_mesh(room, mesh, recipe)
        kinds = {f"boundary_{b['index']}": 'wall' for b in room['boundaries'] if b['flags'] & 0x100}
        kinds.update({f'primitive_{i}': p['kind'] for i, p in enumerate(recipe.get('primitives', []))})
        kinds.update({s['group']: s['material'] for s in recipe.get('structural_surfaces', [])})
        kinds['floor'] = 'floor'
        bake = bake_projected(mesh, room, images, folder / 'appearance.obj', segments=segments,
                              group_kinds=kinds, observed_depths=depths)
        (folder / 'bake.json').write_text(json.dumps(bake, indent=2))
        renders[name] = []
        for camera in room['cameras']:
            image, covered, depth = render(folder / 'appearance.obj', camera, return_depth=True)
            image.save(folder / f"{camera['index']}.png")
            renders[name].append((np.asarray(image), covered, depth))
    scores = {name: [] for name in renders}
    rows = ['<!doctype html><meta charset="utf-8"><style>body{background:#222;color:#eee}img{width:320px}</style>',
            '<h1>Attachment alternatives</h1><p>Original / common scoring region / baseline / alternatives. Source images also supply textures; this is not independent validation.</p>']
    for i, (camera, original) in enumerate(zip(room['cameras'], images, strict=True)):
        _base_image, base_covered, base_depth = renders['baseline'][i]
        region = np.zeros(base_covered.shape, bool)
        for name in renders:
            if name == 'baseline':
                continue
            _, covered, depth = renders[name][i]
            shared = covered & base_covered
            difference = np.zeros(region.shape, bool)
            difference[shared] = np.abs(depth[shared] - base_depth[shared]) > 2
            region |= difference | (covered != base_covered)
        region = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        reference = np.asarray(original.convert('RGB'))
        reference_edges = cv2.Canny(reference, 80, 160).astype(bool) & region
        original.save(out / f"original{camera['index']}.png")
        overlay = reference.copy()
        overlay[region] = (overlay[region].astype(float) * .5 + np.array([127, 0, 0])).astype(np.uint8)
        from PIL import Image
        Image.fromarray(overlay).save(out / f"region{camera['index']}.png")
        for name, views in renders.items():
            pixels, covered, _ = views[i]
            edges = cv2.Canny(pixels, 80, 160) > 0
            distance = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)
            scores[name].append({'camera': camera['index'], 'pixels': int(region.sum()),
                                 'uncovered': int((region & ~covered).sum()),
                                 'rgb_mae': float(np.abs(reference.astype(float) - pixels)[region].mean() / 255) if region.any() else 0.,
                                 'edge_distance': float(distance[reference_edges].mean()) if reference_edges.any() else 0.})
        rows.append(f'<h2>Camera {camera["index"]}</h2><img src="original{camera["index"]}.png"><img src="region{camera["index"]}.png">'
                    + ''.join(f'<img alt="{html.escape(name)}" src="{name}/{camera["index"]}.png">' for name in renders))
    report = {'scores': scores, **rank_alternatives(scores), 'applied': False}
    (out / 'evaluation.json').write_text(json.dumps(report, indent=2))
    rows.append('<p>Recommendation: ' + html.escape(str(report['recommended_candidate'])) + '</p>')
    (out / 'index.html').write_text('\n'.join(rows))
    return report
