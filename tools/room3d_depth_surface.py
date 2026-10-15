"""Editable single-view depth surface draft; never a completed volumetric room."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from room3d_preview import camera_basis, render
from room3d_refine import load_depth


def build_surface(camera, depth, reliable, image, out, step=4, maximum_jump=.15):
    """Unproject a bounded grid; separate uncertain faces and reject depth jumps."""
    depth, reliable = np.asarray(depth, float), np.asarray(reliable, bool)
    if (depth.shape != (240, 320) or reliable.shape != depth.shape
            or image.size != (320, 240) or not np.isfinite(depth).all() or not (depth > 0).all()):
        raise ValueError("Expected native image and finite positive 320x240 depth")
    if isinstance(step, bool) or int(step) != step or not 2 <= step <= 32:
        raise ValueError("Grid step must be an integer from two to 32")
    if not np.isfinite(maximum_jump) or not 0 < maximum_jump <= .5:
        raise ValueError("Relative depth jump must be positive and at most 0.5")
    if out.exists():
        raise FileExistsError(out)
    basis = camera_basis(camera)
    if not np.isfinite(basis).all() or not np.isfinite(camera['origin']).all():
        raise ValueError("Expected a finite camera pose")
    focal = float(camera['fov'])
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("Expected positive focal length")
    xs = np.unique(np.r_[np.arange(0, 320, step), 320])
    ys = np.unique(np.r_[np.arange(0, 240, step), 240])
    xx, yy = np.meshgrid(xs, ys)
    ix, iy = np.minimum(xx, 319).astype(int), np.minimum(yy, 239).astype(int)
    z = depth[iy, ix]
    rays = np.stack([(xx - 160) / focal, (yy - 120) / focal, np.ones_like(xx)], axis=-1)
    points = (rays * z[..., None]) @ basis + np.asarray(camera['origin'])
    vertices = points.reshape(-1, 3)
    reliability = reliable[iy, ix].ravel()
    zs = z.ravel()
    faces, rejected, uncertain = [], 0, 0
    width = len(xs)
    for row in range(len(ys) - 1):
        for col in range(width - 1):
            a = row * width + col
            for indices in ((a, a + 1, a + width), (a + 1, a + width + 1, a + width)):
                values = zs[list(indices)]
                if values.max() / values.min() - 1 > maximum_jump:
                    rejected += 1
                    continue
                known = bool(reliability[list(indices)].all())
                uncertain += not known
                faces.append((indices, known))
    if not faces:
        raise ValueError("No connected surface survives the depth discontinuity check")
    out.mkdir(parents=True)
    image.save(out / 'surface.png')
    (out / 'surface.mtl').write_text(''.join(
        f'newmtl {name}\nKd 1 1 1\nmap_Kd surface.png\n' for name in ('reliable_proposal', 'uncertain_proposal')))
    lines = ['# OPEN SINGLE-VIEW SURFACE DRAFT; hidden geometry and traversal unresolved', 'mtllib surface.mtl']
    lines += ['v ' + ' '.join(format(float(v), '.12g') for v in point) for point in vertices]
    lines += [f'vt {(float(x)-.5)/319:.12g} {1-(float(y)-.5)/239:.12g}'
              for x, y in zip(xx.ravel(), yy.ravel(), strict=True)]
    for indices, known in faces:
        name = 'reliable_proposal' if known else 'uncertain_proposal'
        lines.extend(['g depth_surface_' + name, 'usemtl ' + name,
                      'f ' + ' '.join(f'{i+1}/{i+1}' for i in indices)])
    path = out / 'draft.obj'
    path.write_text('\n'.join(lines) + '\n')
    preview, mask = render(path, camera)
    preview.save(out / 'preview.png')
    return {'status': 'single_view_surface_draft', 'vertices': len(vertices),
            'triangles': len(faces), 'uncertain_faces': uncertain,
            'rejected_discontinuity_faces': rejected, 'screen_coverage': float(mask.mean()),
            'hidden_geometry': 'not_reconstructed', 'navigation': 'not_validated',
            'reliability_meaning': 'Model confidence only; no measured or independently verified surface',
            'qualification': 'Reprojection uses its source image; agreement is not independent fidelity evidence. Open surfaces are a modeling guide, not a volumetric room'}


def export(survey, images_root, depth_root, out):
    if any(out.resolve().is_relative_to(p.resolve()) for p in (images_root, depth_root, survey.parent)):
        raise ValueError("Keep the draft outside all input folders")
    room = json.loads(survey.read_text())
    if len(room['cameras']) != 1:
        raise ValueError("This draft exporter requires exactly one camera")
    camera = room['cameras'][0]
    image = Image.open(images_root / f"camera{camera['index']}-original.png").convert('RGB')
    depth, reliable = load_depth(depth_root, survey, room, [image])[0]
    record = build_surface(camera, depth, reliable, image, out)
    record['scale'] = json.loads((depth_root / 'depth.json').read_text())['scale']
    (out / 'draft.json').write_text(json.dumps(record, indent=2))
    moved = {**camera, 'origin': (np.asarray(camera['origin']) + camera_basis(camera)[0]
                                  * float(np.median(depth)) * .1).tolist()}
    render(out / 'draft.obj', moved)[0].save(out / 'shifted-view.png')
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Single-view surface draft</title>'
                                  '<style>body{background:#17191d;color:#ddd;font-family:sans-serif}img{width:320px}</style>'
                                  '<h1>Superficie provisional de una sola vista</h1><p>Original / reproyección / cámara desplazada. '
                                  'No es una habitación volumétrica: faltan caras ocultas, colisiones y validación de recorrido. '
                                  'La escala es una estimación para editar; la reproyección no mide fidelidad independiente.</p>'
                                  '<img src="surface.png"><img src="preview.png"><img src="shifted-view.png">'
                                  '<p><a href="draft.json">Limitaciones y métricas</a> · <a href="draft.obj">Malla editable</a></p>')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--survey', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--depth', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.survey, args.images, args.depth, args.out)))
