"""Measure retained detail groups in the final mesh, without fitting or selection."""

import hashlib

import numpy as np
from room3d_preview import render
from room3d_shape_fit import compare


def measure(room, recipe, segments, depths, mesh):
    """Keep rejected fit diagnostics separate from the geometry actually exported."""
    views = list(zip(room['cameras'], segments, depths, strict=True))
    if len({camera['index'] for camera, _, _ in views}) != len(views):
        raise ValueError('Expected distinct camera indices')
    references = {}
    scenes = {}
    for camera, entries, (target, valid) in views:
        index = camera['index']
        if target.shape != (240, 320) or valid.shape != target.shape:
            raise ValueError('Expected native depth and confidence dimensions')
        for item, mask in entries:
            key = (index, item['mask'])
            if key in references or mask.shape != target.shape:
                raise ValueError('Expected distinct image-aligned observation masks')
            references[key] = mask
        scenes[index] = render(mesh, camera, return_depth=True)[2]
    objects = []
    for index, primitive in enumerate(recipe.get('primitives', [])):
        observations = primitive.get('evidence', {}).get('shared_observations', [])
        if primitive.get('omit_geometry') or not primitive.get('shape') or not observations:
            continue
        group = f'primitive_{index}'
        wanted = {(obs['camera'], obs['mask']) for obs in observations}
        missing = sorted(wanted-references.keys())
        measurements = []
        for camera, _, (target, valid) in views:
            keys = sorted(key for key in wanted & references.keys() if key[0] == camera['index'])
            if not keys:
                continue
            mask = np.logical_or.reduce([references[key] for key in keys])
            part = render(mesh, camera, return_depth=True, only_groups={group})[2]
            measurements.append({'camera': camera['index'],
                'masks': [{'name': key[1], 'pixels_sha256': hashlib.sha256(references[key].tobytes()).hexdigest()}
                          for key in keys],
                'scores': compare(part, scenes[camera['index']], mask, target, valid)})
        objects.append({'group': group, 'kind': primitive['kind'],
            'status': 'missing_observations' if missing else 'measured_final_geometry',
            'missing_observations': [{'camera': camera, 'mask': mask} for camera, mask in missing],
            'views': measurements})
    return {'schema': 1, 'mesh_sha256': hashlib.sha256(mesh.read_bytes()).hexdigest(),
        'visibility_sha256': hashlib.sha256(mesh.with_suffix('.cameras').read_bytes()).hexdigest()
            if mesh.with_suffix('.cameras').exists() else None,
        'objects': objects,
        'qualification': 'Post-selection measurement of actual final mesh groups after cleanup; no optimization or candidate reselection. Source-mask agreement is not fidelity. Reserved-view status remains in original fit reports; hidden geometry and navigation unverified.'}
